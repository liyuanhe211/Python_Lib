# -*- coding: utf-8 -*-
"""
Lit_Retrieval_2_Metadata_Completion.py — 数据库检索补全文献元数据

══════════════════════════════════════════════════════════════════════════════
  文献检索流水线第 2 步：对第 1 步解析出的每个条目查询 CrossRef REST API，
  补全缺失信息——最重要的是 DOI，同时补全标题、完整作者列表、期刊全名与
  缩写（short-container-title）、卷、期、页码 / 文章号、年份。

  检索与校验逻辑移植自 Zotero_MultiFetcher 插件 src/modules/CrossRefResolver.ts：

  - 条目已带 DOI（语言模型提取）→ 直接按 DOI 取 CrossRef 记录并回填；
    若 CrossRef 查无此 DOI，判为幻觉 / 笔误，标记 doi_not_in_crossref 并
    取消选中（不参与后续下载）。**例外是 JSTOR 的 `10.2307/<数字>` 标识符**：
    JSTOR 只为一部分内容注册过 DOI，旧刊往往没注册，查不到 CrossRef 记录
    是常态而非异常，而影子图书馆照样按这个标识符收录，所以不取消选中；
  - 条目无 DOI → 先用结构化查询（query.author / query.container-title /
    query.bibliographic + 年份过滤），不理想再退回自由文本查询；
  - 候选按 标题（权重最高）/ 年份 / 卷 / 第一作者姓（含 Levenshtein 容错）/
    页码或文章号 / 期刊名 Jaccard 相似度 打分，得分 ≥ 0.5 才采纳；引文与
    候选都有像样标题、而标题相似度过低时直接否决该候选（见下）；
  - 采纳后做事后校验（标题 / 年份 / 卷 / 文章号 / 页码 / 作者 / 期刊），
    问题记入 crossref_issues 供人工复核。

  关于标题参与打分（2026-07-26 修复）：
  原先的打分完全不看标题，且最终得分是「各项得分之和 ÷ 可比字段数」。
  当候选记录字段极少时（例如技术报告只有年份和作者、没有期刊 / 卷 / 页码），
  可比字段数会小到 1~2 个，个别字段的偶然吻合就被放大成高分。实际发生过的
  误匹配：引文「Loop, M. S. 1974. …. Herpetologica 30:123–127.」被匹配到
  10.2172/4327023（美国能源部的《Thermoelectric size effect in noble metals》
  进度报告）——仅凭年份同为 1974 就拿到 0.500 分，正好卡在采纳门槛上，而两者
  标题相似度只有 0.083。因此现在把标题作为权重为 2 的打分因子，并加上标题
  否决与最低可比字段数两道闸。

  产出：
  - 更新 Lit_Retrieval_State.json；
  - 按 ACS 格式列出补全后的全部引文，供人工核对；
  - 为每个条目各写一份 RIS（一个文献一份，不是合并成一个 References.ris），
    落在 <workdir>/<该文献将来的 PDF 文件名>.ris。这份 RIS 含期刊缩写 J2，
    既可直接导入 Zotero，也供 Lit_Retrieval_4_Rename_Ref.py 免语言模型直接
    改名；下载并改名成功后由第 4 步迁移到「PDF texts/<主干>/」，下载失败的
    则留在原地，供将来重跑时免去重新解析与检索。

用法：
    # 交互模式
    python -m Tools.Lit_Retrieval_2_Metadata_Completion

    # 自动化模式
    python -m Tools.Lit_Retrieval_2_Metadata_Completion ^
        --workdir "E:\\My_Program\\Knowledge_Base_Chemistry\\0 New Download"
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import concurrent.futures
import threading
import time
from pathlib import Path
from urllib.parse import quote

import requests

from Tools.Lit_Retrieval_Common import (
    BROWSER_USER_AGENT,
    CONTACT_EMAIL,
    ask_workdir,
    format_acs_citation,
    is_jstor_identifier,
    jaccard_similarity,
    levenshtein,
    load_state,
    normalize_doi,
    ref_short_label,
    save_state,
    sync_pending_ris,
)

CROSSREF_WORKS_URL = "https://api.crossref.org/works"
USER_AGENT = f"ZoteroMultiFetcher-Python/1.0 (mailto:{CONTACT_EMAIL})"
REQUEST_TIMEOUT = 15
REQUEST_DELAY_S = 0.2

#: 同时向 CrossRef 发起检索的条目数（与参考插件 RESOLVE_CONCURRENCY 一致）
RESOLVE_CONCURRENCY = 10

#: 表明「非正文条目」的可疑标题（刊头、封面、目录等），移植自参考插件
_SUSPECT_TITLE_PATTERNS = [
    "masthead", "graphical abstract", "front cover", "back cover",
    "inside front cover", "inside back cover", "table of contents",
    "issue information", "frontispiece",
]

#: 标题在打分中的权重。标题是所有字段里区分度最高的一项，给它两票。
TITLE_WEIGHT = 2.0

#: 引文与候选都有「像样的标题」时，标题相似度低于此值直接否决该候选。
#: 语言模型解析出的标题可能被截断或有排版差异，所以门槛取得比较宽松，
#: 只用来挡住明显风马牛不相及的匹配。
TITLE_VETO_SIMILARITY = 0.3

#: 「像样的标题」指规范化后至少有这么多个有效词。太短的标题（如「Introduction」）
#: 相似度不可靠，不参与否决。
TITLE_VETO_MIN_WORDS = 4

#: 采纳候选所需的最低可比字段数。可比字段太少时，个别字段的偶然吻合会被
#: 「得分之和 ÷ 可比字段数」放大成高分。
MIN_COMPARABLE_FACTORS = 2


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CrossRef API 调用                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_work_by_doi(doi: str, session: requests.Session) -> "dict | None":
    """按 DOI 获取完整 CrossRef 记录；不存在或出错时返回 None。"""
    try:
        url = f"{CROSSREF_WORKS_URL}/{quote(doi, safe='')}?mailto={CONTACT_EMAIL}"
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            return None
        return response.json().get("message")
    except (requests.RequestException, ValueError):
        return None


def _search_crossref_url(url: str, ref: dict, session: requests.Session,
                         log=None) -> "dict | None":
    """请求一个 CrossRef 检索 URL，返回得分最高的候选（match 字典）。

    请求本身失败（限流、超时、服务端故障）与「查询成功但没有合适候选」是
    两回事，但返回值都是 None，所以前者要打一句日志说清楚——否则一次限流
    会静悄悄地让条目失去 DOI，事后完全看不出是网络问题还是真没收录。
    """
    def note(message: str) -> None:
        if log:
            log(f"  ⚠ {ref_short_label(ref)}：{message}")

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            note(f"CrossRef 检索请求失败（HTTP {response.status_code}）"
                 f"，本次未能取得候选"
                 + ("；这是限流，稍后重跑即可" if response.status_code == 429 else ""))
            return None
        items = response.json().get("message", {}).get("items") or []
    except (requests.RequestException, ValueError) as e:
        note(f"CrossRef 检索请求出错（{type(e).__name__}: {e}），本次未能取得候选")
        return None

    best_match, best_score = None, 0.0
    for work in items:
        match = extract_match(work)
        if _is_suspect_title(match["title"]):
            continue
        if title_vetoes_match(ref, match):
            continue
        match["match_score"] = score_match(ref, match)
        if match["match_score"] > best_score:
            best_score = match["match_score"]
            best_match = match
    return best_match


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  查询构建（移植自 buildSearchParams / buildSearchQuery）                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_search_params(ref: dict) -> "dict | None":
    """结构化查询参数：作者 / 期刊 / 其余书目信息分字段，精度更高。"""
    params: dict[str, str] = {"mailto": CONTACT_EMAIL, "rows": "20"}
    has_any = False

    names = ref.get("first_author_last_name") or []
    if names:
        params["query.author"] = str(names[0])
        has_any = True

    journal = ref.get("journal") or (
        (ref.get("journal_abbrev") or "").replace(".", "").replace(",", "") or None
    )
    if journal:
        params["query.container-title"] = journal
        has_any = True

    bib_parts: list[str] = []
    if ref.get("year"):
        bib_parts.append(str(ref["year"]))
    if ref.get("volume"):
        bib_parts.append(str(ref["volume"]))
    if ref.get("article_number"):
        bib_parts.append(str(ref["article_number"]))
    elif ref.get("pages"):
        first_page = str(ref["pages"]).replace("–", "-").split("-")[0].strip()
        if first_page:
            bib_parts.append(first_page)
    if ref.get("title"):
        bib_parts.append(str(ref["title"]))
    if bib_parts:
        params["query.bibliographic"] = " ".join(bib_parts)
        has_any = True

    return params if has_any else None


def build_search_query(ref: dict) -> str:
    """自由文本查询串（结构化查询效果不佳时的兜底方案）。"""
    parts: list[str] = []
    names = ref.get("first_author_last_name") or []
    if names:
        parts.append(str(names[0]))
    if ref.get("journal"):
        parts.append(str(ref["journal"]))
    elif ref.get("journal_abbrev"):
        parts.append(str(ref["journal_abbrev"]).replace(".", "").replace(",", ""))
    if ref.get("year"):
        parts.append(str(ref["year"]))
    if ref.get("volume"):
        parts.append(str(ref["volume"]))
    if ref.get("article_number"):
        parts.append(str(ref["article_number"]))
    elif ref.get("pages"):
        first_page = str(ref["pages"]).replace("–", "-").split("-")[0].strip()
        if first_page:
            parts.append(first_page)
    if ref.get("title"):
        parts.append(str(ref["title"]))
    return " ".join(p for p in parts if p)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  候选提取 / 打分 / 校验（移植自 extractCrossRefMatch 等）                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_match(work: dict) -> dict:
    """从 CrossRef work JSON 中提取本流水线关心的字段。"""
    date_parts = (
        (work.get("published-print") or {}).get("date-parts")
        or (work.get("published-online") or {}).get("date-parts")
        or (work.get("issued") or {}).get("date-parts")
        or [[None]]
    )
    return {
        "doi": work.get("DOI"),
        "title": (work.get("title") or [None])[0],
        "authors": work.get("author") or [],
        "journal": (work.get("container-title") or [None])[0],
        "journal_abbrev": (work.get("short-container-title") or [None])[0],
        "publisher": work.get("publisher"),
        "volume": work.get("volume"),
        "issue": work.get("issue"),
        "pages": work.get("page"),
        "article_number": work.get("article-number"),
        "year": date_parts[0][0] if date_parts and date_parts[0] else None,
        "match_score": 0.0,
        "verification_issues": [],
    }


def _is_suspect_title(title: "str | None") -> bool:
    if not title:
        return False
    lower = title.lower()
    return any(p in lower for p in _SUSPECT_TITLE_PATTERNS)


def _first_page(pages: str) -> str:
    return str(pages).replace("–", "-").split("-")[0].strip()


def _initials(name: str) -> str:
    import re
    return "".join(
        part[0].upper() for part in re.split(r"[\s.\-]+", name) if part
    )


def _significant_words(title: str) -> set:
    """标题里长度大于 1 的词（与 jaccard_similarity 的切词口径一致）。"""
    import re
    return {w for w in re.sub(r"[^\w\s]", "", title or "").split() if len(w) > 1}


def title_similarity(ref: dict, match: dict) -> "float | None":
    """引文标题与候选标题的 Jaccard 相似度；任一方没有标题时返回 None。"""
    ref_title, match_title = ref.get("title"), match.get("title")
    if not ref_title or not match_title:
        return None
    return jaccard_similarity(str(ref_title).lower(), str(match_title).lower())


def title_vetoes_match(ref: dict, match: dict) -> bool:
    """标题是否明确否决这个候选。

    只在双方标题都「像样」（规范化后有足够多的有效词）时才生效——语言模型
    解析出的标题可能被截断，但截断后仍会与真标题共享大量词，所以门槛设得
    很宽松，只用来挡住完全不相干的匹配。
    """
    similarity = title_similarity(ref, match)
    if similarity is None:
        return False
    if len(_significant_words(str(ref["title"]))) < TITLE_VETO_MIN_WORDS:
        return False
    if len(_significant_words(str(match["title"]))) < TITLE_VETO_MIN_WORDS:
        return False
    return similarity < TITLE_VETO_SIMILARITY


def score_match(ref: dict, match: dict) -> float:
    """按 标题 / 年份 / 卷 / 第一作者姓 / 页码或文章号 / 期刊名 打分（0~1）。"""
    score, factors = 0.0, 0

    similarity = title_similarity(ref, match)
    if similarity is not None:
        factors += TITLE_WEIGHT
        score += similarity * TITLE_WEIGHT

    if ref.get("year") and match.get("year"):
        factors += 1
        if int(ref["year"]) == int(match["year"]):
            score += 1
        elif abs(int(ref["year"]) - int(match["year"])) == 1:
            score += 0.5

    if ref.get("volume") and match.get("volume"):
        factors += 1
        if str(ref["volume"]) == str(match["volume"]):
            score += 1

    names = ref.get("first_author_last_name") or []
    if names and match.get("authors"):
        factors += 1
        ref_author = str(names[0]).lower()
        match_author = (match["authors"][0].get("family") or "").lower()
        if ref_author == match_author:
            score += 1
        elif levenshtein(ref_author, match_author) <= 2:
            score += 0.7

    if ref.get("article_number") and match.get("article_number"):
        factors += 1
        if str(ref["article_number"]) == str(match["article_number"]):
            score += 1
    elif ref.get("pages") and match.get("pages"):
        factors += 1
        if _first_page(ref["pages"]) == _first_page(match["pages"]):
            score += 1

    if ref.get("journal") and match.get("journal"):
        factors += 1
        score += jaccard_similarity(
            str(ref["journal"]).lower(), str(match["journal"]).lower()
        )

    # 可比字段太少时，「得分之和 ÷ 可比字段数」会把个别字段的偶然吻合放大成
    # 高分（技术报告之类字段稀疏的记录最容易这样蒙混过关），一律判为不可信。
    if factors < MIN_COMPARABLE_FACTORS:
        return 0.0
    return score / factors if factors else 0.0


def verify_match(ref: dict, match: dict) -> list[str]:
    """事后校验采纳的候选，返回问题列表（空列表 = 全部通过）。"""
    issues: list[str] = []

    similarity = title_similarity(ref, match)
    if similarity is not None and similarity < 0.5:
        issues.append("title_mismatch")

    if ref.get("year") and match.get("year") and int(ref["year"]) != int(match["year"]):
        issues.append("year_mismatch")

    if ref.get("volume") and match.get("volume") \
            and str(ref["volume"]) != str(match["volume"]):
        issues.append("volume_mismatch")

    if ref.get("article_number") and match.get("article_number"):
        ref_num = str(ref["article_number"]).lstrip("eE").strip()
        match_num = str(match["article_number"]).lstrip("eE").strip()
        if ref_num != match_num:
            issues.append("article_number_mismatch")

    if ref.get("pages") and match.get("pages"):
        if _first_page(ref["pages"]) != _first_page(match["pages"]):
            issues.append("pages_mismatch")

    authors = ref.get("authors") or []
    if authors and match.get("authors"):
        ref_initials = _initials(str(authors[0]))
        first = match["authors"][0]
        if first.get("given"):
            match_initials = _initials(f"{first['given']} {first.get('family', '')}")
        else:
            match_initials = _initials(first.get("family") or "")
        if ref_initials and match_initials and ref_initials != match_initials:
            names = ref.get("first_author_last_name") or []
            ref_last = str(names[0]).lower() if names else ""
            match_last = (first.get("family") or "").lower()
            if ref_last and match_last and ref_last != match_last:
                issues.append("author_mismatch")

    if ref.get("journal") and match.get("journal"):
        similarity = jaccard_similarity(
            str(ref["journal"]).lower(), str(match["journal"]).lower()
        )
        if similarity < 0.3:
            issues.append("journal_mismatch")

    return issues


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  回填（移植自 enrichFromCrossRef）                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _authors_to_display(authors: list[dict]) -> list[str]:
    """CrossRef 结构化作者 → 「首字母. 姓」显示形式（与参考插件一致）。"""
    import re
    display: list[str] = []
    for author in authors:
        given, family = author.get("given"), author.get("family")
        if given and family:
            initials = " ".join(
                p[0].upper() + "." for p in re.split(r"[\s\-]+", given) if p
            )
            display.append(f"{initials} {family}")
        else:
            display.append(family or given or "")
    return display


def enrich_from_match(ref: dict, match: dict) -> None:
    """用 CrossRef 已验证数据覆盖语言模型解析出的字段。"""
    if match.get("title"):
        ref["title"] = match["title"]
        ref["title_source"] = "crossref"
    if match.get("journal"):
        ref["journal"] = match["journal"]
    if match.get("journal_abbrev"):
        ref["journal_abbrev"] = match["journal_abbrev"]
    if match.get("publisher") and not ref.get("publisher"):
        ref["publisher"] = match["publisher"]
    for key in ("volume", "issue", "pages", "article_number", "year"):
        if match.get(key):
            ref[key] = match[key]
    if match.get("authors"):
        ref["authors_structured"] = [
            {"family": a.get("family") or "", "given": a.get("given") or ""}
            for a in match["authors"]
        ]
        ref["authors"] = _authors_to_display(match["authors"])
        if not ref.get("first_author_last_name"):
            family = match["authors"][0].get("family")
            if family:
                ref["first_author_last_name"] = [family]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  单条目补全（移植自 resolveSingleReference）                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def resolve_single_reference(ref: dict, session: requests.Session,
                             log=print) -> None:
    """就地补全一个条目：已有 DOI 则取记录回填；否则检索发现 DOI。"""
    label = ref_short_label(ref)

    if ref.get("doi"):
        ref["doi"] = normalize_doi(ref["doi"])
        if not ref.get("doi_source"):
            ref["doi_source"] = "llm"
        work = fetch_work_by_doi(ref["doi"], session)
        if work:
            match = extract_match(work)
            enrich_from_match(ref, match)
            ref["crossref_issues"] = []
            log(f"  ✅ {label}：DOI 已验证并回填元数据（{ref['doi']}）")
        elif is_jstor_identifier(ref["doi"]):
            # JSTOR 的 10.2307/<stable ID> 标识符查不到 CrossRef 记录是常态：
            # JSTOR 只为一部分内容注册过 DOI，旧刊往往没注册。但影子图书馆
            # 一律按这个标识符收录，照样下得到，所以既不标记可疑也不取消选中。
            log(f"  ℹ️ {label}：{ref['doi']} 是 JSTOR 标识符，CrossRef 没有它的"
                f"记录属于正常情况，保留该条目（元数据沿用解析结果）。")
        elif ref["doi_source"] == "llm":
            # 语言模型提取的 DOI 在 CrossRef 查无记录——大概率是幻觉或笔误，
            # 标记后取消选中（用户提供的 DOI 不做此检查）
            issues = ref.setdefault("crossref_issues", [])
            if "doi_not_in_crossref" not in issues:
                issues.append("doi_not_in_crossref")
            ref["selected"] = False
            log(f"  ⚠ {label}：语言模型提取的 DOI 在 CrossRef 中不存在，"
                f"已标记为可疑并取消选中（{ref['doi']}）")
        ref["status"]["resolved"] = True
        return

    params = build_search_params(ref)
    query = build_search_query(ref)
    if not params and not query:
        log(f"  ⚠ {label}：可用书目信息不足，无法检索 DOI。")
        ref["status"]["resolved"] = True
        return

    year_filter = ""
    if ref.get("year"):
        year = int(ref["year"])
        year_filter = f"&filter=from-pub-date:{year - 1},until-pub-date:{year + 1}"

    best_match = None
    best_score = 0.0

    if params:
        query_string = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
        url = f"{CROSSREF_WORKS_URL}?{query_string}{year_filter}"
        result = _search_crossref_url(url, ref, session, log=log)
        if result and result["match_score"] > best_score:
            best_match, best_score = result, result["match_score"]

    if best_score < 0.5 and query:
        time.sleep(REQUEST_DELAY_S)
        url = (f"{CROSSREF_WORKS_URL}?query={quote(query)}"
               f"&rows=20&mailto={CONTACT_EMAIL}{year_filter}")
        result = _search_crossref_url(url, ref, session, log=log)
        if result and result["match_score"] > best_score:
            best_match, best_score = result, result["match_score"]

    if best_match and best_score >= 0.5:
        best_match["verification_issues"] = verify_match(ref, best_match)
        ref["doi"] = normalize_doi(best_match["doi"])
        ref["doi_source"] = "crossref"
        ref["crossref_issues"] = best_match["verification_issues"]
        enrich_from_match(ref, best_match)
        issue_note = (
            f"，校验发现问题：{', '.join(ref['crossref_issues'])}"
            if ref["crossref_issues"] else ""
        )
        log(f"  ✅ {label}：已通过 CrossRef 找到 DOI（{ref['doi']}，"
            f"匹配得分 {best_score:.2f}）{issue_note}")
    else:
        log(f"  ❌ {label}：CrossRef 未找到足够可信的匹配"
            f"（最高得分 {best_score:.2f}）。")

    ref["status"]["resolved"] = True


def resolve_references(refs: list[dict], concurrency: int = RESOLVE_CONCURRENCY,
                       log=print) -> None:
    """有界并发地补全一组条目（就地修改）。"""
    todo = [r for r in refs if not r["status"].get("resolved")]
    if not todo:
        log("  ℹ️ 所有条目此前已完成补全，无需重复检索。")
        return

    log(f"  🔍 开始 CrossRef 检索：共 {len(todo)} 条，最多 {concurrency} 条并发……")
    session = _session()
    progress_lock = threading.Lock()
    completed = 0

    def worker(ref: dict) -> None:
        nonlocal completed
        lines: list[str] = []
        try:
            resolve_single_reference(ref, session, log=lines.append)
        except Exception as e:
            lines.append(f"  ❌ {ref_short_label(ref)}：检索过程出错：{e}")
        with progress_lock:
            completed += 1
            for line in lines:
                log(line)
            log(f"     进度：{completed}/{len(todo)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, r) for r in todo]
        for future in concurrent.futures.as_completed(futures):
            future.result()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  主流程                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def print_acs_citation_list(refs: list[dict], log=print) -> None:
    """按 ACS 格式列出补全后的全部引文，供人工核对识别是否正确。"""
    log(f"\n  {'─' * 58}")
    log(f"  📋 补全后的引文列表（ACS 格式），共 {len(refs)} 条：")
    width = len(str(len(refs)))
    for seq, ref in enumerate(refs, 1):
        log(f"\n  ({seq:>{width}}) {format_acs_citation(ref)}")
        notes: list[str] = []
        if not ref.get("doi"):
            notes.append("未找到 DOI")
        if ref.get("crossref_issues"):
            notes.append(f"校验问题：{', '.join(ref['crossref_issues'])}")
        if not ref.get("selected", True):
            notes.append("已取消选中，不参与下载")
        if notes:
            log(f"  {' ' * (width + 3)}⚠ {'；'.join(notes)}")


def run(workdir: Path, concurrency: int = RESOLVE_CONCURRENCY,
        log=print) -> "list[dict] | None":
    """对状态文件中的条目做补全并写出 RIS；返回条目列表。"""
    state = load_state(workdir)
    if state is None or not state.get("references"):
        log(f"  ❌ 未在 {workdir} 找到解析结果。请先运行第 1 步"
            f"（Lit_Retrieval_1_Reference_List_Parsing）。")
        return None

    refs = state["references"]
    resolve_references(refs, concurrency=concurrency, log=log)

    resolved_with_doi = sum(1 for r in refs if r.get("doi") and r.get("selected", True))
    log(f"\n  📊 补全结果：{len(refs)} 条中 {resolved_with_doi} 条已确定 DOI。")

    print_acs_citation_list(refs, log=log)

    written, removed = sync_pending_ris(workdir, refs, log=log)
    save_state(workdir, state)
    log(f"\n  💾 已为 {written} 个条目各写出一份 RIS（文件名与该文献将来的 PDF "
        f"文件名一致）：{workdir}")
    if removed:
        log(f"  🧹 顺带清理了 {removed} 份不再需要的占位 RIS。")

    problem_refs = [r for r in refs if r.get("crossref_issues")]
    if problem_refs:
        log(f"\n  ⚠ 以下 {len(problem_refs)} 条存在校验问题，建议人工复核：")
        for ref in problem_refs:
            log(f"     {ref_short_label(ref)}：{', '.join(ref['crossref_issues'])}")
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="文献检索流水线第 2 步：CrossRef 补全 DOI 与元数据并输出 RIS。"
    )
    parser.add_argument("--workdir", help="文献存储目录（含状态文件）")
    parser.add_argument("--concurrency", type=int, default=RESOLVE_CONCURRENCY,
                        help=f"并发检索数（默认 {RESOLVE_CONCURRENCY}）")
    args = parser.parse_args()

    print("═" * 62)
    print("  🔍 文献检索流水线 · 第 2 步 · 元数据补全（CrossRef）")
    print("═" * 62)

    workdir = Path(args.workdir).resolve() if args.workdir else ask_workdir()

    if load_state(workdir) is None:
        # 交互模式下状态文件缺失：当场引导用户完成第 1 步
        print("  ℹ️ 该目录还没有解析结果。可以现在粘贴参考文献文本，"
              "先完成第 1 步解析。")
        try:
            answer = input("  是否现在输入文本进行解析？[Enter/y=是, n=退出] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in ("", "y", "yes"):
            from Tools.Lit_Retrieval_1_Reference_List_Parsing import run as parse_run
            from Tools.Lit_Retrieval_Common import read_multiline_until_end
            text = read_multiline_until_end("\n  请粘贴参考文献列表文本：")
            if not text.strip():
                print("  ⚠ 输入为空，退出。")
                return
            parse_run(workdir, text)
        else:
            return

    run(workdir, concurrency=args.concurrency)
    print("\n  👋 第 2 步完成。下一步：运行 Lit_Retrieval_3_Download_PDF 下载 PDF。")


if __name__ == "__main__":
    main()
