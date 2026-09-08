# -*- coding: utf-8 -*-
"""
Lit_Retrieval_Common.py — 文献检索流水线的共享基础设施

══════════════════════════════════════════════════════════════════════════════
  被 Lit_Retrieval_0_1/2/3 与总驱动脚本 Lit_Retrieval_0.py 共用：

  - reference 字典的字段规范与工厂函数（对应 Zotero_MultiFetcher 插件的
    ParsedReference 结构，见参考项目 src/modules/ReferenceTypes.ts）
  - 流水线状态文件 Lit_Retrieval_State.json 的读写
  - DOI 正则与规范化（移植自参考项目 src/utils/identifierPatterns.ts）
  - RIS 的生成与解析（供导入 Zotero 等文献管理软件，也供 Lit_Retrieval_4_Rename_Ref.py
    免语言模型直接改名）
  - ACS 格式引文的生成（第 2 步补全完成后向用户展示识别结果）
  - 标准命名文件名（Lit_Retrieval_4_Rename_Ref.py 产出格式）的解析与库内查重
  - 字符串相似度 Jaccard / Levenshtein（移植自参考项目
    src/modules/CrossRefResolver.ts）
  - 终端多行输入（某一行输入 'end' 时停止）

  目录契约与 Lit_Retrieval_5_PDF_to_Markdown.py / RAG_Lib/Docling.py 保持一致：
      <workdir>/<PDF 文件>
      <workdir>/<待下载条目的标准名>.ris              占位 RIS（见下）
      <workdir>/PDF texts/<stem[:80]>/<stem[:80]>.ris 单篇 RIS（改名后归档）
      <workdir>/PDF texts/<stem[:80]>/<stem[:80]>.md  Lit_Retrieval_5_PDF_to_Markdown 产出

  RIS 一律「一个文献一份」，不再产出合并的 References.ris。每份 RIS 的
  文件名与该文献的 PDF 文件名一致（都由 expected_pdf_filename 决定），
  存放位置随下载状态迁移：

      尚未拿到 PDF  → <workdir>/<标准名>.ris（占位 RIS，供将来重试下载时
                      免去重新解析与 CrossRef 检索）
      已下载并改名  → <workdir>/PDF texts/<stem[:80]>/<stem[:80]>.ris

  这一迁移由 sync_pending_ris 负责，在第 2、3、4 步结束时各调用一次。
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# 本流水线的进度输出含表情符号；在 GBK 等旧编码控制台上让无法编码的字符
# 退化为替换符而不是抛 UnicodeEncodeError 使脚本崩溃
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

# 文本化输出契约（"PDF texts" 目录名、文件夹名截断规则）唯一定义在
# RAG_Lib/Docling.py，此处 import 复用，保证与 Lit_Retrieval_5_PDF_to_Markdown / RAG 流水线互认
from LLM_Lib.RAG_Lib.Docling import OUTPUT_PARENT_NAME, truncated_folder_name

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  常量                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

STATE_FILENAME = "Lit_Retrieval_State.json"

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

#: 对外 API（CrossRef / Unpaywall / OpenAlex）的公开联系邮箱，
#: 与参考插件 Zotero_MultiFetcher 使用的占位邮箱一致（勿填个人邮箱）
CONTACT_EMAIL = "zotero-multifetcher@github.com"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  reference 字典规范                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_STATUS_DEFAULTS: dict = {
    "resolved": False,            # 步骤 2：是否已做过 CrossRef 补全
    "ris_written": False,         # 步骤 2：是否已写出该条目的 RIS
    "pending_ris_path": None,     # 尚未拿到 PDF 时，工作目录下的占位 RIS 路径
    "download": "pending",        # 步骤 3：pending / success / already_exists / failed / skipped
    "download_source": None,      # 成功来源（如 "sci-hub.su" / "OpenAlex"）
    "pdf_path": None,             # 下载得到的 PDF 路径（临时名或改名后）
    "existing_path": None,        # 查重命中的既有 PDF 路径
    "renamed": False,             # 步骤 4：是否已标准化改名
    "renamed_path": None,
    "ris_sidecar_path": None,     # 单篇 RIS 最终归档位置
    "markdown_done": False,       # 步骤 5：Lit_Retrieval_5_PDF_to_Markdown 是否完成
    "error": None,
}

_REFERENCE_DEFAULTS: dict = {
    "id": "",
    "index": None,
    "sub_index": None,
    "first_author_last_name": [],
    "authors": [],
    "authors_structured": [],     # [{"family": ..., "given": ...}]，来自 CrossRef
    "title": None,
    "journal": None,
    "journal_abbrev": None,
    "year": None,
    "volume": None,
    "issue": None,
    "pages": None,
    "article_number": None,
    "doi": None,
    "doi_source": None,           # "llm" / "crossref" / "user"
    "title_source": None,
    "url": None,
    "publisher": None,            # 书籍条目使用
    "confidence": 0.0,
    "item_type": "journalArticle",
    "raw_text_start": None,
    "raw_text_end": None,
    "selected": True,
    "crossref_issues": [],
}


def new_reference(**kwargs) -> dict:
    """创建一个字段齐全的 reference 字典（未给出的字段取默认值）。"""
    ref = copy.deepcopy(_REFERENCE_DEFAULTS)
    ref["status"] = copy.deepcopy(_STATUS_DEFAULTS)
    for key, value in kwargs.items():
        ref[key] = value
    return ref


def normalize_reference(ref: dict) -> dict:
    """补齐缺失字段（就地修改并返回），保证下游代码可以放心取值。"""
    for key, value in _REFERENCE_DEFAULTS.items():
        ref.setdefault(key, copy.deepcopy(value))
    status = ref.setdefault("status", {})
    for key, value in _STATUS_DEFAULTS.items():
        status.setdefault(key, copy.deepcopy(value))
    if ref.get("doi"):
        ref["doi"] = normalize_doi(ref["doi"])
    return ref


def is_book_reference(ref: dict) -> bool:
    """书籍类条目（book / bookSection）走标题搜索而非 DOI 检索。"""
    return ref.get("item_type") in ("book", "bookSection")


def ref_short_label(ref: dict) -> str:
    """用于进度输出的简短标签，如「[62] Deng 2024」。"""
    parts: list[str] = []
    if ref.get("index"):
        sub = ref.get("sub_index") or ""
        parts.append(f"[{ref['index']}{sub}]")
    names = ref.get("first_author_last_name") or []
    if names:
        parts.append(str(names[0]))
    elif ref.get("authors"):
        parts.append(str(ref["authors"][0]))
    if ref.get("year"):
        parts.append(str(ref["year"]))
    return " ".join(parts) if parts else (ref.get("doi") or ref.get("id") or "(未知条目)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DOI 正则与规范化（移植自 identifierPatterns.ts）                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

#: DOI 提取正则。后缀允许出现括号 / 方括号 / 尖括号——例如 BioOne 系期刊用的
#: `10.1670/0022-1511(2002)036[0116:dojvns]2.0.co;2`——末尾多余的标点交给
#: normalize_doi 去掉。
#:
#: 2026-07-26 由原先的五条正则合并成这一条。原先那套有三个缺陷，都会实际
#: 造成下载失败：
#:   1. `10.1111/j.1365-2656.2000.00477.x` 会被同时提取成完整版和截断掉
#:      `.x` 的版本两条，粘贴一个 DOI 凭空变成两个条目，截断的那个必然扑空；
#:   2. 写成 `https://doi.org/10.1111/j.1365-2656.2000.00477.x` 时**只**得到
#:      截断版，完整 DOI 根本提取不出来（末尾锚定的那条正则要求整行只有 DOI）；
#:   3. 带括号 / 方括号的 DOI 一条也匹配不上。
DOI_REGEXP: re.Pattern = re.compile(
    r'(10\.\d{4,9}/[-._;()<>\[\]:/A-Za-z0-9]+)'
)

#: 兼容早先按列表遍历的写法
DOI_REGEXPS: list[re.Pattern] = [DOI_REGEXP]


#: JSTOR 的文章链接，形如 https://www.jstor.org/stable/3892027?seq=1
#: （也认 /stable/pdf/<id> 这种直接指向 PDF 的形式）
JSTOR_STABLE_REGEXP = re.compile(
    r'(?:https?://)?(?:www\.)?jstor\.org/stable/(?:pdf/)?(\d{4,12})',
    re.IGNORECASE,
)


#: 形如 DOI 的 JSTOR 标识符：前缀固定是 10.2307，后缀是纯数字的 stable ID
JSTOR_DOI_REGEXP = re.compile(r'^10\.2307/\d+$', re.IGNORECASE)


def jstor_stable_id_to_identifier(stable_id: str) -> str:
    """JSTOR 的 stable ID 换算成 `10.2307/<stable ID>` 形式的标识符。

    **注意：换出来的东西不一定是注册过的 DOI。** JSTOR 确实为它的很大一部分
    内容注册过 `10.2307/<stable ID>` 形式的 DOI，但并非全部。实测对比：

    - `10.2307/1563325` 是**真的注册过**的 DOI——注册机构是 Crossref，
      doi.org 能解析（302 重定向到 `jstor.org/stable/1563325?origin=crossref`），
      内容协商能取到元数据；
    - `10.2307/3892027`（Herpetologica 1974 年那篇孟加拉巨蜥的文章）**没有
      注册过**——doi.org、CrossRef、DataCite 全部返回 404，DOI 系统的注册
      机构查询接口直接回答 `"status": "DOI does not exist"`。

    但影子图书馆（Sci-Hub / LibGen / Anna's Archive）一律按这个拼出来的
    标识符收录 JSTOR 的内容——Anna's Archive 的 SciDB 页面上就明写着
    「DOI: 10.2307/3892027」——所以**拿它去下载是可行的，拿它去查元数据
    则可能一无所获**。凡是按 CrossRef 有无记录来判断条目真伪的地方，都要
    对这类标识符网开一面（见 is_jstor_identifier）。
    """
    return f"10.2307/{stable_id.strip()}"


def is_jstor_identifier(doi: str) -> bool:
    """判断是不是 `10.2307/<数字>` 这种 JSTOR 标识符。

    这类标识符查不到 CrossRef 记录是常态而非异常，因此不能据此判定它是
    语言模型编造的、更不能把条目取消选中——它照样能从各下载渠道拿到 PDF。
    """
    return bool(JSTOR_DOI_REGEXP.match((doi or "").strip()))


def match_dois(text: str) -> list[str]:
    """从任意文本中提取全部 DOI（去重，保持出现顺序）。

    JSTOR 的文章链接会被换算成对应的 `10.2307/<stable ID>` 一并返回，
    这样粘贴一串 JSTOR 链接与粘贴一串 DOI 效果一样。注意换出来的标识符
    不一定是注册过的 DOI，详见 jstor_stable_id_to_identifier。
    """
    results: list[str] = []
    for match in DOI_REGEXP.finditer(text):
        doi = normalize_doi(match.group(1))
        if doi and doi not in results:
            results.append(doi)
    for match in JSTOR_STABLE_REGEXP.finditer(text):
        doi = jstor_stable_id_to_identifier(match.group(1))
        if doi not in results:
            results.append(doi)
    return results


def normalize_doi(doi: str) -> str:
    """规范化 DOI：去掉 URL 前缀 / "doi:" 前缀 / 首尾空白与结尾标点。"""
    doi = (doi or "").strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi.rstrip(".,;\"'）)")


def is_pure_doi_input(text: str) -> bool:
    """判断输入是否为「纯 DOI 列表」（每个非空行都只是一个 DOI 或 JSTOR 链接）。

    与参考插件 LLMImportDialog 的 isPureDOIInput 行为一致：纯 DOI 输入
    直接构建条目，完全跳过语言模型解析。JSTOR 的文章链接同等对待——
    它能一一换算成 `10.2307/<stable ID>`，没有任何需要语言模型解析的地方。
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    bare = re.compile(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?10\.\d{4,15}/\S+$',
                      re.IGNORECASE)
    jstor = re.compile(r'^(?:https?://)?(?:www\.)?jstor\.org/stable/(?:pdf/)?\d{4,12}'
                       r'(?:[/?#]\S*)?$', re.IGNORECASE)
    return all(bare.match(line) or jstor.match(line) for line in lines)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  状态文件                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def state_path(workdir: str | Path) -> Path:
    return Path(workdir) / STATE_FILENAME


def load_state(workdir: str | Path) -> dict | None:
    """读取状态文件；不存在或损坏时返回 None。"""
    path = state_path(workdir)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ⚠ 状态文件读取失败（{e}），将视为不存在：{path}")
        return None
    for ref in state.get("references", []):
        normalize_reference(ref)
    return state


def save_state(workdir: str | Path, state: dict) -> Path:
    """保存状态文件（自动更新时间戳）。"""
    path = state_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def new_state(workdir: str | Path) -> dict:
    return {
        "version": 1,
        "workdir": str(Path(workdir).resolve()),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "references": [],
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RIS 生成                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

RIS_TYPE_BY_ITEM_TYPE: dict[str, str] = {
    "journalArticle": "JOUR",
    "conferencePaper": "CONF",
    "book": "BOOK",
    "bookSection": "CHAP",
    "thesis": "THES",
    "patent": "PAT",
    "preprint": "UNPB",
    "report": "RPRT",
    "webpage": "ELEC",
}

ITEM_TYPE_BY_RIS_TYPE: dict[str, str] = {v: k for k, v in RIS_TYPE_BY_ITEM_TYPE.items()}


def _author_to_ris(author: "dict | str") -> str:
    """将作者转为 RIS 的「Family, Given」格式。

    - dict（CrossRef 结构化作者）：直接取 family / given；
    - 字符串（LLM 解析的 "S. Deng" 形式）：最后一个词视为姓，其余视为名。
    """
    if isinstance(author, dict):
        family = (author.get("family") or "").strip()
        given = (author.get("given") or "").strip()
        if family and given:
            return f"{family}, {given}"
        return family or given
    text = str(author).strip()
    if "," in text:
        return text  # 已是 Family, Given 格式
    tokens = text.split()
    if len(tokens) >= 2:
        return f"{tokens[-1]}, {' '.join(tokens[:-1])}"
    return text


def reference_to_ris(ref: dict) -> str:
    """将一个 reference 字典序列化为一条 RIS 记录（含结尾 ER 行）。"""
    lines: list[str] = []

    def tag(name: str, value) -> None:
        if value is None:
            return
        value = str(value).strip()
        if value:
            lines.append(f"{name}  - {value}")

    ris_type = RIS_TYPE_BY_ITEM_TYPE.get(ref.get("item_type") or "", "JOUR")
    lines.append(f"TY  - {ris_type}")
    tag("TI", ref.get("title"))

    authors = ref.get("authors_structured") or ref.get("authors") or []
    for author in authors:
        tag("AU", _author_to_ris(author))
    if not authors:
        for name in ref.get("first_author_last_name") or []:
            tag("AU", name)

    tag("T2", ref.get("journal"))
    tag("J2", ref.get("journal_abbrev"))
    tag("PY", ref.get("year"))
    tag("VL", ref.get("volume"))
    tag("IS", ref.get("issue"))

    pages = (ref.get("pages") or "").strip() if ref.get("pages") else ""
    if pages:
        parts = re.split(r"[-–—]", pages, maxsplit=1)
        tag("SP", parts[0].strip())
        if len(parts) > 1 and parts[1].strip():
            tag("EP", parts[1].strip())
    elif ref.get("article_number"):
        # 无页码时用文章号充当起始页（Zotero 导入后显示在 pages 字段）
        tag("SP", ref.get("article_number"))

    tag("PB", ref.get("publisher"))
    tag("DO", ref.get("doi"))
    tag("UR", ref.get("url") or (f"https://doi.org/{ref['doi']}" if ref.get("doi") else None))
    lines.append("ER  - ")
    return "\n".join(lines)


def write_sidecar_ris(pdf_path: str | Path, ref: dict) -> Path:
    """把单篇 RIS 写到该 PDF 对应的「PDF texts/<stem[:80]>/」子文件夹中。"""
    pdf_path = Path(pdf_path)
    base = truncated_folder_name(pdf_path.stem)
    out_dir = pdf_path.parent / OUTPUT_PARENT_NAME / base
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base}.ris"
    out_path.write_text(reference_to_ris(ref) + "\n", encoding="utf-8")
    return out_path


def _quiet_unlink(path: "Path | None") -> bool:
    """删除文件，不存在或删不掉时返回 False（不抛异常）。"""
    if path is None:
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def needs_pending_ris(ref: dict) -> bool:
    """该条目此刻是否应该在工作目录下留一份占位 RIS。

    「还没有拿到 PDF」的条目才需要——即已选中、没有在库中查到既有副本、
    且单篇 RIS 尚未归档到「PDF texts/」。下载失败的条目正好落在这一档，
    占位 RIS 会一直保留到某次重跑真的把 PDF 下载下来为止。
    """
    if not ref.get("selected", True):
        return False
    status = ref.get("status") or {}
    if status.get("download") == "already_exists":
        return False
    return not status.get("ris_sidecar_path")


def sync_pending_ris(workdir: str | Path, refs: list[dict],
                     log=print) -> "tuple[int, int]":
    """维护工作目录下的占位 RIS，使其与各条目当前的下载状态一致。

    对还没拿到 PDF 的条目写出 <workdir>/<标准名>.ris（文件名与该文献将来
    的 PDF 文件名一致）；对已归档单篇 RIS 或库中已存在的条目，删除残留的
    占位 RIS。元数据变化导致标准名改变时，旧文件一并清理。

    Returns:
        (写出的占位 RIS 数量, 清理掉的占位 RIS 数量)
    """
    workdir = Path(workdir)
    written = removed = 0
    for ref in refs:
        status = ref.setdefault("status", {})
        old_path = Path(status["pending_ris_path"]) if status.get("pending_ris_path") else None

        if not needs_pending_ris(ref):
            if old_path and _quiet_unlink(old_path):
                removed += 1
            status["pending_ris_path"] = None
            continue

        target = workdir / (Path(expected_pdf_filename(ref)).stem + ".ris")
        if old_path and old_path != target:
            _quiet_unlink(old_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(reference_to_ris(ref) + "\n", encoding="utf-8")
        except OSError as e:
            log(f"  ⚠ {ref_short_label(ref)}：占位 RIS 写入失败：{e}")
            continue
        status["pending_ris_path"] = str(target)
        status["ris_written"] = True
        written += 1
    return written, removed


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ACS 格式引文                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _given_names_to_initials(given: str) -> str:
    """名 → ACS 首字母形式：「Richard B.」→「R. B.」、「Wei-Hua」→「W.-H.」。"""
    chunks: list[str] = []
    for token in re.split(r"[\s.]+", given):
        if not token:
            continue
        pieces = [p for p in token.split("-") if p]
        if pieces:
            chunks.append("-".join(f"{p[0].upper()}." for p in pieces))
    return " ".join(chunks)


def _author_to_acs(author: "dict | str") -> str:
    """作者 → ACS 引文格式「King, R. B.」。"""
    if isinstance(author, dict):
        family = (author.get("family") or "").strip()
        given = (author.get("given") or "").strip()
    else:
        text = str(author).strip()
        if not text:
            return ""
        if "," in text:
            family, _, given = (part.strip() for part in text.partition(","))
        else:
            tokens = text.split()
            if len(tokens) < 2:
                return text
            family, given = tokens[-1], " ".join(tokens[:-1])
    initials = _given_names_to_initials(given)
    if family and initials:
        return f"{family}, {initials}"
    return family or initials


#: ACS 参考文献表中超过这个作者数就截断并加「et al.」
_ACS_MAX_AUTHORS = 10


def _end_with(text: str, punctuation: str) -> str:
    """给一段引文加上结尾标点；已经以标点收尾的（如「R. B.」「et al.」）不重复加。"""
    text = text.strip()
    if text.endswith(punctuation):
        return text
    return text[:-1] + punctuation if text.endswith(".") else text + punctuation


def format_acs_citation(ref: dict) -> str:
    """把一个条目排成一条 ACS 格式引文（纯文本，不带斜体 / 粗体标记）。

    期刊论文：
        King, R. B. Predicted and Observed Maximum Prey Size–Snake Size
        Allometry. Funct. Ecol. 2002, 16 (6), 766–772.
        https://doi.org/10.1046/j.1365-2435.2002.00678.x
    书籍：
        Author, A. B. Book Title; Publisher: 2002.
    """
    authors = ref.get("authors_structured") or ref.get("authors") or []
    names = [n for n in (_author_to_acs(a) for a in authors) if n]
    if not names:
        names = [str(n).strip() for n in (ref.get("first_author_last_name") or []) if n]
    if len(names) > _ACS_MAX_AUTHORS:
        author_part = "; ".join(names[:_ACS_MAX_AUTHORS]) + "; et al."
    else:
        author_part = "; ".join(names)

    title = (ref.get("title") or "").strip().rstrip(".")
    year = str(ref.get("year") or "").strip()
    doi = (ref.get("doi") or "").strip()

    parts: list[str] = []
    if author_part:
        parts.append(_end_with(author_part, "."))

    if is_book_reference(ref):
        publisher = (ref.get("publisher") or "").strip()
        if title:
            parts.append(_end_with(title, ";" if publisher or year else "."))
        if publisher:
            parts.append(f"{publisher}: {year}." if year else _end_with(publisher, "."))
        elif year:
            parts.append(f"{year}.")
    else:
        if title:
            parts.append(_end_with(title, "."))
        journal = (ref.get("journal_abbrev") or ref.get("journal") or "").strip()
        if journal:
            parts.append(_end_with(journal, "."))

        volume = str(ref.get("volume") or "").strip()
        issue = str(ref.get("issue") or "").strip()
        pages = (ref.get("pages") or "").strip()
        if pages:
            pages = re.sub(r"\s*[-–—]+\s*", "–", pages)
        elif ref.get("article_number"):
            pages = f"No. {ref['article_number']}"

        locator: list[str] = []
        if year:
            locator.append(year)
        if volume:
            locator.append(f"{volume} ({issue})" if issue else volume)
        if pages:
            locator.append(pages)
        if locator:
            parts.append(", ".join(locator) + ".")

    if doi:
        parts.append(f"https://doi.org/{doi}")
    return " ".join(parts) if parts else (ref.get("id") or "(信息不足，无法排版引文)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RIS 解析                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_RIS_TAG_RE = re.compile(r"^([A-Z][A-Z0-9])\s{0,2}-\s?(.*)$")


def parse_ris_text(text: str) -> list[dict[str, list[str]]]:
    """解析 RIS 文本为条目列表；每个条目是 {标签: [值, ...]} 的字典。"""
    entries: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    last_tag: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        match = _RIS_TAG_RE.match(line.strip())
        if match:
            tag_name, value = match.group(1), match.group(2).strip()
            if tag_name == "ER":
                if current:
                    entries.append(current)
                current = {}
                last_tag = None
                continue
            current.setdefault(tag_name, []).append(value)
            last_tag = tag_name
        elif line.strip() and last_tag:
            # 折行续接到上一个标签
            current[last_tag][-1] += " " + line.strip()
    if current:
        entries.append(current)
    return entries


def parse_ris_file(path: str | Path) -> list[dict[str, list[str]]]:
    return parse_ris_text(Path(path).read_text(encoding="utf-8-sig"))


def ris_first(entry: dict[str, list[str]], *tags: str) -> str:
    """按顺序返回第一个非空标签值；全部缺失时返回空字符串。"""
    for tag_name in tags:
        for value in entry.get(tag_name, []):
            if value.strip():
                return value.strip()
    return ""


def ris_year(entry: dict[str, list[str]]) -> str:
    """从 PY / DA / Y1 中提取 4 位年份。"""
    for tag_name in ("PY", "DA", "Y1"):
        for value in entry.get(tag_name, []):
            match = re.search(r"\d{4}", value)
            if match:
                return match.group(0)
    return ""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  字符串相似度（移植自 CrossRefResolver.ts）                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def jaccard_similarity(a: str, b: str) -> float:
    words_a = {w for w in re.sub(r"[^\w\s]", "", a).split() if len(w) > 1}
    words_b = {w for w in re.sub(r"[^\w\s]", "", b).split() if len(w) > 1}
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    return intersection / (len(words_a) + len(words_b) - intersection)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  标准命名文件名解析与库内查重                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

#: Lit_Retrieval_4_Rename_Ref.py 产出的标准文件名：{人名}「{Year} - {Journal}」{其余部分}
_RENAMED_STEM_RE = re.compile(r"^(?P<primary>[^「]*)「(?P<bracket>[^」]*)」(?P<rest>.*)$")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "by",
    "with", "via", "from", "at", "as", "its", "is", "are",
}


def _norm_tokens(text: str) -> set[str]:
    """归一化为小写词集合（去掉标点与常见虚词，用于标题匹配）。"""
    words = re.sub(r"[^\w\s]", " ", text or "").lower().split()
    return {w for w in words if len(w) > 1 and w not in _STOPWORDS}


def parse_renamed_filename(stem: str) -> dict | None:
    """解析标准命名文件名主干。

    返回 {"primary": 人名, "year": "2014" 或 None, "journal": 期刊缩写或 None,
          "rest": 「」后的全部文字}；不符合格式时返回 None。
    """
    match = _RENAMED_STEM_RE.match(stem)
    if not match:
        return None
    bracket = match.group("bracket").strip()
    year_match = re.search(r"\b(\d{4})\b", bracket)
    year = year_match.group(1) if year_match else None
    journal = None
    if "-" in bracket:
        journal = bracket.split("-", 1)[1].strip() or None
    elif year is None and bracket:
        journal = bracket
    return {
        "primary": match.group("primary").strip(),
        "year": year,
        "journal": journal,
        "rest": match.group("rest").strip(),
    }


@dataclass
class LibraryEntry:
    """库内一个既有 PDF 的预解析信息（用于查重）。"""
    path: Path
    stem_tokens: set[str] = field(default_factory=set)
    parsed: "dict | None" = None
    rest_tokens: set[str] = field(default_factory=set)
    journal_norm: str = ""


def build_library_index(library_root: str | Path,
                        log=print) -> list[LibraryEntry]:
    """递归扫描库根目录下所有 PDF 文件名，构建查重索引。

    可以假设既有文件已经过 Lit_Retrieval_4_Rename_Ref.py 标准命名；不符合标准命名的
    文件也会被索引（仅凭词集合匹配，不参与年份 / 期刊比对）。
    """
    library_root = Path(library_root)
    entries: list[LibraryEntry] = []
    if not library_root.is_dir():
        return entries
    for dirpath, dirnames, filenames in os.walk(library_root):
        # "PDF texts" 里只有转换产物，不会有待查重的 PDF，跳过以加速
        dirnames[:] = [d for d in dirnames if d != OUTPUT_PARENT_NAME]
        for filename in filenames:
            if not filename.lower().endswith(".pdf"):
                continue
            stem = filename[:-4]
            parsed = parse_renamed_filename(stem)
            entries.append(LibraryEntry(
                path=Path(dirpath) / filename,
                stem_tokens=_norm_tokens(stem),
                parsed=parsed,
                rest_tokens=_norm_tokens(parsed["rest"]) if parsed else set(),
                journal_norm=re.sub(r"[^a-z0-9]", "", (parsed["journal"] or "").lower())
                if parsed else "",
            ))
    log(f"  📚 已索引库内 PDF 文件 {len(entries)} 个（根目录：{library_root}）")
    return entries


def find_existing_pdf_for_reference(
    ref: dict, library_index: list[LibraryEntry],
) -> "Path | None":
    """在库内查找该条目是否已有对应 PDF（按文件名启发式匹配）。

    匹配条件（注意既有文件名中的标题可能被截断）：
      1. 第一作者（或通讯作者）姓氏出现在文件名词集合中——必要条件；
      2. 年份都已知时必须一致；
      3. 标题匹配：条目标题与文件名「」后的词集合重叠度 ≥ 0.5 且共同词 ≥ 3；
         或条目无标题时，期刊缩写归一化后一致（且年份一致）。
    """
    author_names: list[str] = []
    for name in (ref.get("first_author_last_name") or []):
        author_names.append(name)
    for author in (ref.get("authors_structured") or []):
        if isinstance(author, dict) and author.get("family"):
            author_names.append(author["family"])
    author_tokens = {t for name in author_names for t in _norm_tokens(name)}
    if not author_tokens:
        return None  # 无作者姓氏时不做文件名查重，避免误判

    year = str(ref.get("year")) if ref.get("year") else None
    title_tokens = _norm_tokens(ref.get("title") or "")
    abbrev_norm = re.sub(r"[^a-z0-9]", "", (ref.get("journal_abbrev") or "").lower())

    for entry in library_index:
        if not (author_tokens & entry.stem_tokens):
            continue
        if entry.parsed is not None and year and entry.parsed["year"] \
                and year != entry.parsed["year"]:
            continue

        title_ok = False
        compare_tokens = entry.rest_tokens or entry.stem_tokens
        if title_tokens and compare_tokens:
            common = title_tokens & compare_tokens
            # 分母取文件名一侧：文件名中的标题可能被截断，不能要求覆盖全标题
            denh = compare_tokens - author_tokens
            overlap = len(common) / max(1, len(denh))
            title_ok = len(common) >= 3 and overlap >= 0.5

        journal_ok = bool(
            abbrev_norm and entry.journal_norm and abbrev_norm == entry.journal_norm
        )

        if title_ok or (not title_tokens and journal_ok and year
                        and entry.parsed and entry.parsed["year"] == year):
            return entry.path
    return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  文件名辅助                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def sanitize_filename(name: str) -> str:
    """移除 Windows 文件名不允许的字符并合并多余空格。"""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    return re.sub(r"\s+", " ", name).strip()


def provisional_pdf_filename(ref: dict) -> str:
    """下载阶段使用的临时文件名（改名步骤会将其标准化）。

    形如「Deng - 2024 - Direct Observation of ....pdf」；信息不足时退回
    以 DOI（斜杠替换为下划线）命名。
    """
    names = ref.get("first_author_last_name") or []
    author = str(names[0]).strip() if names else ""
    year = str(ref.get("year") or "").strip()
    title = (ref.get("title") or "").strip()
    parts = [p for p in (author, year, title[:80]) if p]
    if parts:
        return sanitize_filename(" - ".join(parts))[:130] + ".pdf"
    if ref.get("doi"):
        return sanitize_filename(ref["doi"].replace("/", "_"))[:130] + ".pdf"
    return f"reference_{ref.get('id') or 'unknown'}.pdf"


def expected_pdf_filename(ref: dict) -> str:
    """该条目最终应有的标准文件名，形如
    「King「2002 - Functional Ecology」Predicted and Observed ....pdf」。

    直接复用 Lit_Retrieval_4_Rename_Ref 的命名实现（凭 RIS 免语言模型的那一
    条路径），所以下载时就用这个名字落盘、占位 RIS 也用同一个主干，第 4 步
    改名往往只是确认一下文件名无需更改。

    RIS 直读所需的四个字段（第一作者姓 / 期刊缩写 / 年份 / 标题）有缺失时，
    改名步骤本来就要退回语言模型，此处相应退回 provisional_pdf_filename。
    """
    try:
        from Tools.Lit_Retrieval_4_Rename_Ref import standard_filename_from_ris_entry
    except ImportError:
        return provisional_pdf_filename(ref)
    for entry in parse_ris_text(reference_to_ris(ref)):
        filename = standard_filename_from_ris_entry(entry, ".pdf")
        if filename:
            return filename
    return provisional_pdf_filename(ref)


def unused_path(path: Path) -> Path:
    """若路径已被占用，追加 _01 / _02 …… 直到不冲突。"""
    if not path.exists():
        return path
    for number in range(1, 100):
        candidate = path.with_name(f"{path.stem}_{number:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"无法为 {path} 找到未占用的文件名")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  终端交互                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def read_multiline_until_end(prompt: str = "") -> str:
    """读取多行输入，直到某一行恰好输入 end（不区分大小写）为止。"""
    if prompt:
        print(prompt)
    print("  （多行输入，单独一行输入 end 表示结束）")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip().lower() == "end":
            break
        lines.append(line)
    return "\n".join(lines)


def ask_workdir(default: "str | None" = None) -> Path:
    """交互式询问工作目录（下载与产物的存储位置），直到得到有效目录。"""
    while True:
        hint = f"（回车使用默认：{default}）" if default else ""
        try:
            raw = input(f"  请输入文献存储目录{hint} > ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n  已取消。")
        if not raw and default:
            raw = str(default)
        if not raw:
            print("  ⚠ 目录不能为空，请重新输入。")
            continue
        path = Path(raw)
        if not path.exists():
            try:
                answer = input(f"  目录不存在：{path}，是否创建？[Enter/y=创建, n=重新输入] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                raise SystemExit("\n  已取消。")
            if answer in ("", "y", "yes"):
                path.mkdir(parents=True, exist_ok=True)
            else:
                continue
        if not path.is_dir():
            print(f"  ⚠ 不是目录：{path}，请重新输入。")
            continue
        return path.resolve()
