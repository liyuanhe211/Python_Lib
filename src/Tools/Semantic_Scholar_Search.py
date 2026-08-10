# -*- coding: utf-8 -*-
"""
Semantic_Scholar_Search.py —— Semantic Scholar 上 MCP 没有封装的几个端点
========================================================================

`paper-search` MCP server 封装了 Semantic Scholar 的常规检索，但**没有封装**
下面这几个端点，而它们恰恰是主题文献检索里最有价值的几项能力：

- **全文片段检索**（``/graph/v1/snippet/search``）——返回约 500 词的**正文段落**，
  不是只有标题和摘要。功能上等价于"对全世界的文献做一次 RAG"，返回的片段可以
  像本地 RAG 的片段一样直接引用进报告。
- **相似论文推荐**（``/recommendations/v1/papers/forpaper/<paperId>``）——基于
  SPECTER embedding，常能挖到关键词检索够不到的相关工作。
- **引文列表**（``/graph/v1/paper/<paperId>/references``）——这篇论文引用了谁。
  回溯方向的引用追踪：正文里看到 (Terpin, Spotila and Foley 1979) 这类转引时，
  从被收录论文的引文列表里直接拿到完整题录，不用去解析全文文本。
- **被引列表**（``/graph/v1/paper/<paperId>/citations``）——谁引用了这篇论文。
  正向方向的引用追踪：发现某篇文献重要之后，把引用它的后续文献拉出来筛选。
  两个方向都附带 ``contexts``（引用发生处的原文句子），是判断相关性的关键材料。

这些端点都要求把密钥放在 ``x-api-key`` 请求头里，而 `WebFetch` 工具没法设置
请求头——这就是本脚本存在的理由。

用法::

    python src/Tools/Semantic_Scholar_Search.py snippet "查询词" [-n 10]
    python src/Tools/Semantic_Scholar_Search.py recommend <paperId> [-n 10]
    python src/Tools/Semantic_Scholar_Search.py references <paperId> [-n 100]
    python src/Tools/Semantic_Scholar_Search.py citations <paperId> [-n 50]

``paperId`` 接受 Semantic Scholar 的多种标识：``CorpusId:12345``、
``DOI:10.1038/nature12373``、``arXiv:2106.15928``，或 40 位的 S2 论文哈希。

密钥查找顺序：环境变量 ``SEMANTIC_SCHOLAR_API_KEY`` →
``~/.config/paper-search-mcp/.env`` 的 ``PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY``
→ ``LLM_API_KEYS_PRIVATE.py`` 的 ``Semantic_Scholar_S2_API_KEY``。
本文件里不含任何密钥字面值。

配额提醒：本机这把密钥的限额是**每秒 1 次请求，且是跨全部端点累计的**。脚本内部
已经按这个上限自持节流，撞到 429 也会退避重试；但如果同时还在用 MCP 的
``search_semantic`` 工具，两边加起来照样会超。**同一时间只用一条路径。**
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

SNIPPET_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
RECOMMEND_URL = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"

#: 密钥限额是每秒 1 次、跨端点累计，这里留一点余量。
MIN_SECONDS_BETWEEN_REQUESTS = 1.2

_last_request_time = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 一、密钥
# ═════════════════════════════════════════════════════════════════════════════

def find_api_key() -> str:
    """按环境变量 → .env → LLM_API_KEYS_PRIVATE.py 的顺序找密钥，找不到返回空串。"""
    from_env = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if from_env:
        return from_env

    env_file = Path.home() / ".config" / "paper-search-mcp" / ".env"
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY":
                value = value.strip().strip('"').strip("'")
                if value:
                    return value

    keys_dir = os.environ.get("LLM_API_KEYS_DIR")
    candidates = [Path(keys_dir)] if keys_dir else []
    for parent in Path(__file__).resolve().parents:
        if parent.name == "Python_Lib" and (parent / "src").is_dir():
            candidates.append(parent.parent)
            break
    for directory in candidates:
        keys_file = directory / "LLM_API_KEYS_PRIVATE.py"
        if not keys_file.is_file():
            continue
        import ast
        tree = ast.parse(keys_file.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Constant):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Name)
                        and target.id == "Semantic_Scholar_S2_API_KEY"
                        and isinstance(node.value.value, str)):
                    return node.value.value.strip()
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# 二、请求
# ═════════════════════════════════════════════════════════════════════════════

def request_json(url: str, params: dict, api_key: str, *,
                 json_body: dict | None = None, max_retries: int = 4) -> dict:
    """按限额自持节流地发一次请求，撞到 429 或 5xx 时退避重试。

    传了 *json_body* 就发 POST（批量元数据端点要求 POST），否则发 GET。
    """
    global _last_request_time

    headers = {"x-api-key": api_key} if api_key else {}
    delay = 2.0

    for attempt in range(1, max_retries + 1):
        elapsed = time.monotonic() - _last_request_time
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

        if json_body is None:
            response = requests.get(url, params=params, headers=headers, timeout=60)
        else:
            response = requests.post(url, params=params, headers=headers,
                                     json=json_body, timeout=60)
        _last_request_time = time.monotonic()

        if response.status_code == 200:
            return response.json()

        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            print(f"  [第 {attempt} 次请求返回 {response.status_code}，"
                  f"等待 {delay:.0f} 秒后重试]", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
            continue

        raise RuntimeError(
            f"请求失败：HTTP {response.status_code}\n"
            f"URL：{response.url.split('?')[0]}\n"
            f"响应正文：{response.text[:500]}"
        )

    raise RuntimeError(f"重试 {max_retries} 次后仍未成功。")


# ═════════════════════════════════════════════════════════════════════════════
# 三、两个子命令
# ═════════════════════════════════════════════════════════════════════════════

def format_paper_line(paper: dict) -> str:
    """把一条论文题录压成一行，方便扫读。

    ``authors`` 在两个端点上的形状不一样：``/snippet/search`` 给的是字符串列表，
    ``/paper/batch`` 给的是 ``{"authorId":..., "name":...}`` 字典列表，两种都要认。
    """
    authors = paper.get("authors") or []
    first = authors[0] if authors else ""
    first_author = (first.get("name", "") if isinstance(first, dict) else str(first)) or "作者不详"
    if len(authors) > 1:
        first_author += " et al."
    year = paper.get("year") or "年份不详"
    venue = paper.get("venue") or ""
    external = paper.get("externalIds") or {}
    doi = external.get("DOI", "")
    parts = [f"{first_author} ({year})"]
    if venue:
        parts.append(venue)
    if doi:
        parts.append(f"DOI: {doi}")
    if paper.get("corpusId"):
        parts.append(f"CorpusId:{paper['corpusId']}")
    return " | ".join(parts)


def resolve_paper_metadata(corpus_ids: list[int], api_key: str) -> dict[int, dict]:
    """批量补齐题录字段，返回 ``corpusId -> 题录`` 的映射。

    ``/snippet/search`` 只回 corpusId、标题、作者，**没有年份、刊名和 DOI**，
    而这三项恰恰是引用时必需的，也不接受 ``fields`` 参数去多要。所以这里
    额外发一次批量查询把它们补上——多花一次请求，换来片段可以直接规范引用。
    """
    if not corpus_ids:
        return {}
    payload = request_json(
        BATCH_URL,
        {"fields": "corpusId,title,year,venue,externalIds,authors,openAccessPdf"},
        api_key,
        json_body={"ids": [f"CorpusId:{cid}" for cid in corpus_ids]},
    )
    resolved: dict[str, dict] = {}
    for entry in payload or []:
        if isinstance(entry, dict) and entry.get("corpusId") is not None:
            resolved[normalize_corpus_id(entry["corpusId"])] = entry
    return resolved


def normalize_corpus_id(value) -> str:
    """统一 corpusId 的类型再做匹配。

    两个端点给的类型不一样：``/snippet/search`` 回字符串，``/paper/batch`` 回整数。
    不统一的话按原值查表必然落空，题录就补不上——这个坑很安静，
    表现只是年份和 DOI 一直显示"不详"。
    """
    return str(value).strip()


def run_snippet(query: str, limit: int, api_key: str, as_json: bool,
                resolve: bool = True) -> None:
    # 这个端点不接受 fields 参数，默认返回已含 paper.title / authors / openAccessInfo
    payload = request_json(SNIPPET_URL, {"query": query, "limit": limit}, api_key)

    items = payload.get("data") or []

    if resolve:
        corpus_ids = [item["paper"]["corpusId"] for item in items
                      if (item.get("paper") or {}).get("corpusId") is not None]
        metadata = resolve_paper_metadata(corpus_ids, api_key)
        for item in items:
            paper = item.get("paper") or {}
            extra = metadata.get(normalize_corpus_id(paper.get("corpusId")))
            if extra:
                # 只补空缺，不覆盖片段端点自己给的标题与作者
                for key, value in extra.items():
                    paper.setdefault(key, value)

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"查询：{query}")
    print(f"返回片段：{len(items)} 条\n")
    for index, item in enumerate(items, 1):
        snippet = item.get("snippet") or {}
        paper = item.get("paper") or {}
        print("─" * 74)
        print(f"[{index}] {paper.get('title', '标题不详')}")
        print(f"     {format_paper_line(paper)}")
        section = (snippet.get("section") or "").strip()
        kind = snippet.get("snippetKind") or ""
        location = "、".join(x for x in (kind, section) if x)
        if location:
            print(f"     片段位置：{location}")
        pdf = (paper.get("openAccessPdf") or {}).get("url")
        if pdf:
            print(f"     开放获取全文：{pdf}")
        else:
            access = paper.get("openAccessInfo") or {}
            if access.get("status"):
                print(f"     开放获取状态：{access['status']}"
                      f"{'（' + access['license'] + '）' if access.get('license') else ''}")
        print()
        print(f"     {(snippet.get('text') or '').strip()}")
        print()


def run_recommend(paper_id: str, limit: int, api_key: str, as_json: bool) -> None:
    payload = request_json(
        f"{RECOMMEND_URL}/{paper_id}",
        {"limit": limit,
         "fields": "title,authors,year,venue,externalIds,corpusId,abstract,openAccessPdf"},
        api_key,
    )

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    papers = payload.get("recommendedPapers") or []
    print(f"基于论文：{paper_id}")
    print(f"推荐相似论文：{len(papers)} 篇\n")
    for index, paper in enumerate(papers, 1):
        print("─" * 74)
        print(f"[{index}] {paper.get('title', '标题不详')}")
        print(f"     {format_paper_line(paper)}")
        pdf = (paper.get("openAccessPdf") or {}).get("url")
        if pdf:
            print(f"     开放获取全文：{pdf}")
        abstract = (paper.get("abstract") or "").strip()
        if abstract:
            print(f"     摘要：{abstract[:400]}{'……' if len(abstract) > 400 else ''}")
        print()


#: 引用关系端点每条结果里论文本体的字段，外加引用发生处的上下文句
CITATION_LINK_FIELDS = ("contexts,intents,isInfluential,"
                        "title,authors,year,venue,externalIds,corpusId,"
                        "abstract,openAccessPdf")


def run_citation_links(paper_id: str, direction: str, limit: int, offset: int,
                       api_key: str, as_json: bool) -> None:
    """引用关系追踪的两个方向共用一套实现。

    *direction* 为 ``references``（回溯：这篇引用了谁）或 ``citations``
    （正向：谁引用了这篇）。两个端点返回结构相同，只是论文字段的键名不同
    （``citedPaper`` 与 ``citingPaper``）。每条结果的 ``contexts`` 是引用
    发生处的原文句子——筛选相关性时比摘要更直接，原样打印。
    """
    payload = request_json(
        f"{PAPER_URL}/{paper_id}/{direction}",
        {"fields": CITATION_LINK_FIELDS, "limit": limit, "offset": offset},
        api_key,
    )

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    paper_key = "citedPaper" if direction == "references" else "citingPaper"
    items = payload.get("data") or []
    if direction == "references":
        print(f"论文 {paper_id} 的引文列表（它引用了谁）：{len(items)} 条")
    else:
        print(f"引用了论文 {paper_id} 的文献（谁引用了它）：{len(items)} 条")
    if payload.get("next"):
        print(f"（未取完，下一页从 --offset {payload['next']} 继续）")
    print()

    for index, item in enumerate(items, 1):
        paper = item.get(paper_key) or {}
        print("─" * 74)
        influential = "★ 高影响引用  " if item.get("isInfluential") else ""
        print(f"[{index}] {influential}{paper.get('title', '标题不详')}")
        print(f"     {format_paper_line(paper)}")
        pdf = (paper.get("openAccessPdf") or {}).get("url")
        if pdf:
            print(f"     开放获取全文：{pdf}")
        for context in (item.get("contexts") or [])[:2]:
            context = " ".join(context.split())
            print(f"     引用处原文：{context[:300]}{'……' if len(context) > 300 else ''}")
        abstract = (paper.get("abstract") or "").strip()
        if abstract:
            print(f"     摘要：{abstract[:300]}{'……' if len(abstract) > 300 else ''}")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# 四、入口
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic Scholar 全文片段检索与相似论文推荐（MCP 未封装的两个端点）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_snippet = sub.add_parser(
        "snippet", help="全文片段检索：返回约 500 词的正文段落，可直接引用")
    p_snippet.add_argument("query", help="自然语言查询或术语组合（这个端点是语义匹配，长句有帮助）")
    p_snippet.add_argument("-n", "--limit", type=int, default=10, help="返回片段数（默认 10）")
    p_snippet.add_argument("--no-resolve", action="store_false", dest="resolve",
                           help="不额外发一次请求补齐年份 / 刊名 / DOI（省一次配额，但片段将无法规范引用）")

    p_recommend = sub.add_parser(
        "recommend", help="相似论文推荐：基于 SPECTER embedding")
    p_recommend.add_argument(
        "paper_id",
        help="论文标识，如 DOI:10.1038/nature12373 / arXiv:2106.15928 / CorpusId:12345")
    p_recommend.add_argument("-n", "--limit", type=int, default=10, help="返回论文数（默认 10）")

    p_references = sub.add_parser(
        "references", help="引文列表（回溯）：这篇论文引用了谁，含引用处原文句子")
    p_references.add_argument(
        "paper_id",
        help="论文标识，如 DOI:10.1038/nature12373 / arXiv:2106.15928 / CorpusId:12345")
    p_references.add_argument("-n", "--limit", type=int, default=100,
                              help="返回条数（默认 100，单次上限 1000）")
    p_references.add_argument("--offset", type=int, default=0,
                              help="分页起点（上一页输出里给出的 next 值）")

    p_citations = sub.add_parser(
        "citations", help="被引列表（正向）：谁引用了这篇论文，含引用处原文句子")
    p_citations.add_argument(
        "paper_id",
        help="论文标识，如 DOI:10.1038/nature12373 / arXiv:2106.15928 / CorpusId:12345")
    p_citations.add_argument("-n", "--limit", type=int, default=50,
                             help="返回条数（默认 50，单次上限 1000）")
    p_citations.add_argument("--offset", type=int, default=0,
                             help="分页起点（上一页输出里给出的 next 值）")

    for p in (p_snippet, p_recommend, p_references, p_citations):
        p.add_argument("--json", action="store_true", dest="as_json",
                       help="输出原始 JSON，便于脚本处理")

    args = parser.parse_args()

    api_key = find_api_key()
    if not api_key:
        print("找不到 Semantic Scholar 密钥。请确认下列任一位置有值：\n"
              "  - 环境变量 SEMANTIC_SCHOLAR_API_KEY\n"
              "  - ~/.config/paper-search-mcp/.env 的 "
              "PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY\n"
              "  - LLM_API_KEYS_PRIVATE.py 的 Semantic_Scholar_S2_API_KEY\n"
              "填好之后跑一次 src/Tools/Paper_Search_MCP_Setup.py 同步。",
              file=sys.stderr)
        sys.exit(1)

    try:
        if args.command == "snippet":
            run_snippet(args.query, args.limit, api_key, args.as_json, args.resolve)
        elif args.command == "recommend":
            run_recommend(args.paper_id, args.limit, api_key, args.as_json)
        else:
            run_citation_links(args.paper_id, args.command, args.limit,
                               args.offset, api_key, args.as_json)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
