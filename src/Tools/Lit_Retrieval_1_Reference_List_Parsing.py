# -*- coding: utf-8 -*-
"""
Lit_Retrieval_1_Reference_List_Parsing.py — 引用列表 → 结构化条目

══════════════════════════════════════════════════════════════════════════════
  文献检索流水线第 1 步：把论文的参考文献列表原始文本交给语言模型解析成
  结构化 JSON 条目（作者、标题、期刊、年份、卷、页码 / 文章号、DOI 等）。

  - 提示词逐字移植自 Zotero_MultiFetcher 插件（src/modules/LLMPrompts.ts）；
  - 语言模型访问走 LLM_Lib/LLM.py 的 call_claude()（本机 claude CLI +
    订阅额度，无需 API key）；默认模型 claude-sonnet-4-6；
  - 长列表按行分批（每批 25 行，与插件 BATCH_SIZE 一致），多批并发调用；
  - 「纯 DOI 列表」输入（每行一个 DOI）直接构建条目，完全跳过语言模型；
  - 解析结果写入工作目录的 Lit_Retrieval_State.json，供后续步骤使用。

用法：
    # 交互模式（询问工作目录，多行粘贴参考文献文本，单独一行输入 end 结束）
    python -m Tools.Lit_Retrieval_1_Reference_List_Parsing

    # 自动化模式
    python -m Tools.Lit_Retrieval_1_Reference_List_Parsing ^
        --workdir "E:\\My_Program\\Knowledge_Base_Chemistry\\0 New Download" ^
        --input-file refs.txt
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
from pathlib import Path

from LLM_Lib.LLM import call_claude, extract_json
from Tools.Lit_Retrieval_Common import (
    ask_workdir,
    is_pure_doi_input,
    load_state,
    match_dois,
    new_reference,
    new_state,
    normalize_reference,
    read_multiline_until_end,
    ref_short_label,
    save_state,
)

# ── 模型与分批配置 ────────────────────────────────────────────────────────────

#: 默认解析模型（本机 Claude 订阅，无需 API key）
DEFAULT_PARSE_MODEL = "claude-sonnet-4-6"

#: 每批发送给语言模型的最大行数（与参考插件 LLMPrompts.ts 的 BATCH_SIZE 一致）
BATCH_SIZE = 25

#: 并发调用语言模型的批次数
PARSE_CONCURRENCY = 4


# ── 提示词（逐字移植自参考插件 src/modules/LLMPrompts.ts）────────────────────

REFERENCE_PARSER_SYSTEM_PROMPT = """You are a bibliographic reference parser. You receive raw text containing one or more academic references and you output structured JSON.

RULES:
1. Parse EVERY individual reference. If a single numbered entry contains sub-references (e.g. "[15] a) ...; b) ...; c) ..."), split them into separate objects with index="15" and sub_index="a", "b", "c" etc.
2. Expand abbreviated journal names to their full form when you can confidently identify them (e.g. "Adv. Mater." -> "Advanced Materials", "J. Am. Chem. Soc." -> "Journal of the American Chemical Society", "Angew. Chem., Int. Ed." -> "Angewandte Chemie International Edition"). Keep the abbreviation in "journal_abbrev".
3. If a field cannot be determined from the text, set it to null. Do NOT guess or hallucinate.
4. Return only the first 2-3 authors in the "authors" array (e.g. ["S. Deng", "Y. Kuang", "L. Liu"]). Always put the first author's last name in "first_author_last_name" (e.g. ["Deng"]). These are used for CrossRef matching.
5. The "confidence" field (0.0-1.0) reflects how confident you are that you parsed the reference correctly.
6. The "item_type" should be one of: "journalArticle", "conferencePaper", "book", "bookSection", "thesis", "patent", "preprint", "report", "webpage".
7. If a DOI is explicitly present in the reference text (e.g. "10.1039/D4MH00979G"), extract it into the "doi" field.
8. Distinguish between "pages" (e.g. "15860-15870") and "article_number" (e.g. "2309679", "e202113078"). Article numbers are typically a single identifier, not a range.
9. Output ONLY the JSON object. No markdown fences, no explanation, no preamble.
10. The "raw_text_start" field should contain the first ~20 characters of this reference as it appears in the original input. The "raw_text_end" field should contain the last ~30 characters. Include any newlines if present. These are used to locate the reference in the source text for highlighting.
11. A DOI or URL (e.g. "https://doi.org/10.1002/adma.202309679") may appear on the line immediately AFTER a reference. This DOI belongs to that reference — extract it into the "doi" field and include it in "raw_text_end". Do NOT create a separate reference entry for such standalone DOI lines. Each reference should appear exactly once in output.

Output format:
{
  "references": [
    {
      "index": "62",
      "sub_index": null,
      "first_author_last_name": ["Deng"],
      "authors": ["S. Deng", "Y. Kuang", "L. Liu"],
      "title": null,
      "journal": "Advanced Materials",
      "journal_abbrev": "Adv. Mater.",
      "year": 2024,
      "volume": "36",
      "issue": null,
      "pages": null,
      "article_number": "2309679",
      "doi": null,
      "raw_text_start": "62 S. Deng, Y. Kuang",
      "raw_text_end": "Adv. Mater., 2024, 36, 2309679.",
      "confidence": 0.9,
      "item_type": "journalArticle"
    }
  ]
}"""


def build_user_prompt(reference_text: str, format_hint: "str | None" = None) -> str:
    """构建用户消息（与参考插件 buildUserPrompt 一致）。"""
    if format_hint and format_hint != "auto":
        prompt = (f"Parse the following reference list (format: {format_hint}) "
                  f"into structured JSON:\n\n---\n")
    else:
        prompt = "Parse the following reference list into structured JSON:\n\n---\n"
    return prompt + reference_text + "\n---"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  输入预处理与分批                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def clean_reference_text(text: str) -> str:
    """统一换行符、压缩连续空行（与参考插件 startParsing 的预处理一致）。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_batches(text: str) -> list[str]:
    """长列表按行分批：超过 2×BATCH_SIZE 行时每 BATCH_SIZE 行一批。"""
    lines = text.split("\n")
    if len(lines) <= BATCH_SIZE * 2:
        return [text]
    return [
        "\n".join(lines[i:i + BATCH_SIZE])
        for i in range(0, len(lines), BATCH_SIZE)
    ]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  语言模型输出解析（含截断修复）                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _salvage_truncated_json(text: str) -> "dict | None":
    """输出被截断时的补救：截到最后一个完整的对象并闭合括号后重试解析。"""
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:]
    last_complete = fragment.rfind("},")
    if last_complete < 0:
        last_complete = fragment.rfind("}")
        if last_complete <= 0:
            return None
    candidate = fragment[: last_complete + 1] + "]}"
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def parse_llm_response(response: str) -> list[dict]:
    """从语言模型输出中提取 references 数组（宽容处理围栏与截断）。"""
    parsed = extract_json(response)
    if not isinstance(parsed, dict):
        parsed = _salvage_truncated_json(response)
    if not isinstance(parsed, dict):
        return []
    refs = parsed.get("references")
    return refs if isinstance(refs, list) else []


def _to_reference(raw: dict) -> dict:
    """把语言模型返回的单条 JSON 归一化为标准 reference 字典。"""
    ref = new_reference()
    for key in (
        "index", "sub_index", "first_author_last_name", "authors", "title",
        "journal", "journal_abbrev", "year", "volume", "issue", "pages",
        "article_number", "doi", "raw_text_start", "raw_text_end",
        "confidence", "item_type",
    ):
        if key in raw and raw[key] is not None:
            ref[key] = raw[key]
    if not isinstance(ref["first_author_last_name"], list):
        ref["first_author_last_name"] = [str(ref["first_author_last_name"])]
    if not isinstance(ref["authors"], list):
        ref["authors"] = [str(ref["authors"])]
    if ref.get("year") is not None:
        try:
            ref["year"] = int(ref["year"])
        except (TypeError, ValueError):
            ref["year"] = None
    if ref.get("doi"):
        ref["doi_source"] = "llm"
    return normalize_reference(ref)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  解析主函数                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def parse_doi_list(text: str) -> list[dict]:
    """纯 DOI 输入：跳过语言模型，直接为每个 DOI 构建条目。"""
    refs: list[dict] = []
    for doi in match_dois(text):
        ref = new_reference(doi=doi, doi_source="user", confidence=1.0)
        refs.append(normalize_reference(ref))
    return refs


def parse_reference_text(
    text: str,
    model: str = DEFAULT_PARSE_MODEL,
    format_hint: "str | None" = None,
    max_workers: int = PARSE_CONCURRENCY,
    log=print,
) -> list[dict]:
    """解析参考文献文本，返回标准 reference 字典列表。

    - 纯 DOI 列表直接构建条目（不访问语言模型）；
    - 其余情况按行分批并发调用 claude CLI，每批解析后合并。
    """
    text = clean_reference_text(text)
    if not text:
        return []

    if is_pure_doi_input(text):
        refs = parse_doi_list(text)
        log(f"  ✅ 检测到纯 DOI 输入，共 {len(refs)} 条，已跳过语言模型解析。")
        return refs

    batches = split_batches(text)
    log(f"  📤 共 {len(batches)} 批待解析（模型：{model}，最多 {max_workers} 批并发）……")

    results: list[list[dict]] = [[] for _ in batches]
    progress_lock = threading.Lock()
    completed = 0

    def run_batch(batch_index: int) -> None:
        nonlocal completed
        prompt = build_user_prompt(batches[batch_index], format_hint)
        response = call_claude(
            prompt,
            model=model,
            system_prompt=REFERENCE_PARSER_SYSTEM_PROMPT,
            confirm=False,
            verbose=False,
        )
        refs_raw = parse_llm_response(response)
        results[batch_index] = [_to_reference(r) for r in refs_raw if isinstance(r, dict)]
        with progress_lock:
            completed += 1
            log(f"  ✅ 第 {batch_index + 1} 批解析完成"
                f"（{len(results[batch_index])} 条），总进度 {completed}/{len(batches)}")

    if len(batches) == 1:
        run_batch(0)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run_batch, i) for i in range(len(batches))]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 让异常向上抛出

    all_refs: list[dict] = []
    for batch_refs in results:
        all_refs.extend(batch_refs)

    # 分配稳定的条目 id（用于状态跟踪）
    for seq, ref in enumerate(all_refs, 1):
        index_part = ref.get("index") or f"{seq}"
        sub_part = ref.get("sub_index") or ""
        ref["id"] = f"ref_{seq:03d}_{index_part}{sub_part}"

    return all_refs


def print_parse_summary(refs: list[dict], log=print) -> None:
    """打印解析结果概览。"""
    log(f"\n  {'─' * 58}")
    log(f"  📋 解析结果：共 {len(refs)} 条")
    with_doi = sum(1 for r in refs if r.get("doi"))
    low_confidence = [r for r in refs if (r.get("confidence") or 0) < 0.5]
    log(f"     其中文本中已含 DOI 的条目 {with_doi} 条；"
        f"解析置信度低于 0.5 的条目 {len(low_confidence)} 条。")
    for ref in refs:
        doi_part = ref.get("doi") or "（暂无 DOI，待第 2 步检索）"
        journal = ref.get("journal_abbrev") or ref.get("journal") or ""
        log(f"     {ref_short_label(ref)}  {journal}  {doi_part}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  命令行入口                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="文献检索流水线第 1 步：语言模型解析参考文献列表。"
    )
    parser.add_argument("--workdir", help="文献存储目录（状态文件写在这里）")
    parser.add_argument("--input-file", help="参考文献文本文件路径（省略则交互式输入）")
    parser.add_argument("--model", default=DEFAULT_PARSE_MODEL,
                        help=f"解析模型（默认 {DEFAULT_PARSE_MODEL}）")
    parser.add_argument("--format-hint", default=None,
                        choices=["numbered", "bibtex", "doi-list", "free-form"],
                        help="输入格式提示（可选）")
    parser.add_argument("--append", action="store_true",
                        help="追加到状态文件中的既有条目（默认覆盖）")
    return parser


def run(workdir: Path, text: str, model: str = DEFAULT_PARSE_MODEL,
        format_hint: "str | None" = None, append: bool = False,
        log=print) -> list[dict]:
    """解析文本并写入状态文件；返回全部条目（供驱动脚本调用）。"""
    refs = parse_reference_text(text, model=model, format_hint=format_hint, log=log)
    if not refs:
        log("  ⚠ 未解析出任何条目。")
        return []

    state = load_state(workdir)
    if state is None:
        state = new_state(workdir)
    if append and state.get("references"):
        offset = len(state["references"])
        for seq, ref in enumerate(refs, offset + 1):
            ref["id"] = f"ref_{seq:03d}_{ref.get('index') or seq}{ref.get('sub_index') or ''}"
        state["references"].extend(refs)
    else:
        state["references"] = refs
    path = save_state(workdir, state)
    log(f"  💾 解析结果已写入状态文件：{path}")
    print_parse_summary(refs, log=log)
    return state["references"]


def main() -> None:
    args = _build_argument_parser().parse_args()

    print("═" * 62)
    print("  📖 文献检索流水线 · 第 1 步 · 参考文献列表解析")
    print("═" * 62)

    workdir = Path(args.workdir).resolve() if args.workdir else ask_workdir()
    workdir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        text = Path(args.input_file).read_text(encoding="utf-8-sig")
        print(f"  📄 已从文件读取输入：{args.input_file}（{len(text):,} 字符）")
    else:
        text = read_multiline_until_end("\n  请粘贴参考文献列表文本：")

    if not text.strip():
        print("  ⚠ 输入为空，退出。")
        return

    existing = load_state(workdir)
    append = args.append
    if (existing and existing.get("references") and not args.append
            and not args.input_file):
        # 交互模式下发现既有条目时询问覆盖还是追加
        try:
            answer = input(
                f"  状态文件中已有 {len(existing['references'])} 条条目，"
                f"覆盖还是追加？[Enter=覆盖, a=追加] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        append = answer == "a"

    run(workdir, text, model=args.model, format_hint=args.format_hint,
        append=append)
    print("\n  👋 第 1 步完成。下一步：运行 Lit_Retrieval_2_Metadata_Completion 补全 DOI 与元数据。")


if __name__ == "__main__":
    main()
