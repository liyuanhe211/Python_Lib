# -*- coding: utf-8 -*-
"""
Lit_Retrieval_0.py — 文献检索流水线总驱动

══════════════════════════════════════════════════════════════════════════════
  一条命令跑通「引用目录解析 → 元数据补全 → PDF 下载 → 标准化改名 →
  文本化」的完整流程（不依赖 Zotero）：

  第 1 步  Lit_Retrieval_1_Reference_List_Parsing
           语言模型（默认 claude-sonnet-4-6，本机订阅）解析参考文献列表；
  第 2 步  Lit_Retrieval_2_Metadata_Completion
           CrossRef 补全 DOI 与元数据，按 ACS 格式列出识别结果，并为每个
           条目各写一份 RIS（含期刊缩写）；
  第 3 步  Lit_Retrieval_3_Download_PDF
           库内查重后多渠道轮询下载（OpenAlex / Unpaywall / DOI 落地页 /
           Anna's Archive / LibGen / Sci-Hub）。并行的是文献，每篇文献内部
           的各个渠道严格串行；书籍最后逐个处理；
  第 4 步  Lit_Retrieval_4_Rename_Ref.rename_pdf_from_ris
           凭 RIS 免语言模型直接标准化改名（信息不足自动退回语言模型），
           单篇 RIS 归档到「PDF texts/<主干[:80]>/」；
  第 5 步  Lit_Retrieval_5_PDF_to_Markdown.convert_multiple_pdf_to_markdown
           Marker 文本化（幂等，已转换的自动跳过）。

  RIS 一律「一个文献一份」，文件名与该文献的 PDF 文件名一致：还没拿到 PDF
  的条目，RIS 放在 <workdir>/ 下（下载失败时正好留在那儿，供将来重跑时
  免去重新解析与 CrossRef 检索）；下载并改名成功后迁到
  <workdir>/PDF texts/<主干[:80]>/ 归档。

  全程状态记录在 <workdir>/Lit_Retrieval_State.json：
  重复运行时已完成的条目自动跳过（已下载的不再下载、已改名的不再改名、
  已文本化的不再转换），可安全断点续跑。

用法：
    # 交互模式（询问存储目录；粘贴参考文献文本，单独一行输入 end 结束）
    python -m Tools.Lit_Retrieval_0

    # 自动化模式
    python -m Tools.Lit_Retrieval_0 ^
        --workdir "E:\\My_Program\\Knowledge_Base_Chemistry\\0 New Download" ^
        --input-file refs.txt

    # 断点续跑（复用状态文件中的既有条目，跳过已完成部分）
    python -m Tools.Lit_Retrieval_0 --workdir "..." --resume
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Tools import Lit_Retrieval_1_Reference_List_Parsing as step1
from Tools import Lit_Retrieval_2_Metadata_Completion as step2
from Tools import Lit_Retrieval_3_Download_PDF as step3
from Tools.Lit_Retrieval_Common import (
    CONTACT_EMAIL,
    ask_workdir,
    load_state,
    read_multiline_until_end,
    ref_short_label,
    reference_to_ris,
    save_state,
    sync_pending_ris,
    write_sidecar_ris,
)
from Tools.Lit_Retrieval_4_Rename_Ref import rename_pdf_from_ris


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第 4 步：标准化改名 + 单篇 RIS 归档                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def rename_downloaded_pdfs(workdir: Path, state: dict,
                           assume_yes: bool = True, log=print) -> None:
    """对已下载但尚未改名的 PDF 逐个改名，并把该条目的 RIS 迁到归档位置。

    第 2 步已经在工作目录下为每个条目写好一份 RIS（文件名就是这篇文献将来
    的 PDF 文件名），这里直接把它交给 Lit_Retrieval_4_Rename_Ref 的 RIS 直读
    模式使用；占位 RIS 意外缺失时临时补写一份。改名成功后单篇 RIS 归档到
    「PDF texts/<新主干[:80]>/」，工作目录下的占位 RIS 随即由 sync_pending_ris
    清理。RIS 信息不足（例如 CrossRef 没给期刊缩写）时，Lit_Retrieval_4_Rename_Ref
    自动退回语言模型流程。
    """
    todo = [
        r for r in state["references"]
        if r["status"]["download"] == "success"
        and not r["status"]["renamed"]
        and r["status"].get("pdf_path")
        and Path(r["status"]["pdf_path"]).exists()
    ]
    if not todo:
        log("  ℹ️ 没有需要改名的新下载 PDF。")
        return

    log(f"\n  ✏️ 开始标准化改名，共 {len(todo)} 个文件……")
    for seq, ref in enumerate(todo, 1):
        status = ref["status"]
        pdf_path = Path(status["pdf_path"])
        log(f"\n  ── 改名 {seq}/{len(todo)}：{pdf_path.name}")

        ris_path = Path(status["pending_ris_path"]) if status.get("pending_ris_path") else None
        temp_ris: "Path | None" = None
        if ris_path is None or not ris_path.exists():
            temp_ris = pdf_path.with_suffix(".ris")
            try:
                temp_ris.write_text(reference_to_ris(ref) + "\n", encoding="utf-8")
                ris_path = temp_ris
            except OSError as e:
                log(f"  ⚠ 临时 RIS 写入失败（{e}），改名将退回语言模型流程。")
                ris_path, temp_ris = None, None

        new_path = rename_pdf_from_ris(
            pdf_path, ris_path=ris_path, assume_yes=assume_yes,
        )

        if new_path is not None:
            status["renamed"] = True
            status["renamed_path"] = str(new_path)
            status["pdf_path"] = str(new_path)
            # 单篇 RIS 归档到「PDF texts/<新主干[:80]>/」
            try:
                sidecar = write_sidecar_ris(new_path, ref)
                status["ris_sidecar_path"] = str(sidecar)
                log(f"  💾 单篇 RIS 已归档：{sidecar}")
            except OSError as e:
                log(f"  ⚠ 单篇 RIS 归档失败：{e}")
        else:
            log(f"  ⚠ {ref_short_label(ref)}：改名未完成，保留原文件名。")

        if temp_ris and temp_ris.exists():
            try:
                temp_ris.unlink()
            except OSError:
                pass
        save_state(workdir, state)

    # 已归档的条目不再需要工作目录下的占位 RIS
    _, removed = sync_pending_ris(workdir, state["references"], log=log)
    if removed:
        log(f"\n  🧹 已把 {removed} 份 RIS 从工作目录迁入「PDF texts/」归档位置。")
    save_state(workdir, state)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第 5 步：PDF 文本化（Marker）                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def convert_downloaded_pdfs(workdir: Path, state: dict, log=print) -> None:
    """对已下载（且尽量已改名）的 PDF 运行 Lit_Retrieval_5_PDF_to_Markdown。

    转换本身幂等：已存在 .all_pages_processed 完成标记的 PDF 自动跳过。
    """
    pdf_paths: list[str] = []
    pending_refs: list[dict] = []
    for ref in state["references"]:
        status = ref["status"]
        if status["download"] != "success" or status["markdown_done"]:
            continue
        path = status.get("renamed_path") or status.get("pdf_path")
        if path and Path(path).exists():
            pdf_paths.append(path)
            pending_refs.append(ref)

    if not pdf_paths:
        log("  ℹ️ 没有需要文本化的新 PDF。")
        return

    log(f"\n  📝 开始 PDF 文本化（Marker），共 {len(pdf_paths)} 个文件……")
    try:
        from Tools.Lit_Retrieval_5_PDF_to_Markdown import convert_multiple_pdf_to_markdown
    except ImportError as e:
        log(f"  ❌ 无法加载 Lit_Retrieval_5_PDF_to_Markdown（{e}）。"
            f"请确认已安装 marker-pdf，或稍后手动运行：")
        for path in pdf_paths:
            log(f"     python -m Tools.Lit_Retrieval_5_PDF_to_Markdown \"{path}\"")
        return

    results = convert_multiple_pdf_to_markdown(pdf_paths)
    done_inputs = {
        str(Path(r["input_path"]).resolve())
        for r in results if r.get("markdown_output_path")
    }
    for ref, path in zip(pending_refs, pdf_paths):
        if str(Path(path).resolve()) in done_inputs:
            ref["status"]["markdown_done"] = True
    save_state(workdir, state)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  总结                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def print_summary(state: dict, log=print) -> None:
    refs = state["references"]
    log(f"\n{'═' * 62}")
    log("  📊 流水线总结")
    log(f"{'═' * 62}")
    log(f"  条目总数：{len(refs)}")

    def count(predicate) -> int:
        return sum(1 for r in refs if predicate(r))

    log(f"  已确定 DOI：{count(lambda r: bool(r.get('doi')))}")
    log(f"  库中已存在（跳过下载）：{count(lambda r: r['status']['download'] == 'already_exists')}")
    log(f"  本次/历史下载成功：{count(lambda r: r['status']['download'] == 'success')}")
    log(f"  下载失败：{count(lambda r: r['status']['download'] == 'failed')}")
    log(f"  已标准化改名：{count(lambda r: r['status']['renamed'])}")
    log(f"  已文本化：{count(lambda r: r['status']['markdown_done'])}")

    failed = [r for r in refs if r["status"]["download"] == "failed"]
    if failed:
        log("\n  ❌ 以下条目未能下载，可稍后重新运行本脚本重试：")
        for ref in failed:
            log(f"     {ref_short_label(ref)}（{ref.get('doi') or '无 DOI'}）："
                f"{ref['status'].get('error') or '未知原因'}")
        kept = [r for r in failed if r["status"].get("pending_ris_path")]
        if kept:
            log(f"     这 {len(kept)} 条的 RIS 已留在工作目录下，文件名与它们将来的"
                f"PDF 文件名一致，重跑时无需重新解析与检索。")

    unselected = [r for r in refs if not r.get("selected", True)]
    if unselected:
        log("\n  ⚠ 以下条目因 DOI 校验可疑被取消选中，请人工复核：")
        for ref in unselected:
            log(f"     {ref_short_label(ref)}：{', '.join(ref.get('crossref_issues') or [])}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  流水线主函数                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_pipeline(
    workdir: Path,
    text: "str | None" = None,
    model: str = step1.DEFAULT_PARSE_MODEL,
    resolve_concurrency: int = step2.RESOLVE_CONCURRENCY,
    download_concurrency: int = step3.DEFAULT_ARTICLE_CONCURRENCY,
    openalex_key: str = "",
    email: str = CONTACT_EMAIL,
    proxy: "str | None" = None,
    library_root: "Path | None" = None,
    dedup: bool = True,
    skip_markdown: bool = False,
    confirm_renames: bool = False,
    append: bool = False,
    log=print,
) -> None:
    """按顺序执行五个步骤；text 为 None 时复用状态文件中的既有条目。"""
    workdir.mkdir(parents=True, exist_ok=True)

    # ── 第 1 步：解析 ──
    if text and text.strip():
        log(f"\n{'━' * 62}")
        log("  ▶ 第 1 步：参考文献列表解析")
        log(f"{'━' * 62}")
        step1.run(workdir, text, model=model, append=append, log=log)
    else:
        state = load_state(workdir)
        if state is None or not state.get("references"):
            log("  ❌ 既没有新输入文本，状态文件中也没有既有条目，无法继续。")
            return
        log(f"  ℹ️ 复用状态文件中的 {len(state['references'])} 个既有条目。")

    # ── 第 2 步：元数据补全 + RIS ──
    log(f"\n{'━' * 62}")
    log("  ▶ 第 2 步：CrossRef 元数据补全与 RIS 输出")
    log(f"{'━' * 62}")
    if step2.run(workdir, concurrency=resolve_concurrency, log=log) is None:
        return

    # ── 第 3 步：查重 + 下载 ──
    log(f"\n{'━' * 62}")
    log("  ▶ 第 3 步：库内查重与多渠道 PDF 下载")
    log(f"{'━' * 62}")
    if step3.run(
        workdir,
        concurrency=download_concurrency,
        openalex_key=openalex_key,
        email=email,
        proxy=proxy,
        library_root=library_root,
        dedup=dedup,
        log=log,
    ) is None:
        return

    state = load_state(workdir)
    if state is None:
        return

    # ── 第 4 步：标准化改名 ──
    log(f"\n{'━' * 62}")
    log("  ▶ 第 4 步：标准化改名（RIS 优先，免语言模型）")
    log(f"{'━' * 62}")
    rename_downloaded_pdfs(workdir, state, assume_yes=not confirm_renames, log=log)

    # ── 第 5 步：文本化 ──
    if skip_markdown:
        log("\n  ⏭️ 已按参数跳过 PDF 文本化步骤。")
    else:
        log(f"\n{'━' * 62}")
        log("  ▶ 第 5 步：PDF 文本化（Marker）")
        log(f"{'━' * 62}")
        convert_downloaded_pdfs(workdir, state, log=log)

    print_summary(state, log=log)
    log(f"\n  👋 全部完成。状态文件：{workdir / 'Lit_Retrieval_State.json'}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  命令行入口                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="文献检索流水线总驱动：解析 → 补全 → 下载 → 改名 → 文本化。"
    )
    parser.add_argument("--workdir",
                        help="文献存储目录（如 E:\\...\\Knowledge_Base_Chemistry\\0 New Download）")
    parser.add_argument("--input-file", help="参考文献文本文件（省略且未 --resume 时交互式输入）")
    parser.add_argument("--resume", action="store_true",
                        help="复用状态文件中的既有条目（不输入新文本，跳过已完成步骤）")
    parser.add_argument("--append", action="store_true",
                        help="新输入的条目追加到既有条目之后（默认覆盖）")
    parser.add_argument("--model", default=step1.DEFAULT_PARSE_MODEL,
                        help=f"解析模型（默认 {step1.DEFAULT_PARSE_MODEL}）")
    parser.add_argument("--download-concurrency", type=int,
                        default=step3.DEFAULT_ARTICLE_CONCURRENCY,
                        help="文章并发下载数")
    parser.add_argument("--resolve-concurrency", type=int,
                        default=step2.RESOLVE_CONCURRENCY,
                        help="CrossRef 并发检索数")
    parser.add_argument("--openalex-key", default="", help="OpenAlex API key（可选）")
    parser.add_argument("--email", default=CONTACT_EMAIL,
                        help="Unpaywall 等 API 的联系邮箱")
    parser.add_argument("--proxy", default=None,
                        help="代理，如 socks5h://127.0.0.1:1080")
    parser.add_argument("--library-root", default=None,
                        help="查重的库根目录（默认取工作目录的上一级）")
    parser.add_argument("--no-dedup", action="store_true", help="跳过库内查重")
    parser.add_argument("--skip-markdown", action="store_true",
                        help="跳过 PDF 文本化步骤")
    parser.add_argument("--confirm-renames", action="store_true",
                        help="每个改名逐一人工确认（默认自动确认）")
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()

    print("═" * 62)
    print("  📚 文献检索流水线（解析 → 补全 → 下载 → 改名 → 文本化）")
    print("═" * 62)

    workdir = Path(args.workdir).resolve() if args.workdir else ask_workdir()

    text: "str | None" = None
    append = args.append
    if args.resume:
        pass  # 复用既有条目
    elif args.input_file:
        text = Path(args.input_file).read_text(encoding="utf-8-sig")
        print(f"  📄 已从文件读取输入：{args.input_file}（{len(text):,} 字符）")
    else:
        state = load_state(workdir)
        if state and state.get("references"):
            print(f"  ℹ️ 状态文件中已有 {len(state['references'])} 个条目。")
            try:
                answer = input(
                    "  [Enter=继续处理既有条目]  [n=输入新的参考文献列表（覆盖）]  "
                    "[a=输入新列表并追加] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer in ("n", "a"):
                append = answer == "a"
                text = read_multiline_until_end("\n  请粘贴参考文献列表文本：")
        else:
            text = read_multiline_until_end("\n  请粘贴参考文献列表文本：")
            if not text.strip():
                print("  ⚠ 输入为空，退出。")
                return

    run_pipeline(
        workdir,
        text=text,
        model=args.model,
        resolve_concurrency=args.resolve_concurrency,
        download_concurrency=args.download_concurrency,
        openalex_key=args.openalex_key,
        email=args.email,
        proxy=args.proxy,
        library_root=Path(args.library_root) if args.library_root else None,
        dedup=not args.no_dedup,
        skip_markdown=args.skip_markdown,
        confirm_renames=args.confirm_renames,
        append=append,
    )


if __name__ == "__main__":
    main()
