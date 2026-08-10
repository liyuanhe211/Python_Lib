# -*- coding: utf-8 -*-
"""
Convert PDF (typically scientific literature or books) to Markdown using Marker.

Marker (https://github.com/datalab-to/marker) is, as of 2026, the strongest
open-source pipeline for scientific PDFs: Surya OCR is layout-aware
(detects paragraphs, figure captions, headings, reading order), it extracts
embedded images and links them in the Markdown output, and it has an
optional LLM boost for messy scans / complex tables / equations.

Embedded-image OCR (on by default, disable with --no-image-ocr): after the
Markdown is written, every image referenced in it (figures, screenshots)
is OCRed with the same Surya models, and the recognized text is inserted as
an HTML comment right below the image link:

    ![](_page_3_Figure_1.jpeg)

    <!-- OCR _page_3_Figure_1.jpeg:
    <recognized text lines>
    -->

The image link itself is untouched, so the image still renders; the OCR text
is invisible when rendered but fully searchable (Obsidian search, grep, RAG
indexing). The step is idempotent — images already carrying an OCR comment
are skipped on re-runs.

Garbled-text-layer detection (on by default, disable with --no-garbled-check):
some PDFs (notably CJK ones produced by old typesetting software) carry an
embedded text layer whose fonts lack a valid ToUnicode CMap — extracting it
yields gibberish that mixes Greek/Arabic/Devanagari/... codepoints. Marker
trusts the text layer unless its OCR-error-detection model flags a page, and
that model misses this failure mode for CJK. After conversion the output
Markdown is scanned with a script-mixing heuristic; if a significant fraction
of lines look garbled, the PDF is automatically re-converted with force_ocr=True
(every page re-OCRed from pixels, text layer ignored). If the output is STILL
garbled after force OCR, a warning asks the user to verify the document's
languages are supported by Surya (or to try --use-llm).

Install:
    pip install marker-pdf
    # Optional, for non-PDF inputs and extras:
    pip install marker-pdf[full]

For each input PDF, output goes to a "PDF texts" subfolder next to the PDF,
inside a folder named after the PDF (stem truncated to 80 chars, matching the
existing Knowledge_Base convention and RAG_Update_Literature_DBs.py's
_unit_dir_for_pdf):

    <dir>/paper.pdf  ->  <dir>/PDF texts/paper/paper.md
                         <dir>/PDF texts/paper/_page_<n>_Figure_<m>.png
                         <dir>/PDF texts/paper/paper_meta.json
                         <dir>/PDF texts/paper/.all_pages_processed

.all_pages_processed (content: "done") marks a completed conversion; if it
already exists the PDF is skipped unless force=True.

This layout is the SHARED contract with the RAG pipeline. The contract is
DEFINED in LLM_Lib/RAG_Lib/Docling.py (this file is just a tool script and
imports the constants and recognition helpers from there): a PDF converted by
either side is recognized as done by the other (find_existing_conversion), so
it is never converted twice.
"""

import argparse
import os
import re

# Windows 上 HuggingFace Hub 默认用符号链接缓存模型，非管理员/未开开发者模式
# 时会 WinError 1314。改用文件复制（多占一点磁盘，但避免权限问题）。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from Python_Lib.My_Lib_Stock import get_input_with_while_cycle

# 文本化输出契约（输出布局常量 + 产物互认判定）唯一定义在 RAG_Lib/Docling.py，
# 本工具脚本 import 复用，保证与 RAG 流水线双向互认、绝不重复转换
from LLM_Lib.RAG_Lib.Docling import (
    DONE_MARKER_NAME,
    FOLDER_NAME_MAX_LEN,
    OUTPUT_PARENT_NAME,
    find_existing_conversion,
    get_output_dir,
    truncated_folder_name,
)


_MODEL_DICT_CACHE = None

#: 嵌入图片 OCR：低于该置信度的识别行丢弃（纯图形/照片类图片会产出高噪声低分行）
IMAGE_OCR_MIN_CONFIDENCE = 0.6

#: 嵌入图片 OCR 每批张数。图多的手册（几百张截图）一次性全部送入 Surya 会
#: 耗尽显存/内存（实测 321 张图直接把 16 GB 显存打爆），必须分块并逐块释放
IMAGE_OCR_CHUNK_SIZE = 16

#: markdown 图片引用 ![alt](path)
_IMAGE_REF_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# ---------------------------------------------------------------------------
# 坏文字层(乱码)检测
#
# 某些 PDF(常见于老排版软件生成的中文文档)内嵌文字层的字体缺少有效的
# ToUnicode 映射,直接提取会得到希腊文/阿拉伯文/天城文等多种文字混杂的乱码。
# Marker 默认信任文字层,其 OCR 错误检测模型对这类 CJK 乱码经常漏检。
# 真实文本一行内几乎不会同时混杂 3 种以上互不相干的文字区段,也不会包含
# 大量"生僻区段"字符(IPA 扩展、叙利亚文、私用区等)——以此为判据。
# ---------------------------------------------------------------------------

#: 单行字母数达到该值才参与乱码统计(短行统计意义不足)
_GARBLED_MIN_LINE_LETTERS = 20
#: 单行判据:生僻区段字符占比阈值 / 占比≥10% 的文字区段数阈值
_GARBLED_RARE_SHARE = 0.15
_GARBLED_SCRIPT_COUNT = 3
#: 全文判据:可疑行占比与可疑行绝对数同时超过阈值才判定为乱码
GARBLED_LINE_RATIO_THRESHOLD = 0.02
GARBLED_MIN_SUSPICIOUS_LINES = 3

_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


def _char_bucket(code_point):
    """把码点归入粗粒度文字区段;返回 None 表示不参与统计(数字/标点/符号)。

    "rare" 表示正常文档几乎不会成片出现的区段(IPA 扩展、叙利亚文、
    康熙部首、私用区等)——坏 ToUnicode 映射产出的乱码大量落在这些区段。
    """
    cp = code_point
    if cp < 0x80:
        return "latin" if (0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A) else None
    if cp < 0xA0:
        return None
    if cp <= 0x024F:
        return "latin"          # Latin-1 补充 / 扩展 A、B
    if cp <= 0x036F:
        return "rare"           # IPA 扩展、修饰字母、独立出现的组合附加符
    if cp <= 0x03FF:
        return "greek"
    if cp <= 0x052F:
        return "cyrillic"
    if cp <= 0x058F:
        return "rare"           # 亚美尼亚文
    if cp <= 0x05FF:
        return "hebrew"
    if cp <= 0x06FF:
        return "arabic"
    if cp <= 0x074F:
        return "rare"           # 叙利亚文
    if cp <= 0x077F:
        return "arabic"
    if cp <= 0x08FF:
        return "rare"           # 它拿文/西非书面文字/阿拉伯文扩展
    if cp <= 0x097F:
        return "devanagari"
    if cp <= 0x0DFF:
        return "rare"           # 其他印度系文字
    if cp <= 0x0E7F:
        return "thai"
    if cp <= 0x10FF:
        return "rare"           # 老挝文/藏文/缅甸文/格鲁吉亚文
    if cp <= 0x11FF:
        return "cjk"            # 谚文字母
    if cp <= 0x1DFF:
        return "rare"           # 埃塞俄比亚文等/语音扩展/组合附加符补充
    if cp <= 0x1EFF:
        return "latin"          # 拉丁扩展附加
    if cp <= 0x1FFF:
        return "greek"          # 希腊文扩展
    if cp <= 0x2BFF:
        return None             # 标点/上下标/货币/箭头/数学符号
    if cp <= 0x2E7F:
        return "rare" if cp <= 0x2DFF else None
    if cp <= 0x2FDF:
        return "rare"           # CJK 部首补充/康熙部首(正文中出现即可疑)
    if cp <= 0x303F:
        return None             # CJK 标点
    if cp <= 0x33FF:
        return "cjk"            # 假名/注音/谚文兼容/围绕字符/CJK 兼容
    if cp <= 0x4DBF:
        return "cjk"            # 扩展 A
    if cp <= 0x4DFF:
        return None             # 易经卦象
    if cp <= 0x9FFF:
        return "cjk"
    if cp <= 0xABFF:
        return "rare"           # 彝文/拉丁扩展 D 等
    if cp <= 0xD7AF:
        return "cjk"            # 谚文音节
    if cp <= 0xF8FF:
        return "rare"           # 谚文扩展 B/代理区/私用区
    if cp <= 0xFAFF:
        return "cjk"            # CJK 兼容表意文字
    if cp <= 0xFB0F:
        return "latin"          # 拉丁连字 ﬁ ﬂ (PDF 提取常见)
    if cp <= 0xFB4F:
        return "rare"           # 亚美尼亚/希伯来表现形式
    if cp <= 0xFDFF:
        return "arabic"         # 阿拉伯文表现形式 A
    if cp <= 0xFE6F:
        return None             # 变体选择符/CJK 竖排标点/小写变体
    if cp <= 0xFEFF:
        return "arabic"         # 阿拉伯文表现形式 B
    if cp <= 0xFF60:
        return None             # 全角拉丁/全角标点
    if cp <= 0xFFDC:
        return "cjk"            # 半角假名/半角谚文
    if cp <= 0xFFFF:
        return "rare"
    if 0x1D400 <= cp <= 0x1D7FF:
        return None             # 数学字母数字符号
    if 0x1F000 <= cp <= 0x1FAFF:
        return None             # 表情符号等
    if cp >= 0x20000:
        return "cjk"            # CJK 扩展 B 及以后
    return "rare"


def markdown_garbled_ratio(md_text):
    """统计 markdown 文本中疑似乱码行的占比。

    返回 (可疑行占比, 参与统计的行数, 可疑行数, 可疑行示例列表)。
    图片链接与 HTML 注释(含本脚本插入的图片 OCR 注释)不参与统计。
    """
    text = _HTML_COMMENT_PATTERN.sub("", md_text)
    text = _IMAGE_REF_PATTERN.sub("", text)
    considered = 0
    suspicious = 0
    samples = []
    for line in text.splitlines():
        counts = {}
        total = 0
        for ch in line:
            bucket = _char_bucket(ord(ch))
            if bucket:
                counts[bucket] = counts.get(bucket, 0) + 1
                total += 1
        if total < _GARBLED_MIN_LINE_LETTERS:
            continue
        considered += 1
        rare_share = counts.get("rare", 0) / total
        major_scripts = sum(1 for count in counts.values() if count / total >= 0.10)
        if rare_share >= _GARBLED_RARE_SHARE or major_scripts >= _GARBLED_SCRIPT_COUNT:
            suspicious += 1
            if len(samples) < 5:
                samples.append(line.strip()[:80])
    ratio = suspicious / considered if considered else 0.0
    return ratio, considered, suspicious, samples


def detect_garbled_markdown(md_path):
    """判定一份输出 markdown 是否疑似乱码(坏文字层)。

    返回 (是否乱码, 可疑行占比, 可疑行示例列表)。
    """
    with open(md_path, "r", encoding="utf-8") as md_file:
        content = md_file.read()
    ratio, _, suspicious, samples = markdown_garbled_ratio(content)
    is_garbled = (ratio >= GARBLED_LINE_RATIO_THRESHOLD
                  and suspicious >= GARBLED_MIN_SUSPICIOUS_LINES)
    return is_garbled, ratio, samples


def _strip_wrapping_quotes(text):
    return text.strip().strip('"').strip("'")


def _get_model_dict():
    """Load Marker's model dict once and reuse across PDFs (it's heavy)."""
    global _MODEL_DICT_CACHE
    if _MODEL_DICT_CACHE is None:
        from marker.models import create_model_dict
        print("Loading Marker models (first run downloads weights)...")
        _MODEL_DICT_CACHE = create_model_dict()
    return _MODEL_DICT_CACHE


def _build_converter(force_ocr, use_llm, llm_service, output_format):
    from marker.converters.pdf import PdfConverter
    from marker.config.parser import ConfigParser

    config = {
        "output_format": output_format,
        "force_ocr": bool(force_ocr),
        "use_llm": bool(use_llm),
    }
    if use_llm and llm_service:
        config["llm_service"] = llm_service

    config_parser = ConfigParser(config)

    return PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=_get_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service() if use_llm else None,
    )


def _clean_ocr_line(text):
    """清理单行 OCR 文本：去掉 <b>/<i>/<math> 等标签，转义会截断 HTML 注释的 '-->'。"""
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    return text.replace("-->", "→").strip()


def ocr_images_in_markdown(md_path, model_dict=None,
                           min_confidence=IMAGE_OCR_MIN_CONFIDENCE):
    """对 markdown 中引用的图片逐张 OCR，把识别文本以 HTML 注释插到图片链接下方。

    图片链接原样保留（图片照常渲染）；OCR 文本渲染时不可见，但对全文搜索
    （Obsidian / grep / RAG 索引）完全可见。幂等：正文中已存在
    ``<!-- OCR <图片文件名>`` 注释的图片直接跳过，可安全重复运行。

    返回本次实际写入 OCR 注释的图片数。
    """
    md_path = os.path.abspath(md_path)
    md_dir = os.path.dirname(md_path)
    with open(md_path, "r", encoding="utf-8") as md_file:
        content = md_file.read()

    # (插入位置, 图片文件名, 图片绝对路径)；只处理首次出现、磁盘上存在的图片
    pending = []
    seen = set()
    for match in _IMAGE_REF_PATTERN.finditer(content):
        image_name = match.group(1)
        if image_name in seen:
            continue
        seen.add(image_name)
        if os.path.splitext(image_name)[1].lower() not in _IMAGE_EXTS:
            continue
        if f"<!-- OCR {image_name}" in content:
            continue
        image_path = os.path.join(md_dir, image_name)
        if not os.path.exists(image_path):
            continue
        line_end = content.find("\n", match.end())
        insert_pos = line_end if line_end != -1 else len(content)
        pending.append((insert_pos, image_name, image_path))

    if not pending:
        return 0

    from PIL import Image

    if model_dict is None:
        model_dict = _get_model_dict()

    print(f"  OCR of {len(pending)} embedded image(s)...")
    # 分块 OCR，逐块打开/关闭图片，只保留轻量的文本结果
    insertions = []  # (insert_pos, image_name, lines)
    for chunk_start in range(0, len(pending), IMAGE_OCR_CHUNK_SIZE):
        chunk = pending[chunk_start:chunk_start + IMAGE_OCR_CHUNK_SIZE]
        images = [Image.open(image_path).convert("RGB") for _, _, image_path in chunk]
        try:
            ocr_results = model_dict["recognition_model"](
                images, det_predictor=model_dict["detection_model"], sort_lines=True)
        finally:
            for image in images:
                image.close()
        for (insert_pos, image_name, _), ocr_result in zip(chunk, ocr_results):
            lines = [_clean_ocr_line(text_line.text) for text_line in ocr_result.text_lines
                     if (text_line.confidence or 0) >= min_confidence]
            lines = [line for line in lines if line]
            if lines:
                insertions.append((insert_pos, image_name, lines))
        done_count = min(chunk_start + IMAGE_OCR_CHUNK_SIZE, len(pending))
        if done_count < len(pending):
            print(f"    ...OCR {done_count}/{len(pending)} images")

    inserted = 0
    # 从后往前插入，避免前面的插入使后面的位置偏移失效
    for insert_pos, image_name, lines in sorted(insertions, key=lambda item: -item[0]):
        comment = "\n\n<!-- OCR {}:\n{}\n-->".format(image_name, "\n".join(lines))
        content = content[:insert_pos] + comment + content[insert_pos:]
        inserted += 1

    if inserted:
        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(content)
    return inserted


def build_marker_converter(force_ocr=False, use_llm=False, llm_service=None,
                           output_format="markdown"):
    """构建 Marker 转换器（模型字典全局缓存复用）。

    公开给 RAG_Lib/Docling.py 使用：扫描型 PDF、单张图片、逐页扫描书都用
    这里的 Marker + Surya OCR 方案（marker 对 png/jpg 输入有原生支持）。
    """
    return _build_converter(force_ocr, use_llm, llm_service, output_format)


def convert_pdf_to_markdown(
    pdf_path,
    force_ocr=False,
    use_llm=False,
    llm_service=None,
    output_format="markdown",
    converter=None,
    force=False,
    image_ocr=True,
    garbled_check=True,
):
    pdf_path = os.path.abspath(_strip_wrapping_quotes(pdf_path))
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input file does not exist: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"Input file must be a .pdf file: {pdf_path}")

    base_name = truncated_folder_name(os.path.splitext(os.path.basename(pdf_path))[0])
    output_dir = get_output_dir(pdf_path)
    marker_path = os.path.join(output_dir, DONE_MARKER_NAME)

    ext = {"markdown": "md", "json": "json", "html": "html", "chunks": "json"}.get(output_format, "md")
    markdown_output_path = os.path.join(output_dir, f"{base_name}.{ext}")

    if not force:
        existing = find_existing_conversion(pdf_path)
        if existing:
            print(f"[SKIP] Already converted: {os.path.basename(pdf_path)} -> {existing}")
            return {
                "input_path": pdf_path,
                "output_dir": existing if os.path.isdir(existing) else os.path.dirname(existing),
                "markdown_output_path": existing,
                "skipped": True,
            }

    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting '{pdf_path}' to Markdown...")
    if force_ocr:
        print("  force_ocr=True (running OCR on every page)")
    if use_llm:
        print(f"  use_llm=True (service={llm_service or 'default'})")

    from marker.output import save_output

    if converter is None:
        converter = _build_converter(force_ocr, use_llm, llm_service, output_format)

    rendered = converter(pdf_path)
    save_output(rendered, output_dir, base_name)

    garbled_retry = False
    still_garbled = False
    if garbled_check and output_format == "markdown":
        is_garbled, ratio, samples = detect_garbled_markdown(markdown_output_path)
        if is_garbled and not force_ocr:
            # 坏文字层:字体缺 ToUnicode 映射,提取出多文字混杂的乱码。
            # 改用 force_ocr 整页从像素重新识别,不再信任文字层。
            garbled_retry = True
            print(f"  [WARN] {ratio:.0%} of text lines look garbled — the PDF's embedded "
                  f"text layer is broken (font lacks a valid ToUnicode map).")
            print("  Re-converting with force_ocr=True (every page re-OCRed from pixels)...")
            for sample in samples[:3]:
                print(f"    garbled sample: {sample}")
            rendered = _build_converter(True, use_llm, llm_service, output_format)(pdf_path)
            save_output(rendered, output_dir, base_name)
            is_garbled, ratio, samples = detect_garbled_markdown(markdown_output_path)
        if is_garbled:
            # force OCR 之后仍是乱码,才可能真是识别层面的问题(如语言不受支持)
            still_garbled = True
            print(f"  [WARN] Output STILL looks garbled after force OCR ({ratio:.0%} of lines).")
            print("  Please verify the document's languages are supported by Surya OCR, "
                  "or retry with --use-llm.")
            for sample in samples[:3]:
                print(f"    garbled sample: {sample}")

    if image_ocr and output_format == "markdown":
        ocr_count = ocr_images_in_markdown(markdown_output_path)
        if ocr_count:
            print(f"  Inserted OCR text for {ocr_count} embedded image(s)")

    with open(marker_path, "w", encoding="utf-8") as marker_file:
        marker_file.write("done")

    return {
        "input_path": pdf_path,
        "output_dir": output_dir,
        "markdown_output_path": markdown_output_path,
        "skipped": False,
        "garbled_retry": garbled_retry,
        "still_garbled": still_garbled,
    }


def convert_multiple_pdf_to_markdown(
    pdf_paths,
    force_ocr=False,
    use_llm=False,
    llm_service=None,
    output_format="markdown",
    force=False,
    image_ocr=True,
    garbled_check=True,
):
    # The converter is built lazily on the first PDF that actually needs
    # conversion (marker-skipped PDFs never trigger model loading); the heavy
    # model dict is cached globally, so later PDFs reuse it.
    converter = None
    results = []
    for pdf_path in pdf_paths:
        try:
            result = convert_pdf_to_markdown(
                pdf_path,
                force_ocr=force_ocr,
                use_llm=use_llm,
                llm_service=llm_service,
                output_format=output_format,
                converter=converter,
                force=force,
                image_ocr=image_ocr,
                garbled_check=garbled_check,
            )
            results.append(result)
            if converter is None and not result.get("skipped"):
                converter = _build_converter(force_ocr, use_llm, llm_service, output_format)
        except Exception as exc:
            print(f"Failed to convert '{pdf_path}': {exc}")
    return results


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert PDF files (scientific literature/books) to Markdown using Marker. "
            "Each PDF produces a folder 'PDF texts/<pdf name>' next to it containing "
            "the .md, extracted images, and a .all_pages_processed completion marker."
        )
    )
    parser.add_argument("pdf_paths", nargs="*", help="Paths to .pdf files")
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Run OCR on every page (use for scanned PDFs even if they contain a text layer).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if the .all_pages_processed marker already exists.",
    )
    parser.add_argument(
        "--no-image-ocr",
        action="store_true",
        help=(
            "Skip OCR of embedded images. By default every image referenced in the "
            "output Markdown is OCRed and the text inserted as an HTML comment below "
            "the image link, making image content searchable."
        ),
    )
    parser.add_argument(
        "--no-garbled-check",
        action="store_true",
        help=(
            "Skip the garbled-text-layer check. By default the output Markdown is "
            "scanned for gibberish caused by a broken embedded text layer (font "
            "without a valid ToUnicode map, common in CJK PDFs); if detected, the "
            "PDF is automatically re-converted with force OCR."
        ),
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable Marker's LLM boost for higher accuracy on messy layouts, tables, and equations.",
    )
    parser.add_argument(
        "--llm-service",
        default=None,
        help=(
            "Dotted path to the Marker LLM service class, e.g. "
            "'marker.services.gemini.GoogleGeminiService' or "
            "'marker.services.openai.OpenAIService'. "
            "Provider API keys are read from environment variables per Marker's docs."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json", "html", "chunks"],
        default="markdown",
        help="Output format (default: markdown).",
    )
    return parser


def _interactive_main(args):
    print("PDF to Markdown (Marker)")
    print("Input PDF paths, one per line. Submit an empty line to start conversion.")
    pdf_paths = get_input_with_while_cycle(strip_quote=True)
    if not pdf_paths:
        print("No PDF files provided.")
        return []
    return convert_multiple_pdf_to_markdown(
        pdf_paths,
        force_ocr=args.force_ocr,
        use_llm=args.use_llm,
        llm_service=args.llm_service,
        output_format=args.output_format,
        force=args.force,
        image_ocr=not args.no_image_ocr,
        garbled_check=not args.no_garbled_check,
    )


def main():
    parser = _build_argument_parser()
    args = parser.parse_args()

    try:
        if args.pdf_paths:
            results = convert_multiple_pdf_to_markdown(
                args.pdf_paths,
                force_ocr=args.force_ocr,
                use_llm=args.use_llm,
                llm_service=args.llm_service,
                output_format=args.output_format,
                force=args.force,
                image_ocr=not args.no_image_ocr,
                garbled_check=not args.no_garbled_check,
            )
        else:
            results = _interactive_main(args)
    except Exception as exc:
        print(f"Error during conversion: {exc}")
        raise SystemExit(1) from exc

    for result in results:
        print(f"Markdown output: {result['markdown_output_path']}")


if __name__ == "__main__":
    main()
