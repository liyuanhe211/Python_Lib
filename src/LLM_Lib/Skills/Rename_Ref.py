"""
Rename_Literature_PDF.py — 用 LLM 识别文献 PDF 元信息并重命名

══════════════════════════════════════════════════════════════════════════════
  交互式工具：逐个输入 PDF 文件路径或图片文件夹路径，提取文字内容，
  通过 Gemini Flash 模型识别文献元信息（作者、期刊/出版社、年份、标题），
  生成规范化文件名/文件夹名并在用户确认后执行重命名。

  支持输入类型：
    • PDF 文件：直接提取页面文字
    • EPUB 文件：提取正文 HTML 文字
    • DjVu 文件：通过 djvutxt 或 python-djvulibre 提取文字
    • 图片文件夹：文件夹内图片按自然排序（类似 Windows Explorer）视为
      PDF 的各页，对需要的页面进行 OCR 文字识别。
      若文件夹内存在非图片文件，提示用户确认后再处理。

  文件名格式（按优先级）：
    • 找到通讯作者（≤3个带*标记）：
        {最后通讯作者姓}, {第一作者姓}「{Year} - {Journal}」{Title}.pdf
    • 学位论文且找到导师+答辩人姓名：
        {第一个导师姓}, {答辩人姓}「{Year} - {Journal}」{Title}.pdf
    • 默认：
        {第一作者姓}「{Year} - {Journal}」{Title}.pdf
      - 名字首字母大写，标题使用 Title Capitalization
      - 期刊缩写使用 ISO 4，但遵循惯用写法（如 Am. 而非 Amer.）
      - 文件名总长度限制 150 字符
      - 两个人名必须出现在同一页，否则不使用双人名格式

  页面读取策略：
      1. 首先从第一页提取文字
      2. 若第一页内容为空白，先尝试 OCR 识别；OCR 成功则使用 OCR 文字，
         否则将第二页视为第一页，后续页面依次前移
      3. 若信息缺失，依次追加最后一页、第二页+倒数第二页、第三页+倒数第三页
      4. 仍然缺失的字段留空

使用方式：
    python -m LLM_Lib.Skills.Rename_Literature_PDF
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

from LLM_Lib.LLM import call_gemini, extract_json, GEMINI_FLASH, _AVG_CHARS_PER_TOKEN, _MODEL_PRICE_PER_M_TOKENS
from Python_Lib.My_Lib_File import get_unused_filename
from Python_Lib.My_Lib_Stock import title_capitalization

# ── 全局 Token 统计 ─────────────────────────────────────────────────────────
_token_lock = threading.Lock()
_total_input_chars = 0
_total_output_chars = 0

# ── 常用图片文件扩展名 ──────────────────────────────────────────────────────
_IMAGE_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif",
    ".webp", ".avif", ".heic", ".heif",
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  自然排序（模拟 Windows Explorer 的 StrCmpLogicalW）                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _natural_sort_key(s: str) -> list:
    """
    生成自然排序键，用于实现类似 Windows Explorer 的文件名排序。

    将字符串拆分为 文本片段 和 数字片段 的交替列表，
    数字片段按整数值比较，文本片段按大小写不敏感的字典序比较。

    例如：
      "name9"  -> ["", "name", 9, ""]
      "name10" -> ["", "name", 10, ""]
    排序后 name9 在 name10 之前。
    """
    parts = re.split(r'(\d+)', s)
    key: list = []
    for part in parts:
        if part.isdigit():
            # 数字片段：按整数值排序，同值时按原始长度（前导零少的排前面）
            key.append((0, int(part), len(part)))
        else:
            # 文本片段：大小写不敏感
            key.append((1, part.lower(), part))
    return key


def natural_sorted(iterable, key=None):
    """
    对可迭代对象进行自然排序（类似 Windows Explorer）。

    Parameters:
        iterable: 可迭代对象
        key: 可选的键函数，先应用此函数再应用自然排序键。
             若为 None，则对元素本身应用自然排序键。
    """
    if key is not None:
        return sorted(iterable, key=lambda x: _natural_sort_key(str(key(x))))
    return sorted(iterable, key=lambda x: _natural_sort_key(str(x)))

# ── 重命名历史记录 CSV ──────────────────────────────────────────────────────
_RENAME_LOG: Path = Path(__file__).parent.parent / "temp" / "rename_history.csv"


def _log_rename(original: Path, new_name: str) -> None:
    """
    将一条重命名记录追加到 CSV 日志文件（temp/rename_history.csv）。

    列：时间, 原文件名, 改后文件名, 原路径
    """
    _RENAME_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _RENAME_LOG.exists() or _RENAME_LOG.stat().st_size == 0
    with open(_RENAME_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["时间", "原文件名", "改后文件名", "原路径"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            original.name,
            new_name,
            str(original),
        ])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PDF 文字提取                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _extract_page_texts(pdf_path: str | Path) -> list[str]:
    """
    提取 PDF 每一页的文字内容，返回按页索引的字符串列表。

    优先使用 pypdf，失败时退回到 PyMuPDF (fitz)。
    """
    # ── 尝试 pypdf（纯 Python，安装最稳定）──
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return [page.extract_text() or "" for page in reader.pages]
    except ImportError:
        pass
    except Exception as e:
        print(f"  ⚠ pypdf 读取失败（{e}），尝试 PyMuPDF……")

    # ── 退回到 PyMuPDF (fitz) ──
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return pages


def _extract_epub_texts(epub_path: str | Path) -> list[str]:
    """
    提取 EPUB 文件的最前面和最后面的文本。
    返回 [前1600字符, 后1600字符]（如果总长度较短则只返回一个元素）。
    """
    import zipfile
    import re
    import xml.etree.ElementTree as ET
    import urllib.parse
    import html

    try:
        with zipfile.ZipFile(epub_path, 'r') as z:
            # 1. Find OPF file from META-INF/container.xml
            try:
                container_xml = z.read('META-INF/container.xml')
                root = ET.fromstring(container_xml)
                ns = {'n': 'urn:oasis:names:tc:opendocument:xmlns:container'}
                rootfile = root.find('.//n:rootfile', ns)
                opf_path = rootfile.attrib.get('full-path') if rootfile is not None else None
            except Exception:
                opf_path = None
            
            html_files = []
            if opf_path and opf_path in z.namelist():
                # 2. Parse OPF to get spine (reading order)
                try:
                    opf_content = z.read(opf_path)
                    # Remove namespaces for easier parsing
                    opf_content_str = re.sub(b' xmlns="[^"]+"', b'', opf_content, count=1)
                    opf_root = ET.fromstring(opf_content_str)
                    
                    manifest = opf_root.find('.//manifest')
                    spine = opf_root.find('.//spine')
                    
                    if manifest is not None and spine is not None:
                        items = {}
                        for item in manifest.findall('item'):
                            items[item.attrib.get('id')] = item.attrib.get('href')
                        
                        opf_dir = os.path.dirname(opf_path)
                        for itemref in spine.findall('itemref'):
                            idref = itemref.attrib.get('idref')
                            if idref in items:
                                href = items[idref]
                                full_href = f"{opf_dir}/{href}" if opf_dir else href
                                html_files.append(full_href)
                except Exception:
                    pass
            
            if not html_files:
                # Fallback to just reading all html files sorted by name
                html_files = sorted([f for f in z.namelist() if f.endswith(('.html', '.xhtml', '.htm'))])
            
            # 3. Extract text
            texts = []
            for html_file in html_files:
                try:
                    file_path = html_file.split('#')[0]
                    file_path = urllib.parse.unquote(file_path)
                    if file_path in z.namelist():
                        html_content = z.read(file_path).decode('utf-8', errors='ignore')
                        # Extract text from body if possible
                        body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.IGNORECASE | re.DOTALL)
                        if body_match:
                            html_content = body_match.group(1)
                        # Strip HTML tags
                        text = re.sub(r'<[^>]+>', ' ', html_content)
                        # Unescape HTML entities
                        text = html.unescape(text)
                        # Clean up whitespace
                        text = re.sub(r'\s+', ' ', text).strip()
                        if text:
                            texts.append(text)
                except Exception:
                    continue
            
            full_text = " ".join(texts)
            if not full_text:
                return []
            
            # 4. Get first and last weighted characters (Chinese=2, English=1)
            def get_weight(s: str) -> int:
                return sum(2 if ord(c) > 255 else 1 for c in s)
                
            if get_weight(full_text) <= 3200:
                return [full_text]
            else:
                def slice_weighted(s: str, max_w: int, from_start: bool) -> str:
                    w = 0
                    if from_start:
                        for i, c in enumerate(s):
                            w += 2 if ord(c) > 255 else 1
                            if w >= max_w:
                                return s[:i+1]
                        return s
                    else:
                        for i, c in enumerate(reversed(s)):
                            w += 2 if ord(c) > 255 else 1
                            if w >= max_w:
                                return s[len(s)-1-i:]
                        return s
                
                return [slice_weighted(full_text, 1600, True), slice_weighted(full_text, 1600, False)]
                
    except Exception as e:
        print(f"  ⚠ EPUB 读取失败：{e}")
        return []


def _extract_djvu_texts(djvu_path: str | Path) -> list[str]:
    """
    提取 DjVu 文件的文本内容。

    优先使用 djvutxt 命令行工具（DjVuLibre），失败时尝试 python-djvulibre。
    返回 [前1600字符, 后1600字符]（如果总长度较短则只返回一个元素）。
    """
    import subprocess

    djvu_path = Path(djvu_path)
    full_text = ""

    # ── 尝试 djvutxt 命令行工具（DjVuLibre）──
    try:
        result = subprocess.run(
            ["djvutxt", str(djvu_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            full_text = result.stdout.strip()
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        print(f"  ⚠ djvutxt 超时：{djvu_path.name}")
    except Exception as e:
        print(f"  ⚠ djvutxt 调用失败：{e}")

    # ── 若 djvutxt 失败，尝试 python-djvulibre ──
    if not full_text:
        try:
            import djvu.decode  # type: ignore

            ctx = djvu.decode.Context()
            doc = ctx.new_document(djvu.decode.FileURI(str(djvu_path)))
            doc.decoding_job.wait()
            page_texts: list[str] = []
            for page in doc.pages:
                page.get_info()
                sexpr = page.text.sexpr
                if sexpr is not None:
                    page_texts.append(str(sexpr))
            full_text = " ".join(page_texts).strip()
        except ImportError:
            pass
        except Exception as e:
            print(f"  ⚠ python-djvulibre 读取失败：{e}")

    if not full_text:
        return []

    # ── 截取首尾片段（加权字符数，中文=2，英文=1）──
    def get_weight(s: str) -> int:
        return sum(2 if ord(c) > 255 else 1 for c in s)

    if get_weight(full_text) <= 3200:
        return [full_text]

    def slice_weighted(s: str, max_w: int, from_start: bool) -> str:
        w = 0
        if from_start:
            for i, c in enumerate(s):
                w += 2 if ord(c) > 255 else 1
                if w >= max_w:
                    return s[:i + 1]
            return s
        else:
            for i, c in enumerate(reversed(s)):
                w += 2 if ord(c) > 255 else 1
                if w >= max_w:
                    return s[len(s) - 1 - i:]
            return s

    return [slice_weighted(full_text, 1600, True), slice_weighted(full_text, 1600, False)]


def _is_blank(text: str) -> bool:
    """判断页面文字是否为空白（仅空白字符或空串）。"""
    return not text or text.strip() == ""


def _is_image_file(path: Path) -> bool:
    """判断文件是否为常见图片格式。"""
    return path.suffix.lower() in _IMAGE_EXTENSIONS


def _check_image_folder(folder_path: Path) -> tuple[bool, list[Path], list[Path]]:
    """
    检查文件夹是否全部由图片文件组成。

    Returns:
        (all_images, image_files, non_image_files)
        - all_images:      所有文件（不含子文件夹和隐藏文件）都是图片
        - image_files:     按自然排序的图片文件列表
        - non_image_files: 非图片文件列表
    """
    all_files = [
        f for f in folder_path.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ]
    if not all_files:
        return False, [], []

    image_files = [f for f in all_files if _is_image_file(f)]
    non_image_files = [f for f in all_files if not _is_image_file(f)]

    # 使用自然排序
    image_files = natural_sorted(image_files, key=lambda f: f.name)

    all_images = len(non_image_files) == 0 and len(image_files) > 0
    return all_images, image_files, non_image_files


def _extract_text_from_image(image_path: Path) -> str:
    """
    对单张图片进行 OCR 文本识别。

    语言：简体中文 + 繁体中文 + 日文 + 英文 + 德文。
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        print(f"  ⚠ 缺少 OCR 依赖库（{e}），跳过 OCR。")
        return ""

    try:
        img = Image.open(str(image_path))
        ocr_text = pytesseract.image_to_string(
            img,
            lang="chi_sim+chi_tra+jpn+eng+deu",
        )
        return ocr_text.strip() if ocr_text else ""
    except Exception as e:
        print(f"  ⚠ 图片 OCR 识别失败（{image_path.name}）：{e}")
        return ""


def _extract_image_folder_page_texts(
    image_files: list[Path],
) -> list[str]:
    """
    将图片文件夹中的每张图片视为一页，提取每页的 OCR 文字。

    仅对需要的页面进行 OCR（按照与 PDF 相同的分轮策略），
    其余页面暂时留空，后续按需 OCR。

    Returns:
        按图片顺序对应的文字列表（未 OCR 的页面为空字符串）。
    """
    n = len(image_files)
    # 初始化为空字符串，后续按需 OCR
    return ["" for _ in range(n)]


def _apply_blank_first_page_shift(pages: list[str]) -> list[str]:
    """
    如果第一页内容为空白，将第二页视为第一页，后续页面依次前移。
    重复此操作直到第一页有内容或无更多页面。
    """
    while pages and _is_blank(pages[0]):
        pages = pages[1:]
    return pages


def _ocr_single_page(
    pdf_path: str | Path, page_idx: int, text: str,
    image_files: list[Path] | None = None,
) -> tuple[str, bool]:
    """
    若某页文字为空白，对该单页进行 OCR 识别并返回结果。
    返回 (ocr或原文字, 是否应用了OCR)。

    当 image_files 不为 None 时，表示处理的是图片文件夹，
    直接对对应的图片文件进行 OCR。

    语言：简体中文 + 繁体中文 + 日文 + 英文 + 德文。
    """
    if not _is_blank(text):
        return text, False

    # ── 图片文件夹模式：直接 OCR 对应图片 ──
    if image_files is not None:
        if page_idx < 0 or page_idx >= len(image_files):
            return text, False
        img_path = image_files[page_idx]
        ocr_text = _extract_text_from_image(img_path)
        if ocr_text:
            print(f"  🔍 第 {page_idx + 1} 页（{img_path.name}）已进行 OCR 识别")
            return ocr_text, True
        return text, False

    # ── PDF 模式 ──
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as e:
        print(f"  ⚠ 缺少 OCR 依赖库（{e}），跳过 OCR。")
        return text, False

    try:
        doc = fitz.open(str(pdf_path))
        if page_idx >= len(doc):
            doc.close()
            return text, False
        page = doc[page_idx]
        if not (page.get_images() or page.get_drawings()):
            doc.close()
            return text, False
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()
        ocr_text = pytesseract.image_to_string(
            img,
            lang="chi_sim+chi_tra+jpn+eng+deu",
        )
        if ocr_text.strip():
            print(f"  🔍 第 {page_idx + 1} 页无文字但有图片，已进行 OCR 识别")
            return ocr_text.strip(), True
        return text, False
    except Exception as e:
        print(f"  ⚠ 第 {page_idx + 1} 页 OCR 识别失败：{e}")
        return text, False



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  分轮次构建 Prompt 页面内容                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _build_page_content_rounds(pages: list[str]) -> list[list[tuple[int, str]]]:
    """
    按策略生成多轮页面内容，每轮返回 [(原始页码(1-based), 文字内容), ...] 列表。

    轮次：
      1. 前3页 + 最后2页
      2. 第4~5页 + 倒数第3~4页
      3. 第6~8页
    """
    n = len(pages)
    if n == 0:
        return []

    rounds: list[list[tuple[int, str]]] = []
    used_pages = set()

    # 第1轮：前3页 + 最后2页
    round1: list[tuple[int, str]] = []
    # 前3页
    for i in range(min(3, n)):
        if i not in used_pages:
            round1.append((i + 1, pages[i]))
            used_pages.add(i)
    # 最后2页
    for i in range(max(0, n - 2), n):
        if i not in used_pages:
            round1.append((i + 1, pages[i]))
            used_pages.add(i)
    
    if round1:
        round1.sort(key=lambda x: x[0])
        rounds.append(round1)

    # 第2轮：第4~5页 + 倒数第3~4页
    round2: list[tuple[int, str]] = []
    # 第4~5页 (index 3, 4)
    for i in range(3, min(5, n)):
        if i not in used_pages:
            round2.append((i + 1, pages[i]))
            used_pages.add(i)
    # 倒数第3~4页 (index n-4, n-3)
    for i in range(max(0, n - 4), n - 2):
        if i not in used_pages:
            round2.append((i + 1, pages[i]))
            used_pages.add(i)
            
    if round2:
        round2.sort(key=lambda x: x[0])
        rounds.append(round2)

    # 第3轮：第6~8页 (index 5, 6, 7)
    round3: list[tuple[int, str]] = []
    for i in range(5, min(8, n)):
        if i not in used_pages:
            round3.append((i + 1, pages[i]))
            used_pages.add(i)

    if round3:
        round3.sort(key=lambda x: x[0])
        rounds.append(round3)

    return rounds


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LLM 调用                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_SYSTEM_PROMPT = """\
You are a metadata extraction assistant for academic and scientific documents.
Given text content extracted from a document (which may be a PDF, EPUB, book, \
scientific journal article, conference paper, or thesis), identify the following \
metadata fields:

1. "language": First, determine the primary language of the provided text content \
   (e.g., "English", "Chinese", "German"). IMPORTANT: Determine the language based \
   ONLY on the extracted text content, NOT on the original filename. This field \
   serves as a reminder to extract and output all subsequent metadata in this EXACT \
   SAME LANGUAGE. Do NOT translate titles, author names, or journal names into \
   English if they are originally in another language.
2. "first_author_last_name": The last name (family name) of the first author. \
   Use title case (first letter capitalized). \
   IMPORTANT: If the author's name is written in Chinese characters, output their \
   FULL NAME (e.g., "张三" instead of just "张"). \
   SPECIAL CASE — set this field to exactly the string "Skipped" (not empty) when \
   ALL of the following conditions are true: \
   (a) the first author is marked as a corresponding author (has * or similar symbol); \
   (b) the last author in the author list does NOT have a * or similar symbol; \
   (c) you are confident about the full author list on the page. \
   When this special case applies, still include the first author's name in \
   "corresponding_authors_last_names" as usual.
3. "publisher_or_journal": The name of the publisher (for books) or journal / \
   conference name (for articles). Use the standard ISO 4 abbreviated journal \
   name, but follow widely-used conventions over strict ISO 4 when they differ \
   (e.g. "J. Am. Chem. Soc." not "J. Amer. Chem. Soc.", "Angew. Chem. Int. Ed." \
   for Angewandte Chemie International Edition, "Nat. Chem." for Nature Chemistry, \
   "ACS Nano" stays as-is). For books use the publisher name. Use title case. \
   IMPORTANT: "ResearchGate" is a preprint hosting platform, NOT a publisher or journal. \
   If the document has any real publisher or journal name available, use that instead. \
   Only return "ResearchGate" if no other publisher or journal information can be found.
4. "year": The publication year as a 4-digit string.
5. "title": The title of the work. Use APA-style title case capitalization: \
   capitalize all major words (nouns, verbs, adjectives, adverbs, pronouns, \
   and ALL words of four or more letters); lowercase only minor words of \
   three letters or fewer that are articles ("a", "an", "the"), short \
   conjunctions ("and", "as", "but", "for", "if", "nor", "or", "so", \
   "yet"), or short prepositions ("at", "by", "in", "of", "off", "on", \
   "per", "to", "up", "via"). Always capitalize the first word and the \
   first word after a colon or em dash. In hyphenated compounds, capitalize \
   the second part if it is a major word (e.g., "Self-Report"). Example: \
   "Amphibians and Reptiles of Baja California" — note that "and" and \
   "of" are lowercase because they are minor words. Truncate if very long \
   (keep it meaningful).
6. "corresponding_authors_last_names": A JSON array of last names (title case) \
   of authors explicitly marked as corresponding author(s) — i.e. authors whose \
   name in the author list is immediately followed by an asterisk (*) or similar \
   symbol. Include at most 3 entries. Return [] if no such marking is found or \
   if you are not confident. IMPORTANT: only populate this field if both the \
   first author AND the corresponding author(s) appear on the SAME page of the \
   provided text. \
   IMPORTANT: If the author's name is written in Chinese characters, output their \
   FULL NAME (e.g., "张三" instead of just "张"). \
   When the special case for "first_author_last_name" = "Skipped" applies (see \
   field 2), include the first author's name in this array as normal — the \
   array should reflect all * -marked authors regardless.
7. "is_thesis": true if the document is a PhD/master's/bachelor's thesis or \
   dissertation, false otherwise.
8. "advisor_last_name": If is_thesis is true and the thesis advisor / supervisor \
   name is explicitly stated on the same page as the candidate name, provide the \
   advisor's last name (title case). If written in Chinese, provide the FULL NAME. \
   Otherwise use "".
9. "candidate_last_name": If is_thesis is true and the thesis author / candidate \
   name is explicitly stated, provide their last name (title case). If written in \
   Chinese, provide the FULL NAME. Otherwise use "".
10. "chinese_label": If an original file name is provided at the end of this prompt, \
   examine it for whether any user-authored Chinese description of the document content exist\
   (e.g. in "Jaakkola2024, 哺乳动物认知丰容的重要性 animals-14-00949.pdf" the Chinese label is \
   "哺乳动物认知丰容的重要性"). If exist, extract ONLY the meaningful Chinese description text; \
   ignore author names, journal codes, year numbers, DOI fragments, and file \
   extensions. Return "" if no such label is present or if you are not confident. \
   Do NOT copy this label into the "title" field.

Return ONLY a JSON object with exactly these ten keys. If a field cannot be \
determined from the provided text, use "" for string fields, [] for array fields, \
and false for boolean fields. Do not include any explanation outside the JSON. \
IMPORTANT LANGUAGE RULE: Extract and output the metadata in the EXACT SAME LANGUAGE \
as the original text. If the original text is in Chinese, output Chinese; if English, \
output English; if German, output German, etc. Do NOT translate the title, author names, \
or journal names into English if they are originally in another language."""


def _build_prompt(
    accumulated_pages: list[tuple[int, str]],
    missing_fields: list[str],
    original_filename: str = "",
    ocr_used: bool = False,
) -> str:
    """
    构建 LLM 查询 Prompt。

    original_filename: PDF 原文件名（完整，含后缀）。会展示在 Prompt 末尾，
                       供 LLM 处理 chinese_label 字段提取。
    ocr_used:          是否有页面经过了 OCR 识别。
    """
    page_sections: list[str] = []
    for page_num, text in accumulated_pages:
        page_sections.append(
            f"--- Page {page_num} ---\n{text.strip()}\n"
        )

    ocr_hint = ""
    if ocr_used:
        ocr_hint = (
            "\n\nNote: Some pages had no embedded text and were processed with "
            "OCR (optical character recognition). The OCR result may contain "
            "recognition errors — please correct obvious mistakes when extracting metadata."
        )

    fields_hint = ""
    if missing_fields:
        fields_hint = (
            f"\n\nNote: The following fields are still missing and need to be "
            f"identified: {', '.join(missing_fields)}. "
            f"Focus especially on finding these."
        )

    filename_section = ""
    if original_filename:
        filename_section = (
            f"\n\n--- Original file name ---\n{original_filename}\n"
        )

    return (
        "Below is text extracted from a document "
        "(a book, scientific literature, or thesis). "
        "Please identify the metadata and return JSON.\n\n"
        + "\n".join(page_sections)
        + ocr_hint
        + fields_hint
        + filename_section
    )


_REQUIRED_FIELDS = [
    "first_author_last_name",
    "publisher_or_journal",
    "year",
    "title",
]


def _get_missing_fields(info: dict) -> list[str]:
    """返回值为空字符串或缺失的字段名列表。"""
    return [f for f in _REQUIRED_FIELDS if not info.get(f, "").strip()]


def _query_llm_for_metadata(
    pages: list[str],
    pdf_path: "str | Path" = "",
    original_filename: str = "",
    image_files: list[Path] | None = None,
) -> dict:
    """
    分轮次查询 LLM，逐步补充缺失信息。

    pdf_path:          PDF 文件路径，用于按需对空白页进行 OCR。
    original_filename: PDF 原文件名（或文件夹名），传给 LLM 以便它识别文件名中的
                       中文注释并填入 chinese_label 字段。
    image_files:       若非 None，表示处理图片文件夹模式，OCR 直接使用图片文件。
    返回包含各字段的 dict。
    """
    rounds = _build_page_content_rounds(pages)
    if not rounds:
        return {f: "" for f in _REQUIRED_FIELDS}

    info: dict = {}
    accumulated_pages: list[tuple[int, str]] = []
    ocr_used = False  # 跟踪是否任何页已 OCR

    for round_idx, page_group in enumerate(rounds):
        # 按需 OCR 本轮页面再累加
        for page_num, text in page_group:
            page_idx = page_num - 1  # _build_page_content_rounds 返回 1-based
            ocr_text, did_ocr = _ocr_single_page(
                pdf_path, page_idx, text, image_files=image_files
            )
            if did_ocr:
                ocr_used = True
            accumulated_pages.append((page_num, ocr_text))

        missing = _get_missing_fields(info) if info else _REQUIRED_FIELDS

        page_nums = [str(p[0]) for p in page_group]
        if round_idx == 0:
            print(f"  📖 [{original_filename}] 第 {round_idx + 1} 轮查询：页面 {', '.join(page_nums)}")
        else:
            print(
                f"  📖 [{original_filename}] 第 {round_idx + 1} 轮补充查询（缺失: "
                f"{', '.join(missing)}）：追加页面 {', '.join(page_nums)}"
            )

        prompt = _build_prompt(
            accumulated_pages,
            missing if round_idx > 0 else [],
            original_filename=original_filename,
            ocr_used=ocr_used,
        )

        response = call_gemini(
            prompt,
            model=GEMINI_FLASH,
            system_prompt=_SYSTEM_PROMPT,
            reasoning=False,
            confirm=False,
            stream=False,  # 并发时关闭流式输出，避免终端输出混乱
            allow_cancel=False, # 并发时关闭按键中断，避免终端输出混乱
            temperature=0.1,
        )

        with _token_lock:
            global _total_input_chars, _total_output_chars
            _total_input_chars += len(prompt) + len(_SYSTEM_PROMPT)
            _total_output_chars += len(response)

        parsed = extract_json(response)
        if isinstance(parsed, dict):
            # 若 LLM 返回 ResearchGate 作为 publisher_or_journal，视为未识别
            poj = parsed.get("publisher_or_journal", "")
            if re.sub(r'\s+', '', poj).lower() == "researchgate":
                parsed["publisher_or_journal"] = ""
            # 合并必填字段：仅用非空新值填充空旧值
            for field in _REQUIRED_FIELDS:
                new_val = parsed.get(field, "").strip()
                old_val = info.get(field, "").strip()
                if new_val and not old_val:
                    info[field] = new_val
                elif not old_val:
                    info[field] = ""
            # 合并额外字段（首轮直接取，后续轮次不覆盖已有非空值）
            for field in ("language", "corresponding_authors_last_names", "is_thesis",
                          "advisor_last_name", "candidate_last_name", "chinese_label"):
                if field not in info or not info[field]:
                    if field in parsed:
                        info[field] = parsed[field]
        else:
            print(f"  ⚠ [{original_filename}] LLM 未返回有效 JSON，跳过本轮结果。")

        # 检查是否已全部获取
        if not _get_missing_fields(info):
            print(f"  ✅ [{original_filename}] 所有字段已获取。")
            break

    # 确保所有字段都存在
    for f in _REQUIRED_FIELDS:
        info.setdefault(f, "")

    return info


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  文件名生成                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_MAX_FILENAME_LEN = 150  # 含 .pdf 后缀


def _sanitize_filename(name: str) -> str:
    """移除或替换文件名中不允许的字符。"""
    # 替换 Windows 文件名中不允许的字符
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # 合并多余空格
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _resolve_conflict_path(path: Path) -> Path:
    """
    若目标路径已存在，在文件名（或文件夹名）后缀之前追加 _01、_02……
    直到找到不冲突的路径并返回（委托给 get_unused_filename 实现）。
    若路径不存在则直接返回原路径。
    """
    return Path(get_unused_filename(str(path)))


def _build_filename(info: dict, suffix: str = ".pdf") -> str:
    """
    根据元信息 dict 生成文件名，有三种格式：

    1. 找到通讯作者（≤3个带星号）:
       {最后通讯作者姓}, {第一作者姓}「{Year} - {Journal}」{Title}{suffix}
    2. 学位论文且找到导师和答辩人:
       {第一个导师姓}, {答辩人姓}「{Year} - {Journal}」{Title}{suffix}
    3. 默认:
       {第一作者姓}「{Year} - {Journal}」{Title}{suffix}

    若 info["chinese_label"] 非空（由 LLM 从原文件名中提取），则在标题前加入：
       {人名}「{Year} - {Journal}」{中文注释}, {英文标题}{suffix}

    总长度限制 150 字符。
    suffix: 文件后缀，默认 ".pdf"，文件夹传空字符串。
    """
    author = info.get("first_author_last_name", "").strip().title()
    journal = title_capitalization(info.get("publisher_or_journal", "").strip())
    year = info.get("year", "").strip()
    title = title_capitalization(info.get("title", "").strip())

    is_thesis = info.get("is_thesis", False)
    advisor = info.get("advisor_last_name", "").strip().title()
    candidate = info.get("candidate_last_name", "").strip().title()
    corr_list: list[str] = info.get("corresponding_authors_last_names", []) or []
    corr_list = [c.strip().title() for c in corr_list if c.strip()]

    # ── 决定人名部分（不含末尾分隔符） ──
    if is_thesis and advisor and candidate:
        name_part = f"{advisor}, {candidate}"
    elif corr_list and len(corr_list) <= 3:
        last_corr = corr_list[-1]
        if author.lower() == "skipped":
            # First author IS the corresponding author; don't duplicate the name
            name_part = last_corr if last_corr else ""
        elif last_corr and author and last_corr.lower() != author.lower():
            name_part = f"{last_corr}, {author}"
        elif last_corr:
            name_part = last_corr
        else:
            name_part = author
    else:
        name_part = author

    # 格式：{人名}「{Year} - {Journal}」{Title}{suffix}
    if journal and year:
        journal_year = f"「{year} - {journal}」"
    elif journal:
        journal_year = f"「{journal}」"
    elif year:
        journal_year = f"「{year}」"
    else:
        journal_year = "「」"

    prefix = (name_part + journal_year) if name_part else journal_year

    # 若 LLM 提取到中文注释，拼成 "中文注释, 英文标题" 的完整标题部分
    original_chinese = info.get("chinese_label", "").strip()
    if original_chinese:
        # 若中文注释与标题完全相同（忽略大小写和首尾空格），只保留一个
        if original_chinese.strip().lower() == title.strip().lower():
            original_chinese = ""
    if original_chinese:
        sep = ", "
        title_part_template = f"{original_chinese}{sep}{{eng}}"
        # 计算英文标题可用空间
        available_for_eng = (
            _MAX_FILENAME_LEN - len(prefix) - len(suffix)
            - len(original_chinese) - len(sep)
        )
        if available_for_eng > 10 and title:
            if len(title) > available_for_eng:
                truncated = title[:available_for_eng].strip()
                last_space = truncated.rfind(" ")
                if last_space > available_for_eng * 0.5:
                    truncated = truncated[:last_space]
                title = truncated
        elif available_for_eng <= 10:
            title = ""
        title = title_part_template.format(eng=title) if title else original_chinese

    # 计算标题可用空间
    available_for_title = _MAX_FILENAME_LEN - len(prefix) - len(suffix)

    if available_for_title <= 0:
        # 前缀已太长，截断前缀
        filename = prefix[:_MAX_FILENAME_LEN - len(suffix)] + suffix
    elif len(title) > available_for_title:
        # 截断标题
        truncated_title = title[:available_for_title].strip()
        # 尝试在单词边界截断
        last_space = truncated_title.rfind(" ")
        if last_space > available_for_title * 0.5:
            truncated_title = truncated_title[:last_space]
        filename = prefix + truncated_title + suffix
    else:
        filename = prefix + title + suffix

    filename = _sanitize_filename(filename)
    return filename


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  主流程                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _analyze_pdf(pdf_path: Path, current_idx: int = 0, total_count: int = 0, original_filename: str = "") -> "tuple[Path, dict, str, Path] | None":
    """
    提取文字并调用 LLM 识别元信息，返回 (pdf_path, info, new_filename, new_path)。
    失败时返回 None。

    original_filename: 若提供，覆盖传给 LLM 的文件名（可含文件夹名前缀）。
    """
    # 收集输出，最后一次性打印，避免多线程输出混乱
    output_lines = []
    output_lines.append(f"\n{'═' * 62}")
    if total_count > 0:
        output_lines.append(f"  分析文件 ({current_idx}/{total_count})：{pdf_path.name}")
    else:
        output_lines.append(f"  分析文件：{pdf_path.name}")
    output_lines.append(f"  路径：{pdf_path}")
    output_lines.append(f"{'═' * 62}")

    # ── 提取文字 ──
    try:
        all_pages = _extract_page_texts(pdf_path)
    except Exception as e:
        output_lines.append(f"  ❌ 无法读取 PDF：{e}")
        print("\n".join(output_lines))
        return None

    if not all_pages:
        output_lines.append(f"  ❌ PDF 没有任何页面。")
        print("\n".join(output_lines))
        return None

    output_lines.append(f"  📄 共 {len(all_pages)} 页")

    # ── 空白首页 OCR + 前移 ──
    # 对开头连续的空白页先逐页尝试 OCR；一旦 OCR 成功就保留该页并停止，
    # 这样 _apply_blank_first_page_shift 就不会把扫描版 PDF 的页面全部丢弃。
    for _i in range(len(all_pages)):
        if not _is_blank(all_pages[_i]):
            break  # 已有文字，不需要 OCR
        _ocr_text, _did_ocr = _ocr_single_page(pdf_path, _i, all_pages[_i])
        if _did_ocr:
            all_pages[_i] = _ocr_text
            break  # OCR 成功，后续交由 _apply_blank_first_page_shift 处理

    pages = _apply_blank_first_page_shift(all_pages)
    if len(pages) < len(all_pages):
        skipped = len(all_pages) - len(pages)
        output_lines.append(f"  ⚠ 前 {skipped} 页为空白，已跳过（将第 {skipped + 1} 页视为第1页）")

    if not pages:
        output_lines.append(f"  ❌ 所有页面均为空白，无法提取信息。")
        print("\n".join(output_lines))
        return None

    # ── OCR 空白页 ──
    # 注：后续页面的 OCR 已延迟到 _query_llm_for_metadata 内按需执行

    # ── LLM 查询 ──
    # 打印之前的输出，因为 LLM 查询可能耗时较长
    print("\n".join(output_lines))
    
    info = _query_llm_for_metadata(
        pages,
        pdf_path=pdf_path,
        original_filename=original_filename or pdf_path.name,
    )

    # ── 展示识别结果并生成文件名 ──
    new_filename = _display_info_and_build_filename(info, pdf_path.name, ".pdf")
    new_path = pdf_path.parent / new_filename

    # 若目标已存在且不是文件本身，自动加 _01/_02 后缀消除冲突
    if new_path != pdf_path:
        resolved = _resolve_conflict_path(new_path)
        if resolved != new_path:
            print(f"  ⚠ 同名文件已存在，建议名称调整为：{resolved.name}")
            new_path = resolved
            new_filename = new_path.name

    return (pdf_path, info, new_filename, new_path)


def _analyze_epub(epub_path: Path, current_idx: int = 0, total_count: int = 0, original_filename: str = "") -> "tuple[Path, dict, str, Path] | None":
    """
    提取 EPUB 文字并调用 LLM 识别元信息，返回 (epub_path, info, new_filename, new_path)。
    失败时返回 None。

    original_filename: 若提供，覆盖传给 LLM 的文件名（可含文件夹名前缀）。
    """
    output_lines = []
    output_lines.append(f"\n{'═' * 62}")
    if total_count > 0:
        output_lines.append(f"  分析文件 ({current_idx}/{total_count})：{epub_path.name}")
    else:
        output_lines.append(f"  分析文件：{epub_path.name}")
    output_lines.append(f"  路径：{epub_path}")
    output_lines.append(f"{'═' * 62}")

    # ── 提取文字 ──
    pages = _extract_epub_texts(epub_path)

    if not pages:
        output_lines.append(f"  ❌ EPUB 无法提取有效文字。")
        print("\n".join(output_lines))
        return None

    output_lines.append(f"  📄 提取了 EPUB 的首尾文本片段")

    # ── LLM 查询 ──
    print("\n".join(output_lines))
    
    info = _query_llm_for_metadata(
        pages,
        pdf_path=epub_path,
        original_filename=original_filename or epub_path.name,
    )

    # ── 展示识别结果并生成文件名 ──
    new_filename = _display_info_and_build_filename(info, epub_path.name, ".epub")
    new_path = epub_path.parent / new_filename

    # 若目标已存在且不是文件本身，自动加 _01/_02 后缀消除冲突
    if new_path != epub_path:
        resolved = _resolve_conflict_path(new_path)
        if resolved != new_path:
            print(f"  ⚠ 同名文件已存在，建议名称调整为：{resolved.name}")
            new_path = resolved
            new_filename = new_path.name

    return (epub_path, info, new_filename, new_path)


def _analyze_djvu(djvu_path: Path, current_idx: int = 0, total_count: int = 0) -> "tuple[Path, dict, str, Path] | None":
    """
    提取 DjVu 文字并调用 LLM 识别元信息，返回 (djvu_path, info, new_filename, new_path)。
    失败时返回 None。
    """
    output_lines = []
    output_lines.append(f"\n{'═' * 62}")
    if total_count > 0:
        output_lines.append(f"  分析文件 ({current_idx}/{total_count})：{djvu_path.name}")
    else:
        output_lines.append(f"  分析文件：{djvu_path.name}")
    output_lines.append(f"  路径：{djvu_path}")
    output_lines.append(f"{'═' * 62}")

    # ── 提取文字 ──
    pages = _extract_djvu_texts(djvu_path)

    if not pages:
        output_lines.append(f"  ❌ DjVu 无法提取有效文字。")
        print("\n".join(output_lines))
        return None

    output_lines.append(f"  📄 提取了 DjVu 的首尾文本片段")

    # ── LLM 查询 ──
    print("\n".join(output_lines))

    info = _query_llm_for_metadata(
        pages,
        pdf_path=djvu_path,
        original_filename=djvu_path.name,
    )

    # ── 展示识别结果并生成文件名 ──
    new_filename = _display_info_and_build_filename(info, djvu_path.name, ".djvu")
    new_path = djvu_path.parent / new_filename

    # 若目标已存在且不是文件本身，自动加 _01/_02 后缀消除冲突
    if new_path != djvu_path:
        resolved = _resolve_conflict_path(new_path)
        if resolved != new_path:
            print(f"  ⚠ 同名文件已存在，建议名称调整为：{resolved.name}")
            new_path = resolved
            new_filename = new_path.name

    return (djvu_path, info, new_filename, new_path)


def _analyze_image_folder(
    folder_path: Path, image_files: list[Path], current_idx: int = 0, total_count: int = 0
) -> "tuple[Path, dict, str, Path] | None":
    """
    将图片文件夹视为一个 PDF 来分析：每张图片对应一页。
    提取 OCR 文字并调用 LLM 识别元信息。

    返回 (folder_path, info, new_foldername, new_path)，失败时返回 None。
    """
    output_lines = []
    output_lines.append(f"\n{'═' * 62}")
    if total_count > 0:
        output_lines.append(f"  分析图片文件夹 ({current_idx}/{total_count})：{folder_path.name}")
    else:
        output_lines.append(f"  分析图片文件夹：{folder_path.name}")
    output_lines.append(f"  路径：{folder_path}")
    output_lines.append(f"  📄 共 {len(image_files)} 张图片（按自然排序）")
    output_lines.append(f"{'═' * 62}")

    # 列出前几张图片
    show_count = min(5, len(image_files))
    for i, img in enumerate(image_files[:show_count]):
        output_lines.append(f"     第 {i+1} 页: {img.name}")
    if len(image_files) > show_count:
        output_lines.append(f"     ... 共 {len(image_files)} 张")

    # ── 构建页面列表（初始全为空字符串，OCR 在 _query_llm_for_metadata 中按需执行） ──
    all_pages = _extract_image_folder_page_texts(image_files)

    if not all_pages:
        output_lines.append(f"  ❌ 文件夹中没有图片。")
        print("\n".join(output_lines))
        return None

    print("\n".join(output_lines))

    # ── LLM 查询（传入 image_files 以便按需 OCR） ──
    info = _query_llm_for_metadata(
        all_pages,
        pdf_path=str(folder_path),
        original_filename=folder_path.name,
        image_files=image_files,
    )

    # ── 展示识别结果并生成文件夹名 ──
    # 文件夹不需要 .pdf 后缀
    new_filename = _display_info_and_build_filename(info, folder_path.name, suffix="")
    new_path = folder_path.parent / new_filename

    # 若目标已存在且不是文件夹本身，自动加 _01/_02 后缀消除冲突
    if new_path != folder_path:
        resolved = _resolve_conflict_path(new_path)
        if resolved != new_path:
            print(f"  ⚠ 同名文件夹已存在，建议名称调整为：{resolved.name}")
            new_path = resolved
            new_filename = new_path.name

    return (folder_path, info, new_filename, new_path)


def _display_info_and_build_filename(info: dict, original_filename: str, suffix: str = ".pdf") -> str:
    """
    展示 LLM 识别结果并生成文件名/文件夹名。

    Parameters:
        info:   LLM 识别到的元数据字典
        original_filename: 原文件名，用于在输出中标识
        suffix: 文件后缀（对于文件夹传空字符串）
    Returns:
        生成的文件名/文件夹名
    """
    print(f"\n  {'─' * 58}")
    print(f"  📋 [{original_filename}] 识别结果：")
    print(f"     第一作者姓氏:  {info.get('first_author_last_name', '') or '(空)'}")
    print(f"     期刊/出版社:   {info.get('publisher_or_journal', '') or '(空)'}")
    print(f"     年份:          {info.get('year', '') or '(空)'}")
    print(f"     标题:          {info.get('title', '') or '(空)'}")

    # 展示额外字段
    corr_list = info.get("corresponding_authors_last_names", []) or []
    if corr_list:
        print(f"     通讯作者姓氏:  {', '.join(corr_list)}")
    if info.get("is_thesis"):
        print(f"     学位论文:      是")
        if info.get("advisor_last_name"):
            print(f"     导师姓氏:      {info.get('advisor_last_name')}")
        if info.get("candidate_last_name"):
            print(f"     答辩人姓氏:    {info.get('candidate_last_name')}")

    missing = _get_missing_fields(info)
    if missing:
        print(f"     ⚠ 以下字段未能识别：{', '.join(missing)}")
    chinese_label = info.get("chinese_label", "").strip()
    if chinese_label:
        print(f"     中文注释:        {chinese_label}")

    # ── 展示识别结果并生成文件名 ──
    new_filename = _build_filename(info, suffix=suffix)

    print(f"\n  📝 [{original_filename}] 建议名称：{new_filename}")
    print(f"     （{len(new_filename)} 字符）")

    return new_filename


def _confirm_and_rename(
    pdf_path: Path,
    info: dict,
    new_filename: str,
    new_path: Path,
    yes_to_all: bool = False,
    is_folder: bool = False,
    folder_to_rename: "Path | None" = None,
    current_index: int = 0,
    total_count: int = 0,
) -> bool:
    """
    展示建议文件名/文件夹名，让用户确认后执行重命名。

    Parameters:
        is_folder: 若为 True，表示重命名的是文件夹而非文件。
        folder_to_rename: 若不为 None，在文件重命名成功后同步将此文件夹重命名为文件的新名称（无扩展名）。

    Returns:
        True 表示用户选择了「全部确认」，调用方应将后续文件也标记为 yes_to_all。
    """
    item_type = "文件夹" if is_folder else "文件"
    print(f"\n{'═' * 62}")
    print(f"  原{item_type}名：  {pdf_path.name}")
    print(f"  建议{item_type}名：{new_filename}")
    print(f"  （{len(new_filename)} 字符）")

    if new_path == pdf_path:
        print(f"  ✅ {item_type}名无需更改。")
        return yes_to_all

    if new_path.exists():
        print(f"  ⚠ 目标{item_type}已存在：{new_path}")

    # ── yes_to_all 模式：跳过询问直接重命名 ──
    if yes_to_all:
        print(f"  ⚡ 全部确认模式，直接重命名……")
    else:
        # ── 用户确认 ──
        if current_index > 0 and total_count > 0:
            print(f"\n  [Enter / y] 确认重命名 ({current_index}/{total_count})")
        else:
            print(f"\n  [Enter / y] 确认重命名")
        print(f"  [a]          全部确认（对后续所有{item_type}自动确认）")
        print(f"  [e]          手动编辑{item_type}名")
        print(f"  [s]          跳过此{item_type}")

        while True:
            try:
                choice = input("  请选择 > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  [已跳过]")
                return False

            if choice in ("", "y", "yes"):
                break
            elif choice == "a":
                print(f"  ⚡ 已启用全部确认模式，后续文件将自动重命名。")
                yes_to_all = True
                break
            elif choice == "e":
                try:
                    custom_name = input(f"  请输入新{item_type}名（不含路径）> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n  [已跳过]")
                    return False
                if custom_name:
                    if not is_folder and not custom_name.lower().endswith(pdf_path.suffix.lower()):
                        custom_name += pdf_path.suffix.lower()
                    new_filename = _sanitize_filename(custom_name)
                    new_path = pdf_path.parent / new_filename
                    # 自定义名也自动消除冲突
                    if new_path != pdf_path:
                        new_path = _resolve_conflict_path(new_path)
                        new_filename = new_path.name
                    print(f"  📝 使用自定义{item_type}名：{new_filename}")
                break
            elif choice == "s":
                print("  [已跳过]")
                return False
            else:
                print("  无效选项，请输入 y/Enter（确认）、a（全部确认）、e（编辑）、s（跳过）")

    # ── 执行重命名 ──
    # 再次检查冲突（以防确认期间其他文件占用了目标名）
    if new_path != pdf_path:
        resolved = _resolve_conflict_path(new_path)
        if resolved != new_path:
            print(f"  ⚠ 目标{item_type}名已被占用，自动修改为：{resolved.name}")
            new_path = resolved
            new_filename = new_path.name
    while True:
        try:
            pdf_path.rename(new_path)
            print(f"  ✅ 已重命名为：{new_filename}")
            _log_rename(pdf_path, new_filename)
            print(f"  📒 已记录至：{_RENAME_LOG}")
            # ── 同步重命名父文件夹 ──
            if folder_to_rename is not None:
                new_folder_name = _sanitize_filename(new_path.stem)
                new_folder_path = folder_to_rename.parent / new_folder_name
                if new_folder_path != folder_to_rename:
                    if new_folder_path.exists():
                        new_folder_path = _resolve_conflict_path(new_folder_path)
                    try:
                        folder_to_rename.rename(new_folder_path)
                        print(f"  ✅ 文件夹已重命名为：{new_folder_path.name}")
                        _log_rename(folder_to_rename, new_folder_path.name)
                    except OSError as _fe:
                        print(f"  ❌ 文件夹重命名失败：{_fe}")
                else:
                    print(f"  ✅ 文件夹名无需更改。")
            break
        except PermissionError as e:
            print(f"  ⚠ {item_type}被占用，无法重命名：{e}")
            print(f"  （记录将在重命名成功后才写入日志）")
            print(f"  [Enter / r] 重试   [s] 跳过")
            try:
                retry_choice = input("  请选择 > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  [已跳过]")
                break
            if retry_choice in ("", "r"):
                continue
            else:
                print("  [已跳过]")
                break
        except OSError as e:
            print(f"  ❌ 重命名失败：{e}")
            break

    return yes_to_all


def main() -> None:
    """
    主流程：
      1. 用 get_input_with_while_cycle 批量收集 PDF 文件或图片文件夹路径
      2. 依次调用 LLM 分析每个文件/文件夹
      3. 集中让用户确认并执行重命名
    """
    from Python_Lib.My_Lib_Stock import get_input_with_while_cycle

    print("=" * 62)
    print("  📚 文献 PDF / EPUB / 图片文件夹 重命名工具")
    print("  请依次输入 PDF/EPUB 文件路径或图片文件夹路径（支持拖拽），")
    print("  输入空行开始处理。")
    print("=" * 62)
    print()

    raw_inputs = get_input_with_while_cycle(
        input_prompt="  路径 > ",
        strip_quote=True,
    )

    if not raw_inputs:
        print("  未输入任何文件，退出。")
        return

    # ── 验证路径 ──
    # 每个条目为 (path, type) 其中 type = "pdf" 或 "image_folder"
    valid_items: list[tuple[Path, str, list[Path]]] = []  # (path, type, image_files)
    for raw in raw_inputs:
        item_path = Path(raw.strip())
        if not item_path.exists():
            print(f"  ❌ 路径不存在：{item_path}，已跳过。")
            continue

        if item_path.is_dir():
            # ── 文件夹：先检查是否有唯一 PDF 或 EPUB 文件 ──
            pdf_epub_in_folder = [
                f for f in item_path.iterdir()
                if f.is_file() and not f.name.startswith(".")
                and f.suffix.lower() in (".pdf", ".epub")
            ]
            if len(pdf_epub_in_folder) == 1:
                sole_file = pdf_epub_in_folder[0]
                ext = sole_file.suffix.lower()
                file_type = "pdf_in_folder" if ext == ".pdf" else "epub_in_folder"
                print(f"  📁 文件夹中发现唯一{'PDF' if ext == '.pdf' else 'EPUB'}文件：{sole_file.name}")
                valid_items.append((sole_file, file_type, [item_path]))
                continue
            # ── 文件夹：检查是否全部是图片 ──
            all_images, image_files, non_image_files = _check_image_folder(item_path)
            if not image_files:
                print(f"  ❌ 文件夹中没有图片文件：{item_path}，已跳过。")
                continue
            if not all_images:
                print(f"\n  ⚠ 文件夹 {item_path.name} 中存在非图片文件：")
                for nif in non_image_files[:10]:
                    print(f"     • {nif.name}")
                if len(non_image_files) > 10:
                    print(f"     ... 共 {len(non_image_files)} 个非图片文件")
                print(f"  共 {len(image_files)} 张图片、{len(non_image_files)} 个非图片文件。")
                try:
                    confirm = input("  是否仍然将图片文件作为页面处理？[y/N] > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n  [已跳过]")
                    continue
                if confirm not in ("y", "yes"):
                    print(f"  已跳过文件夹：{item_path}")
                    continue
            valid_items.append((item_path, "image_folder", image_files))
            continue

        if item_path.is_file():
            if item_path.suffix.lower() == ".pdf":
                valid_items.append((item_path, "pdf", []))
                continue
            elif item_path.suffix.lower() == ".epub":
                valid_items.append((item_path, "epub", []))
                continue
            elif item_path.suffix.lower() == ".djvu":
                valid_items.append((item_path, "djvu", []))
                continue
            else:
                print(f"  ⚠ 文件扩展名不是 .pdf、.epub 或 .djvu（{item_path.suffix}），已跳过。")
                continue

        print(f"  ❌ 无法识别的路径类型：{item_path}，已跳过。")

    if not valid_items:
        print("  没有有效的文件或文件夹，退出。")
        return

    # ── LLM 分析阶段 ──
    pdf_count = sum(1 for _, t, _ in valid_items if t == "pdf")
    epub_count = sum(1 for _, t, _ in valid_items if t == "epub")
    djvu_count = sum(1 for _, t, _ in valid_items if t == "djvu")
    folder_count = sum(1 for _, t, _ in valid_items if t == "image_folder")
    summary_parts = []
    if pdf_count:
        summary_parts.append(f"{pdf_count} 个 PDF 文件")
    if epub_count:
        summary_parts.append(f"{epub_count} 个 EPUB 文件")
    if djvu_count:
        summary_parts.append(f"{djvu_count} 个 DjVu 文件")
    if folder_count:
        summary_parts.append(f"{folder_count} 个图片文件夹")
    print(f"\n  共 {'、'.join(summary_parts)}，开始并发 LLM 分析（最多 10 线程）……")

    # results 中额外记录 is_folder 标记和可选的待重命名文件夹路径
    results: list[tuple[Path, dict, str, Path, bool, "Path | None"]] = []
    total_items = len(valid_items)
    
    import concurrent.futures

    def _process_item(args: tuple[int, tuple[Path, str, list[Path]]]) -> "tuple[Path, dict, str, Path, bool, Path | None] | None":
        idx, (item_path, item_type, image_files) = args
        if item_type == "pdf":
            res = _analyze_pdf(item_path, current_idx=idx, total_count=total_items)
            if res is not None:
                path, info, fname, new_path = res
                return (path, info, fname, new_path, False, None)
        elif item_type == "epub":
            res = _analyze_epub(item_path, current_idx=idx, total_count=total_items)
            if res is not None:
                path, info, fname, new_path = res
                return (path, info, fname, new_path, False, None)
        elif item_type == "djvu":
            res = _analyze_djvu(item_path, current_idx=idx, total_count=total_items)
            if res is not None:
                path, info, fname, new_path = res
                return (path, info, fname, new_path, False, None)
        elif item_type in ("pdf_in_folder", "epub_in_folder"):
            folder_path = image_files[0]  # 存储原始文件夹路径
            combined_name = f"{folder_path.name} / {item_path.name}"
            if item_type == "pdf_in_folder":
                res = _analyze_pdf(item_path, current_idx=idx, total_count=total_items, original_filename=combined_name)
            else:
                res = _analyze_epub(item_path, current_idx=idx, total_count=total_items, original_filename=combined_name)
            if res is not None:
                path, info, fname, new_path = res
                return (path, info, fname, new_path, False, folder_path)
        else:  # image_folder
            res = _analyze_image_folder(item_path, image_files, current_idx=idx, total_count=total_items)
            if res is not None:
                path, info, fname, new_path = res
                return (path, info, fname, new_path, True, None)
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有任务
        futures = [
            executor.submit(_process_item, (idx, item))
            for idx, item in enumerate(valid_items, start=1)
        ]
        # 收集结果（保持原始顺序）
        for future in futures:
            res = future.result()
            if res is not None:
                results.append(res)

    if not results:
        print("\n  所有文件/文件夹均分析失败，退出。")
        return

    # ── 集中确认阶段 ──
    print(f"\n{'=' * 62}")
    print(f"  ✅ 分析完毕，共 {len(results)} 项，开始逐一确认重命名……")
    print(f"{'=' * 62}")

    yes_to_all = False
    total_count = len(results)
    for idx, (item_path, info, new_filename, new_path, is_folder, folder_to_rename) in enumerate(results, start=1):
        yes_to_all = _confirm_and_rename(
            item_path, info, new_filename, new_path, yes_to_all,
            is_folder=is_folder,
            folder_to_rename=folder_to_rename,
            current_index=idx,
            total_count=total_count,
        )

    # ── 打印 Token 消耗和预估费用 ──
    in_tok = int(_total_input_chars / _AVG_CHARS_PER_TOKEN)
    out_tok = int(_total_output_chars / _AVG_CHARS_PER_TOKEN)
    price = _MODEL_PRICE_PER_M_TOKENS.get(GEMINI_FLASH, (0, 0))
    cost = (in_tok * price[0] + out_tok * price[1]) / 1_000_000

    print(f"\n{'=' * 62}")
    print(f"  💰 总计消耗：输入 ~{in_tok:,} tokens，输出 ~{out_tok:,} tokens")
    print(f"  💸 预估总支出：~${cost:.4f}")
    print(f"{'=' * 62}")

    print("\n  👋 完成！")


if __name__ == "__main__":
    main()
