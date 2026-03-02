"""
Content_Based_Literature_Categorization.py — 基于内容的文献分类工具

══════════════════════════════════════════════════════════════════════════════
  提供两个工作模式：

  1. summarize — 总结分类特征
     扫描根目录下所有子文件夹，读取文件夹名称和其中已有文献的文件名，
     利用 LLM 为每个分类生成描述，保存到 `0 Category Note.json`。
     修改已有 description 前会征得用户同意。

  2. categorize — 自动分类文献
     读取 PDF / EPUB / 图片文件夹的内容，将内容片段 + 文件名 + 所有分类描述
     发送给 LLM（Flash 模型），判断文献应归属的类别（支持多分类）。
     用户确认或修改后，将文件复制到所有目标分类文件夹中，然后删除原文件
     （除非原文件已在某个正确分类中）。

  分类体系说明：
     文件夹名称使用「维度 - 子类」的分层结构，例如：
       对象 - Reptiles - Lizards
       影响 - Vitamin D
       光源 - LED
     LLM 会理解这种层级关系，自动选择最具体的匹配子类。

使用方式：
    python -m LLM_Lib.Skills.Content_Based_Literature_Categorization [mode]
    mode 可选: summarize, categorize

    无参数时交互式选择模式。
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import threading
import unicodedata
from pathlib import Path

from LLM_Lib.LLM import (
    call_gemini,
    extract_json,
    GEMINI_PRO,
    GEMINI_FLASH,
    _AVG_CHARS_PER_TOKEN,
    _MODEL_PRICE_PER_M_TOKENS,
)
from LLM_Lib.Skills.Rename_Ref import (
    _extract_page_texts,
    _extract_epub_texts,
    _extract_djvu_texts,
    _check_image_folder,
    _extract_text_from_image,
    _is_blank,
    _ocr_single_page,
)

# ── Markdown 终端渲染 ──────────────────────────────────────────────────────
_TERM_WIDTH = 100
_INDENT = "     "


def _print_summary_in_terminal(text: str, indent: str = _INDENT) -> None:
    """
    将 LLM 返回的 Markdown 摘要简易渲染后打印到终端：
      - ## / ### 标题 → 加粗标识行
      - **text** → 大写或直接去掉星号
      - 正文段落按终端宽度重新折行
    """
    wrap_width = max(40, _TERM_WIDTH - len(indent))

    def _strip_inline(s: str) -> str:
        # **bold** / *italic* → 去除标记
        s = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', s)
        return s

    paragraphs = re.split(r'\n{2,}', text.strip())
    first = True
    for para in paragraphs:
        lines = para.splitlines()
        if not lines:
            continue
        first_line = lines[0].strip()
        # 标题行
        m = re.match(r'^(#{1,3})\s+(.*)', first_line)
        if m:
            title = _strip_inline(m.group(2))
            print(f"\n{indent}── {title} ──")
            body_lines = lines[1:]
        else:
            body_lines = lines

        # 合并剩余行为一段文字再折行
        body = ' '.join(_strip_inline(l.strip()) for l in body_lines if l.strip())
        if body:
            if not first or m:
                pass
            for wrapped in textwrap.wrap(body, wrap_width):
                print(f"{indent}{wrapped}")
        first = False


# ── 全局 Token 统计 ─────────────────────────────────────────────────────────
_token_lock = threading.Lock()
_total_input_chars = 0
_total_output_chars = 0

# ── 常量 ────────────────────────────────────────────────────────────────────
CATEGORY_NOTE_FILENAME = "0 Category Note.json"


# ── 分类名计算辅助 ───────────────────────────────────────────────────────────

def _strip_num_prefix(name: str) -> str:
    """删除文件夹名称中形如 '0 '、'12 '、'0' 等开头的数字前缀（含后续空格）。"""
    return re.sub(r"^[0-9]+\s*", "", name)


def _find_common_root(paths: list[Path]) -> Path:
    """返回一组路径的最近公共祖先目录。"""
    if not paths:
        return Path(".")
    try:
        common = Path(os.path.commonpath([str(p) for p in paths]))
        return common if common.is_dir() else common.parent
    except ValueError:
        return paths[0].parent


def _folder_to_category_name(folder: Path, root: Path) -> str:
    """
    计算 folder 相对于 root 的分类名。
    - 取相对路径的各个部分
    - 每个部分去除数字前缀
    - 用 ' - ' 连接
    例: root=…/文献_Private/, folder=…/文献_Private/0 光照、体色/测量 - 显色
        → '光照、体色 - 测量 - 显色'
    """
    try:
        rel = folder.relative_to(root)
    except ValueError:
        return _strip_num_prefix(folder.name)
    parts = [_strip_num_prefix(p) for p in rel.parts]
    parts = [p for p in parts if p]
    return " - ".join(parts) if parts else _strip_num_prefix(folder.name)


# 发送给 LLM 的文本字符数上限（节约 token）
_MAX_TEXT_CHARS = 4000
# 每个文件夹最多读取多少个文件名用于生成 description
_MAX_FILES_FOR_SUMMARY = 100


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  工具函数                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _display_width(s: str) -> int:
    """计算字符串的显示宽度（中文/全角字符占 2 列）。"""
    width = 0
    for ch in s:
        cat = unicodedata.east_asian_width(ch)
        width += 2 if cat in ("W", "F") else 1
    return width


def _pad_to_width(s: str, target_width: int) -> str:
    """将字符串用空格填充到目标显示宽度。"""
    current = _display_width(s)
    return s + " " * max(0, target_width - current)


def _print_multi_column(items: list[str], indent: str = "    ", terminal_width: int = 120) -> None:
    """
    以多列格式打印带编号的列表项。
    序号先按列排列（先填满第一列再第二列），列之间对齐。
    """
    if not items:
        return

    # 计算每个条目带编号后的文本
    num_width = len(str(len(items)))
    entries = [f"{i + 1:>{num_width}}. {name}" for i, name in enumerate(items)]

    # 计算单列最大显示宽度
    max_entry_width = max(_display_width(e) for e in entries)
    col_width = max_entry_width + 3  # 列间距

    # 可用宽度
    usable = terminal_width - len(indent)
    num_cols = max(1, usable // col_width)

    # 按列优先排列：计算每列行数
    num_rows = (len(entries) + num_cols - 1) // num_cols

    for row in range(num_rows):
        parts = []
        for col in range(num_cols):
            idx = col * num_rows + row
            if idx < len(entries):
                parts.append(_pad_to_width(entries[idx], col_width))
        print(indent + "".join(parts).rstrip())


def _write_temp_markdown(
    file_path: Path,
    document_summary: str,
    assigned: list[dict],
    maybe_relevant: list[dict],
) -> Path:
    """
    将分类结果写入临时 Markdown 文件并返回路径。
    """
    lines = [
        f"# 📚 文献分类结果",
        f"",
        f"**文件**: `{file_path.name}`  ",
        f"**路径**: `{file_path}`",
        f"",
    ]

    if document_summary:
        lines.append("---")
        lines.append("")
        lines.append("## 📖 文献摘要")
        lines.append("")
        lines.append(document_summary)
        lines.append("")

    if assigned:
        lines.append("---")
        lines.append("")
        lines.append("## ✅ LLM 建议的分类")
        lines.append("")
        for cat in assigned:
            reason = f" — {cat['reason']}" if cat.get("reason") else ""
            lines.append(f"- **{cat['name']}**{reason}")
        lines.append("")

    if maybe_relevant:
        lines.append("---")
        lines.append("")
        lines.append("## 🤔 或许相关的分类")
        lines.append("")
        for cat in maybe_relevant:
            reason = f" — {cat['reason']}" if cat.get("reason") else ""
            lines.append(f"- **{cat['name']}**{reason}")
        lines.append("")

    content = "\n".join(lines)

    # 写入临时文件
    raw_stem = file_path.stem if file_path.is_file() else file_path.name
    # 去除空格、截断到 25 字符
    stem = raw_stem.replace(" ", "_")[:25]
    tmp_dir = Path(tempfile.gettempdir()) / "literature_categorization"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    md_path = tmp_dir / f"{stem}_cat.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    return md_path

def _track_tokens(input_text: str, output_text: str) -> None:
    """累计输入/输出字符数用于费用统计。"""
    global _total_input_chars, _total_output_chars
    with _token_lock:
        _total_input_chars += len(input_text)
        _total_output_chars += len(output_text)


def _save_summary_markdown(file_path: Path, document_summary: str) -> Path | None:
    """
    将文献摘要保存到 {file_path.parent}/LLM Summary/{file_path.stem}_Summary.md。
    若 document_summary 为空则跳过。
    返回保存路径，失败或跳过返回 None。
    """
    if not document_summary:
        return None
    stem = file_path.stem if file_path.is_file() else file_path.name
    summary_dir = file_path.parent / "LLM Summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_path = summary_dir / f"{stem}_Summary.md"
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(document_summary)
        print(f"  💾 摘要已保存到: {save_path}")
        return save_path
    except Exception as e:
        print(f"  ⚠ 摘要保存失败: {e}")
        return None



def _load_category_note(folder: Path) -> dict:
    """从文件夹中读取 0 Category Note.json，失败或不存在返回空 dict。"""
    json_path = folder / CATEGORY_NOTE_FILENAME
    if not json_path.exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"  ⚠ 读取 {json_path} 失败: {e}")
    return {}


def _save_category_note(folder: Path, data: dict) -> None:
    """将分类信息写入 0 Category Note.json。"""
    json_path = folder / CATEGORY_NOTE_FILENAME
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 已保存: {json_path}")


def _list_literature_files(folder: Path, max_count: int = _MAX_FILES_FOR_SUMMARY) -> list[str]:
    """列出文件夹中的文献文件名和子文件夹名（排除 Category Note 和隐藏文件）。"""
    files = []
    try:
        for f in sorted(folder.iterdir()):
            if f.name == CATEGORY_NOTE_FILENAME or f.name.startswith("."):
                continue
            if f.is_file() or f.is_dir():
                files.append(f.name)
                if len(files) >= max_count:
                    break
    except PermissionError:
        pass
    return files


def _collect_all_categories(root: Path) -> tuple[dict[str, dict], dict[str, Path]]:
    """
    扫描根目录下所有子文件夹，返回 ({分类名: category_note_data}, {分类名: Path})。
    分类名为相对 root 的路径各部分去除数字前缀后以 ' - ' 连接的结果。
    排除 Uncategorized 文件夹。
    """
    categories: dict[str, dict] = {}
    paths: dict[str, Path] = {}
    for sub_dir in sorted(root.iterdir()):
        if not sub_dir.is_dir():
            continue
        if sub_dir.name.lower() in ("uncategorized",):
            continue
        cat_name = _folder_to_category_name(sub_dir, root)
        note = _load_category_note(sub_dir)
        categories[cat_name] = note
        paths[cat_name] = sub_dir
    return categories, paths


def _collect_categories_from_txt(txt_path: Path) -> tuple[dict[str, dict], dict[str, Path]]:
    """
    从 Categories txt 文件读取分类文件夹列表。
    每行为一个分类文件夹的绝对或相对路径（相对路径以 txt 文件所在目录为基准）。
    以 # 开头的行视为注释，空行跳过。
    分类名为所有路径的公共根目录到各文件夹的相对路径，各部分去除数字前缀后以 ' - ' 连接。
    返回 ({分类名: category_note_data}, {分类名: Path})。
    """
    categories: dict[str, dict] = {}
    paths: dict[str, Path] = {}

    try:
        with open(txt_path, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  ❌ 读取 Categories 文件失败: {e}")
        return categories, paths

    # 第一遍：收集所有有效文件夹
    valid_folders: list[Path] = []
    for raw_line in lines:
        line = raw_line.strip().strip("\"'")
        if not line or line.startswith("#"):
            continue
        folder = Path(line)
        if not folder.is_absolute():
            folder = txt_path.parent / folder
        if not folder.is_dir():
            print(f"  ⚠ 跳过不存在的文件夹: {folder}")
            continue
        valid_folders.append(folder)

    # 确定公共根目录
    common_root = _find_common_root(valid_folders) if valid_folders else txt_path.parent

    # 第二遍：用完整相对路径构建分类名
    for folder in valid_folders:
        cat_name = _folder_to_category_name(folder, common_root)
        if cat_name in paths:
            print(f"  ⚠ 存在同名分类 [{cat_name}]，")
            print(f"      已有: {paths[cat_name]}")
            print(f"      重复: {folder}  → 跳过后者")
            continue
        note = _load_category_note(folder)
        categories[cat_name] = note
        paths[cat_name] = folder

    return categories, paths


def _ask_categories_source() -> tuple[dict[str, dict], dict[str, Path]] | None:
    """
    询问用户分类来源：根目录 或 Categories txt 文件。
    返回 (all_categories, all_category_paths)，失败或放弃返回 None。
    """
    print(f"\n  请选择分类来源：")
    print(f"    [1] 根目录（扫描子文件夹作为分类）")
    print(f"    [2] Categories 文件（每行一个分类文件夹路径）")
    src_choice = input("  请输入 1 或 2: ").strip()

    if src_choice == "1":
        root_input = input("\n  请输入包含各个分类文件夹的根目录路径: ").strip().strip("\"'")
        if not root_input:
            return None
        root = Path(root_input)
        if not root.is_dir():
            print("  ❌ 路径不存在或不是文件夹！")
            return None
        all_categories, all_category_paths = _collect_all_categories(root)
        if not all_categories:
            print("  ❌ 未找到任何分类子文件夹！")
            return None
        return all_categories, all_category_paths

    elif src_choice == "2":
        txt_input = input("\n  请输入 Categories txt 文件路径: ").strip().strip("\"'")
        if not txt_input:
            return None
        txt_path = Path(txt_input)
        if not txt_path.is_file():
            print("  ❌ 文件不存在！")
            return None
        all_categories, all_category_paths = _collect_categories_from_txt(txt_path)
        if not all_categories:
            print("  ❌ 未从 Categories 文件中读取到任何有效分类文件夹！")
            return None
        return all_categories, all_category_paths

    else:
        print("  无效选择。")
        return None


def _build_category_list_text(categories: dict[str, dict]) -> str:
    """
    将分类信息格式化为 LLM 可读的文本。
    按维度（中文前缀）分组显示。
    """
    # 按维度分组
    groups: dict[str, list[tuple[str, str]]] = {}
    for name, note in categories.items():
        desc = note.get("description", "")
        # 提取维度前缀（如 "对象"、"影响"、"光源" 等）
        parts = name.split(" - ")
        axis = parts[0] if parts else name
        if axis not in groups:
            groups[axis] = []
        groups[axis].append((name, desc))

    lines = []
    for axis, items in groups.items():
        lines.append(f"\n## {axis}")
        for name, desc in items:
            if desc:
                lines.append(f"  - **{name}**: {desc}")
            else:
                lines.append(f"  - **{name}**")
    return "\n".join(lines)


def _extract_text_from_file(file_path: Path, max_chars: int = _MAX_TEXT_CHARS) -> str:
    """
    从 PDF / EPUB / 图片文件夹 中提取文本内容。
    返回截断到 max_chars 的文本。
    """
    text = ""

    if file_path.is_file():
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            try:
                pages = _extract_page_texts(file_path)
            except Exception as e:
                print(f"  ⚠ PDF 读取失败: {e}")
                return ""
            # 从前往后逐页拼接，直到达到字数上限；空白页尝试 OCR
            collected = []
            total_len = 0
            for page_idx, p in enumerate(pages):
                p = p.strip()
                if _is_blank(p):
                    # 页面无文字，尝试对图片页进行 OCR
                    ocr_text, did_ocr = _ocr_single_page(file_path, page_idx, p)
                    if did_ocr:
                        p = ocr_text.strip()
                if not p:
                    continue
                if total_len + len(p) > max_chars:
                    # 取该页的前一部分
                    remaining = max_chars - total_len
                    if remaining > 100:
                        collected.append(p[:remaining])
                    break
                collected.append(p)
                total_len += len(p)
            text = "\n\n".join(collected)

        elif suffix == ".epub":
            try:
                parts = _extract_epub_texts(file_path)
            except Exception as e:
                print(f"  ⚠ EPUB 读取失败: {e}")
                return ""
            text = "\n\n".join(parts)

        elif suffix == ".djvu":
            try:
                parts = _extract_djvu_texts(file_path)
            except Exception as e:
                print(f"  ⚠ DjVu 读取失败: {e}")
                return ""
            text = "\n\n".join(parts)

        else:
            print(f"  ⚠ 不支持的文件格式: {suffix}")
            return ""

    elif file_path.is_dir():
        is_img, img_files, _ = _check_image_folder(file_path)
        if is_img and img_files:
            ocr_parts = []
            total_len = 0
            for img in img_files[:5]:  # 最多 OCR 前5张
                ocr_text = _extract_text_from_image(img)
                if ocr_text.strip():
                    ocr_parts.append(ocr_text.strip())
                    total_len += len(ocr_text)
                    if total_len >= max_chars:
                        break
            text = "\n\n".join(ocr_parts)
        else:
            print(f"  ⚠ 文件夹不是纯图片文件夹。")
            return ""

    # 截断
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  模式 1: 总结分类特征                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_SUMMARIZE_SYSTEM_PROMPT = """\
You are an expert librarian and research literature organizer. Your task is to \
generate concise, strict criteria-based specifications for what literature \
belongs in a given category. You will be provided with the folder's name, its existing \
literature, and descriptions of other related categories.

The folder names follow a hierarchical naming convention using " - " as separator:
  维度 - 子类 - 更细子类
For example:
  "对象 - Reptiles - Lizards" means: Subject → Reptiles → Lizards
  "影响 - Vitamin D" means: Effects/Impact → Vitamin D
  "光源 - LED" means: Light source → LED
  "生理 - 视觉 - 视网膜，传感器" means: Physiology → Vision → Retina, sensors

Output ONLY a JSON object with these keys:
- "description": The strictly actionable criteria for inclusion. \
  CRITICAL RULES FOR "description": \
  1. NO FLUFF WORDS. DO NOT use phrases like "This category contains...", "The literature includes...", "This folder is for...", etc. Start IMMEDIATELY with the core criteria (e.g., "Criteria: Must be ..."). \
  2. State explicit, actionable standards for inclusion. \
  3. Define the UNIQUE ANGLE/PERSPECTIVE of this category. Categories are not mutually exclusive (e.g., one categorizes by species, another by physiology), but the angle must be distinct. \
  4. Explicitly distinguish it from other easily confusable categories provided in the context. \
  5. Provide concise examples of UNQUALIFIED cases (e.g., "Unqualified: specific research on a single snake species, study on reptile lighting"). \
  6. Keep it extremely brief and direct. \
  7. HARD LIMIT: the entire "description" string MUST be 300 characters or fewer (count every character including spaces and punctuation). If your draft is longer, cut ruthlessly until it fits.
- "keywords": An array of 3-8 English keywords that characterize this category.

Do NOT include any markdown formatting around the JSON object or any extra text."""


def _summarize_single_folder(
    folder: Path,
    all_categories: dict[str, dict] | None = None,
    category_name: str | None = None,
) -> dict | None:
    """
    对单个文件夹生成分类描述。

    Args:
        folder: 要总结的文件夹路径。
        all_categories: 所有分类的 {name: note} dict（可选，用于提供上下文）。
        category_name: 当前文件夹的分类名（全路径形式，不提供时退化为 folder.name）。

    Returns:
        生成的分类信息 dict，失败返回 None。
    """
    display_name = category_name or folder.name
    files = _list_literature_files(folder)
    existing = _load_category_note(folder)

    # 构建 Prompt
    file_list = "\n".join(f"  - {f}" for f in files) if files else "  (empty folder)"

    # 提供同级文件夹列表及其现有描述作为上下文，帮助 LLM 理解分类边界
    sibling_context = ""
    if all_categories:
        siblings_info = []
        for name, note in all_categories.items():
            if name != display_name:
                desc = note.get("description", "")
                if desc:
                    siblings_info.append(f"  - {name}: {desc}")
                else:
                    siblings_info.append(f"  - {name}")
                    
        if siblings_info:
            sibling_context = (
                "\n\nFor context, here are the other categories in this collection "
                "(to help you understand the classification boundaries and avoid overlapping):\n"
                + "\n".join(siblings_info)
            )

    prompt = (
        f"Folder path: {folder.resolve()}\n"
        f"Category name: {display_name}\n"
        f"Folder name: {folder.name}\n\n"
        f"Literature files in this folder ({len(files)} shown):\n"
        f"{file_list}"
        f"{sibling_context}"
    )

    response = call_gemini(
        prompt,
        model=GEMINI_PRO,
        system_prompt=_SUMMARIZE_SYSTEM_PROMPT,
        reasoning=True,
        confirm=False,
        stream=True,
        temperature=0.2,
    )
    _track_tokens(prompt + _SUMMARIZE_SYSTEM_PROMPT, response)

    new_data = extract_json(response)
    if not isinstance(new_data, dict) or "description" not in new_data:
        print(f"  ⚠ [{display_name}] LLM 返回格式不正确。")
        return None

    return new_data


def mode_summarize_category() -> None:
    """模式 1: 总结分类特征 — 扫描根目录下所有子文件夹并生成描述。"""
    print(f"\n{'═' * 62}")
    print(f"  📂 模式 1: 总结分类特征")
    print(f"{'═' * 62}")

    result = _ask_categories_source()
    if result is None:
        return
    all_categories, all_category_paths = result

    print(f"\n  找到 {len(all_categories)} 个分类文件夹。")

    # 列出需要生成/更新描述的文件夹
    needs_new: list[str] = []
    has_desc: list[str] = []
    for name, note in all_categories.items():
        if note.get("description"):
            has_desc.append(name)
        else:
            needs_new.append(name)

    if needs_new:
        print(f"  🆕 {len(needs_new)} 个文件夹尚无描述。")
    if has_desc:
        print(f"  ✅ {len(has_desc)} 个文件夹已有描述。")

    # 选择操作范围
    print(f"\n  请选择操作范围：")
    print(f"    [1] 仅为尚无描述的文件夹生成 ({len(needs_new)} 个)")
    print(f"    [2] 为所有文件夹重新生成 ({len(all_categories)} 个)")
    print(f"    [3] 指定多个文件夹（每行一个路径，输入 end 结束）")
    choice = input("  请输入 1/2/3: ").strip()

    if choice == "1":
        target_names = needs_new
    elif choice == "2":
        target_names = list(all_categories.keys())
    elif choice == "3":
        print("  请输入文件夹路径（每行一个），输入 end 结束：")
        target_names = []
        while True:
            line = input("  路径 > ").strip().strip("\"'")
            if line.lower() == "end":
                break
            if not line:
                continue
            folder_path = Path(line)
            if not folder_path.is_dir():
                # 尝试作为已知分类名称
                if line in all_category_paths:
                    folder_path = all_category_paths[line]
                else:
                    print(f"  ❌ 文件夹不存在: {line}，跳过。")
                    continue
            # 从 all_category_paths 反查分类名
            cat_name = next((n for n, p in all_category_paths.items() if p.resolve() == folder_path.resolve()), None)
            if cat_name is None:
                print(f"  ❌ 此文件夹不在已知分类列表中: {folder_path}，跳过。")
                continue
            if cat_name in target_names:
                print(f"  ⚠ 已添加过: {cat_name}，跳过。")
                continue
            target_names.append(cat_name)
            print(f"  ✓ 已添加: {cat_name}")
        if not target_names:
            print("  未指定任何有效文件夹。")
            return
    else:
        print("  无效选择。")
        return

    if not target_names:
        print("  没有需要处理的文件夹。")
        return

    print(f"\n  将为以下 {len(target_names)} 个文件夹生成描述：")
    for name in target_names:
        print(f"    - {name}")
    print()

    def _process_one_summarize(name: str) -> dict | None:
        """调用 LLM 为单个分类生成描述，返回 new_data 或 None。"""
        folder = all_category_paths[name]
        new_data = _summarize_single_folder(folder, all_categories, category_name=name)
        if new_data is None:
            return None
        return new_data

    def _confirm_and_save(name: str, new_data: dict) -> None:
        """展示生成结果，询问用户是否覆盖，保存。"""
        folder = all_category_paths[name]
        existing = _load_category_note(folder)

        print(f"  📋 新生成描述 ({len(new_data['description'])} 字符): {new_data['description']}")
        if new_data.get("keywords"):
            print(f"  🏷️  关键词: {', '.join(new_data['keywords'])}")

        if existing.get("description"):
            print(f"  📋 现有描述: {existing['description']}")
            confirm = input("  是否覆盖现有 description? [y/N]: ").strip().lower()
            if confirm not in ("y", "yes"):
                new_data["description"] = existing["description"]
                print("  ℹ️  保留原有 description。")

        merged = {**existing, **new_data}
        _save_category_note(folder, merged)

    # ── 处理第一个文件夹 ────────────────────────────────────────────
    first_name = target_names[0]
    first_folder = all_category_paths[first_name]
    first_existing = _load_category_note(first_folder)

    print(f"\n{'─' * 62}")
    print(f"  [1/{len(target_names)}] {first_name}")
    if first_existing.get("description"):
        print(f"  📋 现有描述: {first_existing['description']}")
    print(f"  ⏳ 正在调用 LLM...")

    first_data = _process_one_summarize(first_name)
    if first_data is not None:
        _confirm_and_save(first_name, first_data)

    remaining_names = target_names[1:]
    if not remaining_names:
        return  # 只有一个，已处理完

    # ── 若只有两个分类，直接顺序处理第二个 ─────────────────────────
    if len(target_names) == 2:
        name = remaining_names[0]
        folder = all_category_paths[name]
        existing = _load_category_note(folder)
        print(f"\n{'─' * 62}")
        print(f"  [2/2] {name}")
        if existing.get("description"):
            print(f"  📋 现有描述: {existing['description']}")
        print(f"  ⏳ 正在调用 LLM...")
        d = _process_one_summarize(name)
        if d is not None:
            _confirm_and_save(name, d)
        return

    # ── 超过两个分类：询问是否并行处理剩余 ──────────────────────────
    parallel_ans = input(
        f"\n  还有 {len(remaining_names)} 个分类待处理，"
        f"是否并行发送所有请求（10 线程）？[Y/n]: "
    ).strip().lower()

    if parallel_ans in ("n", "no"):
        # 顺序处理
        for i, name in enumerate(remaining_names, 2):
            folder = all_category_paths[name]
            existing = _load_category_note(folder)
            print(f"\n{'─' * 62}")
            print(f"  [{i}/{len(target_names)}] {name}")
            if existing.get("description"):
                print(f"  📋 现有描述: {existing['description']}")
            print(f"  ⏳ 正在调用 LLM...")
            d = _process_one_summarize(name)
            if d is not None:
                _confirm_and_save(name, d)
    else:
        # 并行处理所有剩余
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"\n  ⏳ 并行调用 LLM（{len(remaining_names)} 个，10 线程）...")
        parallel_results: dict[str, dict | None] = {}
        _par_lock = threading.Lock()
        _par_done = 0

        def _par_worker(cat_name: str) -> None:
            nonlocal _par_done
            result = _process_one_summarize(cat_name)
            with _par_lock:
                parallel_results[cat_name] = result
                _par_done += 1
                print(f"  ✓ 已完成 {_par_done}/{len(remaining_names)}: {cat_name}")

        with ThreadPoolExecutor(max_workers=10) as pool:
            fts = [pool.submit(_par_worker, n) for n in remaining_names]
            for ft in fts:
                ft.result()

        # 统一向用户确认
        print(f"\n{'═' * 62}")
        print(f"  📋 并行生成完成，逐一确认后保存...")
        print(f"{'═' * 62}")
        for i, name in enumerate(remaining_names, len(target_names) - len(remaining_names) + 1):
            d = parallel_results.get(name)
            print(f"\n{'─' * 62}")
            print(f"  [{i}/{len(target_names)}] {name}")
            if d is None:
                print(f"  ❌ LLM 调用失败，跳过。")
                continue
            _confirm_and_save(name, d)



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  模式 2: 自动分类文献                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_CATEGORIZE_SYSTEM_PROMPT = """\
You are an expert academic literature classifier. Given a document's content \
excerpt and filename, determine which categories it belongs to.

IMPORTANT classification rules:
1. A document can belong to MULTIPLE categories across different dimensions \
   (e.g., it can be about a specific animal AND about a specific wavelength \
   AND about a specific physiological effect simultaneously).
2. The categories use a hierarchical naming convention: "维度 - 子类 - 更细子类".
   When a document matches a more specific subcategory, assign it to the \
   SPECIFIC subcategory (e.g., "对象 - Reptiles - Lizards" rather than just \
   "对象 - Reptiles"). However, if it covers the broader topic in general, \
   assign the parent category.
3. Consider ALL dimensions when classifying:
   - 对象 (Subject): What organism(s) is the study about?
   - 波段 (Wavelength band): Does it focus on UV, infrared, etc.?
   - 影响 (Effects): What biological effects are studied?
   - 生理 (Physiology): What physiological systems are involved?
   - 光源 (Light source): Does it discuss specific light sources?
   - 测量 (Measurement): Does it cover measurement methodologies?
4. Be thorough — a paper about "UV vision in frogs" should at minimum match \
   subject (amphibians), wavelength (UV), and effect/physiology categories. \
   A paper studying spectral sensitivity of photoreceptors relates to both \
   vision physiology AND color vision effects.
5. Err on the side of inclusion: assign a category if the document's content \
   is meaningfully relevant, even if the category is not the paper's primary \
   focus. Only exclude a category if it is merely tangentially mentioned \
   (e.g., a single passing reference).

Output ONLY a JSON object with these keys:
- "document_summary": 用中文撰写的论文内容摘要，必须使用结构良好的 Markdown 格式。\
  格式要求如下：\
  (1) 首段：用1-2句话总述（包含作者、年份、期刊），点明研究核心，对**关键术语**加粗。\
  (2) 正文：按内容拆分为若干 ### 编号小节（如：### 1. 研究背景 / ### 2. 实验方法 / \
  ### 3. 主要结果 / ### 4. 结论与意义），每节内容充实，层次分明。\
  (3) 数据呈现：若文献包含对比数据，优先使用 Markdown 表格展示；\
  列表项用 "* **关键词**：说明" 格式，数值带单位。\
  (4) 数学/物理量：行内公式用 $...$ （如 $D_3$、$\mu W/cm^2$），独立公式用 $$...$$。\
  (5) 详细程度：类似于向同行解释这篇文献的价值和核心发现，忠实原文数据，\
  避免泛泛而谈。整体长度视文献信息量而定，通常 300–600 字。
- "categories": An array of objects, each with keys "name" (exact category \
  folder name) and "reason" (用中文1句话说明为何归入此分类).
- "maybe_relevant": An array of objects with the same structure as "categories", \
  listing categories that have LOWER relevance — the document touches on the \
  topic but it is not a primary focus. These are categories the user might \
  want to include but that you are less confident about.

Do NOT include any text outside the JSON object."""


def _classify_single_file(
    file_path: Path,
    categories_text: str,
    category_names: list[str],
) -> tuple[list[dict], list[dict], str]:
    """
    对单个文件进行分类。

    Returns:
        (assigned_categories, maybe_relevant, document_summary)
        其中 assigned_categories 和 maybe_relevant 都是 [{"name": str, "reason": str}, ...] 列表。
    """
    print(f"  ⏳ 提取文本...")
    text = _extract_text_from_file(file_path)
    if not text.strip():
        print(f"  ⚠ 未能提取到有效文本。")
        return [], [], ""

    print(f"  📄 提取了 {len(text):,} 字符的文本")

    prompt = (
        f"## Document Information\n\n"
        f"**Filename**: {file_path.name}\n"
        f"**File location**: {file_path.parent}\n\n"
        f"## Document Content (first ~{_MAX_TEXT_CHARS} characters)\n\n"
        f"{text}\n\n"
        f"## Available Categories\n"
        f"{categories_text}\n"
    )

    print(f"  ⏳ 正在调用 LLM 进行分类...")
    response = call_gemini(
        prompt,
        model=GEMINI_PRO,
        system_prompt=_CATEGORIZE_SYSTEM_PROMPT,
        reasoning=False,
        confirm=False,
        stream=False,
        temperature=0.1,
        allow_cancel=False,
    )
    _track_tokens(prompt + _CATEGORIZE_SYSTEM_PROMPT, response)

    parsed = extract_json(response)
    if not isinstance(parsed, dict):
        print(f"  ⚠ LLM 返回格式不正确。")
        return [], [], ""

    raw_categories = parsed.get("categories", [])
    raw_maybe = parsed.get("maybe_relevant", [])
    document_summary = parsed.get("document_summary", "")

    # 标准化: 支持旧格式 (字符串列表) 和新格式 (对象列表)
    def _normalize_cat_list(raw: list) -> list[dict]:
        result = []
        for item in raw:
            if isinstance(item, str):
                result.append({"name": item, "reason": ""})
            elif isinstance(item, dict) and "name" in item:
                result.append({"name": item["name"], "reason": item.get("reason", "")})
        return result

    assigned = _normalize_cat_list(raw_categories if isinstance(raw_categories, list) else [])
    maybe = _normalize_cat_list(raw_maybe if isinstance(raw_maybe, list) else [])

    # 验证类别名称是否存在
    category_set = set(category_names)
    valid_assigned = [c for c in assigned if c["name"] in category_set]
    valid_maybe = [c for c in maybe if c["name"] in category_set]

    invalid = [c["name"] for c in assigned + maybe if c["name"] not in category_set]
    if invalid:
        print(f"  ⚠ LLM 返回了无效类别: {invalid}")

    return valid_assigned, valid_maybe, document_summary


def _parse_user_nums(text: str) -> list[str]:
    """将用户输入按逗号和空格拆分为 token 列表。"""
    import re
    return [t for t in re.split(r"[,\s]+", text.strip()) if t]


def _move_to_waste(file_path: Path) -> bool:
    """将文献移入其所在文件夹下的 Waste 子文件夹。返回 True 表示成功。"""
    waste_dir = file_path.parent / "Waste"
    waste_dir.mkdir(parents=True, exist_ok=True)
    target = waste_dir / file_path.name
    if target.exists():
        print(f"  ⚠ Waste 中已有同名文件: {file_path.name}，跳过。")
        return True
    try:
        if file_path.is_file():
            shutil.move(str(file_path), str(target))
        else:
            shutil.move(str(file_path), str(target))
        print(f"  🗑️  已移入 Waste: {target}")
        return True
    except Exception as e:
        print(f"  ❌ 移入 Waste 失败: {e}")
        return False


def _display_selection_state(
    all_numbered: list[dict],
    selected_names: set[str],
    original_assigned_names: set[str],
) -> None:
    """
    显示当前选中状态。

    - 建议分类中被选中的显示 ✅，被移除的显示 ❌
    - 或许相关中被选中的显示 ✅，未选中的显示 ❓
    """
    # 分成两组显示
    assigned_items = [c for c in all_numbered if c["section"] == "assigned"]
    maybe_items = [c for c in all_numbered if c["section"] == "maybe"]

    # 计算所有条目名称的最大显示宽度，用于理由列对齐
    all_items = assigned_items + maybe_items
    max_name_width = max((_display_width(c["name"]) for c in all_items), default=0)

    def _fmt_line(num: int, icon: str, name: str, reason: str) -> str:
        padded = _pad_to_width(name, max_name_width)
        reason_str = f"  （{reason}）" if reason else ""
        return f"    [{num:>2}] {icon} {padded}{reason_str}"

    print(f"\n  📋 LLM 建议的分类:")
    for cat in assigned_items:
        num = cat["_num"]
        in_selected = cat["name"] in selected_names
        icon = "✅" if in_selected else "❌"
        print(_fmt_line(num, icon, cat["name"], cat.get("reason") or ""))

    if maybe_items:
        print(f"\n  🤔 或许相关的分类:")
        for cat in maybe_items:
            num = cat["_num"]
            in_selected = cat["name"] in selected_names
            icon = "✅" if in_selected else "❓"
            print(_fmt_line(num, icon, cat["name"], cat.get("reason") or ""))


def mode_categorize_files() -> None:
    """模式 2: 自动分类文献 — 读取文件内容并自动分类到文件夹。"""
    from Python_Lib.My_Lib_Stock import get_input_with_while_cycle
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"\n{'═' * 62}")
    print(f"  📚 模式 2: 自动分类文献")
    print(f"{'═' * 62}")

    result = _ask_categories_source()
    if result is None:
        return
    all_categories, all_category_paths = result

    # 检查多少分类有描述
    with_desc = sum(1 for v in all_categories.values() if v.get("description"))
    print(f"\n  找到 {len(all_categories)} 个分类（{with_desc} 个有描述）。")

    if with_desc < len(all_categories):
        print(f"  ⚠ {len(all_categories) - with_desc} 个分类尚无描述。")
        print(f"     建议先用 summarize 模式生成描述以提高分类准确度。")
        print(f"     （即使没有描述，也会根据文件夹名称进行分类。）")

    # 构建分类列表文本（只构建一次，所有文件共享）
    categories_text = _build_category_list_text(all_categories)
    category_names = list(all_categories.keys())

    # 收集要分类的文件
    print(f"\n  请输入要分类的文件路径（PDF/EPUB/图片文件夹），")
    print(f"  每行一个，输入空行开始处理。")
    print()

    raw_inputs = get_input_with_while_cycle(
        input_prompt="  📄 路径 > ",
        strip_quote=True,
    )

    if not raw_inputs:
        print("  未输入任何文件，退出。")
        return

    # 验证路径
    valid_files: list[Path] = []
    for raw in raw_inputs:
        p = Path(raw.strip())
        if not p.exists():
            print(f"  ❌ 路径不存在: {p}")
            continue
        if p.is_file():
            if p.suffix.lower() in (".pdf", ".epub", ".djvu"):
                valid_files.append(p)
            else:
                print(f"  ⚠ 不支持的文件格式: {p.suffix}，跳过 {p.name}")
        elif p.is_dir():
            is_img, img_files, _ = _check_image_folder(p)
            if is_img and img_files:
                valid_files.append(p)
            else:
                print(f"  ⚠ 文件夹不是纯图片文件夹: {p}")
        else:
            print(f"  ❌ 无法识别: {p}")

    if not valid_files:
        print("  没有有效文件，退出。")
        return

    # ══════════════════════════════════════════════════════════════
    #  阶段 1: 并行 LLM 调用（10 线程），收集所有结果
    # ══════════════════════════════════════════════════════════════
    print(f"\n  共 {len(valid_files)} 个有效文件，开始并行调用 LLM（10 线程）...\n")

    # results[i] = (assigned_cats, maybe_cats, document_summary) 或 None（失败）
    results: list[tuple[list[dict], list[dict], str] | None] = [None] * len(valid_files)
    _completed_count = 0
    _count_lock = threading.Lock()

    def _worker(idx: int, fp: Path) -> None:
        nonlocal _completed_count
        try:
            result = _classify_single_file(fp, categories_text, category_names)
            results[idx] = result
        except Exception as e:
            print(f"  ❌ [{fp.name}] LLM 调用失败: {e}")
            results[idx] = None
        with _count_lock:
            _completed_count += 1
            print(f"  ✓ 已完成 {_completed_count}/{len(valid_files)}: {fp.name}")

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = []
        for idx, fp in enumerate(valid_files):
            futures.append(pool.submit(_worker, idx, fp))
        # 等待所有任务完成
        for fut in futures:
            fut.result()

    # ══════════════════════════════════════════════════════════════
    #  阶段 2: 逐个让用户确认/修改分类
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 62}")
    print(f"  📋 LLM 分类完成，开始用户确认...")
    print(f"{'═' * 62}")

    for i, file_path in enumerate(valid_files):
        result = results[i]

        print(f"\n{'═' * 62}")
        print(f"  [{i + 1}/{len(valid_files)}] {file_path.name}")
        print(f"  路径: {file_path}")
        print(f"{'═' * 62}")

        if result is None:
            assigned_cats, maybe_cats, document_summary = [], [], ""
        else:
            assigned_cats, maybe_cats, document_summary = result

        if not assigned_cats and not maybe_cats:
            if document_summary:
                print(f"\n  📖 文献摘要:")
                _print_summary_in_terminal(document_summary)
            print(f"\n  ❌ 未能识别任何匹配的分类。")
            print(f"  输入类别编号（逗号/空格分隔）手动分类，或按 Enter 跳过。")
            print(f"  输入 [a] 显示所有分类，输入 [d] 删除文献。")
            _file_deleted = False
            while True:
                user_input = input("  > ").strip()
                if user_input.lower() == "d":
                    if _move_to_waste(file_path):
                        _file_deleted = True
                        break
                    continue  # 移动失败，重新询问
                break
            if _file_deleted:
                continue
            if user_input.lower() == "a":
                _print_multi_column(category_names)
                user_input = input("  请输入编号（逗号/空格分隔）: ").strip()
            if user_input:
                tokens = _parse_user_nums(user_input)
                manual = []
                for t in tokens:
                    if t.isdigit():
                        idx = int(t) - 1
                        if 0 <= idx < len(category_names):
                            manual.append(category_names[idx])
                    elif t in category_names:
                        manual.append(t)
                if manual:
                    assigned_cats = [{"name": c, "reason": "手动指定"} for c in manual]
                else:
                    print(f"  输入的类别均无效，跳过。")
                    continue
            else:
                continue

        # ── 写入临时 Markdown 文件 ──
        md_path = _write_temp_markdown(file_path, document_summary, assigned_cats, maybe_cats)

        # ── 显示文献摘要 ──
        if document_summary:
            print(f"\n  📖 文献摘要:")
            _print_summary_in_terminal(document_summary)

        # ── 编号映射: 建议分类 + 或许相关分类 统一编号 ──
        all_numbered: list[dict] = []

        for cat in assigned_cats:
            num = len(all_numbered) + 1
            all_numbered.append({**cat, "section": "assigned", "_num": num})

        for cat in maybe_cats:
            num = len(all_numbered) + 1
            all_numbered.append({**cat, "section": "maybe", "_num": num})

        # 初始选中集合 = 建议分类
        original_assigned_names = {c["name"] for c in assigned_cats}
        selected_names: set[str] = set(original_assigned_names)

        # 显示初始状态
        _display_selection_state(all_numbered, selected_names, original_assigned_names)

        # ── 检查文件当前所在分类 ──
        current_cat_name = next(
            (n for n, p in all_category_paths.items() if p.resolve() == file_path.parent.resolve()),
            None,
        )
        if current_cat_name and current_cat_name in selected_names:
            print(f"\n  ℹ️  文件当前已在分类 [{current_cat_name}] 中。")

        # ── 用户交互 ──
        print(f"\n  📝 详细结果已写入: {md_path}")
        print(f"\n  操作选项:")
        print(f"    [Enter]    确认当前选中分类")
        print(f"    [编号]     切换选中/取消（已选→取消，未选→选中）")
        print(f"    [a]        显示所有其他分类供选择")
        print(f"    [n]        创建新分类（输入绝对文件夹路径）")
        print(f"    [d]        删除文献（移入 Waste 文件夹）")
        print(f"    [s]        跳过此文件")

        action = ""  # 最终操作: "" = 确认, "s" = 跳过, "d" = 删除

        while True:
            try:
                choice = input("  请选择 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  [已跳过]")
                action = "s"
                break

            if choice.lower() == "s":
                action = "s"
                break

            if choice.lower() == "d":
                if _move_to_waste(file_path):
                    action = "d"
                    break
                continue  # 移动失败，重新询问

            if choice == "":
                # 确认当前选中
                action = ""
                break

            if choice.lower() == "n":
                # 创建新分类
                new_path_input = input("  请输入新分类的绝对文件夹路径: ").strip().strip("\"'")
                if not new_path_input:
                    print("    ⚠ 路径为空，已取消。")
                    continue
                new_cat_path = Path(new_path_input)
                new_cat_name = new_cat_path.name
                if not new_cat_name:
                    print("    ⚠ 无法解析文件夹名称，已取消。")
                    continue
                if new_cat_name in all_categories:
                    print(f"    ⚠ 分类 [{new_cat_name}] 已存在，将直接加入选中。")
                else:
                    try:
                        new_cat_path.mkdir(parents=True, exist_ok=True)
                        all_categories[new_cat_name] = {"description": ""}
                        all_category_paths[new_cat_name] = new_cat_path
                        category_names.append(new_cat_name)
                        print(f"    ✅ 新分类已创建: {new_cat_path}")
                    except Exception as e:
                        print(f"    ❌ 创建文件夹失败: {e}")
                        continue
                if new_cat_name not in selected_names:
                    selected_names.add(new_cat_name)
                    num = len(all_numbered) + 1
                    all_numbered.append({"name": new_cat_name, "reason": "新建分类", "section": "maybe", "_num": num})
                    print(f"    ✅ 已加入选中: {new_cat_name}")
                else:
                    print(f"    ℹ️  [{new_cat_name}] 已在选中列表中。")
                _display_selection_state(all_numbered, selected_names, original_assigned_names)
                print(f"\n  按 Enter 确认，或继续调整:")
                continue

            if choice.lower() == "a":
                # 显示所有未涉及的分类
                covered_names = {c["name"] for c in all_numbered}
                other_names = [n for n in category_names if n not in covered_names]
                if not other_names:
                    print("  所有分类已在上方列出。")
                    continue
                print(f"\n  📂 其他所有分类:")
                _print_multi_column(other_names)
                print()
                print(f"  输入编号（逗号/空格分隔）可加入选中（编号对应上方列表），直接按 Enter 返回。")
                extra_input = input("  > ").strip()
                if extra_input:
                    tokens = _parse_user_nums(extra_input)
                    for t in tokens:
                        if t.isdigit():
                            idx = int(t) - 1
                            if 0 <= idx < len(other_names):
                                name = other_names[idx]
                                selected_names.add(name)
                                # 也加入 all_numbered 以便后续显示
                                num = len(all_numbered) + 1
                                all_numbered.append({"name": name, "reason": "手动加入", "section": "maybe", "_num": num})
                                print(f"    ✅ 已加入: {name}")
                _display_selection_state(all_numbered, selected_names, original_assigned_names)
                print(f"\n  按 Enter 确认，或继续调整:")
                continue

            # 解析编号（逗号/空格分隔，无需 +/- 前缀，toggle 模式）
            tokens = _parse_user_nums(choice)
            changed = False
            for t in tokens:
                # 去除可能的 +/- 前缀（兼容旧用法）
                force_add = t.startswith("+")
                force_remove = t.startswith("-")
                num_str = t.lstrip("+-")

                if num_str.isdigit():
                    num = int(num_str)
                    if 1 <= num <= len(all_numbered):
                        name = all_numbered[num - 1]["name"]
                        if force_add:
                            selected_names.add(name)
                            changed = True
                        elif force_remove:
                            selected_names.discard(name)
                            # 如果从建议分类中移除，确保它在 maybe 区域可见
                            if name in original_assigned_names:
                                entry = all_numbered[num - 1]
                                if entry["section"] == "assigned":
                                    entry["section"] = "maybe"
                            changed = True
                        else:
                            # toggle
                            if name in selected_names:
                                selected_names.discard(name)
                                # 从建议分类中移除 → 放入或许相关
                                if name in original_assigned_names:
                                    entry = all_numbered[num - 1]
                                    if entry["section"] == "assigned":
                                        entry["section"] = "maybe"
                            else:
                                selected_names.add(name)
                            changed = True
                    else:
                        print(f"    ⚠ 编号 {num} 超出范围 (1-{len(all_numbered)})")
                else:
                    # 尝试作为类别名
                    if num_str in category_names:
                        if num_str in selected_names:
                            selected_names.discard(num_str)
                        else:
                            selected_names.add(num_str)
                        changed = True
                    else:
                        print(f"    ⚠ 无法识别: {t}")

            if changed:
                _display_selection_state(all_numbered, selected_names, original_assigned_names)
                print(f"\n  按 Enter 确认，或继续调整:")

        if action == "s":
            continue

        if action == "d":
            continue

        # ── 保存文献摘要到输入文件所在目录的 "LLM Summary" 子文件夹 ──
        _save_summary_markdown(file_path, document_summary)

        # ── 询问文章说明（可选，插入到文件名第一个」后面） ──
        try:
            note_input = input("\n  为文件添加说明（直接回车跳过）> ").strip()
        except (EOFError, KeyboardInterrupt):
            note_input = ""
        if note_input:
            # 去除不能出现在文件名中的字符（Windows 非法字符：\ / : * ? " < > |）
            note_input = re.sub(r'[\\/:*?"<>|]', "", note_input).strip()
        if note_input:
            old_name = file_path.name
            bracket_pos = old_name.find("」")
            if bracket_pos != -1:
                # 在第一个「」后插入：note，{原后缀部分}
                new_name = (
                    old_name[: bracket_pos + 1]
                    + note_input
                    + "，"
                    + old_name[bracket_pos + 1 :]
                )
            else:
                # 找不到」，插在扩展名前
                stem = file_path.stem if file_path.is_file() else file_path.name
                suffix = file_path.suffix if file_path.is_file() else ""
                new_name = stem + note_input + "，" + suffix
            new_file_path = file_path.parent / new_name
            try:
                file_path.rename(new_file_path)
                print(f"  ✏️  已重命名为：{new_name}")
                file_path = new_file_path
            except Exception as e:
                print(f"  ⚠ 重命名失败: {e}")

        # 验证最终分类
        final_categories = [c for c in selected_names if c in category_names]
        if not final_categories:
            print(f"  没有有效分类，跳过。")
            continue

        # 执行复制
        copied_to: list[str] = []
        for cat in final_categories:
            target_dir = all_category_paths[cat]
            target_path = target_dir / file_path.name
            if target_path.exists():
                # 检查是否就是同一个文件（文件已在此分类中）
                if file_path.resolve() == target_path.resolve():
                    print(f"  ℹ️  [{cat}] 文件已在此分类中，无需复制。")
                    copied_to.append(cat)
                    continue
                print(f"  ⚠ [{cat}] 同名文件已存在，跳过复制。")
                continue
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                if file_path.is_file():
                    shutil.copy2(file_path, target_path)
                else:
                    shutil.copytree(file_path, target_path)
                print(f"  ✅ 已复制到 [{cat}]")
                copied_to.append(cat)
            except Exception as e:
                print(f"  ❌ 复制到 [{cat}] 失败: {e}")

        # 删除原文件（仅当原文件不在任何一个目标分类中时）
        if copied_to:
            is_in_target = any(
                file_path.resolve() == (all_category_paths[cat] / file_path.name).resolve()
                for cat in final_categories
            )
            if not is_in_target:
                try:
                    if file_path.is_file():
                        file_path.unlink()
                    else:
                        shutil.rmtree(file_path)
                    print(f"  🗑️  已删除原文件。")
                except Exception as e:
                    print(f"  ⚠ 删除原文件失败: {e}")
            else:
                print(f"  ℹ️  原文件已在目标分类中，保留不删。")



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  主入口                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "summarize":
            mode_summarize_category()
        elif mode == "categorize":
            mode_categorize_files()
        else:
            print(f"  未知模式: {mode}")
            print(f"  可选: summarize, categorize")
    else:
        print(f"\n{'═' * 62}")
        print(f"  📚 基于内容的文献分类工具")
        print(f"{'═' * 62}")
        print(f"\n  请选择工作模式:")
        print(f"    [1] 总结分类特征 (Summarize Category)")
        print(f"    [2] 自动分类文献 (Categorize Files)")
        choice = input("\n  请输入 1 或 2: ").strip()
        if choice == "1":
            mode_summarize_category()
        elif choice == "2":
            mode_categorize_files()
        else:
            print("  无效选择。")


if __name__ == "__main__":
    main()
