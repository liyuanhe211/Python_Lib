"""
Lit_Categorization.py — 基于内容的文献分类工具

══════════════════════════════════════════════════════════════════════════════
  提供三个工作模式：

  1. summarize — 总结分类特征
     扫描根目录下所有子文件夹，读取文件夹名称和其中已有文献的文件名，
     利用 LLM 为每个分类生成描述，保存到 `0 Category Note.json`。
     修改已有 description 前会征得用户同意。

  2. categorize — 自动分类文献
     读取 PDF / EPUB / 图片文件夹的内容，将内容片段 + 文件名 + 所有分类描述
     发送给 LLM（Flash 模型），判断文献应归属的类别（支持多分类）。
     用户确认或修改后，将文件复制到所有目标分类文件夹中，然后删除原文件
     （除非原文件已在某个正确分类中）。

  3. summary (lit-summary) — 总结文献内容
     读取文献全文（去除参考文献段落），生成详细的中文总结。
     总结长度动态计算：每 10 个英文字符对应 1 个汉字，全篇不少于 1000 字，
     取较长的限制。若 PDF 超过 20 页会询问用户确认（可能是书籍）。
     总结结果保存到 LLM Summary 文件夹（Markdown + JSON 缓存）。

  分类体系说明：
     文件夹名称使用「维度 - 子类」的分层结构，例如：
       对象 - Reptiles - Lizards
       影响 - Vitamin D
       光源 - LED
     LLM 会理解这种层级关系，自动选择最具体的匹配子类。

  附属文件契约（与 Lit_Retrieval 流水线 / RAG 流水线互认）：
     一个文献文件（PDF / EPUB / DjVu / 图片文件夹）可能带有下列附属产物，
     位置全部相对于文献所在目录。本工具在复制 / 移动 / 删除 / 重命名文献时
     会同步处理这些附属文件，保持其相对位置与文件名主干一致：

       <dir>/<主干>.ris                     同目录同名 RIS（Lit_Retrieval_2 的
                                            占位 RIS；Rename_Ref 自动发现用）
       <dir>/PDF texts/<主干[:80]>/         文本化单元文件夹，内含：
           <主干[:80]>.ris                    归档的单篇 RIS
           <主干[:80]>.md                     Marker / Docling 文本化产物
           _page_*.png 等                     提取的图片
           .all_pages_processed               文本化完成标记
           _rag_index.parquet 等              RAG 索引产物
       <dir>/PDF texts/<主干>.md / .txt     旧版平铺产物（旧 Docling 平铺布局 /
                                            DeepSeek OCR 合并文本）
       <dir>/LLM Summary/<主干>_Summary.*   本工具自己的摘要产物（md + json）

     「PDF texts」目录名与主干截断规则（80 字符、去尾部空格点号）的唯一定义处
     是 LLM_Lib/RAG_Lib/Docling.py（经 Tools/Lit_Retrieval_Common 转发）。
     主干改变时，单元文件夹本身与其中以旧主干开头的文件都要跟着换名。

使用方式：
    python src/Tools/Lit_Categorization.py [mode]
    mode 可选: summarize, categorize, summary

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
    call_claude,
    extract_json,
    CLAUDE_OPUS,
    CLAUDE_SONNET,
    _AVG_CHARS_PER_TOKEN,
    _MODEL_PRICE_PER_M_TOKENS,
)
from Tools.Lit_Retrieval_4_Rename_Ref import (
    _extract_page_texts,
    _extract_epub_texts,
    _extract_djvu_texts,
    _check_image_folder,
    _extract_text_from_image,
    _is_blank,
    _ocr_single_page,
)
from Tools.Lit_Retrieval_Common import OUTPUT_PARENT_NAME, truncated_folder_name

# ── Markdown 终端渲染 ──────────────────────────────────────────────────────
_INDENT = "     "


def _print_summary_in_terminal(text: str, indent: str = _INDENT) -> None:
    """
    将 LLM 返回的 Markdown 摘要简易渲染后打印到终端：
      - ## / ### 标题 → 加粗标识行
      - **text** → 大写或直接去掉星号
      - 正文段落按终端宽度重新折行（正确计算 CJK 双宽字符）
    """
    term_width = shutil.get_terminal_size(fallback=(120, 40)).columns
    wrap_width = max(40, term_width - len(indent))

    def _strip_inline(s: str) -> str:
        # **bold** / *italic* → 去除标记
        s = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', s)
        return s

    def _char_w(ch: str) -> int:
        return 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1

    def _display_wrap(s: str, width: int) -> list[str]:
        """按显示宽度折行（CJK 字符占 2 列），允许在任意字符处换行。"""
        lines: list[str] = []
        cur = ''
        cur_w = 0
        for ch in s:
            cw = _char_w(ch)
            if cur_w + cw > width and cur:
                lines.append(cur)
                cur = ch
                cur_w = cw
            else:
                cur += ch
                cur_w += cw
        if cur:
            lines.append(cur)
        return lines

    def _join_lines(raw_lines: list[str]) -> str:
        """拼接段落内各行，CJK 字符间不插入空格。"""
        result = ''
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if not result:
                result = stripped
            else:
                last, first = result[-1], stripped[0]
                if unicodedata.east_asian_width(last) in ('W', 'F') or \
                        unicodedata.east_asian_width(first) in ('W', 'F'):
                    result += stripped
                else:
                    result += ' ' + stripped
        return result

    paragraphs = re.split(r'\n{2,}', text.strip())
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

        body = _join_lines([_strip_inline(l) for l in body_lines])
        if body:
            for wrapped in _display_wrap(body, wrap_width):
                print(f"{indent}{wrapped}")


# ── 全局 Token 统计 ─────────────────────────────────────────────────────────
_token_lock = threading.Lock()
_total_input_chars = 0
_total_output_chars = 0

# ── 常量 ────────────────────────────────────────────────────────────────────
CATEGORY_NOTE_FILENAME = "0 Category Note.json"
SUMMARY_CACHE_VERSION = 1

# ── 分类来源持久化状态文件 ───────────────────────────────────────────────────
_STATE_FILE = Path(__file__).resolve().parent.parent.parent.parent / "src_Saves" / "Last_Categories.json"

# ── LLM Summary 集中存储路径（使用 txt 文件作为分类来源时设置）─────────────────
_categories_txt_path: Path | None = None


def _load_last_source() -> dict:
    """读取上次分类来源设置，失败返回空 dict。"""
    try:
        if _STATE_FILE.is_file():
            with open(_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_last_source(choice: str, path: str) -> None:
    """保存本次分类来源设置。"""
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"choice": choice, "path": path}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


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
    例: root=…/文献_Priva2e/, folder=…/文献_Private/0 光照、体色/测量 - 显色
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
_TRUNCATION_MARKER = "[Content truncated here.]"


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


def _print_multi_column(items: list[str], indent: str = "    ", terminal_width: int = 0) -> None:
    """
    以多列格式打印带编号的列表项。
    序号先按列排列（先填满第一列再第二列），列之间对齐。
    terminal_width=0 时动态获取当前终端宽度。
    """
    if not items:
        return

    # 动态获取终端宽度
    if terminal_width <= 0:
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 120

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


def _get_summary_dir(file_path: Path) -> Path:
    """返回当前文献对应的 LLM Summary 文件夹。
    若使用 Categories txt 文件作为分类来源，则集中存储到 txt 同级的 LLM Summary 文件夹。
    """
    if _categories_txt_path is not None:
        return _categories_txt_path.parent / "LLM Summary"
    return file_path.parent / "LLM Summary"


def _get_summary_stem(file_path: Path) -> str:
    """返回文献摘要文件使用的 stem；文件用 stem，文件夹用 name。"""
    return file_path.stem if file_path.suffix else file_path.name


def _get_summary_markdown_path(file_path: Path) -> Path:
    """返回当前文献对应的 Markdown 摘要路径。"""
    return _get_summary_dir(file_path) / f"{_get_summary_stem(file_path)}_Summary.md"


def _get_summary_json_path(file_path: Path) -> Path:
    """返回当前文献对应的 JSON 摘要缓存路径。"""
    return _get_summary_dir(file_path) / f"{_get_summary_stem(file_path)}_Summary.json"


def _normalize_category_response(raw: list) -> list[dict]:
    """标准化 LLM 返回的分类列表。"""
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append({"name": item, "reason": ""})
        elif isinstance(item, dict) and "name" in item:
            result.append({"name": item["name"], "reason": item.get("reason", "")})
    return result


def _save_summary_cache(
    file_path: Path,
    document_summary: str,
    assigned_categories: list[dict],
    maybe_relevant: list[dict],
    raw_response: str = "",
) -> Path | None:
    """
    将 LLM 摘要与分类结果缓存到 JSON。

    额外响应字段统一放在文件末尾的 __llm_*__ 键下，便于后续快速识别和断点恢复。
    """
    if not document_summary and not assigned_categories and not maybe_relevant and not raw_response:
        return None

    summary_dir = _get_summary_dir(file_path)
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_path = _get_summary_json_path(file_path)

    payload = {
        "cache_version": SUMMARY_CACHE_VERSION,
        "source_file_name": file_path.name,
        "source_file_path": str(file_path),
        "document_summary": document_summary,
        "__llm_categories__": assigned_categories,
        "__llm_maybe_relevant__": maybe_relevant,
        "__llm_raw_response__": raw_response,
    }

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  💾 摘要缓存已保存到: {save_path}")
        return save_path
    except Exception as e:
        print(f"  ⚠ 摘要缓存保存失败: {e}")
        return None


def _find_summary_json(file_path: Path) -> Path | None:
    """
    从文献所在目录的 LLM Summary 子文件夹开始，逐级向上查找摘要 JSON 文件，
    直到文件系统根目录为止。

    这样可以处理不同模式下 Summary 存储位置不同的情况：
    - 模式 3 / 根目录分类：存放在文献旁边的 LLM Summary/
    - Categories txt 分类：存放在 txt 同级的 LLM Summary/（位于更上层目录）
    """
    stem = _get_summary_stem(file_path)
    filename = f"{stem}_Summary.json"
    search_dir = file_path.parent
    while True:
        candidate = search_dir / "LLM Summary" / filename
        if candidate.exists():
            return candidate
        parent = search_dir.parent
        if parent == search_dir:  # 已到文件系统根目录
            break
        search_dir = parent
    return None


def _load_summary_cache(
    file_path: Path,
    category_names: list[str],
) -> tuple[list[dict], list[dict], str] | None:
    """从 JSON 摘要缓存中恢复文献摘要与分类结果。
    向上逐级搜索 LLM Summary 目录，兼容不同模式下的存储位置。
    """
    json_path = _find_summary_json(file_path)
    if json_path is None:
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except Exception as e:
        print(f"  ⚠ 读取摘要缓存失败: {e}")
        return None

    if not isinstance(cached, dict):
        return None

    document_summary = cached.get("document_summary", "")
    assigned = _normalize_category_response(cached.get("__llm_categories__", []))
    maybe = _normalize_category_response(cached.get("__llm_maybe_relevant__", []))

    category_set = set(category_names)
    valid_assigned = [c for c in assigned if c["name"] in category_set]
    valid_maybe = [c for c in maybe if c["name"] in category_set]

    invalid = [c["name"] for c in assigned + maybe if c["name"] not in category_set]
    if invalid:
        print(f"  ⚠ 摘要缓存中存在无效类别: {invalid}")

    # 必须至少有分类结果才视为有效的分类缓存；
    # 仅有 document_summary（如模式 3 总结文献内容）不算分类缓存。
    if not valid_assigned and not valid_maybe:
        return None

    print(f"  ♻️  已复用摘要缓存: {json_path}")
    return valid_assigned, valid_maybe, document_summary


def _load_existing_summary(file_path: Path) -> str | None:
    """
    从 JSON 缓存中仅读取 document_summary 字段。
    向上逐级搜索 LLM Summary 目录，兼容不同模式下的存储位置。
    用于在分类时复用已有的文献总结，避免重新提取原文。
    返回总结文本，无缓存或无总结返回 None。
    """
    json_path = _find_summary_json(file_path)
    if json_path is None:
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except Exception:
        return None
    if not isinstance(cached, dict):
        return None
    summary = cached.get("document_summary", "")
    return summary if summary else None


def _save_summary_markdown(file_path: Path, document_summary: str) -> Path | None:
    """
    将文献摘要保存到 {file_path.parent}/LLM Summary/{file_path.stem}_Summary.md。
    若 document_summary 为空则跳过。
    返回保存路径，失败或跳过返回 None。
    """
    if not document_summary:
        return None
    summary_dir = _get_summary_dir(file_path)
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_path = _get_summary_markdown_path(file_path)
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(document_summary)
        print(f"  💾 摘要已保存到: {save_path}")
        return save_path
    except Exception as e:
        print(f"  ⚠ 摘要保存失败: {e}")
        return None


def _copy_summary_artifacts(source_file_path: Path, target_file_path: Path) -> None:
    """将源文献已有的摘要文件复制到目标文献对应位置。"""
    for source_path, target_path in (
        (_get_summary_markdown_path(source_file_path), _get_summary_markdown_path(target_file_path)),
        (_get_summary_json_path(source_file_path), _get_summary_json_path(target_file_path)),
    ):
        if not source_path.exists() or source_path == target_path:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()
        shutil.copy2(source_path, target_path)


def _delete_summary_artifacts(file_path: Path) -> None:
    """删除文献对应的摘要文件。"""
    for summary_path in (_get_summary_markdown_path(file_path), _get_summary_json_path(file_path)):
        try:
            if summary_path.exists():
                summary_path.unlink()
        except Exception as e:
            print(f"  ⚠ 删除摘要文件失败: {summary_path} ({e})")


def _rename_summary_artifacts(old_file_path: Path, new_file_path: Path) -> None:
    """在文献重命名或移动时，同步移动其摘要文件。"""
    for old_path, new_path in (
        (_get_summary_markdown_path(old_file_path), _get_summary_markdown_path(new_file_path)),
        (_get_summary_json_path(old_file_path), _get_summary_json_path(new_file_path)),
    ):
        if not old_path.exists() or old_path == new_path:
            continue
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if new_path.exists():
                new_path.unlink()
            shutil.move(str(old_path), str(new_path))
        except Exception as e:
            print(f"  ⚠ 移动摘要文件失败: {old_path} -> {new_path} ({e})")


# ── 附属文件处理（Lit_Retrieval 流水线目录契约，见模块 docstring）─────────────
#
# 覆盖：同目录同名 .ris、「PDF texts/<主干[:80]>/」文本化单元文件夹（内含单篇
# RIS、Markdown、提取图片、完成标记、RAG 索引产物）、旧版平铺产物
# （PDF texts/<主干>.md / .txt，含 DeepSeek OCR 合并文本）。
# LLM Summary 摘要产物的存储位置随分类来源变化，由上面的
# _copy/_delete/_rename_summary_artifacts 单独处理。

def _iter_companion_pairs(
    old_file_path: Path, new_file_path: Path,
) -> list[tuple[Path, Path]]:
    """列出 (现存附属文件, 目标位置) 对。

    目标位置 = 新文献所在目录下的相同相对位置、按新主干命名。
    old 与 new 传同一路径时（删除场景），目标位置与来源相同。
    """
    old_stem = _get_summary_stem(old_file_path)
    new_stem = _get_summary_stem(new_file_path)
    pairs: list[tuple[Path, Path]] = []

    # 1. 同目录同名 RIS
    ris = old_file_path.parent / f"{old_stem}.ris"
    if ris.is_file():
        pairs.append((ris, new_file_path.parent / f"{new_stem}.ris"))

    old_pt = old_file_path.parent / OUTPUT_PARENT_NAME
    new_pt = new_file_path.parent / OUTPUT_PARENT_NAME

    # 2. 文本化单元文件夹。截断规则认两种：当前契约 truncated_folder_name
    #    （去尾部空格和点号）与旧 DeepSeek OCR 流水线（只去尾部空格），
    #    取第一个命中的；目标名一律用当前契约。
    for base in dict.fromkeys((truncated_folder_name(old_stem), old_stem[:80].rstrip())):
        unit_dir = old_pt / base
        if unit_dir.is_dir():
            pairs.append((unit_dir, new_pt / truncated_folder_name(new_stem)))
            break

    # 3. 旧版平铺产物（旧 Docling 平铺布局 .md / DeepSeek OCR 合并文本 .txt）
    for ext in (".md", ".txt"):
        flat = old_pt / (old_stem + ext)
        if flat.is_file():
            pairs.append((flat, new_pt / (new_stem + ext)))

    return pairs


def _rename_unit_dir_contents(unit_dir: Path, old_base: str, new_base: str) -> None:
    """单元文件夹换主干后，把夹内以旧主干开头的文件改成新主干开头。

    夹内的 <主干[:80]>.ris / .md 与文件夹同主干，是 Docling / Lit_Retrieval
    各脚本互认产物的依据；提取图片（_page_*.png）与完成标记
    （.all_pages_processed）不含主干，保持原名。
    """
    if old_base == new_base or not unit_dir.is_dir():
        return
    for child in list(unit_dir.iterdir()):
        if not child.name.startswith(old_base):
            continue
        target = child.with_name(new_base + child.name[len(old_base):])
        if target.exists():
            continue
        try:
            child.rename(target)
        except OSError as e:
            print(f"  ⚠ 附属文件改名失败: {child} -> {target} ({e})")


def _copy_companion_files(source_file_path: Path, target_file_path: Path) -> None:
    """复制文献后，把其附属文件复制到目标目录下的相同相对位置。"""
    for src, dst in _iter_companion_pairs(source_file_path, target_file_path):
        if src.resolve() == dst.resolve():
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                _rename_unit_dir_contents(dst, src.name, dst.name)
            else:
                if dst.exists():
                    dst.unlink()
                shutil.copy2(src, dst)
        except Exception as e:
            print(f"  ⚠ 复制附属文件失败: {src} -> {dst} ({e})")


def _move_companion_files(old_file_path: Path, new_file_path: Path) -> None:
    """文献移动或重命名后，同步迁移其附属文件（保持相对位置与主干一致）。"""
    for src, dst in _iter_companion_pairs(old_file_path, new_file_path):
        if src.resolve() == dst.resolve():
            continue
        was_dir = src.is_dir()
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(src), str(dst))
            if was_dir:
                _rename_unit_dir_contents(dst, src.name, dst.name)
        except Exception as e:
            print(f"  ⚠ 移动附属文件失败: {src} -> {dst} ({e})")


def _delete_companion_files(file_path: Path) -> None:
    """删除文献对应的全部附属文件。"""
    for src, _ in _iter_companion_pairs(file_path, file_path):
        try:
            if src.is_dir():
                shutil.rmtree(src)
            else:
                src.unlink()
        except Exception as e:
            print(f"  ⚠ 删除附属文件失败: {src} ({e})")


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
    global _categories_txt_path
    last = _load_last_source()
    last_choice = last.get("choice", "")
    last_path = last.get("path", "")

    print(f"\n  请选择分类来源：")
    print(f"    [1] 根目录（扫描子文件夹作为分类）")
    print(f"    [2] Categories 文件（每行一个分类文件夹路径）")
    if last_choice and last_path:
        print(f"    [Enter] 上次选择：{last_path}")
        choice_prompt = "  请输入: "
    else:
        choice_prompt = "  请输入 1 或 2: "
    src_choice = input(choice_prompt).strip()
    if not src_choice and last_choice:
        src_choice = last_choice
        # 直接复用上次路径，无需再次询问
        if src_choice == "1":
            root = Path(last_path)
            if not root.is_dir():
                print("  ❌ 路径不存在或不是文件夹！")
                return None
            _categories_txt_path = None
            all_categories, all_category_paths = _collect_all_categories(root)
            if not all_categories:
                print("  ❌ 未找到任何分类子文件夹！")
                return None
            return all_categories, all_category_paths
        elif src_choice == "2":
            txt_path = Path(last_path)
            if not txt_path.is_file():
                print("  ❌ 文件不存在！")
                return None
            _categories_txt_path = txt_path
            all_categories, all_category_paths = _collect_categories_from_txt(txt_path)
            if not all_categories:
                print("  ❌ 未从 Categories 文件中读取到任何有效分类文件夹！")
                return None
            return all_categories, all_category_paths

    if src_choice == "1":
        if last_choice == "1" and last_path:
            path_prompt = f"\n  请输入包含各个分类文件夹的根目录路径 [上次: {last_path}，回车确认]: "
        else:
            path_prompt = "\n  请输入包含各个分类文件夹的根目录路径: "
        root_input = input(path_prompt).strip().strip("\"'")
        if not root_input:
            if last_choice == "1" and last_path:
                root_input = last_path
            else:
                return None
        root = Path(root_input)
        if not root.is_dir():
            print("  ❌ 路径不存在或不是文件夹！")
            return None
        _save_last_source("1", root_input)
        _categories_txt_path = None
        all_categories, all_category_paths = _collect_all_categories(root)
        if not all_categories:
            print("  ❌ 未找到任何分类子文件夹！")
            return None
        return all_categories, all_category_paths

    elif src_choice == "2":
        if last_choice == "2" and last_path:
            path_prompt = f"\n  请输入 Categories txt 文件路径 [上次: {last_path}，回车确认]: "
        else:
            path_prompt = "\n  请输入 Categories txt 文件路径: "
        txt_input = input(path_prompt).strip().strip("\"'")
        if not txt_input:
            if last_choice == "2" and last_path:
                txt_input = last_path
            else:
                return None
        txt_path = Path(txt_input)
        if not txt_path.is_file():
            print("  ❌ 文件不存在！")
            return None
        _save_last_source("2", txt_input)
        _categories_txt_path = txt_path
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


def _truncate_text_with_marker(text: str, max_chars: int) -> str:
    """将文本截断到 max_chars，并在截断位置插入英文标记。"""
    if len(text) <= max_chars:
        return text

    marker = f"\n\n{_TRUNCATION_MARKER}"
    if max_chars <= len(marker):
        return _TRUNCATION_MARKER[:max_chars]
    return text[: max_chars - len(marker)] + marker


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
            truncated = False
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
                    if remaining > len(_TRUNCATION_MARKER) + 20:
                        collected.append(_truncate_text_with_marker(p, remaining))
                    elif collected:
                        collected.append(_TRUNCATION_MARKER)
                    else:
                        collected.append(_truncate_text_with_marker(p, max_chars))
                    truncated = True
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
            truncated = False
            for img in img_files[:5]:  # 最多 OCR 前5张
                ocr_text = _extract_text_from_image(img)
                if ocr_text.strip():
                    ocr_parts.append(ocr_text.strip())
                    total_len += len(ocr_text)
                    if total_len >= max_chars:
                        truncated = True
                        break
            if len(img_files) > 5:
                truncated = True
            if truncated and ocr_parts:
                ocr_parts.append(_TRUNCATION_MARKER)
            text = "\n\n".join(ocr_parts)
        else:
            print(f"  ⚠ 文件夹不是纯图片文件夹。")
            return ""

    # 截断
    text = _truncate_text_with_marker(text, max_chars)

    return text


def _get_pdf_page_count(file_path: Path) -> int | None:
    """返回 PDF 的页数，失败返回 None。"""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(file_path))
        return len(reader.pages)
    except Exception:
        pass
    try:
        import fitz
        doc = fitz.open(str(file_path))
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return None


# ── 参考文献段落检测 ──────────────────────────────────────────────────────────
_REFERENCE_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"references|bibliography|works?\s+cited|literature\s+cited"
    r"|参考文献|引用文献|文献目录"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_references_section(text: str) -> str:
    """
    去除文本末尾的 References / Bibliography 段落。
    从最后一次出现的参考文献标题行开始截断。
    """
    matches = list(_REFERENCE_HEADER_RE.finditer(text))
    if not matches:
        return text
    # 取最后一个匹配（通常 References 在末尾）
    last_match = matches[-1]
    # 只在文本后半部分才认为是正式的参考文献段
    if last_match.start() < len(text) * 0.5:
        return text
    return text[: last_match.start()].rstrip()


def _extract_full_text_no_refs(file_path: Path) -> str:
    """
    提取文献的全部文本内容（不截断），并去除参考文献段落。
    仅支持 PDF / EPUB / DjVu / 图片文件夹。
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
            collected = []
            for page_idx, p in enumerate(pages):
                p = p.strip()
                if _is_blank(p):
                    ocr_text, did_ocr = _ocr_single_page(file_path, page_idx, p)
                    if did_ocr:
                        p = ocr_text.strip()
                if p:
                    collected.append(p)
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
            for img in img_files:
                ocr_text = _extract_text_from_image(img)
                if ocr_text.strip():
                    ocr_parts.append(ocr_text.strip())
            text = "\n\n".join(ocr_parts)
        else:
            print(f"  ⚠ 文件夹不是纯图片文件夹。")
            return ""

    # 去除参考文献段落
    text = _strip_references_section(text)
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

    response = call_claude(
        prompt,
        model=CLAUDE_OPUS,
        system_prompt=_SUMMARIZE_SYSTEM_PROMPT,
        reasoning=True,       # Claude 由模型自身决定
        confirm=False,
        stream=True,          # Claude 后端忽略（不支持流式）
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
            if not _overwrite_all[0]:
                confirm = input("  是否覆盖现有 description? [y/a(全部)/N]: ").strip().lower()
                if confirm == "a":
                    _overwrite_all[0] = True
                elif confirm not in ("y", "yes"):
                    new_data["description"] = existing["description"]
                    print("  ℹ️  保留原有 description。")

        merged = {**existing, **new_data}
        _save_category_note(folder, merged)

    _overwrite_all: list[bool] = [False]

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
6. If the excerpt contains the marker "[Content truncated here.]", it means \
    part of the document or some pages were omitted. Treat it strictly as an \
    omission marker and do not invent details for the omitted content.

Output ONLY a JSON object with these keys:
- "document_summary": 用中文撰写的论文内容摘要，必须使用结构良好的 Markdown 格式。\
  格式要求如下：\
  (1) 首段：用1-2句话总述（包含作者、年份、期刊），点明研究核心，对**关键术语**加粗。\
  (2) 正文：按内容拆分为若干 ### 编号小节（如：### 1. 研究背景 / ### 2. 实验方法 / \
  ### 3. 主要结果 / ### 4. 结论与意义），每节内容充实，层次分明。\
  (3) 数据呈现：若文献包含对比数据，优先使用 Markdown 表格展示；\
  列表项用 "* **关键词**：说明" 格式，数值带单位。\
    (4) 数学/物理量：行内公式用 $...$ （如 $D_3$、$\\mu W/cm^2$），独立公式用 $$...$$。\
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
    cached = _load_summary_cache(file_path, category_names)
    if cached is not None:
        return cached

    # ── 尝试复用已有的文献总结（模式 3 生成的），避免重新提取原文 ──
    existing_summary = _load_existing_summary(file_path)
    if existing_summary:
        print(f"  ♻️  复用已有文献总结进行分类（{len(existing_summary):,} 字符）")
        prompt = (
            f"## Document Information\n\n"
            f"**Filename**: {file_path.name}\n"
            f"**File location**: {file_path.parent}\n\n"
            f"## Existing Document Summary (produced by a prior LLM analysis)\n\n"
            f"{existing_summary}\n\n"
            f"## Available Categories\n"
            f"{categories_text}\n"
        )
    else:
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
            f"## Document Content Excerpt (~{_MAX_TEXT_CHARS} characters max; "
            f'"{_TRUNCATION_MARKER}" marks omitted content/pages)\n\n'
            f"{text}\n\n"
            f"## Available Categories\n"
            f"{categories_text}\n"
        )

    print(f"  ⏳ 正在调用 LLM 进行分类...")
    response = call_claude(
        prompt,
        model=CLAUDE_OPUS,
        system_prompt=_CATEGORIZE_SYSTEM_PROMPT,
        reasoning=False,       # Claude 由模型自身决定
        confirm=False,
        stream=False,          # Claude 后端忽略
        allow_cancel=False,    # Claude 后端忽略
    )
    _track_tokens(prompt + _CATEGORIZE_SYSTEM_PROMPT, response)

    parsed = extract_json(response)
    if not isinstance(parsed, dict):
        print(f"  ⚠ LLM 返回格式不正确。")
        return [], [], ""

    raw_categories = parsed.get("categories", [])
    raw_maybe = parsed.get("maybe_relevant", [])
    # 若已有高质量总结（模式 3），优先保留；否则使用 LLM 分类时生成的摘要
    document_summary = existing_summary or parsed.get("document_summary", "")

    assigned = _normalize_category_response(raw_categories if isinstance(raw_categories, list) else [])
    maybe = _normalize_category_response(raw_maybe if isinstance(raw_maybe, list) else [])

    # 验证类别名称是否存在
    category_set = set(category_names)
    valid_assigned = [c for c in assigned if c["name"] in category_set]
    valid_maybe = [c for c in maybe if c["name"] in category_set]

    invalid = [c["name"] for c in assigned + maybe if c["name"] not in category_set]
    if invalid:
        print(f"  ⚠ LLM 返回了无效类别: {invalid}")

    _save_summary_cache(file_path, document_summary, valid_assigned, valid_maybe, response)

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
        _rename_summary_artifacts(file_path, target)
        _move_companion_files(file_path, target)
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
        print(f"\n  路径: {file_path}")
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
                print(f"\n{'═' * 62}")
                print(f"  [{i + 1}/{len(valid_files)}] {file_path.name}")
                print(f"{'═' * 62}")
                print(f"\n  📝 详细结果已写入: {md_path}")
                print(f"\n  路径: {file_path}")
                print(f"\n  操作选项:")
                print(f"    [Enter]    确认当前选中分类")
                print(f"    [编号]     切换选中/取消（已选→取消，未选→选中）")
                print(f"    [a]        显示所有其他分类供选择")
                print(f"    [n]        创建新分类（输入绝对文件夹路径）")
                print(f"    [d]        删除文献（移入 Waste 文件夹）")
                print(f"    [s]        跳过此文件")
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
                _rename_summary_artifacts(file_path, new_file_path)
                _move_companion_files(file_path, new_file_path)
                print(f"  ✏️  已重命名为：{new_name}")
                file_path = new_file_path
            except Exception as e:
                print(f"  ⚠ 重命名失败: {e}")

        # ── 保存文献摘要到最终文件所在目录的 "LLM Summary" 子文件夹 ──
        _save_summary_markdown(file_path, document_summary)

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
                _copy_summary_artifacts(file_path, target_path)
                _copy_companion_files(file_path, target_path)
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
                    _delete_summary_artifacts(file_path)
                    _delete_companion_files(file_path)
                    print(f"  🗑️  已删除原文件。")
                except Exception as e:
                    print(f"  ⚠ 删除原文件失败: {e}")
            else:
                print(f"  ℹ️  原文件已在目标分类中，保留不删。")



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  模式 3: 总结文献内容                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_LITERATURE_SUMMARY_SYSTEM_PROMPT = """\
You are an expert academic literature analyst. Your task is to produce a \
comprehensive Chinese-language summary of a research document.

CRITICAL RULES:
1. The summary must cover ALL sections of the document provided to you — \
   introduction, methods, results, discussion, conclusions, supplementary \
   materials, etc. Do NOT skip any section. The reference list has already \
   been removed before sending to you, so summarize everything you receive.
2. Use well-structured Markdown format:
   - 首段：1-2句总述（包含作者、年份、期刊 — 如果可以从文本中推断），点明研究核心，\
     对**关键术语**加粗。
   - 正文：按内容拆分为若干 ### 编号小节（如：### 1. 研究背景 / ### 2. 实验方法 / \
     ### 3. 主要结果 / ### 4. 结论与意义），每节内容充实，层次分明。
   - 数据呈现：若文献包含对比数据，优先使用 Markdown 表格展示；\
     列表项用 "* **关键词**：说明" 格式，数值带单位。
   - 数学/物理量：行内公式用 $...$ （如 $D_3$、$\\mu W/cm^2$），独立公式用 $$...$$。
3. Be faithful to the original data and findings. Do NOT fabricate details.
4. The summary must be in Chinese (中文).
5. **Length requirement**: You MUST produce a summary of at least {min_chars} 个汉字. \
   This is a hard minimum — do not produce anything shorter. Aim for thorough \
   coverage; if the document is information-dense, produce a proportionally \
   longer summary. Be detailed, not vague.

Output ONLY the Markdown summary text, no JSON wrapping, no extra commentary."""


def _summarize_single_literature(file_path: Path) -> str | None:
    """
    对单个文献进行全文总结（去除参考文献后）。

    Returns:
        总结文本（Markdown），失败返回 None。
    """
    # ── 检查已有缓存 ──
    json_path = _get_summary_json_path(file_path)
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            existing_summary = cached.get("document_summary", "")
            if existing_summary:
                print(f"  ♻️  已有总结缓存: {json_path}")
                print(f"      （总结长度: {len(existing_summary)} 字符）")
                reuse = input("  是否复用已有总结？[Y/n]: ").strip().lower()
                if reuse not in ("n", "no"):
                    return existing_summary
                print("  将重新生成总结。")
        except Exception:
            pass

    # ── PDF 超过 20 页确认 ──
    if file_path.is_file() and file_path.suffix.lower() == ".pdf":
        page_count = _get_pdf_page_count(file_path)
        if page_count is not None:
            print(f"  📄 PDF 共 {page_count} 页")
            if page_count > 20:
                print(f"  ⚠ 此 PDF 超过 20 页，可能是书籍而非论文。")
                confirm = input("  是否继续总结？[y/N]: ").strip().lower()
                if confirm not in ("y", "yes"):
                    print("  已跳过。")
                    return None

    # ── 提取全文 ──
    print(f"  ⏳ 提取全文（去除参考文献）...")
    text = _extract_full_text_no_refs(file_path)
    if not text.strip():
        print(f"  ⚠ 未能提取到有效文本。")
        return None

    char_count = len(text)
    print(f"  📄 提取了 {char_count:,} 字符的文本（已去除参考文献）")

    # ── 计算最小字数要求 ──
    # 每 10 个英文字符对应 1 个汉字，全篇不少于 1000 字，取较长的限制
    dynamic_min = char_count // 30
    min_chars = max(1000, dynamic_min)
    print(f"  📏 总结最低字数要求: {min_chars} 字"
          f"（动态计算 {dynamic_min}，下限 1000，取较大值）")

    # ── 构建 Prompt ──
    system_prompt = _LITERATURE_SUMMARY_SYSTEM_PROMPT.replace("{min_chars}", str(min_chars))

    prompt = (
        f"## Document Information\n\n"
        f"**Filename**: {file_path.name}\n\n"
        f"## Full Document Content (references removed)\n\n"
        f"{text}\n"
    )

    print(f"  ⏳ 正在调用 LLM 生成总结...")
    response = call_claude(
        prompt,
        model=CLAUDE_OPUS,
        system_prompt=system_prompt,
        reasoning=True,             # Claude 由模型自身决定
        confirm=False,
        stream=True,                # Claude 后端忽略
        allow_cancel=True,          # Claude 后端忽略
        max_output_tokens=65536,    # Claude 后端忽略
    )
    _track_tokens(prompt + system_prompt, response)

    if not response.strip():
        print(f"  ⚠ LLM 未返回有效内容。")
        return None

    # ── 保存 ──
    _save_summary_cache(file_path, response, [], [], "")
    _save_summary_markdown(file_path, response)

    return response


def mode_summarize_literature() -> None:
    """模式 3: 总结文献内容 — 读取文献全文并生成详细中文总结。"""
    from Python_Lib.My_Lib_Stock import get_input_with_while_cycle

    print(f"\n{'═' * 62}")
    print(f"  📖 模式 3: 总结文献内容")
    print(f"{'═' * 62}")

    print(f"\n  请输入要总结的文件路径（PDF/EPUB/DjVu/图片文件夹），")
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

    print(f"\n  共 {len(valid_files)} 个有效文件，开始逐个总结...\n")

    for i, file_path in enumerate(valid_files):
        print(f"\n{'═' * 62}")
        print(f"  [{i + 1}/{len(valid_files)}] {file_path.name}")
        print(f"{'═' * 62}")

        summary = _summarize_single_literature(file_path)

        if summary:
            print(f"\n  📖 文献总结:")
            _print_summary_in_terminal(summary)
            print(f"\n  ✅ 总结已保存。")
        else:
            print(f"  ❌ 总结失败或已跳过。")

    # 费用统计
    if _total_input_chars > 0 or _total_output_chars > 0:
        print(f"\n{'═' * 62}")
        print(f"  📊 Token 用量统计")
        print(f"{'═' * 62}")
        est_input_tokens = _total_input_chars / _AVG_CHARS_PER_TOKEN
        est_output_tokens = _total_output_chars / _AVG_CHARS_PER_TOKEN
        # Claude 走订阅额度、不按 token 计费；只显示 token 用量，不显示美元预估
        price = _MODEL_PRICE_PER_M_TOKENS.get(CLAUDE_OPUS)
        if price is not None:
            input_price, output_price = price
            cost = (est_input_tokens * input_price + est_output_tokens * output_price) / 1_000_000
            print(f"  输入: ~{est_input_tokens:,.0f} tokens | 输出: ~{est_output_tokens:,.0f} tokens")
            print(f"  估算费用: ${cost:.4f}")
        else:
            # 价格表里没有 Claude 条目 → 静默跳过美元预估行
            print(f"  输入: ~{est_input_tokens:,.0f} tokens | 输出: ~{est_output_tokens:,.0f} tokens")
            print(f"  ℹ️  Claude 走订阅额度计费，此处不显示美元预估")


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
        elif mode in ("summary", "lit-summary"):
            mode_summarize_literature()
        else:
            print(f"  未知模式: {mode}")
            print(f"  可选: summarize, categorize, summary")
    else:
        print(f"\n{'═' * 62}")
        print(f"  📚 基于内容的文献分类工具")
        print(f"{'═' * 62}")
        print(f"\n  请选择工作模式:")
        print(f"    [1] 总结分类特征 (Summarize Category)")
        print(f"    [2] 自动分类文献 (Categorize Files)")
        print(f"    [3] 总结文献内容 (Summarize Literature)")
        choice = input("\n  请输入 1/2/3: ").strip()
        if choice == "1":
            mode_summarize_category()
        elif choice == "2":
            mode_categorize_files()
        elif choice == "3":
            mode_summarize_literature()
        else:
            print("  无效选择。")


if __name__ == "__main__":
    main()
