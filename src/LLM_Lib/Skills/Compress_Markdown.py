"""
compress_md.py — 基于 LLM 的 Markdown 文档压缩工具

══════════════════════════════════════════════════════════════════════════════
  利用 Gemini Pro 模型将 Markdown 文档压缩至指定字数或比例。

  特性：
    • 保留 Markdown 所有标题结构（# ## ### 等），仅压缩标题下的内容
    • 逐段压缩，不合并或拆分段落，不破坏章节结构
    • 以 JSON dict 形式将段落 ID 与内容发送给模型
    • 长文档自动分块处理，每块输出不超过 5000 字，
      输入截止点为整自然段，优先在章节分节处切断
    • 支持 API 直接调用或网页手动粘贴两种模式
    • 同时向模型提供压缩比例和目标字数

  使用方式：
    python -m LLM_Lib.Skills.compress_md

    交互式输入：
      1. Markdown 文件路径
      2. 目标字数（>1）或压缩比例（<1，如 0.5 表示保留 50%；也支持 50%）
      3. 选择 API 调用或网页粘贴模式
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from LLM_Lib.LLM import call_gemini, extract_json, GEMINI_PRO


# ── 常量 ──────────────────────────────────────────────────────────────────────

# 每块 LLM 输出的字数上限
MAX_OUTPUT_CHARS_PER_CHUNK = 5000

# 原始字数不超过此值的段落直接原样输出，不发送给 LLM
SHORT_PARA_THRESHOLD = 40

SYSTEM_PROMPT = (
    "你是一位专业的学术文本编辑，专注于在保留核心含义和学术规范的前提下"
    "压缩中文学术文本。你的输出必须是严格的 JSON 格式。"
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  预处理：删去链接                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _remove_links(text: str) -> str:
    """
    删去 Markdown 文本中的所有链接，保留链接文字。

    处理以下链接形式：
      - [text](url)         → text
      - [text](url "title") → text
      - [text][ref]         → text
      - ![alt](url)         → （整体删去，图片无文字价值）
      - <http://...>        → （整体删去）
      - Reference 定义行    → （整体删去）[ref]: url
    """
    # 图片链接 ![alt](url) → 完全删去
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    # 图片引用 ![alt][ref] → 完全删去
    text = re.sub(r'!\[[^\]]*\]\[[^\]]*\]', '', text)
    # 行内链接 [text](url) 或 [text](url "title") → text
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # 引用链接 [text][ref] 或 [text][] → text
    text = re.sub(r'\[([^\]]*)\]\[[^\]]*\]', r'\1', text)
    # 自动链接 <http://...> 或 <https://...> → 完全删去
    text = re.sub(r'<https?://[^>]+>', '', text)
    # Reference 定义行 [ref]: url ... → 整行删去
    text = re.sub(r'^\s*\[[^\]]+\]:\s+\S.*$', '', text, flags=re.MULTILINE)
    return text


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  解析 Markdown 文档                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _parse_markdown(text: str) -> list[dict]:
    """
    将 Markdown 文本解析为有序元素列表。

    元素类型：
      - heading:   {"type": "heading",   "level": int, "text": str}
      - paragraph: {"type": "paragraph", "id": str,    "text": str, "char_count": int}

    段落以空行分隔。以 ``#`` 开头的行识别为标题。
    围栏代码块（```）内的空行不会拆分段落。
    """
    elements: list[dict] = []
    para_counter = 0
    current_lines: list[str] = []
    in_code_block = False

    def flush() -> None:
        nonlocal para_counter
        body = "\n".join(current_lines).strip()
        if body:
            para_counter += 1
            elements.append({
                "type": "paragraph",
                "id": f"P{para_counter:03d}",
                "text": body,
                "char_count": len(body),
            })
        current_lines.clear()

    for line in text.splitlines():
        # 围栏代码块检测
        if re.match(r"^`{3,}", line.strip()):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        if in_code_block:
            current_lines.append(line)
            continue

        # 标题行
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            elements.append({
                "type": "heading",
                "level": len(m.group(1)),
                "text": m.group(2).strip(),
            })
            continue

        # 空行 → 段落分隔
        if line.strip() == "":
            flush()
            continue

        current_lines.append(line)

    flush()
    return elements


def _assign_section_paths(elements: list[dict]) -> None:
    """
    为每个段落计算其所属的章节路径。

    根据标题层级构建路径，例如：
      ## A → ### B → 段落  →  section_path = "A > B"
    """
    heading_stack: dict[int, str] = {}

    for elem in elements:
        if elem["type"] == "heading":
            level = elem["level"]
            heading_stack[level] = elem["text"]
            # 清除更深层级的标题
            for lv in list(heading_stack):
                if lv > level:
                    del heading_stack[lv]
        elif elem["type"] == "paragraph":
            parts = [heading_stack[lv] for lv in sorted(heading_stack)]
            elem["section_path"] = " > ".join(parts)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  分块                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _build_sections(elements: list[dict]) -> list[dict]:
    """
    将元素按标题分组为「章节」。

    每个章节包含：
      - headings:    此节起始处的标题元素列表（连续标题合为一组）
      - paragraphs:  此节的段落元素列表
      - total_chars: 段落总字数
    """
    sections: list[dict] = []
    cur_headings: list[dict] = []
    cur_paras: list[dict] = []

    for elem in elements:
        if elem["type"] == "heading":
            if cur_paras:
                sections.append({
                    "headings": cur_headings,
                    "paragraphs": cur_paras,
                    "total_chars": sum(p["char_count"] for p in cur_paras),
                })
                cur_headings = []
                cur_paras = []
            cur_headings.append(elem)
        elif elem["type"] == "paragraph":
            cur_paras.append(elem)

    # 末尾残余
    if cur_paras:
        sections.append({
            "headings": cur_headings,
            "paragraphs": cur_paras,
            "total_chars": sum(p["char_count"] for p in cur_paras),
        })

    return sections


def _create_chunks(elements: list[dict], ratio: float) -> list[list[dict]]:
    """
    将段落分块，使每块压缩后的输出不超过 MAX_OUTPUT_CHARS_PER_CHUNK 字。

    分块策略：
      1. 按章节为单位打包：优先将整个章节放入同一块
      2. 章节边界切分：当前块容纳不下下一个完整章节时，在章节边界切分
      3. 段落边界切分：当单个章节本身超过字数限制时，在段落边界处切分

    返回列表，每个元素是该块包含的段落元素列表。
    """
    input_limit = int(MAX_OUTPUT_CHARS_PER_CHUNK / ratio)
    sections = _build_sections(elements)

    chunks: list[list[dict]] = []
    cur_chunk: list[dict] = []
    cur_chars = 0

    for section in sections:
        sec_chars = section["total_chars"]
        if sec_chars == 0:
            continue

        if cur_chars + sec_chars <= input_limit:
            # 整个章节放入当前块
            cur_chunk.extend(section["paragraphs"])
            cur_chars += sec_chars

        elif sec_chars > input_limit:
            # 章节本身超过字数限制 → 在段落边界逐一拆分
            if cur_chunk:
                chunks.append(cur_chunk)
                cur_chunk = []
                cur_chars = 0

            for para in section["paragraphs"]:
                pc = para["char_count"]
                if cur_chars + pc > input_limit and cur_chars > 0:
                    chunks.append(cur_chunk)
                    cur_chunk = []
                    cur_chars = 0
                cur_chunk.append(para)
                cur_chars += pc

        else:
            # 当前块容不下 → 在章节边界切分
            if cur_chunk:
                chunks.append(cur_chunk)
            cur_chunk = list(section["paragraphs"])
            cur_chars = sec_chars

    if cur_chunk:
        chunks.append(cur_chunk)

    return chunks


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Prompt 构建                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _build_compress_prompt(
    chunk: list[dict],
    ratio: float,
    chunk_original_chars: int,
    chunk_target_chars: int,
) -> str:
    """为一个分块构建 LLM 压缩 Prompt，包含详细规则和示范案例。"""

    # ── 构建段落 JSON ──
    para_data: dict = {}
    for para in chunk:
        target = max(10, int(para["char_count"] * ratio))
        para_data[para["id"]] = {
            "所属章节": para.get("section_path", ""),
            "原文": para["text"],
            "原始字数": para["char_count"],
            "目标字数": target,
        }

    json_str = json.dumps(para_data, ensure_ascii=False, indent=2)

    # ── 构建 Prompt ──
    # 注意：f-string 中 {{ / }} 产生字面量 { / }
    prompt = f"""\
# 任务：Markdown 文档段落级压缩

## 任务说明

请对以下各段落进行**独立压缩**。每个段落用唯一 ID 标识，需压缩至各自的目标字数。

## 全局压缩参数

| 参数 | 值 |
|------|------|
| 总压缩比例 | 保留原文的 **{ratio:.0%}**（即删减 {1 - ratio:.0%} 的内容） |
| 本批段落原始总字数 | {chunk_original_chars:,} 字 |
| 本批段落压缩目标总字数 | {chunk_target_chars:,} 字 |

## 压缩原则

### 必须遵守的规则

1. **逐段独立压缩**：每个段落 ID 对应一个压缩结果，**严禁合并**不同段落，**严禁拆分**段落，**严禁重排**段落顺序
2. **字数控制**：每段压缩后字数应接近其「目标字数」（允许 ±20% 偏差）
3. **结构保持**：不得改变段落的章节归属关系

### 保留优先级（从高到低）

1. 核心论点、关键结论、因果关系
2. 人名、书名、年代、专有名词、关键数据
3. 引用标注（如 `[[6]](#footnote-6)`、脚注编号等）
4. 逻辑连接词和论证链条
5. 背景说明和上下文铺垫

### 可删减内容

- 同义重复和冗余修饰
- 过度展开的背景解释
- 非关键性的举例和补充说明
- 过渡性套话和语气填充词

## 压缩示例

### 示例输入

```json
{{
  "DEMO": {{
    "所属章节": "研究对象及范围 > 明中后期的分期界定",
    "原文": "第二种观点则将明代中期的起点置于正德年间，这也是社会史与绘画史研究中较为流行的一种看法。陈宝良在《明代社会转型与文化变迁》一书中指出，正德时期（1506—1520）是明代社会由传统农业社会向早期商业社会转型的分水岭。[[7]](#footnote-7)正德朝以后，随着商品经济的蓬勃发展，封建礼教束缚逐渐松动，尚奢、从众的世俗风气日渐兴盛，社会结构发生了深刻的质变。与之观点相呼应，单国强在《明代绘画史》中亦将明代中期的起始时间大致定为正德元年（1506）。[[8]](#footnote-8)这一分期的合理性在于，正德朝不仅是明代政治生态由治转乱的关键转折点，也是明代绘画风格由院体、浙派向吴门画派转移的重要阶段。",
    "原始字数": 254,
    "目标字数": 127
  }}
}}
```

### 示例输出

```json
{{
  "DEMO": "第二种观点以正德年间为起点，在社会史与绘画史研究中较为流行。陈宝良指出正德时期（1506—1520）是向早期商业社会转型的分水岭，商品经济促使礼教松动与世俗风气兴盛。单国强《明代绘画史》亦定于正德元年（1506），[[8]](#footnote-8)因其既是政治转折点，也是绘画风格由院体、浙派转向吴门画派的重要阶段。"
}}
```

**效果分析**：原文 254 字 → 压缩后 119 字（约 47%）。保留了两位学者的核心观点、关键年代（1506—1520）和分期理由，删减了冗余修饰（"蓬勃发展""深刻的质变"等）和重复性表述。引用标注 `[[8]](#footnote-8)` 得到保留。

## 待压缩段落

```json
{json_str}
```

## 输出要求

请以 JSON 格式返回压缩结果。key 为段落 ID，value 为压缩后的纯文本字符串：

```json
{{
  "P001": "压缩后的文本...",
  "P002": "压缩后的文本..."
}}
```

**重要**：
- 必须包含所有输入段落 ID，不得遗漏任何一个
- value 为纯文本字符串，不要嵌套 JSON 对象
- 仅返回上述 JSON 对象即可，无需额外说明文字"""

    return prompt


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  预生成所有 Prompt 并保存到 Markdown 文件                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _save_chunk_prompts(
    chunks: list[list[dict]],
    ratio: float,
    base_path: Path,
) -> list[Path]:
    """
    为所有分块预先构建 Prompt 并保存为 Markdown 文件。

    文件保存在 ``{base_path.stem}_Prompts/`` 子目录下：
      - 需要 LLM 的块：``Chunk_{i+1:02d}_of_{total:02d}.md``
      - 全为短段落无需 LLM 的块：``Chunk_{i+1:02d}_of_{total:02d}_skipped.md``

    返回所有文件路径列表（与 chunks 等长）。
    """
    total = len(chunks)
    prompt_dir = base_path.parent / f"{base_path.stem}_Prompts"
    prompt_dir.mkdir(exist_ok=True)

    prompt_paths: list[Path] = []

    for i, chunk in enumerate(chunks):
        to_compress = [p for p in chunk if p["char_count"] > SHORT_PARA_THRESHOLD]

        if not to_compress:
            prompt_path = prompt_dir / f"Chunk_{i + 1:02d}_of_{total:02d}_skipped.md"
            prompt_path.write_text(
                f"# 第 {i + 1}/{total} 块\n\n全部为短段落（≤{SHORT_PARA_THRESHOLD} 字），无需 LLM 压缩。\n",
                encoding="utf-8",
            )
        else:
            chunk_original = sum(p["char_count"] for p in to_compress)
            chunk_target = int(chunk_original * ratio)
            prompt = _build_compress_prompt(to_compress, ratio, chunk_original, chunk_target)
            prompt_path = prompt_dir / f"Chunk_{i + 1:02d}_of_{total:02d}.md"
            prompt_path.write_text(prompt, encoding="utf-8")

        prompt_paths.append(prompt_path)

    return prompt_paths


def _print_prompt_file_list(prompt_paths: list[Path], current_index: int) -> None:
    """打印所有 Prompt 文件路径，并在当前处理块旁标注箭头。"""
    print()
    for i, p in enumerate(prompt_paths):
        marker = "  ◀ 当前" if i == current_index else ""
        prefix = "  →" if i == current_index else "   "
        print(f"{prefix} [{i + 1}/{len(prompt_paths)}] {p}{marker}")
    print()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LLM 调用与响应解析                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _try_parse_json_loose(text: str) -> dict | None:
    """宽松模式尝试从文本中提取 JSON 对象（找最外层 { ... }）。"""
    brace_start = text.find("{")
    if brace_start == -1:
        return None
    brace_end = text.rfind("}")
    if brace_end == -1 or brace_end <= brace_start:
        return None
    try:
        return json.loads(text[brace_start : brace_end + 1])
    except json.JSONDecodeError:
        return None


def _compress_chunk(
    chunk: list[dict],
    ratio: float,
    *,
    manual_paste: bool = False,
    confirm: bool = False,
    chunk_index: int = 0,
    total_chunks: int = 1,
    prebuilt_prompt: str | None = None,
) -> dict[str, str]:
    """
    调用 LLM 压缩一个分块的段落。

    返回 {paragraph_id: compressed_text}。
    JSON 解析失败时返回空字典（对应段落将保留原文）。

    Args:
        confirm: 为 True 时每块发送前显示完整交互确认界面（可切换模型、切换手动粘贴等）。
    """
    # ── 跳过短段落（原样保留，不发送给 LLM）──
    short_paras = {p["id"]: p["text"] for p in chunk if p["char_count"] <= SHORT_PARA_THRESHOLD}
    to_compress  = [p for p in chunk if p["char_count"] > SHORT_PARA_THRESHOLD]

    if short_paras:
        print(
            f"  ℹ  {len(short_paras)} 个短段落（≤{SHORT_PARA_THRESHOLD} 字）将原样保留: "
            f"{', '.join(sorted(short_paras))}"
        )

    # 若全部段落都是短段落，跳过 LLM 调用
    if not to_compress:
        print(f"\n  ▸ 第 {chunk_index + 1}/{total_chunks} 块全部为短段落，跳过 LLM 调用")
        return short_paras

    chunk_original = sum(p["char_count"] for p in to_compress)
    chunk_target = int(chunk_original * ratio)

    if prebuilt_prompt is not None:
        prompt = prebuilt_prompt
    else:
        prompt = _build_compress_prompt(to_compress, ratio, chunk_original, chunk_target)

    print(
        f"\n  ▸ 发送第 {chunk_index + 1}/{total_chunks} 块 "
        f"({len(to_compress)} 段需压缩, {chunk_original:,} 字 → 目标 {chunk_target:,} 字)"
    )

    response = call_gemini(
        prompt,
        model=GEMINI_PRO,
        confirm=confirm,
        manual_paste=manual_paste,
        system_prompt=SYSTEM_PROMPT,
    )

    # ── 解析 JSON 响应 ──
    result = extract_json(response)
    if not isinstance(result, dict):
        print("  ⚠ extract_json 未成功，尝试宽松解析...")
        result = _try_parse_json_loose(response)

    if not isinstance(result, dict):
        print("  ✗ 无法解析 LLM 响应为 JSON，此块将保留原文")
        return short_paras

    # ── 校验结果完整性 ──
    expected_ids = {p["id"] for p in to_compress}
    returned_ids = set(result.keys())
    missing = expected_ids - returned_ids
    extra = returned_ids - expected_ids

    if missing:
        print(
            f"  ⚠ 缺少 {len(missing)} 个段落 ID: "
            f"{', '.join(sorted(missing))}（将保留原文）"
        )
    if extra:
        print(f"  ⚠ 多余 {len(extra)} 个段落 ID: {', '.join(sorted(extra))}")
        for eid in extra:
            del result[eid]

    # ── 确保 value 为字符串 ──
    cleaned: dict[str, str] = {}
    for pid, val in result.items():
        if isinstance(val, str):
            cleaned[pid] = val
        elif isinstance(val, dict):
            # 模型可能返回了嵌套对象，尝试提取 "原文" 或直接 json.dumps
            cleaned[pid] = val.get("原文", json.dumps(val, ensure_ascii=False))
            print(f"  ⚠ {pid} 的值为嵌套对象，已尝试提取文本")
        else:
            cleaned[pid] = str(val)

    # ── 压缩统计 ──
    compressed_chars = sum(len(v) for v in cleaned.values())
    if chunk_original > 0:
        actual_ratio = compressed_chars / chunk_original
        print(
            f"  ✓ 压缩完成: {chunk_original:,} → {compressed_chars:,} 字 "
            f"(实际 {actual_ratio:.0%})"
        )

    # ── 合并短段落（原样）到返回结果 ──
    cleaned.update(short_paras)
    return cleaned


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  重建 Markdown                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _reconstruct_markdown(
    elements: list[dict],
    compressed: dict[str, str],
) -> str:
    """
    用原始标题结构和压缩后的段落文本重建 Markdown 文档。

    未包含在 compressed 中的段落 ID 将保留原文。
    """
    lines: list[str] = []

    for elem in elements:
        if elem["type"] == "heading":
            if lines:
                lines.append("")
            lines.append("#" * elem["level"] + " " + elem["text"])
        elif elem["type"] == "paragraph":
            if lines:
                lines.append("")
            pid = elem["id"]
            text = compressed[pid] if pid in compressed else str(elem["text"])
            lines.append(text)

    return "\n".join(lines) + "\n"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  断点恢复                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _collect_resume_data() -> dict[str, str]:
    """
    交互式收集断点恢复数据：让用户粘贴之前的 LLM 响应，提取已完成的段落压缩结果。

    用法：
      - 每条响应粘贴完后，在新行输入 END 并回车确认；
      - 所有响应粘贴完毕后，输入 DONE 并回车结束。

    返回 {paragraph_id: compressed_text}。
    """
    print()
    print("  ℹ  请逐条粘贴之前的 LLM 响应 JSON。")
    print("     每条粘贴完后，在新行输入 END 并回车；所有粘贴完毕后输入 DONE 结束。")
    print()

    recovered: dict[str, str] = {}
    response_count = 0

    while True:
        print(f"  [响应 {response_count + 1}] 粘贴 LLM 响应（输入 DONE 结束）:")
        lines: list[str] = []
        finished = False
        while True:
            try:
                line = input()
            except EOFError:
                finished = True
                break
            stripped = line.strip()
            if stripped == "DONE":
                finished = True
                break
            if stripped == "END":
                break
            lines.append(line)

        if lines:
            text = "\n".join(lines)
            parsed = _try_parse_json_loose(text)
            if isinstance(parsed, dict):
                new_ids: list[str] = []
                for pid, val in parsed.items():
                    if isinstance(val, str):
                        recovered[pid] = val
                    elif isinstance(val, dict):
                        recovered[pid] = val.get("原文", json.dumps(val, ensure_ascii=False))
                    else:
                        recovered[pid] = str(val)
                    new_ids.append(pid)
                response_count += 1
                print(
                    f"  ✓ 解析成功，获取 {len(new_ids)} 个段落: "
                    f"{', '.join(sorted(new_ids))}"
                )
            else:
                print("  ⚠ 无法从该响应中解析 JSON，已跳过。")

        if finished:
            break

    return recovered


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  主流程                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compress_markdown(
    file_path: str | Path,
    target: float,
    *,
    manual_paste: bool = False,
    confirm: bool = False,
) -> Path:
    """
    压缩 Markdown 文档。

    Args:
        file_path:    输入 Markdown 文件路径。
        target:       >1 为目标字数，<1 为压缩比例（如 0.5 = 保留 50%）。
        manual_paste: 是否使用网页手动粘贴模式（confirm=False 时有效）。
        confirm:      为 True 时每块发送前显示完整交互确认界面，可临时切换
                      模型或手动粘贴模式；为 False 时直接发送（或使用
                      manual_paste 标志）。

    Returns:
        输出文件路径。
    """
    path = Path(file_path)
    text = path.read_text(encoding="utf-8-sig")

    # ── 删去链接 ──
    text = _remove_links(text)

    # ── 解析 ──
    elements = _parse_markdown(text)
    _assign_section_paths(elements)

    # ── 统计 ──
    paragraphs = [e for e in elements if e["type"] == "paragraph"]
    headings = [e for e in elements if e["type"] == "heading"]
    total_chars = sum(p["char_count"] for p in paragraphs)

    if total_chars == 0:
        print("  文档无正文段落内容，无需压缩。")
        return path

    # ── 计算压缩参数 ──
    if target < 1:
        ratio = target
        target_chars = int(total_chars * ratio)
    else:
        target_chars = int(target)
        ratio = target_chars / total_chars
        if ratio >= 1:
            print(
                f"  目标字数 ({target_chars:,}) ≥ 原文正文字数 ({total_chars:,})，"
                f"无需压缩。"
            )
            return path

    print(f"\n{'═' * 62}")
    print(f"  Markdown 文档压缩")
    print(f"{'═' * 62}")
    print(f"  文件:       {path.name}")
    print(f"  标题数:     {len(headings)}")
    print(f"  段落数:     {len(paragraphs)}")
    print(f"  正文总字数: {total_chars:,}")
    print(f"  目标字数:   {target_chars:,}")
    print(f"  压缩比例:   {ratio:.1%}")
    print(f"{'─' * 62}")

    # ── 分块 ──
    chunks = _create_chunks(elements, ratio)
    print(f"  分块数:     {len(chunks)}")
    for i, chunk in enumerate(chunks):
        c_chars = sum(p["char_count"] for p in chunk)
        print(f"    第 {i + 1} 块: {len(chunk)} 段, {c_chars:,} 字")

    if manual_paste and len(chunks) > 1:
        print(
            f"\n  ℹ  手动粘贴模式下需逐块处理，共 {len(chunks)} 块，"
            f"每块需分别粘贴响应。"
        )

    # ── 预生成所有 Prompt 并保存到 Markdown 文件 ──
    prompt_paths = _save_chunk_prompts(chunks, ratio, path)
    print(f"\n  ✓ 已将所有 {len(chunks)} 块的 Prompt 保存至：")
    for p in prompt_paths:
        print(f"    {p}")

    # ── 读取预生成的 Prompt（跳过 skipped 块）──
    prebuilt_prompts: list[str | None] = []
    for p in prompt_paths:
        if "_skipped" in p.name:
            prebuilt_prompts.append(None)
        else:
            prebuilt_prompts.append(p.read_text(encoding="utf-8"))

    # ── 断点恢复 ──
    all_compressed: dict[str, str] = {}

    resume_answer = input("\n  是否从之前的任务断点恢复？[y/N] > ").strip().lower()
    if resume_answer in ("y", "yes", "是"):
        all_compressed = _collect_resume_data()
        if all_compressed:
            print(f"\n  ✓ 共恢复 {len(all_compressed)} 个段落的压缩结果")
            for i, chunk in enumerate(chunks):
                long_ids = {p["id"] for p in chunk if p["char_count"] > SHORT_PARA_THRESHOLD}
                recovered_in_chunk = long_ids & set(all_compressed)
                if long_ids and long_ids == recovered_in_chunk:
                    print(f"    第 {i + 1} 块: 已全部恢复 ({len(long_ids)} 段)，将跳过 LLM 调用")
                elif recovered_in_chunk:
                    print(
                        f"    第 {i + 1} 块: 已恢复 {len(recovered_in_chunk)}/{len(long_ids)} 段，"
                        f"剩余 {len(long_ids) - len(recovered_in_chunk)} 段待压缩"
                    )
        else:
            print("  ℹ  未解析到任何段落，将完整执行所有块。")

    # ── 逐块压缩 ──
    for i, chunk in enumerate(chunks):
        _print_prompt_file_list(prompt_paths, i)

        # 过滤掉已恢复的段落（保留短段落占位符，它们由 _compress_chunk 内部处理）
        filtered_chunk = [p for p in chunk if p["id"] not in all_compressed]

        if not filtered_chunk:
            print(f"\n  ▸ 第 {i + 1}/{len(chunks)} 块已全部恢复，跳过 LLM 调用")
            continue

        recovered_count = len(chunk) - len(filtered_chunk)
        if recovered_count > 0:
            print(
                f"\n  ℹ  第 {i + 1}/{len(chunks)} 块已恢复 {recovered_count} 段，"
                f"仅压缩剩余 {len(filtered_chunk)} 段"
            )
        # 若块被过滤则需重建 Prompt，否则沿用预生成的
        use_prebuilt = prebuilt_prompts[i] if recovered_count == 0 else None

        result = _compress_chunk(
            filtered_chunk,
            ratio,
            manual_paste=manual_paste,
            confirm=confirm,
            chunk_index=i,
            total_chunks=len(chunks),
            prebuilt_prompt=use_prebuilt,
        )
        all_compressed.update(result)

    # ── 重建文档 ──
    output_text = _reconstruct_markdown(elements, all_compressed)

    # ── 最终统计 ──
    final_chars = sum(
        len(all_compressed[p["id"]] if p["id"] in all_compressed else str(p["text"])) for p in paragraphs
    )
    actual_ratio = final_chars / total_chars if total_chars > 0 else 0

    # ── 写入输出文件 ──
    ratio_pct = round(actual_ratio * 100)
    output_path = (
        path.parent
        / f"{path.stem}_Compressed_CharCount_{final_chars}_Ratio_{ratio_pct}{path.suffix}"
    )
    output_path.write_text(output_text, encoding="utf-8")

    print(f"\n{'═' * 62}")
    print(f"  压缩完成")
    print(f"  原始正文字数: {total_chars:,}")
    print(f"  压缩后字数:   {final_chars:,}")
    print(f"  目标压缩比:   {ratio:.1%}")
    print(f"  实际压缩比:   {actual_ratio:.1%}")
    print(f"  输出文件:     {output_path}")
    print(f"{'═' * 62}")

    return output_path


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  交互式入口                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    """交互式入口：依次输入文件路径、压缩目标、调用方式。"""

    print(f"\n{'═' * 62}")
    print(f"  📄 Markdown 文档压缩工具")
    print(f"{'═' * 62}")

    # ── 1. 输入文件路径 ──
    file_input = input("\n  请输入 Markdown 文件路径: ").strip().strip('"').strip("'")
    path = Path(file_input)
    if not path.exists():
        print(f"  错误：文件不存在 — {path}")
        return
    if not path.is_file():
        print(f"  错误：路径不是文件 — {path}")
        return

    # ── 2. 输入压缩目标 ──
    target_input = input(
        "  请输入目标字数（>1）或压缩比例（<1，如 0.5 或 50%）: "
    ).strip()

    try:
        if target_input.endswith("%"):
            target_value = float(target_input.rstrip("%")) / 100
        else:
            target_value = float(target_input)
    except ValueError:
        print(f"  错误：无法解析输入 — {target_input!r}")
        return

    if target_value <= 0:
        print("  错误：目标值必须为正数")
        return

    # ── 3. 选择调用方式 ──
    print(f"\n  请选择 LLM 调用方式：")
    print(f"    [a / Enter]  直接调用 API（Gemini Pro，无需确认）")
    print(f"    [i]          交互模式（每块发送前确认，可切换模型 / 手动粘贴）")
    print(f"    [w]          网页手动粘贴模式（Notepad++ 打开 Prompt）")
    choice = input("  请选择 > ").strip().lower()

    confirm = choice == "i"
    manual_paste = choice == "w"

    # ── 4. 执行压缩 ──
    compress_markdown(path, target_value, manual_paste=manual_paste, confirm=confirm)


if __name__ == "__main__":
    main()
