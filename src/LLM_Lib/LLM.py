"""
LLM.py — Gemini API 通用访问层

══════════════════════════════════════════════════════════════════════════════
  提供跨项目可复用的 Gemini API 访问功能：

  - API Key 加载（从 E:\\My_Program\\LLM_API_KEYS_PRIVATE.py 或环境变量）
  - 模型名称常量（GEMINI_PRO, GEMINI_FLASH 等）
  - 通用 call_gemini() 函数：
      • prompt / model / system_prompt / temperature / max_output_tokens
      • reasoning（默认 True）— 启用模型思考模式
      • confirm（默认 True）— 发送前终端交互确认 + 可切换模型
  - 基于内容哈希的 LLM 调用缓存（SQLite，存于 LLM.py 同目录 temp/llm_cache.db）
  - LLM 输出 JSON 提取
  - 模型显示名称映射
  - Flash 模型默认不开启思考模式（reasoning 参数默认 None = 自动判断）

使用方式:
    from LLM_Lib.LLM import call_gemini, GEMINI_PRO, GEMINI_FLASH

    response = call_gemini(
        "请分析……",
        model=GEMINI_PRO,
        reasoning=True,   # Flash 模型传 None 时自动禁用
        confirm=False,
    )
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import string
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 确保 stdout 实时写入（消除流式输出在终端"一下子出现"的问题）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(write_through=True)
    except Exception:
        pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  API Key 加载                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_KEYS_DIR = r"E:\My_Program"
if _KEYS_DIR not in sys.path:
    sys.path.insert(0, _KEYS_DIR)

try:
    from LLM_API_KEYS_PRIVATE import GEMINI_API_KEY  # type: ignore[import-untyped]
except ImportError:
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  模型常量                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

GEMINI_PRO:   str = "gemini-3.1-pro-preview"
GEMINI_FLASH: str = "gemini-3-flash-preview"
DEFAULT_MODEL: str = GEMINI_PRO

# 旧名称别名（兼容现有代码中 MODEL_PRO / MODEL_FLASH 的引用）
MODEL_PRO   = GEMINI_PRO
MODEL_FLASH = GEMINI_FLASH

_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-2.5-pro":         "Gemini 2.5 Pro",
    "gemini-2.5-flash":       "Gemini 2.5 Flash",
}

# 默认启用思考（reasoning）模式的模型白名单。
# 未列出的模型在 reasoning=None 时默认关闭 reasoning。
_MODELS_DEFAULT_REASONING: frozenset[str] = frozenset({
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
})


def model_display_name(model: str) -> str:
    """将模型 ID 转换为人类可读的显示名称。"""
    return _MODEL_DISPLAY_NAMES.get(model, model)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  费用估算                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 每 100 万 token 的价格（美元），(输入价格, 输出价格)
# 使用短提示（<=200k token）档位的付费层价格
_MODEL_PRICE_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    "gemini-3.1-pro-preview":                  (2.00,  12.00),
    "gemini-3.1-pro-preview-customtools":      (2.00,  12.00),
    "gemini-3-pro-preview":                    (2.00,  12.00),
    "gemini-3-flash-preview":                  (0.50,   3.00),
    "gemini-2.5-pro":                          (1.25,  10.00),
    "gemini-2.5-flash":                        (0.30,   2.50),
    "gemini-2.5-flash-lite":                   (0.10,   0.40),
    "gemini-2.5-flash-lite-preview-09-2025":   (0.10,   0.40),
    "gemini-2.0-flash":                        (0.10,   0.40),
    "gemini-2.0-flash-lite":                   (0.075,  0.30),
}

# 平均每个 token 对应的字符数（粗略估算）
# 英文约 4 chars/token，中文约 1.5 chars/token；混合文本取约 2.5
_AVG_CHARS_PER_TOKEN = 2.5

# 本次程序运行期间累计的 LLM 调用费用（美元）
_session_cost_usd: float = 0.0


def get_session_cost() -> float:
    """返回本次程序运行期间累计的 LLM API 调用费用（美元）。"""
    return _session_cost_usd


def reset_session_cost() -> None:
    """重置本次程序运行期间的累计费用计数器。"""
    global _session_cost_usd
    _session_cost_usd = 0.0


def estimate_cost_usd(model: str, input_text: str, output_text: str) -> float | None:
    """
    根据输入/输出文本的字符数粗略估算本次 API 调用费用（美元浮点数）。

    使用字符数 / _AVG_CHARS_PER_TOKEN 估算 token 数，
    再乘以对应模型的每百万 token 价格。

    不在价格表中的模型返回 None。
    """
    price = _MODEL_PRICE_PER_M_TOKENS.get(model)
    if price is None:
        # 尝试前缀匹配（应对 -001 等后缀变体）
        for key, val in _MODEL_PRICE_PER_M_TOKENS.items():
            if model.startswith(key) or key.startswith(model):
                price = val
                break
    if price is None:
        return None

    input_tokens  = len(input_text)  / _AVG_CHARS_PER_TOKEN
    output_tokens = len(output_text) / _AVG_CHARS_PER_TOKEN
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def estimate_cost(model: str, input_text: str, output_text: str) -> str | None:
    """
    根据输入/输出文本的字符数粗略估算本次 API 调用费用。

    返回格式化的费用字符串（如 "~$0.0032"），
    不在价格表中的模型返回 None。
    """
    cost = estimate_cost_usd(model, input_text, output_text)
    if cost is None:
        return None
    return f"~${cost:.4f}"


def _print_cost(model: str, input_text: str, output_text: str, show_cost: bool = True) -> None:
    """
    更新会话累计费用；若 show_cost=True，在终端打印本次及会话累计费用。
    无论 show_cost 如何，只要有价格数据就更新全局累计值。
    """
    global _session_cost_usd
    cost = estimate_cost_usd(model, input_text, output_text)
    if cost is None:
        return
    _session_cost_usd += cost
    if show_cost:
        in_tok  = int(len(input_text)  / _AVG_CHARS_PER_TOKEN)
        out_tok = int(len(output_text) / _AVG_CHARS_PER_TOKEN)
        print(
            f"  💰 本次费用：~${cost:.4f}  "
            f"（输入 ~{in_tok:,} tokens，输出 ~{out_tok:,} tokens）"
        )
        print(
            f"  💰 会话累计费用：~${_session_cost_usd:.4f}"
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  google-genai 延迟导入                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_genai: Any = None
_genai_types: Any = None


def _ensure_genai():
    """延迟导入 google-genai SDK，首次调用时加载。"""
    global _genai, _genai_types
    if _genai is None:
        try:
            from google import genai
            from google.genai import types as genai_types
            _genai = genai
            _genai_types = genai_types
        except ImportError:
            raise ImportError(
                "google-genai 未安装。请执行：uv add google-genai"
            )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Gemini 客户端（单例）                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_client: Any = None


def _get_client() -> Any:
    """获取或创建 Gemini API 客户端单例。"""
    global _client
    if _client is None:
        _ensure_genai()
        api_key = GEMINI_API_KEY
        if not api_key or api_key == "your-gemini-api-key-here":
            raise ValueError(
                "Gemini API Key 未配置。\n"
                "请在 E:\\My_Program\\LLM_API_KEYS_PRIVATE.py 中设置 GEMINI_API_KEY，\n"
                "或设置环境变量：$env:GEMINI_API_KEY = 'your-key-here'"
            )
        _client = _genai.Client(api_key=api_key)
    return _client


def get_genai_client() -> Any:
    """公开获取 Gemini Client 实例（供需要直接操作客户端的场景使用）。"""
    return _get_client()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SQLite 缓存系统                                                            ║
# ║  默认存储位置：LLM.py 同目录下的 temp/llm_cache.db                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_DEFAULT_CACHE_DB: Path = Path(__file__).parent / "temp" / "llm_cache.db"
_cache_db_path: Path = _DEFAULT_CACHE_DB


def set_cache_db(path: Path | str) -> None:
    """覆盖默认缓存数据库路径。必须在首次调用 call_gemini 前设置。"""
    global _cache_db_path
    _cache_db_path = Path(path)


def get_cache_db() -> Path:
    """返回当前缓存数据库路径。"""
    return _cache_db_path


def is_cache_enabled() -> bool:
    """缓存是否启用（环境变量 LLM_CACHE_DISABLED=1 可禁用）。"""
    return os.environ.get("LLM_CACHE_DISABLED", "").strip() not in ("1", "true", "yes")


def _get_db_connection():
    """获取 SQLite 连接，首次调用时自动建表并创建目录。"""
    import sqlite3

    _cache_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_cache_db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            hash          TEXT PRIMARY KEY,
            model         TEXT NOT NULL,
            system_prompt TEXT,
            temperature   REAL NOT NULL,
            reasoning     INTEGER NOT NULL,
            prompt        TEXT NOT NULL,
            response      TEXT NOT NULL,
            timestamp     TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _cache_key(
    model: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    reasoning: bool,
) -> str:
    """根据调用参数（含 reasoning）生成 SHA-256 哈希键。"""
    content = (
        f"{model}|{system_prompt or ''}|{temperature:.4f}"
        f"|{int(reasoning)}|{prompt}"
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_cached(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 1.0,
    reasoning: bool = False,
) -> str | None:
    """查找 SQLite 缓存。命中时返回响应文本，未命中返回 None。"""
    if not is_cache_enabled():
        return None
    import sqlite3

    try:
        conn = _get_db_connection()
        key = _cache_key(model, prompt, system_prompt, temperature, reasoning)
        row = conn.execute(
            "SELECT response, timestamp FROM llm_cache WHERE hash = ?", (key,)
        ).fetchone()
        conn.close()
        if row:
            preview = prompt[:50].replace("\n", " ")
            print(
                f"  💾 [缓存命中] 模型={model_display_name(model)} "
                f"时间={row[1]} prompt=「{preview}…」"
            )
            return row[0]
    except sqlite3.Error as e:
        print(f"  ⚠ 缓存读取失败：{e}")
    return None


def save_cache(
    model: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    reasoning: bool,
    response: str,
) -> None:
    """将 LLM 响应写入 SQLite 缓存（仅在请求成功完成后调用）。"""
    if not is_cache_enabled():
        return
    import sqlite3

    try:
        conn = _get_db_connection()
        key = _cache_key(model, prompt, system_prompt, temperature, reasoning)
        conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache
                (hash, model, system_prompt, temperature, reasoning, prompt, response, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                model,
                system_prompt or "",
                temperature,
                int(reasoning),
                prompt,
                response,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"  ⚠ 缓存写入失败：{e}")


def clear_cache() -> int:
    """清空所有缓存条目。返回删除的记录数。"""
    import sqlite3

    try:
        conn = _get_db_connection()
        count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        conn.execute("DELETE FROM llm_cache")
        conn.commit()
        conn.close()
        return count
    except sqlite3.Error:
        return 0


def cache_stats() -> dict[str, int | str]:
    """返回缓存统计信息（count, size_mb）。"""
    import sqlite3

    try:
        conn = _get_db_connection()
        count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        conn.close()
        size_mb = (
            f"{_cache_db_path.stat().st_size / 1024 / 1024:.2f}"
            if _cache_db_path.exists()
            else "0.00"
        )
        return {"count": count, "size_mb": size_mb}
    except sqlite3.Error:
        return {"count": 0, "size_mb": "0.00"}


# ── 向后兼容旧接口 ─────────────────────────────────────────────────────────────
def set_cache_dir(path: Path | str) -> None:
    """已废弃（旧 JSON 文件缓存接口），现改为设置 DB 文件的父目录。"""
    set_cache_db(Path(path) / "llm_cache.db")


def get_cache_dir() -> Path | None:
    """已废弃（旧接口），返回 DB 文件的父目录。"""
    return _cache_db_path.parent


# ── 向后兼容：为现有 prompt_*.md 文件补写 Response 节 ─────────────────────────

def _parse_prompt_md(filepath: Path) -> "dict | None":
    """
    解析 temp/ 下的 prompt_*.md 文件，提取 model、system_prompt、prompt。

    Returns:
        包含以下键的 dict，解析失败时返回 None：
          - model (str)
          - system_prompt (str | None)
          - prompt (str)
          - has_response (bool)  — 文件中是否已含 "## Response" 节
    """
    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception:
        return None

    has_response = "## Response" in text

    # 提取 model
    model_match = re.search(r"\*\*Model:\*\*\s*`([^`]+)`", text)
    model = model_match.group(1) if model_match else ""

    # 提取 system prompt（## System Prompt 节，若有）
    sys_match = re.search(
        r"## System Prompt\n\n(.*?)\n\n(?=##)", text, re.DOTALL
    )
    system_prompt: "str | None" = sys_match.group(1).strip() if sys_match else None

    # 提取 user prompt（## User Prompt 节直到 ---、## Response 或文件末尾）
    user_match = re.search(
        r"## User Prompt\n\n(.*?)(?=\n\n---|\n## Response|\Z)", text, re.DOTALL
    )
    prompt = user_match.group(1).strip() if user_match else None

    if not model or not prompt:
        return None

    return {
        "model": model,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "has_response": has_response,
    }


def backfill_md_responses_from_cache() -> int:
    """
    向后兼容工具函数：扫描 temp/ 下所有 prompt_*.md 文件，
    对尚未包含响应（缺少 "## Response" 节）的文件，尝试从 SQLite 缓存中
    查找对应的 LLM 响应并写入。

    匹配逻辑：
      - temperature 固定使用默认值 1.0（旧文件均使用默认值）
      - reasoning 会依次尝试 False 和 True（取最先命中的）

    Returns:
        成功补写响应的文件数量。
    """
    import sqlite3

    temp_dir = Path(__file__).parent / "temp"
    if not temp_dir.exists():
        print("  temp/ 目录不存在，跳过。")
        return 0

    md_files = sorted(temp_dir.glob("prompt_*.md"))
    if not md_files:
        print("  temp/ 中没有 prompt_*.md 文件。")
        return 0

    print(f"  扫描到 {len(md_files)} 个 prompt_*.md 文件……")

    try:
        conn = _get_db_connection()
    except Exception as e:
        print(f"  ⚠ 无法连接数据库：{e}")
        return 0

    updated = 0
    skipped_has_response = 0
    skipped_no_match = 0
    skipped_parse_error = 0

    for filepath in md_files:
        info = _parse_prompt_md(filepath)
        if info is None:
            print(f"  ⚠ 解析失败，跳过：{filepath.name}")
            skipped_parse_error += 1
            continue

        if info["has_response"]:
            skipped_has_response += 1
            continue

        # 尝试 reasoning=False 和 True，temperature 固定 1.0
        found_response: "str | None" = None
        for reasoning in (False, True):
            key = _cache_key(
                info["model"], info["prompt"],
                info["system_prompt"], 1.0, reasoning,
            )
            try:
                row = conn.execute(
                    "SELECT response FROM llm_cache WHERE hash = ?", (key,)
                ).fetchone()
            except sqlite3.Error:
                row = None
            if row:
                found_response = row[0]
                break

        if found_response is None:
            skipped_no_match += 1
            continue

        _append_response_to_md(filepath, found_response)
        updated += 1

    conn.close()

    print(f"  ✅ 补写完成：{updated} 个文件已补写响应")
    print(f"     已有响应（跳过）：{skipped_has_response}")
    print(f"     未找到匹配缓存（跳过）：{skipped_no_match}")
    if skipped_parse_error:
        print(f"     解析失败（跳过）：{skipped_parse_error}")
    return updated


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  用户确认交互                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Prompt 预览阈值：超过此字数时写入临时 .md 并由 VS Code 打开预览
_PROMPT_PREVIEW_THRESHOLD = 1000


def _open_prompt_preview_in_vscode(prompt: str, model: str) -> None:
    """
    将 Prompt 写入临时 .md 文件并在 VS Code 中打开。
    文件会在进程退出时自动清理（通过 atexit）。
    """
    import atexit

    # 写入临时文件（delete=False，以便 VS Code 能读取）
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        prefix="llm_prompt_preview_",
        delete=False,
        encoding="utf-8",
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmp.write(f"# LLM Prompt 预览\n\n")
    tmp.write(f"- **模型**: {model_display_name(model)}\n")
    tmp.write(f"- **时间**: {timestamp}\n")
    tmp.write(f"- **字符数**: {len(prompt):,}\n\n")
    tmp.write("---\n\n")
    tmp.write(prompt)
    tmp.flush()
    tmp.close()
    tmp_path = tmp.name

    # 进程退出时自动删除临时文件
    atexit.register(lambda p=tmp_path: Path(p).unlink(missing_ok=True))

    # 直接用 code 命令打开文件（用户可自行 Ctrl+Shift+V 预览 Markdown）
    try:
        subprocess.Popen(["code", tmp_path])
        print(f"  📄 Prompt 已在 VS Code 中打开：{tmp_path}")
        print(f"     提示：按 Ctrl+Shift+V 可预览 Markdown 格式")
    except FileNotFoundError:
        print(f"  ⚠ 无法启动 VS Code，Prompt 已保存至：{tmp_path}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  手动粘贴模式 — 请求 ID / Notepad++ / 终端收集                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _generate_request_id() -> str:
    """生成 6 位随机小写字母 ID，用于手动粘贴模式的请求追踪。"""
    return "".join(random.choices(string.ascii_lowercase, k=6))


def _write_prompt_for_manual_paste(
    prompt: str,
    request_id: str,
    model: str,
    system_prompt: str | None = None,
) -> Path:
    """
    将 Prompt 写入临时 .md 文件（供用户复制到网页版 LLM）。

    文件名包含请求 ID，Prompt 首尾标注 ID 以便模型回复时重复。
    返回文件路径。
    """
    import atexit

    tmp_dir = Path(tempfile.gettempdir())
    filename = f"llm_prompt_{request_id}.md"
    filepath = tmp_dir / filename

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# LLM Prompt [{request_id}]\n\n")
        f.write(f"- **请求 ID**: `{request_id}`\n")
        f.write(f"- **模型**: {model_display_name(model)}\n")
        f.write(f"- **时间**: {timestamp}\n\n")
        f.write("---\n\n")
        if system_prompt:
            f.write(f"## System Prompt\n\n{system_prompt}\n\n---\n\n")
        f.write(f"## User Prompt\n\n")
        f.write(prompt)
        f.write(f"\n\n---\n\n")
        f.write(
            f"**⚠ 请将上面的 User Prompt 部分粘贴到网页版模型中。"
            f"回复结束后，请在最后一行写上：ID={request_id}**\n"
        )

    atexit.register(lambda p=str(filepath): Path(p).unlink(missing_ok=True))
    return filepath


def _open_with_notepadpp(*file_paths: str | Path) -> None:
    """
    用 Notepad++ 打开一个或多个文件。

    按优先级尝试：notepad++ (PATH) → 默认安装路径 → 退化为 notepad。
    """
    path_args = [str(p) for p in file_paths]
    # 尝试 PATH 中的 notepad++
    try:
        subprocess.Popen(["notepad++"] + path_args)
        return
    except FileNotFoundError:
        pass
    # 尝试默认安装路径
    npp_exe = r"C:\Program Files\Notepad++\notepad++.exe"
    try:
        subprocess.Popen([npp_exe] + path_args)
        return
    except FileNotFoundError:
        pass
    # 退化为 notepad（每个文件单独打开）
    for p in path_args:
        try:
            subprocess.Popen(["notepad", p])
        except FileNotFoundError:
            print(f"  ⚠ 无法打开文件：{p}")


def _collect_manual_response(request_id: str) -> str:
    """
    在终端收集用户粘贴的 LLM 响应。

    用户逐行输入/粘贴，输入单行 `end` 表示结束。
    收集完成后校验响应末尾是否包含匹配的请求 ID。
    若用户拒绝不匹配的响应，会继续要求重新粘贴，直到用户接受。

    Returns:
        清理后的 LLM 响应文本。
    """
    while True:
        print(f"\n{'═' * 62}")
        print(f"  ▸ 请粘贴模型响应（请求 ID: {request_id}）")
        print(f"{'═' * 62}")
        print(f"  粘贴或输入模型的响应内容。")
        print(f"  输入完成后，在新的一行输入 end 并回车。")
        print(f"{'─' * 62}")

        lines: list[str] = []
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip().lower() == "end":
                break
            lines.append(line)

        response = "\n".join(lines).strip()

        # 校验 ID
        if request_id not in response:
            print(f"\n  ⚠ 响应中未找到请求 ID「{request_id}」")
            print(f"  请确认这是对请求 {request_id} 的响应。")
            try:
                choice = input("  继续接受此响应? [y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "n"
            if choice != "y":
                print(f"  已拒绝此响应。请重新粘贴正确的响应。")
                continue  # 循环回去重新收集响应
        else:
            print(f"  ✓ ID 验证通过: {request_id}")

        # 移除响应末尾的 ID 标记行
        cleaned_lines = response.splitlines()
        if cleaned_lines:
            last_line = cleaned_lines[-1].strip()
            if request_id in last_line and len(last_line) < 80:
                cleaned_lines.pop()
        response = "\n".join(cleaned_lines).strip()

        print(f"  ✅ 收到响应（{len(response):,} 字符）")
        return response


def _do_manual_paste_call(
    prompt: str,
    model: str,
    system_prompt: str | None = None,
    request_id: str = "",
) -> tuple[str, str]:
    """
    执行一次手动粘贴模式调用。

    1. 生成/使用 ID → 2. 在 Prompt 尾部追加 ID 指令 → 3. 写文件并用 Notepad++ 打开
    → 4. 终端收集响应 → 5. 校验 ID

    Returns:
        (response, request_id) 元组。
    """
    rid = request_id or _generate_request_id()
    # 在 Prompt 中追加 ID 指令
    id_instruction = (
        f"\n\n---\n**请求 ID: {rid}**\n"
        f"请在你的回复的最后一行重复此 ID，格式为: ID={rid}"
    )
    prompt_with_id = prompt + id_instruction

    filepath = _write_prompt_for_manual_paste(
        prompt_with_id, rid, model, system_prompt
    )
    _open_with_notepadpp(filepath)
    print(f"\n  📝 Prompt 已用 Notepad++ 打开：{filepath}")
    print(f"  请求 ID: {rid}")

    response = _collect_manual_response(rid)
    return response, rid


def _confirm_before_send(prompt: str, model: str) -> tuple[str, bool]:
    """
    在终端显示 Prompt 预览并等待用户确认。

    - Prompt ≤ 1000 字符：直接在终端全文展示
    - Prompt > 1000 字符：写入临时 .md 文件并用 VS Code 打开

    用户可选择：
      [Enter / y]  确认发送（API）
      [p]          改用 Pro 模型
      [f]          改用 Flash 模型
      [w]          使用网页手动粘贴模式
      [c]          取消

    Returns:
        (model, manual_paste) — 确认后的模型名称 + 是否使用手动粘贴。

    Raises:
        RuntimeError: 用户取消时。
    """
    separator = "─" * 62

    print(f"\n{'═' * 62}")
    print(f"  ▸ LLM 调用确认")
    print(f"{'═' * 62}")
    print(f"  模型:  {model_display_name(model)}")
    print(f"  Prompt 长度:  {len(prompt):,} 字符")
    print(f"{separator}")

    if len(prompt) <= _PROMPT_PREVIEW_THRESHOLD:
        # 短 Prompt：直接在终端全文展示
        print("  Prompt 全文：\n")
        for line in prompt.splitlines():
            print(f"    {line}")
    else:
        # 长 Prompt：写入临时 .md 并在 VS Code 打开预览
        _open_prompt_preview_in_vscode(prompt, model)

    print(f"\n{separator}")
    print("  快捷选项：")
    print(f"    [Enter / y]  确认发送（模型: {model_display_name(model)}）")
    print(f"    [p]          改用 Pro  模型 → {GEMINI_PRO}")
    print(f"    [f]          改用 Flash 模型 → {GEMINI_FLASH}")
    print(f"    [w]          使用网页手动粘贴模式（用 Notepad++ 打开 Prompt）")
    print(f"    [c]          取消此次调用")
    print(f"{separator}")

    while True:
        try:
            choice = input("  请选择 > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  [已取消]")
            raise RuntimeError("用户取消了 LLM 调用")

        if choice in ("", "y", "yes"):
            return model, False
        elif choice == "p":
            print(f"  ✓ 已切换为 Pro 模型: {GEMINI_PRO}")
            return GEMINI_PRO, False
        elif choice == "f":
            print(f"  ✓ 已切换为 Flash 模型: {GEMINI_FLASH}")
            return GEMINI_FLASH, False
        elif choice == "w":
            print(f"  ✓ 将使用网页手动粘贴模式")
            return model, True
        elif choice == "c":
            print("  [已取消]")
            raise RuntimeError("用户取消了 LLM 调用")
        else:
            print(
                "  无效选项，请输入 y/Enter（确认）、"
                "p（Pro）、f（Flash）、w（网页粘贴）、c（取消）"
            )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  按键取消 API 调用（msvcrt / termios）                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _key_pressed_c() -> bool:
    """
    非阻塞检测用户是否按下了 'c' 键（不需要 Enter）。

    Windows: 使用 msvcrt.kbhit() + msvcrt.getwch()。
    其他平台: 退化为始终返回 False（不支持无 Enter 按键检测）。
    """
    try:
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            return ch.lower() == "c"
    except ImportError:
        pass
    return False


def _api_call_with_cancel(
    client: Any,
    model: str,
    prompt: str,
    config: Any,
    stream: bool,
    confirm: bool,
    allow_cancel: bool = True,
    reasoning: bool = False,
    show_cost: bool = True,
) -> "str | None":
    """
    执行 API 调用，支持按 'c' 键（无需 Enter）随时取消。

    流式模式：在主线程直接迭代，每个 chunk 之间检查 'c' 键，确保输出实时可见。
    非流式模式：在子线程执行请求，主线程轮询 'c' 键。
    取消时返回 None，正常完成返回响应文本，其他异常向上抛出。
    """
    reasoning_label = "思考 ON" if reasoning else "思考 OFF"
    if stream:
        # ── 流式：主线程直接迭代，chunk 间检查按键 ──
        # 不用线程，避免 Debug Console / IDE 对子线程输出的缓冲问题
        print(f"\n  ⏳ {model_display_name(model)} 正在生成（流式输出，{reasoning_label}）……")
        if allow_cancel:
            print(f"  （按 c 键可随时中断请求）")
        print(f"{'=' * 62}")

        chunks: list[str] = []
        _in_thinking = False  # 当前是否处于 thinking 片段

        _t_stream_start = time.time()
        print(f"  🕐 [计时] 开始调用 generate_content_stream …")
        _first_chunk_received = False
        for chunk in client.models.generate_content_stream(
            model=model, contents=prompt, config=config
        ):
            if not _first_chunk_received:
                _elapsed = time.time() - _t_stream_start
                print(f"  🕐 [计时] 第一个 chunk 到达，耗时 {_elapsed:.3f} 秒")
                _first_chunk_received = True
            # 每个 chunk 到达后先检查是否按了 c
            if allow_cancel and _key_pressed_c():
                print(f"\n\n  ⚠ 流式请求已中断（c 键）")
                return None

            # ── 尝试逐 part 处理，区分思考 / 正文 ──
            parts = []
            try:
                if chunk.candidates:
                    parts = chunk.candidates[0].content.parts or []
            except (AttributeError, IndexError):
                pass

            if parts:
                for part in parts:
                    is_thought = getattr(part, "thought", False)
                    text = part.text or ""
                    if not text:
                        continue
                    if is_thought:
                        if not _in_thinking:
                            print(f"  💭 思考过程：")
                            print(f"{'─' * 62}")
                            _in_thinking = True
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    else:
                        if _in_thinking:
                            print(f"\n{'─' * 62}")
                            print(f"  📝 生成响应：")
                            _in_thinking = False
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        chunks.append(text)
            else:
                # 回退：SDK 版本不支持 parts，直接用 chunk.text
                piece = chunk.text or ""
                if piece:
                    sys.stdout.write(piece)
                    sys.stdout.flush()
                    chunks.append(piece)

        result = "".join(chunks)
        print(f"\n{'=' * 62}")
        print(f"  ✅ 响应完成（{len(result):,} 字符）")
        _print_cost(model, prompt, result, show_cost)
        return result

    else:
        # ── 非流式：子线程请求，主线程轮询 'c' 键 ──
        import threading

        result_box: list = [None]
        error_box: list = [None]
        done_event = threading.Event()

        def _worker_non_stream():
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                result_box[0] = resp.text or ""
            except Exception as e:
                error_box[0] = e
            finally:
                done_event.set()

        if confirm:
            print(f"  ⏳ 正在调用 {model_display_name(model)}（{reasoning_label}）……")
        if allow_cancel:
            print(f"  （按 c 键可随时中断请求）")

        t = threading.Thread(target=_worker_non_stream, daemon=True)
        t.start()

        user_cancelled = False
        while not done_event.wait(timeout=0.05):
            if allow_cancel and _key_pressed_c():
                user_cancelled = True
                break

        if user_cancelled:
            print(f"\n  ⚠ 请求已中断（c 键）")
            return None
        if error_box[0]:
            raise error_box[0]
        result = result_box[0]
        if confirm:
            print(f"  ✅ 响应完成（{len(result):,} 字符）")
        _print_cost(model, prompt, result, show_cost)
        return result


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Prompt 临时文件存储                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _save_prompt_to_md(
    prompt: str,
    model: str,
    system_prompt: str | None = None,
) -> Path:
    """
    将 system_prompt（如有）和 prompt 写入 LLM_Lib/temp/ 下的临时 .md 文件。
    文件名含时间戳，路径不含空格，便于审核。
    返回写入的文件路径。
    """
    temp_dir = Path(__file__).parent / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒精度，避免同秒冲突
    filename = f"prompt_{timestamp}.md"
    filepath = temp_dir / filename

    lines: list[str] = []
    lines.append(f"# Prompt — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")
    lines.append(f"**Model:** `{model}`")
    lines.append(f"")
    if system_prompt:
        lines.append("## System Prompt")
        lines.append("")
        lines.append(system_prompt)
        lines.append("")
    lines.append("## User Prompt")
    lines.append("")
    lines.append(prompt)

    filepath.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [Prompt saved] {filepath}")
    return filepath


def _append_response_to_md(filepath: Path, response: str) -> None:
    """
    将 LLM 响应追加到已存在的 Prompt .md 文件中。
    若文件已含 "## Response" 节则跳过（避免重复追加）。
    """
    try:
        existing = filepath.read_text(encoding="utf-8")
        if "## Response" in existing:
            return
        with open(filepath, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n## Response\n\n")
            f.write(response)
        print(f"  [Response saved] {filepath}")
    except Exception as e:
        print(f"  ⚠ 无法写入响应到 Markdown 文件：{e}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  通用 call_gemini                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def call_gemini(
    prompt: str,
    *,
    model: str = "",
    reasoning: bool | None = None,
    confirm: bool = True,
    stream: bool = True,
    manual_paste: bool = False,
    system_prompt: str | None = None,
    temperature: float = 1.0,
    max_output_tokens: int = 65536,
    allow_cancel: bool = True,
    show_cost: bool = True,
) -> str:
    """
    通用 Gemini API 调用函数。

    Args:
        prompt:            用户 Prompt 文本。
        model:             模型名称（留空则使用 DEFAULT_MODEL）。
        reasoning:         是否启用思考模式。默认 None = 自动判断：
                           Pro 模型默认开启，Flash 模型默认关闭。
                           传入 True/False 可显式覆盖。
        confirm:           是否在发送前要求用户终端确认（默认 True）。
        stream:            是否使用流式输出，实时打印模型响应（默认 True）。
                           缓存命中时无论此参数如何均直接返回，不走流式。
        manual_paste:      是否使用手动粘贴模式（默认 False）。为 True 时
                           跳过 API 调用，改为用 Notepad++ 打开 Prompt 供用户
                           复制到网页版 LLM，然后在终端粘贴响应。
        system_prompt:     系统提示词（可选）。
        temperature:       采样温度（0.0–2.0）。
        max_output_tokens: 最大输出 token 数。
        allow_cancel:      是否允许按 'c' 键中断请求（默认 True）。并发调用时建议设为 False。
        show_cost:         是否在每次 API 调用后打印本次及会话累计费用（默认 True）。
                           无论此参数如何，费用均会计入全局会话累计值。

    Returns:
        LLM 响应完整文本。

    Raises:
        RuntimeError: confirm=True 时用户取消。
        ValueError:   API Key 未配置。
    """
    model = model or DEFAULT_MODEL

    # ── 用户确认（可能切换到手动粘贴模式）──
    if confirm:
        model, manual_paste_chosen = _confirm_before_send(prompt, model)
        manual_paste = manual_paste or manual_paste_chosen

    # ── 自动决定 reasoning：仅白名单中的 Pro 模型默认开启 ──
    if reasoning is None:
        reasoning = model in _MODELS_DEFAULT_REASONING

    # ── 缓存检查 ──
    cached = get_cached(model, prompt, system_prompt, temperature, reasoning)
    if cached is not None:
        return cached

    # ── 保存 Prompt 到临时 .md 文件（供审核）──
    _md_filepath = _save_prompt_to_md(prompt, model, system_prompt)

    # ── 手动粘贴模式 ──
    if manual_paste:
        result, _rid = _do_manual_paste_call(prompt, model, system_prompt)
        save_cache(model, prompt, system_prompt, temperature, reasoning, result)
        _append_response_to_md(_md_filepath, result)
        return result

    # ── API 调用 ──
    _ensure_genai()

    # ── 构建配置 ──
    config = _genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    if system_prompt:
        config.system_instruction = system_prompt

    if reasoning:
        try:
            config.thinking_config = _genai_types.ThinkingConfig(
                thinking_budget=8192
            )
        except (AttributeError, TypeError):
            pass  # SDK 版本不支持 thinking config

    client = _get_client()

    while True:
        result = _api_call_with_cancel(client, model, prompt, config, stream, confirm, allow_cancel, reasoning, show_cost)
        if result is not None:
            break
        # 用户按 c 取消，询问是否重试
        try:
            retry = input("  是否重新发送请求？[Enter / y = 重试，n = 取消] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            retry = "n"
        if retry in ("", "y", "yes"):
            print(f"  🔄 重新发送……")
        else:
            raise RuntimeError("用户中断了 LLM 调用")

    # ── 写入缓存（仅成功完成时）──
    save_cache(model, prompt, system_prompt, temperature, reasoning, result)
    _append_response_to_md(_md_filepath, result)

    return result


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  JSON 提取                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_json(text: str) -> dict | list | None:
    """
    从 LLM 输出中提取 JSON 对象或数组。

    支持:
      - Markdown 代码块 (``​`json ... ``​`)
      - 裸 JSON 对象 ({...})
      - 裸 JSON 数组 ([...])

    Returns:
        解析后的 dict / list，解析失败时返回 None。
    """
    # 1. Markdown 代码块
    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. 裸 JSON 对象
    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 裸 JSON 数组
    match = re.search(r'(\[[\s\S]*\])', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Markdown 标题级别洗白                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def adjust_heading_levels(text: str, min_level: int) -> str:
    """
    将 Markdown 文本中所有标题的级别整体向深处平移，使最浅层的标题恰好
    展示为 `min_level` 级（其余标题按相对深度比例顺延）。

    应用场景: 将外部文件内容嵌入到 Prompt 时，避免内容
    自身的 `#` / `##` 标题突破 Prompt 框架的标题层级结构。

    例:
        嵌入位置的 Prompt 样式标题是 `##`，min_level=3:

        输入 Markdown:       输出 Markdown:
          # 大标题               ### 大标题
          ## 章          →      #### 章
          ### 节                 ##### 节

    Args:
        text:      包含 Markdown 标题的原文本。
        min_level: 目标最浅标题级别（1–6）。当文本中最浅标题当前已经
                   >= min_level 时，返回原文本不作磁。

    Returns:
        标题级别调整后的文本。非标题行不变。
    """
    if not text:
        return text

    lines = text.splitlines()

    # 找到当前最浅（# 数最少）的标题级别
    current_min: int | None = None
    for line in lines:
        m = re.match(r'^(#+)\s', line)
        if m:
            level = len(m.group(1))
            if current_min is None or level < current_min:
                current_min = level

    if current_min is None or current_min >= min_level:
        return text  # 文本中无标题，或所有标题已经 >= min_level

    shift = min_level - current_min
    result: list[str] = []
    for line in lines:
        m = re.match(r'^(#+)(\s.*)', line)
        if m:
            new_level = min(len(m.group(1)) + shift, 6)  # 最深不超过 6 级
            result.append('#' * new_level + m.group(2))
        else:
            result.append(line)
    return '\n'.join(result)



# from LLM_Lib.LLM import backfill_md_responses_from_cache
# backfill_md_responses_from_cache()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  __main__ Interactive Mode                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _interactive_simple_llm() -> None:
    """交互式简单 LLM 访问：选择模型 → 输入多行 Prompt → 获取结果。"""

    separator = "─" * 62

    # ── 模型选择 ──
    available_models = list(_MODEL_DISPLAY_NAMES.items())
    print(f"\n{'═' * 62}")
    print("  可用模型：")
    print(separator)
    for i, (model_id, display) in enumerate(available_models, 1):
        default_tag = " (默认)" if model_id == DEFAULT_MODEL else ""
        print(f"    [{i}] {display}  ({model_id}){default_tag}")
    print(separator)

    while True:
        try:
            choice = input(f"  请选择模型编号 [默认=1] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [已退出]")
            return
        if choice == "":
            chosen_model = available_models[0][0]
            break
        if choice.isdigit() and 1 <= int(choice) <= len(available_models):
            chosen_model = available_models[int(choice) - 1][0]
            break
        print(f"  无效选择，请输入 1-{len(available_models)}")

    print(f"  ✓ 已选择: {model_display_name(chosen_model)}")

    # ── 多行 Prompt 输入 ──
    print(f"\n{separator}")
    print("  请输入 Prompt（多行输入，单独一行输入 end 结束）：")
    print(separator)

    lines: list[str] = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print("\n  [已中断]")
            return
        if line.strip() == "end":
            break
        lines.append(line)

    prompt = "\n".join(lines)
    if not prompt.strip():
        print("  ⚠ Prompt 为空，已跳过。")
        return

    print(f"\n{separator}")
    print(f"  Prompt 长度: {len(prompt):,} 字符")
    print(f"  正在调用 {model_display_name(chosen_model)}……")
    print(separator)

    # ── 调用 LLM ──
    response = call_gemini(
        prompt,
        model=chosen_model,
        confirm=False,
        stream=True,
    )

    print(f"\n{'═' * 62}")
    print("  ✓ 调用完成")
    print(f"{'═' * 62}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  LLM Interactive Mode                                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    while True:
        print()
        print("  功能列表：")
        print("    [1] 简单 LLM 访问")
        print()
        try:
            mode = input("  请选择功能编号（q 退出）> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break
        if mode in ("q", "quit", "exit"):
            print("  再见！")
            break
        elif mode == "1":
            _interactive_simple_llm()
        else:
            print("  无效选择，请输入 1 或 q")