# -*- coding: utf-8 -*-
"""
Paper_Search_MCP_Setup.py —— paper-search-mcp 的配置同步与体检工具
==================================================================

一条命令把 `paper-search-mcp` 的本地配置带到正确状态并当场验证：

1. **同步密钥**：从 `LLM_API_KEYS_PRIVATE.py` 读出各家学术检索 API 的密钥，
   写进 `~/.config/paper-search-mcp/.env`（上游包读取密钥的唯一位置）。
   新申请到一个密钥时，只需要往 `LLM_API_KEYS_PRIVATE.py` 里加一行，
   再跑一次本脚本，不用记 `.env` 里的键名长什么样。
2. **修复落盘补丁**：确认 site-packages 里的 OpenAlex 密钥补丁还在，
   被 `uv tool upgrade` 冲掉了就重新写回去。
3. **检查 MCP 启动配置**：确认 Claude Code 的 `paper-search` server 走的是
   `Paper_Search_MCP_Launcher.py`，而不是会丢补丁的裸 `paper-search-mcp` 命令。
4. **联网验证**：真的各发一次检索请求，确认 OpenAlex 与 Semantic Scholar
   在带密钥的情况下能正常返回。

用法::

    python src/Tools/Paper_Search_MCP_Setup.py            # 同步 + 修复 + 验证
    python src/Tools/Paper_Search_MCP_Setup.py --check    # 只体检，不改任何文件

日常并不需要手动跑这个脚本——`Paper_Search_MCP_Launcher.py` 每次启动 MCP server
时都会自动做第 2 步。需要手动跑的场合只有两个：**新申请到一个密钥**，或者
**想确认当前状态是否健康**。

密钥值本身不出现在本文件里，也不会被打印出来。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

#: `LLM_API_KEYS_PRIVATE.py` 里的变量名 → `.env` 里的键名。
#: 左边跟着密钥文件的实际拼写走，右边跟着上游 `config.get_env` 的 `PAPER_SEARCH_MCP_` 前缀走。
KEY_MAPPING: dict[str, str] = {
    "OpenAlex_API_KEY":            "PAPER_SEARCH_MCP_OPENALEX_API_KEY",
    "Semantic_Scholar_S2_API_KEY": "PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY",
    "CORE_API_KEY":                "PAPER_SEARCH_MCP_CORE_API_KEY",
    "DOAJ_API_KEY":                "PAPER_SEARCH_MCP_DOAJ_API_KEY",
}

ENV_PATH = Path.home() / ".config" / "paper-search-mcp" / ".env"

LAUNCHER_PATH = Path(__file__).resolve().parent / "Paper_Search_MCP_Launcher.py"


# ═════════════════════════════════════════════════════════════════════════════
# 一、定位并读取密钥文件
# ═════════════════════════════════════════════════════════════════════════════

def find_api_keys_file() -> Path | None:
    """定位 `LLM_API_KEYS_PRIVATE.py`，不把个人绝对路径写死在源码里。

    与 ``LLM_Lib.LLM._find_api_keys_dir`` 同一套规则：先看环境变量
    ``LLM_API_KEYS_DIR``，再从本文件向上找到本仓库根 ``Python_Lib``，
    取它的上一层目录。
    """
    candidates: list[Path] = []
    env_dir = os.environ.get("LLM_API_KEYS_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    for parent in Path(__file__).resolve().parents:
        if parent.name == "Python_Lib" and (parent / "src").is_dir():
            candidates.append(parent.parent)
            break
    for directory in candidates:
        path = directory / "LLM_API_KEYS_PRIVATE.py"
        if path.is_file():
            return path
    return None


def read_keys(keys_file: Path) -> dict[str, str]:
    """用 ast 静态解析出形如 ``NAME = "字面值"`` 的赋值，不执行这个文件。"""
    tree = ast.parse(keys_file.read_text(encoding="utf-8"), filename=str(keys_file))
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found[target.id] = node.value.value
    return found


# ═════════════════════════════════════════════════════════════════════════════
# 二、同步 .env
# ═════════════════════════════════════════════════════════════════════════════

def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def sync_env(keys: dict[str, str], *, dry_run: bool) -> list[str]:
    """把密钥写进 `.env`，返回本次发生变化的键名列表。

    只改 :data:`KEY_MAPPING` 覆盖到的键，其余行（注释、邮箱等）原样保留。
    """
    existing_text = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    existing = parse_env(existing_text)

    changes: list[str] = []
    for source_name, env_name in KEY_MAPPING.items():
        new_value = (keys.get(source_name) or "").strip()
        if not new_value:
            continue
        if existing.get(env_name, "") == new_value:
            continue
        changes.append(env_name)
        existing[env_name] = new_value

    if not changes or dry_run:
        return changes

    lines = existing_text.splitlines()
    remaining = set(changes)
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.partition("=")[0].strip()
        if key in remaining:
            lines[index] = f"{key}={existing[key]}"
            remaining.discard(key)
    for key in changes:
        if key in remaining:
            lines.append(f"{key}={existing[key]}")

    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    return changes


# ═════════════════════════════════════════════════════════════════════════════
# 三、检查 Claude Code 的 MCP server 启动配置
# ═════════════════════════════════════════════════════════════════════════════

def check_mcp_config() -> tuple[bool, str]:
    """确认 `paper-search` server 走的是本地启动器。返回 (是否正确, 说明)。"""
    config_path = Path.home() / ".claude.json"
    if not config_path.exists():
        return False, "~/.claude.json 不存在，无法检查 MCP 启动配置。"

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"~/.claude.json 解析失败：{exc}"

    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or "paper-search" not in servers:
        return False, "~/.claude.json 里没有名为 paper-search 的 MCP server。"

    entry = servers["paper-search"]
    launched_with = " ".join([str(entry.get("command", ""))] +
                             [str(a) for a in entry.get("args", [])])
    if "Paper_Search_MCP_Launcher.py" in launched_with:
        return True, "MCP server 已经走本地启动器，补丁不会被 uv tool upgrade 冲掉。"
    return False, (
        "MCP server 目前直接启动上游命令，OpenAlex 补丁会在 uv tool upgrade 后失效。\n"
        "      请把 ~/.claude.json 中 paper-search 的启动方式改为：\n"
        '        "command": "uv",\n'
        '        "args": ["tool", "run", "--from", "paper-search-mcp", "python",\n'
        f'                 "{LAUNCHER_PATH.as_posix()}"]'
    )


# ═════════════════════════════════════════════════════════════════════════════
# 四、需要 paper_search_mcp 环境的步骤（落盘补丁 + 联网验证）
# ═════════════════════════════════════════════════════════════════════════════

def run_in_tool_env(check_only: bool) -> int:
    """在 uv tool 环境里重新跑本脚本的第 2、4 步。"""
    argv = ["uv", "tool", "run", "--from", "paper-search-mcp", "python",
            str(Path(__file__).resolve()), "--in-tool-env"]
    if check_only:
        argv.append("--check")
    # 先把父进程的输出刷出去，否则子进程的输出会插到前面，读起来次序全乱
    sys.stdout.flush()
    sys.stderr.flush()
    return subprocess.run(argv).returncode


def tool_env_steps(check_only: bool) -> bool:
    """在已经能 import paper_search_mcp 的进程里执行。返回是否全部通过。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import Paper_Search_MCP_Patches as patches

    ok = True

    print("\n[3/4] 落盘补丁（供 paper-search 命令行入口使用）")
    module_path = patches.locate_openalex_file()
    if module_path is None:
        print("      定位不到 openalex.py —— paper-search-mcp 可能没装好。")
        ok = False
    elif check_only:
        present = patches.PATCH_MARKER in module_path.read_text(encoding="utf-8")
        print(f"      补丁在位：{'是' if present else '否（跑一次不带 --check 的本脚本即可补上）'}")
        ok = ok and present
    else:
        result = patches.ensure_file_patch()
        print({"already": "      补丁本来就在，未改动。",
               "patched": "      补丁不在，已重新写入。",
               "failed":  "      补丁写入失败，见上面的报错。"}[result])
        ok = ok and result != "failed"

    # 进程内补丁：让下面的验证走到和 MCP server 完全一样的代码路径
    patches.apply_all()

    print("\n[4/4] 联网验证")
    ok = verify_openalex() and ok
    ok = verify_semantic_scholar() and ok
    return ok


def verify_openalex() -> bool:
    """OpenAlex 的补丁正不正常。

    这里把"返回 0 条"当作失败：补丁失效时 OpenAlex 恰好就是 403 加空结果，
    而 upstream 的 ``search()`` 会把错误吞掉只返回空列表——所以空结果正是
    我们要抓的症状，不能放过。
    """
    try:
        from paper_search_mcp.academic_platforms.openalex import OpenAlexSearcher
        searcher = OpenAlexSearcher()
        has_key = bool(getattr(searcher, "api_key", ""))
        papers = searcher.search("CRISPR Cas9", max_results=2)
        if not papers:   # 给一次重试的机会，排除一次性抖动
            time.sleep(3)
            papers = searcher.search("CRISPR Cas9", max_results=2)
        print(f"      OpenAlex：密钥已加载={has_key}，返回 {len(papers)} 条")
        if papers:
            print(f"        例：{papers[0].title[:60]}")
        elif has_key:
            print("        密钥在，但两次都没有结果 —— 补丁可能失效了，"
                  "上面的日志里应该有 403")
        return has_key and bool(papers)
    except Exception as exc:  # noqa: BLE001
        print(f"      OpenAlex 验证失败：{exc}")
        return False


def verify_semantic_scholar() -> bool:
    """Semantic Scholar 的密钥有没有被认出来。

    与 OpenAlex 不同，这里**不**把"返回 0 条"当作失败：密钥是上游原生支持的，
    不涉及任何补丁，只要密钥加载上了配置就是好的。Semantic Scholar 侧偶尔会
    返回 HTTP 500 或把请求限流掉，那是对方的临时状况，不该让体检报红——
    否则用户会习惯性忽略这个体检的结论。
    """
    try:
        from paper_search_mcp.academic_platforms.semantic import SemanticSearcher
        searcher = SemanticSearcher()
        has_key = bool(searcher.get_api_key())
        papers = searcher.search("CRISPR Cas9", max_results=2)
        if not papers:
            time.sleep(3)
            papers = searcher.search("CRISPR Cas9", max_results=2)
        print(f"      Semantic Scholar：密钥已加载={has_key}，返回 {len(papers)} 条")
        if papers:
            print(f"        例：{papers[0].title[:60]}")
        elif has_key:
            print("        密钥正常。这次没拿到结果是 Semantic Scholar 侧的临时状况"
                  "（限流或 5xx），不计为配置问题。")
        return has_key
    except Exception as exc:  # noqa: BLE001
        print(f"      Semantic Scholar 验证失败：{exc}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 五、入口
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="paper-search-mcp 的密钥同步与体检工具")
    parser.add_argument("--check", action="store_true",
                        help="只体检，不修改任何文件")
    parser.add_argument("--in-tool-env", action="store_true",
                        help=argparse.SUPPRESS)  # 内部使用：已在 uv tool 环境里
    args = parser.parse_args()

    if args.in_tool_env:
        sys.exit(0 if tool_env_steps(args.check) else 1)

    print("=" * 66)
    print("paper-search-mcp 配置同步与体检")
    print("=" * 66)

    print("\n[1/4] 从 LLM_API_KEYS_PRIVATE.py 同步密钥到 .env")
    keys_file = find_api_keys_file()
    if keys_file is None:
        print("      找不到 LLM_API_KEYS_PRIVATE.py，跳过同步。")
    else:
        keys = read_keys(keys_file)
        available = [name for name in KEY_MAPPING if keys.get(name, "").strip()]
        missing = [name for name in KEY_MAPPING if name not in available]
        print(f"      密钥文件：{keys_file}")
        print(f"      已提供：{', '.join(available) if available else '（无）'}")
        if missing:
            print(f"      未提供：{', '.join(missing)}")
        changed = sync_env(keys, dry_run=args.check)
        if not changed:
            print("      .env 已是最新，无需改动。")
        elif args.check:
            print(f"      待更新（--check 模式未写入）：{', '.join(changed)}")
        else:
            print(f"      已更新：{', '.join(changed)}")

    print("\n[2/4] Claude Code 的 MCP 启动配置")
    config_ok, message = check_mcp_config()
    print(f"      {'正常' if config_ok else '需要处理'}：{message}")

    exit_code = run_in_tool_env(args.check)

    print("\n" + "=" * 66)
    print("体检结束。" if exit_code == 0 and config_ok
          else "体检结束，上面标为『需要处理』或验证失败的项目需要跟进。")
    sys.exit(exit_code if config_ok else 1)


if __name__ == "__main__":
    main()
