# -*- coding: utf-8 -*-
"""
Paper_Search_MCP_Patches.py —— paper-search-mcp 的本地补丁集合
================================================================

`paper-search-mcp` 是外部包（用 `uv tool install` 装的），它有一处能力缺失需要
本地补上：**OpenAlex 自 2026-02-13 起强制要求 API key**（免费申请），而上游
0.1.4 版的 `academic_platforms/openalex.py` 还停留在已被废除的 polite pool 机制，
完全没有密钥相关代码，`User-Agent` 里写的还是上游作者的占位邮箱。

这个文件解决的是"补丁怎么活下来"的问题
------------------------------------------

历史做法是直接改写 site-packages 里的 `openalex.py`。这么做的毛病是
**`uv tool upgrade paper-search-mcp` 会把整个包重装一遍，补丁随之消失**，而且
消失得很安静——OpenAlex 检索开始返回 403，但没人会立刻把它和几天前跑过的
一次升级联系起来。

所以改成两条腿走路：

1. **进程内打补丁（主路径，升级绝对冲不掉）**：MCP server 不再由
   `paper-search-mcp` 这个命令直接启动，而是由 `Paper_Search_MCP_Launcher.py`
   启动。启动器先调用本模块的 :func:`apply_all` 在内存里把补丁打好，再拉起上游的
   server。补丁代码住在本仓库里，升级外部包不会影响它。
2. **落盘补丁（辅路径，服务于命令行入口）**：`paper-search` 命令行工具走的是
   上游自己的入口，进程内补丁够不着它。所以 :func:`ensure_file_patch` 会把同一段
   补丁代码**追加**到 site-packages 的 `openalex.py` 末尾。启动器每次启动时顺手
   检查一次，发现补丁没了就自动补上——也就是说升级之后的第一次 Claude Code
   会话就会把落盘补丁自动修复，不需要人记得跑什么脚本。

补丁代码本身（:data:`OPENALEX_KEY_PATCH_CODE`）只写一份，上面两条路径共用：
进程内路径把它 `exec` 进 `openalex` 模块的命名空间，落盘路径把同一段文本追加到
模块文件末尾。两种情形下它的运行环境完全一致（都是 `openalex` 模块的全局命名
空间），所以不存在"两份实现走偏"的问题。

补丁的写法刻意不依赖上游 `search()` 的内部实现——它包装的是
`OpenAlexSearcher.__init__` 与该实例持有的 `requests.Session`，让这个 session
发往 `api.openalex.org` 的**每个**请求都自动带上 `api_key` 查询参数。上游哪怕重构
了参数拼装逻辑，补丁依然有效。

密钥本身不在这个文件里，运行时从 `~/.config/paper-search-mcp/.env` 读取
（上游 `config.get_env` 的既有机制），所以本文件可以安全地进版本控制。
"""

from __future__ import annotations

import sys

#: 追加到 openalex.py 末尾、或 exec 进该模块命名空间的补丁代码。
#: 依赖的名字（``OpenAlexSearcher``、包相对导入 ``..config``）都由 openalex
#: 模块的命名空间提供，所以这段代码在两种运行方式下行为一致。
OPENALEX_KEY_PATCH_CODE = '''

# ── 本地补丁：OpenAlex API key 支持 ──────────────────────────────────────────
# OpenAlex 自 2026-02-13 起要求 API key（免费申请），上游没有相关代码。
# 本段由 Python_Lib/src/Tools/Literature_Management/Paper_Search_MCP_Patches.py 生成，两种方式注入：
#   - 进程内：Paper_Search_MCP_Launcher.py 在拉起 MCP server 前 exec 本段
#   - 落盘：  追加到本文件末尾，供 paper-search 命令行入口使用
# 密钥从 ~/.config/paper-search-mcp/.env 读取，不出现在任何源码里。

def _local_patch_inject_openalex_key(session, api_key):
    """让 *session* 发往 api.openalex.org 的每个请求都带上 api_key 查询参数。"""
    if getattr(session, "_local_patch_key_injected", False):
        return
    original_request = session.request

    def request_with_key(method, url, **kwargs):
        if "api.openalex.org" in str(url):
            params = dict(kwargs.get("params") or {})
            params.setdefault("api_key", api_key)
            kwargs["params"] = params
        return original_request(method, url, **kwargs)

    session.request = request_with_key
    session._local_patch_key_injected = True


def _local_patch_apply_openalex_key():
    """包装 OpenAlexSearcher.__init__，注入密钥与真实邮箱。可重复调用。"""
    if getattr(OpenAlexSearcher.__init__, "_local_patch_openalex_key", False):
        return

    from ..config import get_env

    original_init = OpenAlexSearcher.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        api_key = (get_env("OPENALEX_API_KEY", "") or "").strip()
        email = (get_env("OPENALEX_EMAIL", "") or "").strip()
        self.api_key = api_key
        user_agent = "paper-search-mcp/1.0"
        if email:
            user_agent += " (mailto:%s)" % email
        self.session.headers.update({"User-Agent": user_agent})
        if api_key:
            _local_patch_inject_openalex_key(self.session, api_key)

    patched_init._local_patch_openalex_key = True
    OpenAlexSearcher.__init__ = patched_init


_local_patch_apply_openalex_key()
# ── 本地补丁结束 ────────────────────────────────────────────────────────────
'''

#: 判断补丁是否已经落盘的标记串。改动补丁内容时不要改它，否则旧补丁会被重复追加。
PATCH_MARKER = "本地补丁：OpenAlex API key 支持"


def apply_all(verbose: bool = False) -> list[str]:
    """在**当前进程内**打上全部本地补丁，返回已生效的补丁名列表。

    只在内存里生效，不改动磁盘上的任何文件。由启动器在拉起 MCP server 前调用。
    """
    applied: list[str] = []

    try:
        from paper_search_mcp.academic_platforms import openalex as openalex_module
    except Exception as exc:
        print(f"[本地补丁] 无法导入 openalex 模块，跳过 OpenAlex 补丁：{exc}",
              file=sys.stderr)
        return applied

    try:
        exec(OPENALEX_KEY_PATCH_CODE, openalex_module.__dict__)
        applied.append("openalex-api-key")
        if verbose:
            print("[本地补丁] OpenAlex API key 补丁已在进程内生效。", file=sys.stderr)
    except Exception as exc:
        print(f"[本地补丁] OpenAlex 补丁应用失败：{exc}", file=sys.stderr)

    return applied


def locate_openalex_file():
    """返回 site-packages 里 openalex.py 的路径；定位不到时返回 ``None``。"""
    from pathlib import Path

    try:
        from paper_search_mcp.academic_platforms import openalex as openalex_module
    except Exception:
        return None
    path = getattr(openalex_module, "__file__", None)
    return Path(path) if path else None


def ensure_file_patch(verbose: bool = False) -> str:
    """确保补丁已经**落盘**到 site-packages 的 openalex.py 里。

    这一份是给 ``paper-search`` 命令行入口用的——命令行走上游自己的入口，
    进程内补丁够不着。启动器每次启动都会调用本函数，所以
    ``uv tool upgrade`` 冲掉补丁之后，下一次 Claude Code 会话会自动修复。

    Returns:
        ``"already"``（本来就在） / ``"patched"``（这次补上了） /
        ``"failed"``（定位或写入失败，已在 stderr 说明原因）。
    """
    module_path = locate_openalex_file()
    if module_path is None or not module_path.exists():
        print("[本地补丁] 定位不到 openalex.py，跳过落盘补丁。", file=sys.stderr)
        return "failed"

    try:
        source = module_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[本地补丁] 读取 openalex.py 失败：{exc}", file=sys.stderr)
        return "failed"

    if PATCH_MARKER in source:
        if verbose:
            print(f"[本地补丁] {module_path.name} 已含补丁，无需改动。", file=sys.stderr)
        return "already"

    try:
        backup_path = module_path.with_suffix(".py.upstream")
        if not backup_path.exists():
            backup_path.write_text(source, encoding="utf-8")
        module_path.write_text(source + OPENALEX_KEY_PATCH_CODE, encoding="utf-8")
    except Exception as exc:
        print(f"[本地补丁] 写入 openalex.py 失败：{exc}", file=sys.stderr)
        return "failed"

    print(f"[本地补丁] 已把 OpenAlex 密钥补丁重新写入 {module_path}"
          "（多半是刚跑过 uv tool upgrade）。", file=sys.stderr)
    return "patched"
