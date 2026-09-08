# -*- coding: utf-8 -*-
"""
Paper_Search_MCP_Launcher.py —— 带本地补丁的 paper-search MCP server 启动器
==========================================================================

替代直接运行 `paper-search-mcp` 命令。启动顺序：

1. 调用 :func:`Paper_Search_MCP_Patches.apply_all` 在**进程内**打上本地补丁
   （目前只有一项：OpenAlex API key 支持，上游 0.1.4 版缺失）；
2. 顺手检查 site-packages 里的落盘补丁还在不在，不在就自动补回来
   （供 `paper-search` 命令行入口使用，见 Patches 模块的说明）；
3. 拉起上游的 MCP server（stdio 传输）。

**必须让第 1 步早于导入 `paper_search_mcp.server`**：server 模块在导入时就
执行了 `openalex_searcher = OpenAlexSearcher()`，补丁晚一步就作用不到这个
已经建好的实例上。

怎么跑
------

这个脚本要用**安装了 paper-search-mcp 的那个 Python** 来跑，也就是 uv tool
的环境。Claude Code 的 MCP server 配置写成：

    "paper-search": {
      "type": "stdio",
      "command": "uv",
      "args": ["tool", "run", "--from", "paper-search-mcp", "python",
               "E:/My_Program/Python_Lib/src/Tools/Literature_Management/Paper_Search_MCP_Launcher.py"]
    }

`uv tool run` 的额外开销实测约 0.2 秒，可以忽略。

为什么值得这么绕
----------------

补丁如果直接写进 site-packages，`uv tool upgrade paper-search-mcp` 会把它冲掉，
而且是安静地冲掉——OpenAlex 开始返回 403，但不会有人立刻联想到几天前的那次
升级。走这个启动器之后，补丁的正本住在本仓库里，每次启动重新应用一遍，
升级外部包影响不到它。

一切正常时本脚本不往 stderr 写任何东西（MCP 的 stdio 传输只用 stdout 通信，
stderr 会进 Claude Code 的日志）。只有补丁失败或落盘补丁被修复时才输出。
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    # 补丁相关的提示是中文的，固定用 UTF-8 写 stderr，避免在 GBK 终端里成为乱码。
    # 不动 stdout —— 那是 MCP 的 stdio 通道，交给上游 server 自己处理。
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    # 让 Paper_Search_MCP_Patches 可导入。用完立刻把这个目录从 sys.path 移除，
    # 避免 src/Tools 下的其他模块名意外遮蔽上游依赖。
    tools_dir = str(Path(__file__).resolve().parent)
    inserted = False
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
        inserted = True

    try:
        import Paper_Search_MCP_Patches as patches
    finally:
        if inserted and tools_dir in sys.path:
            sys.path.remove(tools_dir)

    # 第 1 步：进程内补丁——必须早于导入 server（它在导入时就实例化 searcher）
    patches.apply_all()

    # 第 2 步：落盘补丁自愈——服务于 paper-search 命令行入口，失败不阻塞启动
    try:
        patches.ensure_file_patch()
    except Exception as exc:  # noqa: BLE001 —— 任何异常都不该拦住 server 启动
        print(f"[本地补丁] 落盘补丁检查异常（不影响本次启动）：{exc}", file=sys.stderr)

    # 第 3 步：拉起上游 server
    from paper_search_mcp.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
