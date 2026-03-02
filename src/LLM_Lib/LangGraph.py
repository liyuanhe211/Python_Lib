"""
LangGraph.py — LangGraph 工作流可复用组件

══════════════════════════════════════════════════════════════════════════════
  提供跨项目可复用的 LangGraph 工作流辅助工具：

  - PromptLogger:       会话级 LLM 调用日志管理器
                        每次 LLM 调用前交互确认、切换模型、
                        记录 Prompt/Response 到 Markdown 文件
  - get_current_session / set_current_session:
                        全局 PromptLogger 会话管理（单进程单线程）

使用方式:
    from LLM_Lib.LangGraph import PromptLogger, set_current_session

    logger = PromptLogger(log_dir=some_dir, description="撰写某章节")
    set_current_session(logger)

    response = logger.confirm_and_call(
        node_name="generate_text",
        prompt=user_prompt,
        default_model=GEMINI_PRO,
        call_fn=lambda model: call_gemini(prompt, model=model, confirm=False),
    )
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Callable

from LLM_Lib.LLM import (
    GEMINI_PRO,
    GEMINI_FLASH,
    model_display_name,
    adjust_heading_levels,
    _open_prompt_preview_in_vscode,
    _PROMPT_PREVIEW_THRESHOLD,
    _generate_request_id,
    _write_prompt_for_manual_paste,
    _open_with_notepadpp,
    _collect_manual_response,
    _do_manual_paste_call,
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  系统打开文件                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _open_with_system(path: Path | str) -> None:
    """使用系统关联程序打开文件。"""
    path_str = str(path)
    try:
        os.startfile(path_str)  # type: ignore[attr-defined]  # Windows only
    except AttributeError:
        try:
            subprocess.Popen(["open", path_str])
        except FileNotFoundError:
            subprocess.Popen(["xdg-open", path_str])
    except Exception:
        pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SkipNodeError — 用户跳过节点时抛出                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class SkipNodeError(Exception):
    """用户在交互确认时选择跳过当前节点的 LLM 调用。"""
    pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PromptLogger — 会话级日志管理器                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PromptLogger:
    """
    管理一次工作流运行的 Prompt/Response 日志文件。

    每次 LLM 调用前：
      1. 在终端显示 Prompt 预览（短 Prompt 全文；长 Prompt 在 VS Code 打开）
      2. 显示将要使用的模型
      3. 等待用户确认或切换模型：
           [Enter / y]  确认并发送（API）
           [a]          确认，后续调用自动确认
           [p]          改用 Pro 模型
           [f]          改用 Flash 模型
           [w]          使用网页手动粘贴模式（Notepad++）
           [c]          取消此次调用
      4. 确认后：调用 LLM / 手动粘贴 → 记录日志

    支持特性:
      - auto_confirm: 后续调用跳过交互确认
      - manual_paste: 粘性手动粘贴模式（首次选择后自动沿用）
      - 日志中用 adjust_heading_levels 替代代码块包裹 Prompt

    使用方式:
        logger = PromptLogger(log_dir=Path("./logs"), description="工作流日志")
        set_current_session(logger)

        response = logger.confirm_and_call(
            node_name="generate_text",
            prompt=prompt,
            default_model=GEMINI_PRO,
            call_fn=lambda model: call_gemini(prompt, model=model, confirm=False),
        )
    """

    def __init__(
        self,
        log_dir: Path | str,
        description: str = "工作流运行日志",
    ) -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_desc = description[:25]
        self.file_path: Path = log_dir / f"{ts} {safe_desc}.md"

        # 写文件头
        header = (
            f"# {safe_desc}\n\n"
            f"**会话时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n"
            f"---\n\n"
        )
        self.file_path.write_text(header, encoding="utf-8")

        self._call_count: int = 0
        # 粘性状态：首次交互后记忆用户选择
        self._use_manual_paste: bool | None = None  # None = 尚未决定
        self._auto_confirm_remaining: bool = False

    # ── 公共方法：带交互确认的 LLM 调用 ────────────────────────────────────────

    def confirm_and_call(
        self,
        node_name: str,
        prompt: str,
        default_model: str,
        call_fn: Callable[[str], str],
        *,
        auto_confirm: bool = False,
        system_prompt: str = "",
        allow_skip: bool = False,
    ) -> str:
        """
        交互式确认 LLM 调用，确认后发送并记录日志。

        Args:
            node_name:     节点名称（用于日志标题）。
            prompt:        将要发送的完整 Prompt 文本。
            default_model: 默认使用的模型名称（可由用户切换）。
            call_fn:       实际发起 LLM 调用的函数，接受一个 model 字符串参数，
                           返回响应文本字符串。
                           签名: (model: str) -> str
            auto_confirm:  是否跳过确认（由调用方指定，例如连续调用的第 2+ 次）。
            system_prompt: 系统提示词（手动粘贴模式写入文件时使用）。
            allow_skip:    是否允许用户跳过此节点（可选，默认 False）。
                           若为 True，交互确认时显示 [s] 跳过选项。
                           用户选择跳过时抛出 SkipNodeError。

        Returns:
            LLM 的响应文本。

        Raises:
            RuntimeError: 用户选择取消时。
            SkipNodeError: 用户选择跳过时（仅 allow_skip=True）。
        """
        self._call_count += 1
        current_model = default_model
        separator = "─" * 62

        # 判断是否需要交互
        should_auto = auto_confirm or self._auto_confirm_remaining
        use_paste = self._use_manual_paste  # 可能为 None、True、False

        if should_auto and use_paste is not None:
            # 已经做过选择且设为自动 → 跳过确认
            print(f"\n  ▸ 自动确认 LLM 调用 #{self._call_count}  节点: {node_name}")
            print(f"    模型: {model_display_name(current_model)}")
            print(f"    Prompt 长度: {len(prompt):,} 字符")
            if use_paste:
                print(f"    模式: 网页手动粘贴")
        else:
            # 需要交互确认
            current_model, use_paste = self._interactive_confirm(
                node_name, prompt, current_model, allow_skip=allow_skip
            )

        # ── 追加 Prompt 到日志文件 ──
        self._append_prompt(node_name, current_model, prompt)

        # ── 打开日志文件供审阅 ──
        _open_with_system(self.file_path)

        # ── 执行调用 ──
        if use_paste:
            response, _rid = _do_manual_paste_call(
                prompt, current_model, system_prompt or None
            )
        else:
            response = call_fn(current_model)

        # ── 追加响应到日志文件 ──
        self._append_response(response)

        return response

    # ── 批量手动粘贴：同时打开多个 Notepad++ 窗口 ─────────────────────────────

    def prepare_manual_paste_batch(
        self,
        batch: list[dict],
        model: str,
        system_prompt: str = "",
    ) -> list[dict]:
        """
        为多个 Prompt 同时准备手动粘贴文件，一次性打开所有 Notepad++ 窗口。

        Args:
            batch: 列表，每个元素 {"node_name": str, "prompt": str}
            model: 模型名称。
            system_prompt: 系统提示词。

        Returns:
            列表 [{"node_name", "prompt", "request_id", "filepath"}]
        """
        prepared: list[dict] = []
        file_paths: list[Path] = []

        for item in batch:
            rid = _generate_request_id()
            prompt = item["prompt"]
            # 追加 ID 指令
            id_instruction = (
                f"\n\n---\n**请求 ID: {rid}**\n"
                f"请在你的回复的最后一行重复此 ID，格式为: ID={rid}"
            )
            prompt_with_id = prompt + id_instruction
            filepath = _write_prompt_for_manual_paste(
                prompt_with_id, rid, model, system_prompt or None
            )
            file_paths.append(filepath)

            self._call_count += 1
            self._append_prompt(item["node_name"], model, prompt)

            prepared.append({
                "node_name": item["node_name"],
                "prompt": prompt,
                "request_id": rid,
                "filepath": filepath,
            })

        # 一次性打开所有文件
        if file_paths:
            _open_with_notepadpp(*file_paths)
            print(f"\n  📝 已同时打开 {len(file_paths)} 个 Prompt 文件：")
            for info in prepared:
                print(f"     [{info['request_id']}] {info['node_name']}")

        _open_with_system(self.file_path)
        return prepared

    def collect_manual_paste_response(self, request_id: str) -> str:
        """收集一个手动粘贴响应并追加到日志。"""
        response = _collect_manual_response(request_id)
        self._append_response(response)
        return response

    # ── 私有方法 ────────────────────────────────────────────────────────────────

    def _interactive_confirm(
        self,
        node_name: str,
        prompt: str,
        current_model: str,
        *,
        allow_skip: bool = False,
    ) -> tuple[str, bool]:
        """
        交互式确认。返回 (model, use_manual_paste)。
        会更新 self._use_manual_paste 和 self._auto_confirm_remaining。

        Raises:
            SkipNodeError: 用户选择跳过时（仅 allow_skip=True）。
        """
        separator = "─" * 62

        print(f"\n{'═' * 62}")
        print(f"  ▸ LLM 调用 #{self._call_count}   节点: {node_name}")
        print(f"{'═' * 62}")
        print(f"  模型:  {model_display_name(current_model)}")
        print(f"  Prompt 长度:  {len(prompt):,} 字符")
        print(f"{separator}")

        if len(prompt) <= _PROMPT_PREVIEW_THRESHOLD:
            print("  Prompt 全文：\n")
            for line in prompt.splitlines():
                print(f"    {line}")
        else:
            _open_prompt_preview_in_vscode(prompt, current_model)

        print(f"\n{separator}")
        print("  快捷选项：")
        print(
            f"    [Enter / y]  确认发送（模型: "
            f"{model_display_name(current_model)}）"
        )
        print(f"    [a]          确认发送，后续调用自动确认")
        print(f"    [p]          改用 Pro  模型 → {GEMINI_PRO}")
        print(f"    [f]          改用 Flash 模型 → {GEMINI_FLASH}")
        print(f"    [w]          使用网页手动粘贴模式（Notepad++）")
        if allow_skip:
            print(f"    [s]          跳过此节点（视为高分通过）")
        print(f"    [c]          取消此次调用")
        print(f"{separator}")

        while True:
            try:
                choice = input("  请选择 > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  [已取消]")
                raise RuntimeError(
                    f"用户取消了节点 [{node_name}] 的 LLM 调用"
                )

            if choice in ("", "y", "yes"):
                self._use_manual_paste = False
                return current_model, False
            elif choice == "a":
                self._auto_confirm_remaining = True
                self._use_manual_paste = False
                print(f"  ✓ 后续调用将自动确认")
                return current_model, False
            elif choice == "p":
                current_model = GEMINI_PRO
                self._use_manual_paste = False
                print(f"  ✓ 已切换为 Pro 模型: {GEMINI_PRO}")
                return current_model, False
            elif choice == "f":
                current_model = GEMINI_FLASH
                self._use_manual_paste = False
                print(f"  ✓ 已切换为 Flash 模型: {GEMINI_FLASH}")
                return current_model, False
            elif choice == "w":
                self._use_manual_paste = True
                self._auto_confirm_remaining = True  # 手动粘贴模式自动沿用
                print(f"  ✓ 将使用网页手动粘贴模式（后续调用自动沿用）")
                return current_model, True
            elif choice == "s" and allow_skip:
                print("  ✓ 跳过此节点，视为高分通过")
                raise SkipNodeError(
                    f"用户跳过了节点 [{node_name}]"
                )
            elif choice == "c":
                print("  [已取消]")
                raise RuntimeError(
                    f"用户取消了节点 [{node_name}] 的 LLM 调用"
                )
            else:
                skip_hint = "、s（跳过）" if allow_skip else ""
                print(
                    "  无效选项，请输入 y/Enter（确认）、"
                    "a（自动确认）、p（Pro）、f（Flash）、"
                    f"w（网页粘贴）{skip_hint}、c（取消）"
                )

    def _append_prompt(self, node_name: str, model: str, prompt: str) -> None:
        """将 Prompt 追加到日志文件。使用 heading 级别调整替代代码块包裹。"""
        ts = datetime.now().strftime("%H:%M:%S")
        # 将 Prompt 内容的标题级别调整到 #### 起步（日志中 ### 是 Prompt 标题）
        adjusted_prompt = adjust_heading_levels(prompt, 4)
        block = (
            f"\n## 调用 #{self._call_count} — `{node_name}` [{ts}]\n\n"
            f"**模型**: `{model}`  \n"
            f"**Prompt 长度**: {len(prompt):,} 字符\n\n"
            f"### Prompt\n\n"
            f"{adjusted_prompt}\n\n"
            f"### 响应\n\n"
        )
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(block)

    def _append_response(self, response: str) -> None:
        block = f"{response}\n\n---\n"
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(block)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  模块级当前会话管理（单进程单线程，工作流运行时设置）                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_current_session: PromptLogger | None = None


def get_current_session() -> PromptLogger | None:
    """获取当前工作流会话的 PromptLogger 实例（若无则返回 None）。"""
    return _current_session


def set_current_session(session: PromptLogger | None) -> None:
    """设置当前工作流会话（在工作流开始时调用）。"""
    global _current_session
    _current_session = session
