"""
Compact_LLM_Conversation.py — 自定义的对话历史压缩工具（compact_lyh 技能的执行脚本）

══════════════════════════════════════════════════════════════════════════════
  设计原则（与 Claude 官方 /compact 的最大区别）：

    用户的 prompt 永不压缩、原文保留，并在输出文件中反复强调（首尾共两遍）；
    助手给用户的结论性文字逐字保留；工具调用与工具输出是过程记录，按全局
    字数预算程序化分摊，超预算的部分摘除后由模型提炼成"关键发现"。
    理由：用户输入与助手结论是不可再生的信息，工具输出总是可以重新获取。

  输入（自动识别两种格式）：
    1. Claude Code 会话 transcript（.jsonl，位于 ~/.claude/projects/<项目>/）
    2. 简单 JSON：[{"role": "user"|"assistant", "content": "..."}, ...]
       （content 也可以是 Claude API 风格的内容块列表；也接受
         {"messages": [...]} 包一层的形式）

  输出：一个 JSON 文件，结构为
    _说明      —— 告诉下一个语言模型如何阅读本文件
    用户prompt —— 全部用户输入原文
    对话历史   —— 用户输入原文 + 逐轮压缩后的助手输出交替排列
    当前状态   —— 压缩时刻的进行中工作 / 待办事项 / 建议下一步
    _元数据    —— 来源、模型、统计、预算执行情况、外置文件清单

  压缩策略（2026-08-02 起为"全局预算"方案，取代按单条阈值的分层方案）：

    • 整份输出文件受一个总预算约束（--total-budget-chars，默认 50,000 字符），
      预算的分配全部是程序化的算术，模型只做语义压缩。分配顺序：

      ① 先扣除"保护内容"——用户 prompt（首尾两遍）、助手文字块（逐字原位
         保留；单回合总量超过 --assistant-text-chars 时最早的块先削减、最后
         一块永远完整）、用户对 AskUserQuestion 的回答、框架（_说明 / 当前
         状态 / _元数据）、以及保留原文的回合（最后 --keep-last-turns 个
         回合与短于 --short-turn-chars 的回合）。
         **保护内容永不为凑预算而被砍**：即使保护内容本身已超预算，也只把
         可压缩材料降到最低配置并如实记录，绝不削减保护内容。
      ② 再扣除每条工具调用的强制"痕迹行"（工具名 + 地址类字段的一行式
         摘要，必留——读者必须能知道每次调用在做什么）与每个含工具输出的
         回合预留的提炼小节配额（--turn-cap-chars）。
      ③ 剩余预算按各条工具材料（调用输入的完整参数、工具输出正文）的
         原始体积做水位分摊：配额够放全文就放全文，不够的从中间挖空或
         降级为痕迹行 / 占位行。
      ④ **不凑满**：全部材料本来就装得下时逐字保留、零削减、零模型调用，
         不为填满预算多留任何东西。

    • 输出文件里的内容块用 ASCII 的 XML 式短标签标注（`<say>` / `<call …/>` /
      `<out>` / `<out cut>` / `<err/>` / `<findings from=N>` 等，全部含义
      在输出文件的 `_说明` 字段里逐条写明）。这些标记每份存档要重复上千次，
      所以：标签一律用 ASCII（中文标记「[工具输出从略] 原文 3,713 字符」要
      14 个 token）；工具材料的标签**不记录字符数**（这类数字对续接工作没有
      帮助）；被摘除的正常工具输出**连痕迹行都不留**（调用痕迹已在紧挨着的
      `<call …/>` 里，约定"调用之后没有 `<out>` 就是输出被摘除"），只有报错
      `<err/>` 与确实为空的 `<out empty/>` 保留标记，因为这两者是信号；
    • **同一回合内渲染完全相同的非保护块只保留一条**，标注 ` xN` 表示重复了
      N 次（`_collapse_dups`）。连续几十次 Edit 改同一个文件、几十条被摘除
      的输出，从第二条起读者得不到任何新信息。被合并掉的块挂在代表条的
      `dups` 上而不是丢弃——它们的输出原文仍要送进该回合的提炼，否则"两次
      读同一个文件、内容不同"这类信息会凭空消失；
    • 每个有材料被摘除的回合，被摘除的输出合并成**一次**模型提炼（「关键
      发现」小节，上限 --turn-cap-chars，必须点名每个报错的实质；设为 0
      则不提炼、直接丢弃）；
    • 超长用户输入（> --long-prompt-chars 字符）用**一次**结构化调用完成
      "指令 / 粘贴资料"的分离与压缩：模型返回 JSON，逐块给出锚点、处理
      方式与压缩结果；"外置"块由程序按锚点从原文逐字抄写成旁路文件（不
      信任模型复述原文），"压缩"块直接取模型给出的压缩结果；
    • 思考（thinking）块一律丢弃；
    • 模型调用只发生在三处：每个超预算回合至多一次提炼、每条超长用户输入
      恰好一次、「当前状态」块一次——结构上不超过"轮次数 × 2 + 1"，
      天然低于"轮次数三倍"的硬上限（脚本内有断言兜底）。

  运行方式（两种，二选一）：

    A. 位置参数（简单场景）：
       python .../Compact_LLM_Conversation.py <对话文件.jsonl|.json> \
              [--output 输出.json] [--dry-run]

    B. jobfile 模式（推荐，用于外部启动器；避免 shell 转义问题）：
       python .../Compact_LLM_Conversation.py --jobfile <spec.json>
       spec.json 结构（字段名用下划线；未列字段沿用默认值；命令行同名参数
       仍会覆盖 jobfile）：
       {
         "input":  "<对话文件绝对路径>",
         "output": "<输出 JSON 绝对路径>",
         "model":  "claude-sonnet-4-6",
         "total_budget_chars":     50000,
         "keep_last_turns":        1,
         "short_turn_chars":       1000,
         "turn_cap_chars":         1000,
         "long_prompt_chars":      4000,
         "assistant_text_chars":   20000,
         "no_cache":               false,
         "dry_run":                false
       }

  运行结束后会打印固定的交接提示（要求终止当前会话、把输出 JSON 提供给
  新会话的语言模型），并在输出文件旁写一个 <输出>.done 哨兵文件，方便
  外部轮询等待。
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# 允许直接以文件路径运行本脚本（不要求先安装包）：把 src/ 加进 sys.path
try:
    from LLM_Lib.LLM import call_claude, extract_json
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from LLM_Lib.LLM import call_claude, extract_json


# ── 默认参数 ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-6"

TOTAL_BUDGET_CHARS = 50_000       # 整份输出文件的目标上限（保护内容超出时接受超出）
SHORT_TURN_KEEP_CHARS = 1000      # 助手回合渲染后短于此值 → 原文保留，零处理
TOOL_DIGEST_CAP_CHARS = 1000      # 每回合被摘除材料 LLM 提炼的目标上限（0 = 不提炼直接丢弃）
LONG_PROMPT_SPLIT_CHARS = 4000    # 用户输入长于此值 → 触发"指令/粘贴资料"分离
PASTED_BLOCK_COMPRESS_CAP = 1500  # "压缩"处理的粘贴资料块的目标上限（字符）
KEEP_LAST_TURNS = 1               # 最后 N 个助手回合保留原文

TOOL_TRACE_CAP = 200              # 工具调用痕迹行的长度上限（强制保留的一行式摘要）
TOOL_INPUT_FULL_CAP = 2000        # 工具调用输入"全文档位"的渲染上限（超出的大字段占位）
TOOL_INPUT_VALUE_KEEP = 500       # 全文档位里，非地址字段的值超过此长度 → 占位
TOOL_INPUT_ADDRESS_CAP = 4000     # 地址类字段值的保留上限（超过则中间挖空）
ERROR_RESULT_KEEP_CHARS = 2000    # 报错工具输出"全文档位"的保留上限
TOOL_RESULT_RENDER_CAP = 12_000   # 送入提炼前，单条长工具输出的截断上限
TURN_RENDER_CAP = 80_000          # 单个助手回合渲染 / 提炼输入的体积兜底上限
ASSISTANT_TEXT_CAP = 20_000       # 每回合助手文字逐字保留的总上限（最后一块永远完整；≤0 = 不设上限）
RESULT_PARTIAL_MIN = 300          # 工具输出分到的配额低于此值 → 不做截取、直接占位
FRAME_OVERHEAD_EST = 2200         # _说明 + 当前状态 + _元数据 + JSON 结构符的体积估计
STATE_TAIL_CHARS = 30_000         # 生成"当前状态"块时回看的历史末尾长度

# 工具输入里的"地址类"字段：标识"对哪个文件 / 跑什么命令 / 查什么模式"，
# 痕迹行由它们构成；全文档位里即使整条输入超长也要完整保留其值
ADDRESS_KEYS = frozenset({
    "file_path", "notebook_path", "path", "command", "description",
    "pattern", "glob", "type", "url", "query", "name", "skill", "args",
    "hpc_name", "remote_path", "local_path", "input", "output",
})

# transcript 中记在 user 角色名下、但并非用户亲手输入的内容块前缀
_META_PREFIXES = (
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "Caveat: The messages below",
    "<system-reminder>",
    "<ide_",                      # <ide_opened_file> / <ide_selection> 等
    "<task-notification>",
    "[SYSTEM NOTIFICATION",
    "[Request interrupted",
)
_SYSREM_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)

# 斜杠命令消息的两种起始标签（不同 Claude Code 版本顺序不同）
_SLASH_CMD_PREFIXES = ("<command-name>", "<command-message>")
# 触发本次压缩的斜杠命令——这一轮记录的是压缩动作自身，不是对话内容
_SELF_COMPACT_RE = re.compile(r"compact[_-]?lyh|/compact\b", re.I)


# ── 数据模型 ──────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """一轮对话：一条用户输入 + 其后直到下一条用户输入之间的全部助手内容。"""
    user_text: str | None = None        # 用户亲手输入的原文；None = 无
    marker: str | None = None           # 一行式记录（如斜杠命令），不进 prompt 列表
    payload: list[dict] = field(default_factory=list)  # 助手回合内容块


@dataclass
class Item:
    """预算分摊的最小单位：一个已渲染成两档（最低 / 全文）的内容块。

    protected 的块两档相同、全额保留；其余块默认取最低档（痕迹行 / 占位行），
    水位分摊到的配额够大时升到全文档，工具输出还可取中间档（截取）。
    """
    kind: str                 # "text" | "ask" | "tool_use" | "result"
    min_render: str
    full_render: str
    protected: bool = False
    raw: str = ""             # 提炼用的输出原文（result 专用）
    label: str = ""           # 提炼来源标注（result 专用）
    error: bool = False
    final: str = ""           # 分摊后的最终渲染
    dups: list = field(default_factory=list)  # 被合并到本条的重复 Item


# ── 输入解析 ──────────────────────────────────────────────────────────────────

def _clip(s: str, cap: int) -> str:
    if len(s) <= cap:
        return s
    return s[:cap] + "\n<clip/>"


def _render_tool_input(inp, keep: int = TOOL_INPUT_FULL_CAP) -> str:
    """渲染工具调用输入的"全文档位"。

    整条输入 JSON 不超过 keep → 全文保留；超过时保留地址类字段（路径、
    命令、模式等，见 ADDRESS_KEYS）的完整值，其余大段内容字段（如 Edit
    的 old_string / new_string、Write 的 content）换成 `<omit N>`
    占位——"对哪个文件、跑什么命令"完整可见，丢的只是随调用附带的大段
    内容，那些内容已经落在文件里，需要时重新读文件即可。
    """
    try:
        s = json.dumps(inp, ensure_ascii=False)
    except (TypeError, ValueError):
        return _squeeze(str(inp), keep)
    if len(s) <= keep:
        return s
    if not isinstance(inp, dict):
        return _squeeze(s, keep)
    slim = {}
    for k, v in inp.items():
        vs = v if isinstance(v, str) else json.dumps(
            v, ensure_ascii=False, default=str)
        if len(vs) > TOOL_INPUT_VALUE_KEEP and k not in ADDRESS_KEYS:
            slim[k] = "<omit>"
        elif isinstance(v, str) and k in ADDRESS_KEYS and len(v) > TOOL_INPUT_ADDRESS_CAP:
            slim[k] = _squeeze(v, TOOL_INPUT_ADDRESS_CAP)
        else:
            slim[k] = v
    s = json.dumps(slim, ensure_ascii=False)
    return s if len(s) <= keep * 2 else _squeeze(s, keep * 2)


def _trace_tool_use(name: str, inp) -> str:
    """工具调用的强制痕迹行：一行式摘要，读者由此知道这次调用在做什么。

    由地址类字段的值构成（路径 / 命令 / 模式 / 描述…），压到单行、
    总长不超过 TOOL_TRACE_CAP。
    """
    pieces: list[str] = []  # 返回 "工具名: 摘要"，不带外层的 <call …/> 标记
    if isinstance(inp, dict):
        for k in ("description", "command", "file_path", "notebook_path",
                  "path", "pattern", "glob", "url", "query", "skill",
                  "name", "hpc_name", "remote_path", "local_path",
                  "input", "output", "args", "type"):
            v = inp.get(k)
            if isinstance(v, str) and v.strip():
                pieces.append(v.strip())
    if not pieces and inp is not None:
        try:
            pieces.append(json.dumps(inp, ensure_ascii=False))
        except (TypeError, ValueError):
            pieces.append(str(inp))
    line = " | ".join(pieces)
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) > TOOL_TRACE_CAP:
        line = line[:TOOL_TRACE_CAP - 1] + "…"
    return f"{name}: {line}" if line else str(name)


def _render_tool_result(block: dict) -> str:
    c = block.get("content")
    if isinstance(c, list):
        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return _clip(str(c or "").strip(), TOOL_RESULT_RENDER_CAP)


def _clean_user_texts(texts: list[str]) -> str:
    """从用户消息的文本块中剔除 IDE 通知、system-reminder 等非用户输入的部分。"""
    clean = []
    for t in texts:
        s = _SYSREM_RE.sub("", t).strip()
        if not s or any(s.startswith(p) for p in _META_PREFIXES):
            continue
        clean.append(s)
    return "\n\n".join(clean).strip()


def parse_claude_transcript(path: Path) -> list[Turn]:
    """解析 Claude Code 会话 transcript（.jsonl）。"""
    turns: list[Turn] = []
    cur: Turn | None = None
    tool_names: dict[str, str] = {}   # tool_use id → 工具名（供 tool_result 回查）
    tool_inputs: dict[str, object] = {}  # tool_use id → 输入（供结果标注来源调用）

    def ensure_turn() -> Turn:
        nonlocal cur
        if cur is None:
            cur = Turn()
            turns.append(cur)
        return cur

    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("type")
        if t not in ("user", "assistant") or d.get("isSidechain"):
            continue

        if t == "assistant":
            blocks = (d.get("message") or {}).get("content") or []
            if isinstance(blocks, str):
                blocks = [{"type": "text", "text": blocks}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text" and (b.get("text") or "").strip():
                    ensure_turn().payload.append(
                        {"kind": "text", "text": b["text"].strip()})
                elif bt == "tool_use":
                    if b.get("id"):
                        tool_names[b["id"]] = b.get("name", "?")
                        tool_inputs[b["id"]] = b.get("input")
                    # 原始输入对象原样保存，渲染（含大字段占位）推迟到输出阶段
                    ensure_turn().payload.append(
                        {"kind": "tool_use", "name": b.get("name", "?"),
                         "input": b.get("input")})
                # thinking 块一律丢弃
            continue

        # ── user 角色 ──
        if d.get("isMeta"):
            continue
        ct = (d.get("message") or {}).get("content")

        # 工具结果（记在 user 名下，实际属于助手回合）
        if isinstance(ct, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in ct):
            for b in ct:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    ensure_turn().payload.append(
                        {"kind": "tool_result", "text": _render_tool_result(b),
                         "error": bool(b.get("is_error")),
                         "tool": tool_names.get(b.get("tool_use_id", ""), ""),
                         "call_input": tool_inputs.get(b.get("tool_use_id", ""))})
            continue

        texts = ([ct] if isinstance(ct, str) else
                 [b.get("text", "") for b in ct
                  if isinstance(b, dict) and b.get("type") == "text"]
                 if isinstance(ct, list) else [])
        joined = "\n".join(texts)

        # 此前经历过官方 /compact 的会话：把官方摘要当作可压缩的助手内容
        if d.get("isCompactSummary"):
            ensure_turn().payload.append(
                {"kind": "text", "text": "[此前官方 /compact 的摘要]\n" + joined})
            continue

        # 斜杠命令：记一行标记，不算用户 prompt。两种标签都要认——不同版本的
        # Claude Code 会把 <command-message> 排在 <command-name> 前面，只认后者
        # 会让整条命令噪音混进"用户 prompt"里。
        if joined.lstrip().startswith(_SLASH_CMD_PREFIXES):
            name = re.search(r"<command-name>(.*?)</command-name>", joined, re.S)
            args = re.search(r"<command-args>(.*?)</command-args>", joined, re.S)
            msg = re.search(r"<command-message>(.*?)</command-message>", joined, re.S)
            marker = "[斜杠命令] " + " ".join(
                x.group(1).strip() for x in (name or msg, args)
                if x and x.group(1).strip())
            cur = Turn(marker=marker)
            turns.append(cur)
            continue

        text = _clean_user_texts(texts)
        if not text:
            continue
        cur = Turn(user_text=text)
        turns.append(cur)

    return turns


def parse_simple_json(data) -> list[Turn]:
    """解析简单 JSON 格式：[{"role": ..., "content": ...}, ...]。"""
    if isinstance(data, dict):
        data = data.get("messages", [])
    if not isinstance(data, list):
        raise ValueError("简单 JSON 格式应为消息列表或 {'messages': [...]}")

    turns: list[Turn] = []
    cur: Turn | None = None

    def ensure_turn() -> Turn:
        nonlocal cur
        if cur is None:
            cur = Turn()
            turns.append(cur)
        return cur

    for msg in data:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", msg.get("speaker", ""))).lower()
        ct = msg.get("content", msg.get("text", ""))
        blocks = ct if isinstance(ct, list) else [{"type": "text", "text": str(ct)}]

        if role in ("user", "human"):
            texts = [b.get("text", "") for b in blocks
                     if isinstance(b, dict) and b.get("type", "text") == "text"]
            text = "\n\n".join(x.strip() for x in texts if x.strip())
            if text:
                cur = Turn(user_text=text)
                turns.append(cur)
        elif role in ("assistant", "model", "ai"):
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type", "text")
                if bt == "text" and (b.get("text") or "").strip():
                    ensure_turn().payload.append(
                        {"kind": "text", "text": b["text"].strip()})
                elif bt == "tool_use":
                    ensure_turn().payload.append(
                        {"kind": "tool_use", "name": b.get("name", "?"),
                         "input": b.get("input")})
                elif bt == "tool_result":
                    ensure_turn().payload.append(
                        {"kind": "tool_result", "text": _render_tool_result(b),
                         "error": bool(b.get("is_error"))})
                # thinking 丢弃
    return turns


def load_turns(path: Path) -> list[Turn]:
    """自动识别输入格式并解析成 Turn 列表。"""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith(("[", "{")):
        try:
            data = json.loads(text)
            # 整个文件是一个 JSON 值 → 简单格式；transcript 是每行一个 JSON
            if not (isinstance(data, dict) and data.get("type")):
                return parse_simple_json(data)
        except json.JSONDecodeError:
            pass
    return parse_claude_transcript(path)


# ── 渲染与预算工具 ────────────────────────────────────────────────────────────

def _squeeze(s: str, cap: int) -> str:
    """把单块内容压到 cap 以内：保留头尾，从中间挖空。"""
    if len(s) <= cap:
        return s
    if cap < 120:
        return s[:max(cap, 0)]
    note = "\n…<gap>…\n"
    keep = cap - len(note)
    head = keep * 2 // 3
    return s[:head] + note + s[-(keep - head):]


def _water_fill(sizes: list[int], budget: int) -> list[int]:
    """把 budget 按"水位"分给各块：装得下的全额满足，省下的额度再由大块均分。

    比"每块一律 budget/N"更省——大量短块不会白占配额，长块能拿到更多。
    """
    quota = [0] * len(sizes)
    left = len(sizes)
    for i in sorted(range(len(sizes)), key=lambda k: sizes[k]):
        take = min(sizes[i], budget // left if left else 0)
        quota[i] = take
        budget -= take
        left -= 1
    return quota


def _fit_parts(parts: list[str], protected: list[bool], cap: int) -> str:
    """把已渲染的内容块拼成一段文本，体积超限时按块裁剪（兜底安全网）。

    受保护的块（助手文字、用户对提问的回答等）全额保留，剩余额度由其他
    块分摊、单块超额的从中间挖空——而不是把拼好的长串从尾部一刀切。尾部
    一刀切正是历史上"最终答复丢失"的成因：工具输出占满额度，收尾的总结
    被切在了线外。
    """
    sep = 2 * max(len(parts) - 1, 0)
    if sum(map(len, parts)) + sep <= cap:
        return "\n\n".join(parts)

    keep = [i for i, p in enumerate(protected) if p]
    free = [i for i, p in enumerate(protected) if not p]
    protected_len = sum(len(parts[i]) for i in keep)

    if protected_len + sep >= cap:
        # 极端情况：光是受保护内容就超额。丢掉其余全部块，保护内容保留
        # 末尾（结论总在最后），并留一行说明避免读者误以为对话就是这样短。
        merged = "\n\n".join(parts[i] for i in keep)
        return f"<dropped-blocks {len(free)}/>\n\n" + _squeeze(merged, cap)

    quota = _water_fill([len(parts[i]) for i in free], cap - protected_len - sep)
    parts = list(parts)
    for i, q in zip(free, quota):
        parts[i] = _squeeze(parts[i], q)
    return "\n\n".join(parts)


def render_payload(payload: list[dict], cap: int = TURN_RENDER_CAP) -> str:
    """把助手回合的内容块按"全文档位"渲染成一段带标记的纯文本。"""
    parts, flags = [], []
    for it in payload:
        if it["kind"] == "text":
            parts.append(f"<say>\n{it['text']}")
            flags.append(True)
        elif it["kind"] == "tool_use":
            parts.append(f"<call {it['name']}: "
                         f"{_render_tool_input(it['input'])} />")
            flags.append(False)
        elif it.get("tool") == "AskUserQuestion":
            parts.append("<user-reply>（用户对提问的回答，是用户指令，"
                         "要点必须完整保留）\n" + it["text"])
            flags.append(True)
        elif it.get("error"):
            parts.append(f"<err>\n{it['text']}")
            flags.append(False)
        else:
            parts.append(f"<out>\n{it['text']}" if it["text"] else "<out empty/>")
            flags.append(False)
    return _fit_parts(parts, flags, cap)


def _text_layout(texts: list[str], cap: int) -> list[int]:
    """给各助手文字块分配逐字保留额度。

    总量不超过 cap（≤0 视为不设上限）→ 全部全额；超过时最后一块（最终
    总结）无条件全额，其余从后往前分配剩余额度——也就是从最早的块开始
    削减。结论总在最后，越早的进度叙述越可牺牲。
    """
    if cap <= 0 or sum(map(len, texts)) <= cap:
        return [len(t) for t in texts]
    quotas = [0] * len(texts)
    quotas[-1] = len(texts[-1])
    budget = max(cap - quotas[-1], 0)
    for i in range(len(texts) - 2, -1, -1):
        take = min(len(texts[i]), budget)
        quotas[i] = take
        budget -= take
    return quotas


def build_items(payload: list[dict], assistant_text_cap: int) -> list[Item]:
    """把一个助手回合的内容块渲染成两档（最低 / 全文）的 Item 列表。"""
    texts = [it["text"] for it in payload if it["kind"] == "text"]
    quotas = _text_layout(texts, assistant_text_cap)
    items: list[Item] = []
    ti = 0
    for it in payload:
        if it["kind"] == "text":
            t, q = it["text"], quotas[ti]
            ti += 1
            if q >= len(t):
                r = f"<say>\n{t}"
            elif q < 200:
                r = f"<say drop={len(t)}/>"
            else:
                r = f"<say orig={len(t)} keep={q}>\n" + _squeeze(t, q)
            items.append(Item("text", r, r, protected=True))
            continue
        if it["kind"] == "tool_use":
            trace = f"<call {_trace_tool_use(it['name'], it['input'])} />"
            full = (f"<call {it['name']}: "
                    f"{_render_tool_input(it['input'])} />")
            items.append(Item("tool_use", trace, full))
            continue
        # tool_result
        txt = it["text"]
        if it.get("tool") == "AskUserQuestion":
            r = ("<user-reply>（用户对提问的回答，是用户指令，要点必须完整保留）\n"
                 + txt)
            items.append(Item("ask", r, r, protected=True))
            continue
        if it.get("call_input") is not None:
            label = _trace_tool_use(it.get("tool") or "?", it["call_input"])
        else:
            label = str(it.get("tool") or "?")
        if it.get("error"):
            full = f"<err>\n{_clip(txt, ERROR_RESULT_KEEP_CHARS)}"
            items.append(Item("result", "<err/>", full, raw=txt, label=label,
                              error=True))
        elif txt:
            # 被摘除的正常输出**不留任何痕迹行**：调用痕迹已经在紧挨着的
            # <call …/> 里，一条内容为零的 <out/> 只是重复宣告"这次调用有过
            # 输出"，一份存档里要重复上百次。约定改为：调用之后没有出现
            # <out>，就表示输出已被摘除、要点在该回合的 <findings> 里；
            # 报错仍保留 <err/>，空输出仍保留 <out empty/>，两者都是信号。
            full = f"<out>\n{txt}"
            items.append(Item("result", "", full, raw=txt, label=label))
        else:
            r = "<out empty/>"
            items.append(Item("result", r, r))
    return _collapse_dups(items)


def _mark_dup(render: str, n: int) -> str:
    """在首行标签的**标签名之后**插入重复次数标记 ` xN`。

    位置紧跟标签名（`<call x9 Edit: … />`）而不是放在行尾，读者扫一眼标签
    就知道这条记录代表多少次调用，不用读到长路径的末尾。
    """
    if not render:
        return render
    head, sep, rest = render.partition("\n")
    m = re.match(r"<\w+", head)
    if not m:
        return f"{head} x{n}{sep}{rest}"
    return f"{head[:m.end()]} x{n}{head[m.end():]}{sep}{rest}"


def _collapse_dups(items: list[Item]) -> list[Item]:
    """合并同一回合内渲染完全相同的非保护块，只留第一条并标注 ` xN`。

    典型场景：连续几十次 Edit 改同一个文件、几十条被摘除的工具输出——痕迹行
    逐条重复，读者从第二条起得不到任何新信息，却要为每条付一次 token。

    去重键是「最低档渲染 + 输出原文」两者。**输出原文必须进键**：去掉字符数
    之后，所有被摘除的输出都渲染成同一个 `<out/>`，只按渲染去重会把内容互不
    相同的输出并成一条，既误导读者，又使它们再也无法各自升档为全文。加上原文
    以后，只有真正一模一样的输出（例如同一个文件的多次 Edit 都回同一句成功
    提示）才会合并。

    被合并掉的块仍挂在代表条的 `dups` 上而不是直接丢弃，其输出原文照样送进
    该回合的提炼。
    """
    reps: dict[tuple[str, str], Item] = {}
    kept: list[Item] = []
    for it in items:
        if it.protected:
            kept.append(it)
            continue
        key = (it.min_render, it.raw)
        rep = reps.get(key)
        if rep is not None:
            rep.dups.append(it)
            continue
        reps[key] = it
        kept.append(it)
    for it in kept:
        if it.dups:
            n = len(it.dups) + 1
            it.min_render = _mark_dup(it.min_render, n)
            it.full_render = _mark_dup(it.full_render, n)
    return kept


# ── LLM 压缩 ──────────────────────────────────────────────────────────────────

PROMPT_COMPRESS_TOOL_OUTPUTS = """\
你在为一段人机对话做历史压缩。这个"助手回合"的结构与文字都已保留（助手的回复\
原文、每一次工具调用的痕迹行），只有其中体积较大的工具输出被摘除。下面给出这些\
被摘除的工具输出（每条都标注了产生它的工具调用）。请把它们提炼成"关键发现"\
要点，提炼结果将附在该回合末尾，供后续的语言模型续接工作时阅读。

必须保留：
- 输出中的关键事实、数字、文件内容要点；
- 每一条报错的实质（报什么错、表现如何）——被摘除的输出里有报错时一条都不能漏；
- 影响后续工作判断的发现。

可以丢弃：
- 与后续工作无关的细节、重复内容、格式噪音；
- 助手在回复原文里已经说过的结论（下方附有助手原话供对照，不要复述）。

输出要求：直接输出要点本身，不要任何前言、标题或解释；长度不超过约 {cap} 个字符；\
使用与原文一致的语言（原文以中文为主则用中文）；代码标识符、文件路径、命令原样保留。

【该回合对应的用户输入（仅供理解上下文，不要复述进结果）】
{user_text}

【助手在该回合对用户说的话（已逐字保留在存档里，不要复述）】
{assistant_text}

【被摘除的工具输出】
{outputs}
"""

PROMPT_SPLIT_LONG_USER = """\
下面是一条很长的用户消息。它可能由两类内容混合而成：
A. 用户自己撰写的指令 / 问题 / 说明（必须逐字保留）；
B. 用户粘贴进来的大段资料（如日志、报错输出、文档、代码文件、网页内容等）。

请识别出其中的"粘贴资料块"（B 类），并对"处理"为"压缩"的块直接给出压缩结果。\
只输出如下 JSON，不要输出其他任何内容：
{{
  "粘贴块": [
    {{
      "描述": "这段资料是什么（一句话）",
      "首锚": "该块开头的前 30 个字符，必须与原文完全逐字一致",
      "尾锚": "该块结尾的最后 30 个字符，必须与原文完全逐字一致",
      "处理": "外置" 或 "压缩",
      "压缩结果": "处理为「压缩」时必填：该块的要点记录（保留关键事实、数字、\
报错信息与结论，不超过约 {cap} 个字符）；处理为「外置」时填空字符串"
    }}
  ]
}}

"处理"方式的判断标准：这段资料后续可能需要逐字查阅（代码、配置、结构化数据）→ \
"外置"（程序会按锚点把原文逐字抄写成旁路文件，不需要你复述原文）；只需要其要点\
（长日志、报错堆栈、文章正文）→ "压缩"。
如果整条消息全部是用户撰写的指令、没有粘贴资料块，输出 {{"粘贴块": []}}。
锚点必须逐字精确复制原文（包括标点与空格），这是后续程序定位的依据。

【用户消息原文】
{user_text}
"""

PROMPT_CURRENT_STATE = """\
下面是一段人机对话历史的末尾部分（助手侧内容已被压缩转述）。请判断这段对话被压缩\
存档的时刻，工作进行到了哪一步。只输出如下 JSON，不要输出其他任何内容：
{{
  "当前工作": "压缩前正在做的具体事情",
  "待办事项": ["尚未完成的事项…"],
  "建议下一步": "紧接着应该做的一步；如果任务已经完成则填 \\"无\\""
}}

【对话历史末尾】
{tail}
"""


class Compactor:
    def __init__(self, args, out_path: Path):
        self.args = args
        self.out_path = out_path
        self.llm_calls = 0
        self.side_files: list[str] = []

    def _llm(self, prompt: str) -> str:
        self.llm_calls += 1
        t0 = time.time()
        resp = call_claude(prompt, model=self.args.model,
                           cache=not self.args.no_cache,
                           verbose=False, timeout=600)
        print(f"    模型调用 #{self.llm_calls} 完成（{time.time() - t0:.0f}s，"
              f"输入 {len(prompt)} 字符 → 输出 {len(resp)} 字符）")
        return resp

    # ── 超长用户输入：一次结构化调用完成分离与压缩 ──
    def process_user_prompt(self, text: str, turn_no: int) -> str:
        if len(text) <= self.args.long_prompt_chars:
            return text
        print(f"  轮次 {turn_no}：用户输入 {len(text)} 字符，超长，"
              f"分离指令与粘贴资料（一次结构化调用）…")
        resp = self._llm(PROMPT_SPLIT_LONG_USER.format(
            cap=PASTED_BLOCK_COMPRESS_CAP, user_text=text))
        parsed = extract_json(resp)
        blocks = (parsed or {}).get("粘贴块") if isinstance(parsed, dict) else None
        if not blocks:
            print("    未识别出粘贴资料块（或解析失败），整条原文保留。")
            return text

        # 先定位全部块，再从后往前替换，避免索引失效
        located = []
        pos = 0
        for b in blocks:
            head, tail = str(b.get("首锚", "")), str(b.get("尾锚", ""))
            if not head or not tail:
                continue
            start = text.find(head, pos)
            end = text.find(tail, start + len(head)) if start >= 0 else -1
            if start < 0 or end < 0:
                print(f"    ⚠️ 锚点定位失败（{b.get('描述', '?')}），该块原文保留。")
                continue
            end += len(tail)
            located.append((start, end, b))
            pos = end
        result = text
        for i, (start, end, b) in enumerate(reversed(located)):
            n = len(located) - i
            block_text = text[start:end]
            desc = str(b.get("描述", "粘贴资料"))
            comp = str(b.get("压缩结果", "")).strip()
            if b.get("处理") == "压缩" and comp:
                repl = (f"【粘贴资料已压缩（原文 {len(block_text)} 字符）："
                        f"{desc}】\n{comp}")
            else:
                # "外置"，或模型漏给压缩结果时的兜底：逐字抄写原文，不丢信息
                side = self.out_path.with_name(
                    f"{self.out_path.stem}_外置_{turn_no:02d}_{n}.txt")
                side.write_text(block_text, encoding="utf-8")
                self.side_files.append(str(side))
                repl = (f"【粘贴资料已逐字外置 → {side}：{desc}"
                        f"（{len(block_text)} 字符，需要时读取该文件）】")
            result = result[:start] + repl + result[end:]
            print(f"    资料块「{desc}」→ {b.get('处理', '外置')}")
        return result

    # ── 单回合提炼 ──
    def digest_turn(self, dropped: list[tuple[str, str]],
                    user_text: str | None, assistant_text: str,
                    turn_no: int) -> str:
        total = sum(len(t) for _, t in dropped)
        print(f"  轮次 {turn_no}：提炼 {len(dropped)} 条被摘除的工具输出"
              f"（共 {total:,} 字符）…")
        bodies = [_clip(t, TOOL_RESULT_RENDER_CAP) for _, t in dropped]
        quota = _water_fill([len(b) for b in bodies], TURN_RENDER_CAP)
        merged = "\n\n".join(f"[输出 {i + 1}｜来自 {label}]\n{_squeeze(b, q)}"
                             for i, ((label, _), b, q)
                             in enumerate(zip(dropped, bodies, quota)))
        resp = self._llm(PROMPT_COMPRESS_TOOL_OUTPUTS.format(
            cap=self.args.turn_cap_chars,
            user_text=_clip(user_text or "（无）", 3000),
            assistant_text=assistant_text[-3000:] or "（无）",
            outputs=merged))
        return resp.strip()

    # ── 当前状态块 ──
    def current_state(self, history: list[dict]) -> dict:
        tail_parts = []
        used = 0
        for entry in reversed(history):
            s = f"〔{entry['角色']}〕{entry['内容']}"
            used += len(s)
            tail_parts.append(s)
            if used > STATE_TAIL_CHARS:
                break
        tail = "\n\n".join(reversed(tail_parts))
        print("  生成「当前状态」块…")
        resp = self._llm(PROMPT_CURRENT_STATE.format(tail=tail))
        parsed = extract_json(resp)
        if isinstance(parsed, dict) and parsed.get("当前工作"):
            return parsed
        return {"当前工作": resp.strip() or "（生成失败）",
                "待办事项": [], "建议下一步": "无"}


# ── 主流程 ────────────────────────────────────────────────────────────────────

README_TEXT = (
    "本文件是上一段人机对话的压缩存档，由 Compact_LLM_Conversation.py 生成。"
    "『用户prompt』中的条目是用户输入的原文（标注了外置 / 已压缩的粘贴资料"
    "除外），未经任何改写，是最高优先级的指令来源，请充分、反复地遵循。"
    "『对话历史』中角色为『助手』的条目由若干内容块组成，每块以一个 XML 式"
    "标签开头，标签后到下一个标签之间是该块的正文。标签含义："
    "<say> 助手对用户说的原话，逐字未改写，可信度与用户 prompt 同级；"
    "<say drop=N/> 该段原话因回合内文字总量超限被略去，N 为原文字符数；"
    "<say orig=N keep=M> 同上但保留了 M 字符（正文中间用 <gap> 挖空）；"
    "<user-reply> 用户对 AskUserQuestion 的回答，属于用户指令；"
    "<call 工具名: 摘要 /> 一次工具调用的痕迹，摘要给出对哪个文件、跑什么命令、"
    "查什么模式，参数中的 <omit> 表示该参数的大段内容已从略（内容在对应"
    "文件里）；"
    "<out> 工具输出原文；<out cut> 该输出被截取、只保留了首尾；"
    "<out empty/> 该次调用确实没有任何输出；"
    "<err> 报错的工具输出原文，<err/> 为其被摘除的形式。"
    "**一次 <call …/> 之后如果既没有 <out…> 也没有 <err…>，表示该次调用正常"
    "返回、但输出受总预算约束被整条摘除**，要点见该回合的 <findings>；"
    "<clip/> 表示此处原文更长、已在该点截断；"
    "<dropped-blocks N/> 表示该回合另有 N 个内容块因体积超限全部略去；"
    "<findings from=N> 是模型对本回合被摘除输出的提炼，仅供了解经过、不是原文。"
    "任何标签里出现 xN（如 <call x9 Edit: 某文件 /> 或 <out x37/>）表示这条"
    "记录在本回合原样重复了 N 次、只保留了一条；展开的参数只属于第一次，"
    "重复各次的输出内容若有价值已并入该回合的 <findings>。"
    "除 <say …> 外的标签不记录字符数——这类数字对续接工作没有帮助，"
    "却要为每一条重复付出 token。"
    "凡标注被摘除或截取的地方，需要精确细节时应重新读取对应文件或重新执行命令。"
    "已压缩=true 表示该回合有内容被削减或摘除，false 的回合为完整原文。"
    "『当前状态』给出了压缩时刻正在进行的工作与待办事项，请从这里续接。"
)

FINAL_MESSAGE = ("此前对话已使用 Compact_LLM_Conversation.py 压缩至文件中，"
                 "请终止此会话，将这个json文件提供给语言模型：\n`{path}`")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="把 LLM 对话历史压缩成 JSON：用户 prompt 与助手文字原文保留，"
                    "工具调用痕迹必留，工具材料按全局预算程序化分摊，超预算部分"
                    "摘除后由模型提炼。")
    # 显式默认值集中在这里，方便下面的 jobfile 合并判断"命令行未显式提供"
    defaults = {
        "input": None,
        "output": None,
        "model": DEFAULT_MODEL,
        "total_budget_chars": TOTAL_BUDGET_CHARS,
        "keep_last_turns": KEEP_LAST_TURNS,
        "short_turn_chars": SHORT_TURN_KEEP_CHARS,
        "turn_cap_chars": TOOL_DIGEST_CAP_CHARS,
        "long_prompt_chars": LONG_PROMPT_SPLIT_CHARS,
        "assistant_text_chars": ASSISTANT_TEXT_CAP,
        "no_cache": False,
        "dry_run": False,
    }
    ap.add_argument("input", nargs="?", default=defaults["input"],
                    help="对话文件：Claude Code transcript(.jsonl) 或简单 JSON "
                         "消息列表。给了 --jobfile 时可省略；两处都给以命令行为准。")
    ap.add_argument("--jobfile", help="JSON 描述文件，字段 input/output/model/... "
                                      "见文件顶部说明。推荐外部启动器使用此模式，"
                                      "避免 shell 转义与多参数续行问题。")
    ap.add_argument("--output", default=defaults["output"],
                    help="输出 JSON 路径（默认：输入文件旁 <名称>_compacted.json）")
    ap.add_argument("--model", default=defaults["model"],
                    help=f"压缩用模型（默认 {defaults['model']}）")
    ap.add_argument("--total-budget-chars", type=int,
                    default=defaults["total_budget_chars"],
                    help="整份输出文件的目标上限；保护内容（用户原文、助手文字、"
                         "痕迹行等）超出时接受超出、绝不削减保护内容；材料装得下"
                         f"时也不凑满（默认 {defaults['total_budget_chars']}）")
    ap.add_argument("--keep-last-turns", type=int, default=defaults["keep_last_turns"],
                    help=f"最后 N 个助手回合保留原文（默认 {defaults['keep_last_turns']}）")
    ap.add_argument("--short-turn-chars", type=int, default=defaults["short_turn_chars"],
                    help=f"短于此字符数的助手回合保留原文（默认 {defaults['short_turn_chars']}）")
    ap.add_argument("--turn-cap-chars", type=int, default=defaults["turn_cap_chars"],
                    help="每回合被摘除材料 LLM 提炼的目标上限"
                         f"（默认 {defaults['turn_cap_chars']}，0 = 不提炼直接丢弃）")
    ap.add_argument("--long-prompt-chars", type=int, default=defaults["long_prompt_chars"],
                    help=f"用户输入超过此长度时分离粘贴资料（默认 {defaults['long_prompt_chars']}）")
    ap.add_argument("--assistant-text-chars", type=int,
                    default=defaults["assistant_text_chars"],
                    help="每回合助手文字块逐字保留的总上限，超限时从最早的块开始"
                         "削减、最后一块（总结）永远完整"
                         f"（默认 {defaults['assistant_text_chars']}，≤0 = 不设上限）")
    ap.add_argument("--no-cache", action="store_true", help="禁用 LLM 结果缓存")
    ap.add_argument("--dry-run", action="store_true",
                    help="只解析并打印预算分摊计划，不调用模型、不写输出")
    args = ap.parse_args()

    # 合并 jobfile：只填命令行未显式提供（仍等于默认值）的字段
    if args.jobfile:
        job_path = Path(args.jobfile).expanduser().resolve()
        if not job_path.is_file():
            print(f"❌ jobfile 不存在：{job_path}")
            return 2
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"❌ jobfile 读取/解析失败：{exc}")
            return 2
        if not isinstance(job, dict):
            print(f"❌ jobfile 顶层必须是 JSON 对象，得到 {type(job).__name__}")
            return 2
        unknown = set(job) - set(defaults)
        if unknown:
            print(f"❌ jobfile 含未知字段：{sorted(unknown)}；"
                  f"允许的字段：{sorted(defaults)}")
            return 2
        for key, default in defaults.items():
            if key in job and getattr(args, key) == default:
                setattr(args, key, job[key])

    if not args.input:
        print("❌ 必须提供 input（位置参数或 jobfile 的 input 字段）")
        return 2

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.is_file():
        print(f"❌ 输入文件不存在：{in_path}")
        return 1
    out_path = (Path(args.output).expanduser().resolve() if args.output
                else in_path.with_name(in_path.stem + "_compacted.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"▶ 解析对话文件：{in_path}")
    turns = [t for t in load_turns(in_path)
             if t.user_text or t.marker or t.payload]

    # 压缩"当前会话"时，最后一轮往往就是发起本次压缩的斜杠命令；它记录的是
    # 压缩动作自身（读 skill、写 jobfile、跑 dry-run），既无保留价值，又会
    # 顶掉 --keep-last-turns 的名额，害得真正干活的那一轮反被压缩。
    if turns and turns[-1].marker and _SELF_COMPACT_RE.search(turns[-1].marker):
        print(f"  末轮是本次压缩的触发命令（{turns[-1].marker}），整轮丢弃。")
        turns.pop()

    n_prompts = sum(1 for t in turns if t.user_text)
    total_in = sum(len(t.user_text or "") + len(render_payload(t.payload))
                   for t in turns)
    print(f"  共 {len(turns)} 轮，其中用户 prompt {n_prompts} 条，"
          f"总量约 {total_in:,} 字符，总预算 {args.total_budget_chars:,} 字符。")

    comp = Compactor(args, out_path)

    # ── 第一步：逐回合渲染成两档 Item，划定"原文保留"的回合 ──
    with_payload = [j for j, t in enumerate(turns) if t.payload]
    n_keep = max(args.keep_last_turns, 0)
    verbatim_idx = set(with_payload[len(with_payload) - n_keep:]) if n_keep else set()

    @dataclass(eq=False)   # 身份比较——Plan 只用于 in / 集合判断，不比内容
    class Plan:
        turn_no: int
        turn: Turn
        verbatim_text: str | None = None      # 原文保留的回合：完整渲染
        items: list[Item] = field(default_factory=list)  # 参与预算分摊的回合
        priority: bool = False                # 最后 N 个回合：分摊时优先满足
        processed_user: str | None = None     # 分离粘贴资料后的用户输入

    plans: list[Plan] = []
    for i, t in enumerate(turns):
        if not t.payload:
            plans.append(Plan(i + 1, t))
            continue
        rendered = render_payload(t.payload)
        if len(rendered) <= args.short_turn_chars:
            plans.append(Plan(i + 1, t, verbatim_text=rendered))
        else:
            # 最后 N 个回合不再无条件原文保留（单个大回合会独吞全部预算），
            # 而是参与分摊但享有最高优先级：剩余预算先满足它们。
            plans.append(Plan(i + 1, t,
                              items=build_items(t.payload,
                                                args.assistant_text_chars),
                              priority=i in verbatim_idx))

    # ── 第二步：处理超长用户输入（每条一次结构化调用；先于预算核算，
    #    否则待外置的粘贴资料会按原始长度白白吃掉预算）──
    for p in plans:
        if p.turn.user_text:
            if args.dry_run:
                p.processed_user = p.turn.user_text   # dry-run 不调模型，用原文估算
            else:
                p.processed_user = comp.process_user_prompt(
                    p.turn.user_text, p.turn_no)

    # ── 第三步：全局预算核算（纯算术，零模型调用） ──
    # 保护成本：用户原文两遍（prompt 区 + 历史区）、框架、原文保留的回合、
    # 分摊回合里的保护块与全部最低档；提炼预留：每个含可摘除输出的回合一份。
    prompt_cost = 2 * sum(len(p.processed_user or "") for p in plans)
    marker_cost = sum(len(t.marker or "") for t in turns)
    verbatim_cost = sum(len(p.verbatim_text or "") for p in plans)
    base_cost = sum(len(it.min_render) + 2
                    for p in plans for it in p.items)
    digest_turns = [p for p in plans
                    if any((it.kind == "result"
                            and it.full_render != it.min_render)
                           or any(d.kind == "result" and d.raw
                                  for d in it.dups)
                           for it in p.items)]
    digest_reserve = (args.turn_cap_chars + 100) * len(digest_turns) \
        if args.turn_cap_chars > 0 else 0

    protected_total = (FRAME_OVERHEAD_EST + prompt_cost + marker_cost
                       + verbatim_cost + base_cost)
    extra = args.total_budget_chars - protected_total - digest_reserve
    budget_saturated = extra <= 0
    extra = max(extra, 0)

    # 可升档的材料（全档 > 最低档的非保护块），两级水位分摊：
    # 最后 N 个回合（priority）先分，剩下的额度再给更早的回合。
    def _upgradable(pp):
        return [it for p in pp for it in p.items
                if not it.protected and len(it.full_render) > len(it.min_render)]

    upg_prio = _upgradable([p for p in plans if p.priority])
    upg_rest = _upgradable([p for p in plans if not p.priority])
    deltas_prio = [len(it.full_render) - len(it.min_render) for it in upg_prio]
    deltas_rest = [len(it.full_render) - len(it.min_render) for it in upg_rest]
    quotas_prio = _water_fill(deltas_prio, extra)
    quotas_rest = _water_fill(deltas_rest, extra - sum(quotas_prio))
    upgradable = upg_prio + upg_rest
    deltas = deltas_prio + deltas_rest
    quotas = quotas_prio + quotas_rest

    n_full = n_partial = n_min = 0
    for it, q, d in zip(upgradable, quotas, deltas):
        if q >= d:
            it.final = it.full_render
            n_full += 1
        elif (it.kind == "result" and not it.error
              and q >= RESULT_PARTIAL_MIN):
            body = _squeeze(it.raw, len(it.min_render) + q)
            it.final = f"<out cut>\n{body}"
            n_partial += 1
        else:
            it.final = it.min_render
            n_min += 1
    for p in plans:
        for it in p.items:
            if not it.final:
                it.final = it.min_render if it.protected is False else it.full_render
            if it.protected:
                it.final = it.full_render

    print(f"  预算核算：保护内容与最低配置 {protected_total:,} 字符 + "
          f"提炼预留 {digest_reserve:,} 字符，剩余 {extra:,} 字符可分摊；"
          f"材料 {len(upgradable)} 条 → 全文 {n_full}、截取 {n_partial}、"
          f"占位 {n_min}。")
    if budget_saturated:
        print("  ⚠️ 预算已被保护内容占满：可压缩材料全部降为最低配置，"
              "保护内容不削减、接受总量超出。")

    if args.dry_run:
        for p in plans:
            t = p.turn
            u = (t.marker or (t.user_text or "").replace("\n", " ")[:60]
                 or "（无用户输入）")
            if p.verbatim_text is not None:
                print(f"  轮次 {p.turn_no:3d}: 原文保留 "
                      f"{len(p.verbatim_text):>7,} 字符 | {u}")
            elif p.items:
                mn = sum(len(it.min_render) for it in p.items)
                fl = sum(len(it.full_render) for it in p.items)
                dg = "，计划提炼 1 次" if p in digest_turns else ""
                print(f"  轮次 {p.turn_no:3d}: 最低 {mn:>7,} / 全文 {fl:>7,}"
                      f" 字符{dg} | {u}")
            else:
                print(f"  轮次 {p.turn_no:3d}: （无助手内容） | {u}")
        n_long_prompts = sum(1 for t in turns
                             if t.user_text
                             and len(t.user_text) > args.long_prompt_chars)
        print(f"（dry-run 结束，未调用模型。正式运行预计调用："
              f"提炼 {len(digest_turns)} + 超长用户输入 {n_long_prompts} + "
              f"当前状态 1 = {len(digest_turns) + n_long_prompts + 1} 次，"
              f"上限为轮次数三倍 = {3 * len(turns)} 次。）")
        return 0

    # ── 第三步：组装输出（含全部模型调用） ──
    prompts: list[dict] = []
    history: list[dict] = []
    n_compressed = 0
    for p in plans:
        t = p.turn
        if t.marker:
            history.append({"轮次": p.turn_no, "角色": "用户", "内容": t.marker})
        elif t.user_text:
            processed = p.processed_user or t.user_text
            prompts.append({"轮次": p.turn_no, "内容": processed})
            history.append({"轮次": p.turn_no, "角色": "用户", "内容": processed})
        if p.verbatim_text is not None:
            print(f"  轮次 {p.turn_no}：助手回合 {len(p.verbatim_text):,} 字符，"
                  "保留原文。")
            history.append({"轮次": p.turn_no, "角色": "助手",
                            "内容": p.verbatim_text, "已压缩": False})
            continue
        if not p.items:
            continue
        reduced = any(it.final != it.full_render or it.dups for it in p.items)
        # final 为空串的块（被摘除的正常工具输出）整条不落地
        parts = [it.final for it in p.items if it.final]
        flags = [it.protected for it in p.items if it.final]
        # 被合并掉的重复块（it.dups）一律送去提炼——它们的渲染虽然与代表条
        # 相同，输出原文却可能完全不同（例如两次读同一个文件）。
        dropped = [(it.label, it.raw) for it in p.items
                   if it.kind == "result" and it.raw
                   and it.final != it.full_render]
        dropped += [(d.label, d.raw) for it in p.items for d in it.dups
                    if d.kind == "result" and d.raw]
        if dropped and args.turn_cap_chars > 0:
            assistant_text = "\n\n".join(
                it["text"] for it in t.payload if it["kind"] == "text")
            digest = comp.digest_turn(dropped, t.user_text, assistant_text,
                                      p.turn_no)
            if digest:
                parts.append(f"<findings from={len(dropped)}>"
                             "（以下是模型对本回合被摘除的工具输出的提炼，"
                             "非原文）\n" + digest)
                flags.append(True)
        elif dropped:
            print(f"  轮次 {p.turn_no}：{len(dropped)} 条被摘除的输出已按设置"
                  "直接丢弃（turn_cap_chars=0）。")
        out = _fit_parts(parts, flags, TURN_RENDER_CAP)
        n_compressed += reduced
        state_word = "分摊后有削减" if reduced else "全部装入预算，逐字保留"
        print(f"  轮次 {p.turn_no}：{state_word}（{len(out):,} 字符）。")
        history.append({"轮次": p.turn_no, "角色": "助手",
                        "内容": out, "已压缩": reduced})

    state = comp.current_state(history)

    # 调用次数硬约束：结构上 ≤ 提炼(≤轮次) + 超长输入(≤轮次) + 状态 1
    call_limit = 3 * max(len(turns), 1)
    assert comp.llm_calls <= call_limit, (
        f"模型调用 {comp.llm_calls} 次，超过轮次数三倍上限 {call_limit}")

    result = {
        "_说明": README_TEXT,
        "用户prompt": prompts,
        "对话历史": history,
        "当前状态": state,
        "_元数据": {
            "来源文件": str(in_path),
            "压缩模型": args.model,
            "压缩时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "轮次数": len(turns),
            "用户prompt条数": n_prompts,
            "压缩的助手回合数": n_compressed,
            "模型调用次数": comp.llm_calls,
            "总预算字符数": args.total_budget_chars,
            "预算已被保护内容占满": budget_saturated,
            "原始总字符数": total_in,
            "外置文件": comp.side_files,
        },
    }
    out_json = json.dumps(result, ensure_ascii=False, indent=2)
    result["_元数据"]["压缩后总字符数"] = len(out_json)
    out_json = json.dumps(result, ensure_ascii=False, indent=2)
    out_path.write_text(out_json, encoding="utf-8")

    print(f"\n✅ 完成：{n_prompts} 条用户 prompt 原文保留，"
          f"{n_compressed} 个助手回合被压缩，共调用模型 {comp.llm_calls} 次"
          f"（上限 {call_limit}）。")
    print(f"   原始约 {total_in:,} 字符 → 输出 {len(out_json):,} 字符"
          f"（预算 {args.total_budget_chars:,}）。")
    if comp.side_files:
        print("   外置文件：")
        for f in comp.side_files:
            print(f"     {f}")

    final_message = FINAL_MESSAGE.format(path=out_path)
    out_path.with_suffix(out_path.suffix + ".done").write_text(
        final_message, encoding="utf-8")
    print("\n" + final_message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
