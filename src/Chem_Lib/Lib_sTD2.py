# -*- coding: utf-8 -*-
"""
Lib_sTD2 — std2 简化 TD-DFT 激发态计算的输入生成、运行与结果解析
==================================================================

std2（stda 程序的更名升级版，Grimme 组）用简化 TD-DFT 家族方法计算
激发态与响应性质：sTDA、sTD-DFT（``-rpa``）、精确积分的 XsTDA /
XsTD-DFT（``-XsTD``，2.0.0 新增，经 libcint 积分库原生支持若干
range-separated hybrid 泛函），以及基于 xtb4stda 基态波函数的
sTDA-xTB / sTD-DFT-xTB。手册的拆分树在
``Chem_Lib_Manuals/std2_20260816/``（入口 ``_INDEX.md``），全文备份与
转换说明见同目录的 ``Manual_std2_README.md``。

输入：**Molden 文件**（必须是 Cartesian GTO 基组；TURBOMOLE / MOLPRO /
TERACHEM 直接生成，Gaussian 经随附的 ``g2molden`` 工具从输出文件转换，
Q-CHEM 经 ``qc2molden.sh``），或 xtb4stda 写出的二进制波函数
``wfn.xtb``（sTDA-xTB 方案，手册第 7 节）。Gaussian 侧的路由要求
（手册第 5 节）：``#P``、``gfinput``、``pop=full``（或 ``IOp(6/7=3)``）、
``6D 10F``（强制笛卡尔基函数）——:func:`std2_input_from_Gaussian_output`
会在转换前检查这些关键词。

产物：标准输出日志（本模块存成 ``<名>.std2.log``）与谱数据文件
``tda.dat``（每个态一行：能量 eV 与长度 / 速度表象的振子强度、旋转
强度；本模块另存稳定名副本 ``<名>.std2_tda.dat``）。随附的绘谱工具
``g_spec``（手册第 4 节；PDF 排版把名字印成了 ``g spec``）从
``tda.dat`` 生成展宽谱 ``spec.dat`` 与棒状谱 ``rots.dat``，其选项写在
``tda.dat`` 的头部（见 :func:`set_tda_dat_g_spec_options`）。

可执行文件（``std2`` / ``g_spec`` / ``g2molden``）随本包分发，位于与
本文件同目录的 ``Executable_sTD2/Linux/bin/``——**只有 Linux 版**
（v2.0.1 源码编译的全静态可移植二进制，静态
MKL、零 glibc 版本依赖；无 Windows 构建）。因此本模块所有需要启动
二进制的函数只能在 Linux（含集群）上运行，Windows 上直接报错
（fail early）；Windows 上仍可使用命令行构造（脚本模式）与结果解析。
文件夹不进版本控制（.gitignore 的 ``Executable_*`` 模式排除），由
``A0_HPC_Sync_Python_Lib.py`` 同步上集群，pip / uv 安装经 pyproject 的
package-data 带入；一律相对于 Chem_Lib 包定位，不写死绝对路径。

**xtb4stda（sTDA-xTB / sTD-DFT-xTB 的基态程序）也随本包分发**
（2026-08-16 起，官方 GitHub v1.0 发行版的静态 Linux 二进制）：
``Executable_sTD2/Linux/bin/xtb4stda`` + 参数目录
``Executable_sTD2/xtb4stda_home/``（``.param_stda1.xtb`` 与
``.param_stda2.xtb`` 为 2018 扩展参数集、覆盖到 Z = 85 含 Se；
``.param_gbsa_*`` 为 GBSA 隐式溶剂参数，含氯仿）。运行时经环境变量
``XTB4STDAHOME`` 指向该参数目录（程序会自动补尾部斜杠；不设置时它会
回落到 ``~/``——本模块因此**强制**设置，绝不依赖回落）。

**80 字符路径陷阱**（集群实测）：xtb4stda 源码里
路径一律是定长 ``character*80``——``XTB4STDAHOME`` 与拼出的参数文件
全路径超过 80 字符就被**静默截断**，报「parameter file ... not
found」。随包参数目录的真实路径远超此限，因此本模块一律经家目录下的
短符号链接 ``~/.xtb4stda_home`` 使用参数目录（运行时自动创建 / 刷新，
见 :func:`ensure_short_xtb4stda_home`；作业脚本里由
``ln -sfn`` 行完成同样的事）。调用形式
``xtb4stda <几何.xyz> [-gbsa <溶剂>]``，电荷与未配对电子数从工作目录的
``.CHRG`` / ``.UHF`` 文件读取（同 xTB 家族约定），产物是二进制波函数
文件 ``wfn.xtb``，交给 ``std2 -xtb`` 做激发态。完整复合链见
:func:`run_sTD_DFT_xTB_folder`（本地）与
:func:`build_sTD_DFT_xTB_command_lines`（HPC 脚本模式）。

三个已知限制：

1. **XsTD 方法不能用 xTB 基态**（手册第 3.2 / 7.2 节），方法名的划分
   本身已排除这种组合。
2. std2 的命令行解析用**子串匹配**（源码 ``main.f``），文件名里含
   连字符可能被误读成旗标，因此传给 std2 的文件名禁止含 ``-``
   （见 :func:`_validate_std2_argument_filename`；自动生成的文件名会
   把 ``-`` 替换成 ``_``）。
3. 线性代数后端是 32 位整数接口（LP64，与已编译的 std2 一致）：超过
   33,000 个基函数（约 2,000–3,000 个原子）会整数溢出，需按 ILP64
   重新编译两个程序（上游 meson 的 ``-Dinterface=64``，仅 MKL 后端
   支持）。数百原子的常规体系远在安全范围内。
4. **tda.dat 文件名越界 bug**（集群实测，上游 std2 v2.0.1）：
   ``print_tdadat`` 的文件名哑元声明成定长 ``character*80``，调用处传
   7 字符字面量 ``'tda.dat'``，越界读进常量池的相邻字符串——本库编译
   的二进制实际写出的文件名是 ``tda.dat`` 后跟一串垃圾字符（**内容
   完整无损**，只有名字坏；``-rpa`` 路径必中，sTDA 的 A+B/2 路径经
   字面量 open 不受影响）。两种执行模式都在 std2 结束后做文件名
   规范化（:func:`_normalize_garbled_tda_dat` / 作业脚本里的 ``mv``
   循环行），把它改名回 ``tda.dat``。值得向上游报告；将来源码修复
   （改成 ``character*(*)``）重编后，规范化自动变成无操作。

与 :mod:`Chem_Lib.Lib_xTB` / :mod:`Chem_Lib.Lib_g_xTB` 同构的分层：
合法值表与解析器（关键参数无默认值，fail early）→ 可执行文件与环境
变量 → 命令构造（脚本模式，供 HPC 作业脚本使用）→ 本地直接运行
（subprocess）→ 结果解析类。
"""

from __future__ import annotations

__author__ = "LiYuanhe"

import json
import os
import re

from .Lib_xTB import chem_lib_directory, ensure_executable_bit

#: 标记 + provenance 文件名（记录输入生成时的参数；运行函数不读取它，
#: 关键参数必须在运行时显式给出——与 xTB / g-xTB 家族的约定一致）。
STD2_CONFIG_FILE_NAME = ".std2_config"

#: 本模块封装的 std2 方法。响应函数（``-resp`` / ``-2PA`` / ``-oprot`` /
#: ``-s2s``）、自旋翻转（``-sf``）、NTO 分析（``-nto``）等不在封装范围，
#: 需要时直接调用二进制（旗标见手册第 3.2 节）。
VALID_STD2_METHODS = ("sTDA", "sTD-DFT", "XsTDA", "XsTD-DFT",
                      "sTDA-xTB", "sTD-DFT-xTB")

#: 使用 xtb4stda 基态波函数（``wfn.xtb``）的方法。
_XTB_GROUND_STATE_METHODS = ("sTDA-xTB", "sTD-DFT-xTB")

#: 使用精确积分（``-XsTD``）的方法。
_EXACT_INTEGRAL_METHODS = ("XsTDA", "XsTD-DFT")

#: 全响应（``-rpa``）的方法。
_FULL_RESPONSE_METHODS = ("sTD-DFT", "XsTD-DFT", "sTD-DFT-xTB")

#: Molden 文件风格（``-sty``，手册第 3.1 节）→ 生成程序。
#: 正确取值可以先用 :func:`run_std2_molden_check`（``-chk``）确认——
#: 检查通过时 std2 会在日志里打出应当使用的 ``-sty`` 旗标。
VALID_MOLDEN_STYLES = {
    1: "TURBOMOLE",
    2: "MOLPRO",
    3: "TERACHEM / GAUSSIAN (g2molden) / Q-CHEM (qc2molden.sh)",
}

#: XsTD 方法原生支持的 range-separated hybrid 泛函 → std2 专用旗标
#: （手册第 3.2 节；旗标大小写敏感，且自带 ``-XsTD`` 与全部参数含义，
#: 使用时不再需要 ``-ax`` / ``-al`` / ``-be``）。
XSTD_RSH_FUNCTIONAL_FLAGS = {
    "CAM-B3LYP": "-CAMB3LYP",
    "wB97X-D2": "-wB97XD2",
    "wB97X-D3": "-wB97XD3",
    "wB97M-V": "-wB97MV",
    "SRC2-R1": "-SRC2R1",
    "SRC2-R2": "-SRC2R2",
}

#: sTDA / sTD-DFT 配 range-separated hybrid 泛函时的经验参数
#: （手册第 3.3.2 节表格；alpha → ``-al``、beta → ``-be``、
#: ax → ``-ax``）。
STD_RSH_PARAMETERS = {
    "CAM-B3LYP": {"ax": 0.38, "alpha": 0.90, "beta": 1.86},
    "LC-BLYP": {"ax": 0.53, "alpha": 4.50, "beta": 8.00},
    "wB97": {"ax": 0.61, "alpha": 4.41, "beta": 8.00},
    "wB97X": {"ax": 0.56, "alpha": 4.58, "beta": 8.00},
    "wB97X-D3": {"ax": 0.51, "alpha": 4.51, "beta": 8.00},
}

#: 光子能量与波长的乘积 E[eV] × λ[nm]（hc，CODATA）。
PHOTON_ENERGY_WAVELENGTH_PRODUCT__eV_nm = 1239.84198

#: xtb4stda 的 GBSA 隐式溶剂（随包参数目录里的 ``.param_gbsa_<名>``
#: 文件，v1.0 发行版全集）。"none" 表示气相（不加 ``-gbsa`` 旗标）。
VALID_XTB4STDA_GBSA_SOLVENTS = (
    "acetone", "acetonitrile", "benzene", "ch2cl2", "chcl3", "cs2",
    "dmso", "ether", "h2o", "methanol", "thf", "toluene",
)

#: 常见等价写法 → GBSA 参数文件的规范名
_XTB4STDA_SOLVENT_ALIASES = {
    "chloroform": "chcl3",
    "water": "h2o",
    "dcm": "ch2cl2",
    "dichloromethane": "ch2cl2",
}


def _normalize_name(name: str) -> str:
    """归一化方法 / 泛函名用于查表：小写、去掉空格与连字符、ω → w。"""
    return (str(name).strip().lower().replace("ω", "w")
            .replace("-", "").replace("_", "").replace(" ", ""))


def resolve_std2_method(method: str) -> str:
    """校验并规范化 std2 方法名，返回 :data:`VALID_STD2_METHODS` 中的写法。

    大小写不敏感，连字符 / 下划线 / 空格可省略（``"stddft"`` 与
    ``"sTD-DFT"`` 等价）。
    """
    if method is None:
        raise ValueError(
            "method is required — pass one of: "
            + ", ".join(VALID_STD2_METHODS)
        )
    normalized_to_canonical = {_normalize_name(x): x for x in VALID_STD2_METHODS}
    normalized = _normalize_name(method)
    if normalized not in normalized_to_canonical:
        raise ValueError(
            f"Unsupported std2 method: {method!r}. "
            f"Valid options: {', '.join(VALID_STD2_METHODS)}"
        )
    return normalized_to_canonical[normalized]


def resolve_molden_style(molden_style: int) -> int:
    """校验 ``-sty`` 取值（Molden 文件的生成程序风格）。

    关键参数、无默认值：std2 自身默认 ``-sty 1``（TURBOMOLE），对
    Gaussian 经 g2molden 转换的文件是**错的**（应为 3），静默沿用默认
    会得到错误结果，因此必须显式给出。不确定时先用
    :func:`run_std2_molden_check` 做输入检查，std2 会在日志里打出应当
    使用的取值。
    """
    if molden_style is None:
        raise ValueError(
            "molden_style is required and has NO default — std2 reads the "
            "Molden file differently depending on the generating program: "
            + "; ".join(f"{key} = {value}"
                        for key, value in VALID_MOLDEN_STYLES.items())
            + ". 不确定时先用 run_std2_molden_check（-chk）确认。"
        )
    style = int(molden_style)
    if style not in VALID_MOLDEN_STYLES:
        raise ValueError(
            f"Unsupported molden_style: {molden_style!r}. Valid values: "
            + "; ".join(f"{key} = {value}"
                        for key, value in VALID_MOLDEN_STYLES.items())
        )
    return style


def _validate_std2_argument_filename(filename: str) -> str:
    """校验将出现在 std2 命令行上的文件名。

    std2 的参数解析对每个命令行 token 做**子串匹配**（源码 ``main.f``，
    如 ``index(dummy,'-e')``），文件名里的连字符可能命中 ``-e`` / ``-t``
    / ``-f`` 等旗标并把后续参数吞掉，因此一律拒绝含 ``-`` 的文件名
    （改用 ``_``）；含空白字符的文件名在脚本模式下也无法安全传递，
    一并拒绝。
    """
    base_name = os.path.basename(filename)
    if "-" in base_name:
        raise ValueError(
            f"Filename passed to std2 must not contain '-': {base_name!r}. "
            "std2 parses its command line by substring matching, so a "
            "hyphenated filename can be misread as a flag — rename the "
            "file, replacing '-' with '_'."
        )
    if re.search(r"\s", base_name):
        raise ValueError(
            f"Filename passed to std2 must not contain whitespace: "
            f"{base_name!r}."
        )
    return base_name


# =====================================================================
# 命令行旗标构造（两种执行模式共用）
# =====================================================================

def build_std2_flags(
    *,
    method: str,
    energy_threshold_eV: float,
    molden_file: str | None = None,
    molden_style: int | None = None,
    fock_exchange_ax: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    rsh_functional: str | None = None,
    triplet: bool = False,
) -> list[str]:
    """构造一次 std2 激发态计算的命令行旗标列表（不含可执行文件路径）。

    参数组合规则（全部 fail early，绝不静默用默认值代答）：

    - ``method`` / ``energy_threshold_eV``：**必填**。能量阈值（``-e``，
      eV）决定纳入组态空间的能量窗口，std2 自身默认 7 eV 但手册明确
      建议按目标能区调整，故本库要求显式给出。
    - 基于 Molden 文件的方法（sTDA / sTD-DFT / XsTDA / XsTD-DFT）：
      ``molden_file`` 与 ``molden_style`` 必填；泛函参数二选一——
      给 ``fock_exchange_ax``（全局杂化泛函的 Fock 交换比例，如 PBE0
      为 0.25；sTDA / sTD-DFT 配 range-separated hybrid 时可再配
      ``alpha`` + ``beta``，成对给出，取值见
      :data:`STD_RSH_PARAMETERS`），或给 ``rsh_functional``（按方法
      自动查表：sTDA / sTD-DFT 用 :data:`STD_RSH_PARAMETERS` 展开成
      ``-ax -al -be``；XsTDA / XsTD-DFT 用
      :data:`XSTD_RSH_FUNCTIONAL_FLAGS` 的专用旗标）。两路同时给出
      即报错。
    - 基于 xTB 基态的方法（sTDA-xTB / sTD-DFT-xTB）：方法参数由 std2
      自动设定（手册第 3.2 节），上述 Molden / 泛函参数一律禁止给出；
      运行目录中须已有 xtb4stda 写出的 ``wfn.xtb``。
    - ``triplet``：计算 singlet-triplet 激发（``-t``，要求自旋限制性
      基态）。

    Returns:
        旗标字符串列表，如 ``["-f", "pbe0.molden.inp", "-ax", "0.25",
        "-sty", "3", "-e", "6"]``。
    """
    method = resolve_std2_method(method)

    if energy_threshold_eV is None:
        raise ValueError(
            "energy_threshold_eV is required and has NO default — std2's "
            "own default (7 eV) is rarely what you want; the manual "
            "recommends adjusting it to the energy range of interest."
        )
    energy_threshold_eV = float(energy_threshold_eV)
    if energy_threshold_eV <= 0:
        raise ValueError(
            f"energy_threshold_eV must be positive: {energy_threshold_eV}"
        )

    if method in _XTB_GROUND_STATE_METHODS:
        forbidden = {"molden_file": molden_file, "molden_style": molden_style,
                     "fock_exchange_ax": fock_exchange_ax, "alpha": alpha,
                     "beta": beta, "rsh_functional": rsh_functional}
        given = [key for key, value in forbidden.items() if value is not None]
        if given:
            raise ValueError(
                f"Method {method} reads the xtb4stda wavefunction (wfn.xtb) "
                f"and sets ax / alpha / beta automatically — do not pass: "
                f"{', '.join(given)}."
            )
        core_flags = ["-xtb"]
    else:
        if molden_file is None:
            raise ValueError(
                f"Method {method} requires a Molden input file — pass "
                "molden_file."
            )
        molden_file = _validate_std2_argument_filename(molden_file)
        molden_style = resolve_molden_style(molden_style)

        if rsh_functional is not None:
            explicitly_given = [name for name, value in
                                (("fock_exchange_ax", fock_exchange_ax),
                                 ("alpha", alpha), ("beta", beta))
                                if value is not None]
            if explicitly_given:
                raise ValueError(
                    "Pass EITHER rsh_functional OR explicit parameters "
                    f"({', '.join(explicitly_given)} given alongside "
                    f"rsh_functional={rsh_functional!r}) — not both."
                )

        if method in _EXACT_INTEGRAL_METHODS:
            if alpha is not None or beta is not None:
                raise ValueError(
                    f"alpha / beta (-al / -be) are sTDA / sTD-DFT parameters; "
                    f"{method} supports range-separated hybrids natively via "
                    "rsh_functional (see XSTD_RSH_FUNCTIONAL_FLAGS)."
                )
            if rsh_functional is not None:
                normalized_to_flag = {_normalize_name(key): flag for key, flag
                                      in XSTD_RSH_FUNCTIONAL_FLAGS.items()}
                normalized = _normalize_name(rsh_functional)
                if normalized not in normalized_to_flag:
                    raise ValueError(
                        f"RSH functional {rsh_functional!r} is not available "
                        f"natively for {method}. Valid options: "
                        f"{', '.join(XSTD_RSH_FUNCTIONAL_FLAGS)}"
                    )
                # 专用旗标自带 -XsTD 与全部参数（手册第 3.2 节）
                functional_flags = [normalized_to_flag[normalized]]
            else:
                if fock_exchange_ax is None:
                    raise ValueError(
                        f"Method {method} requires either fock_exchange_ax "
                        "(global hybrid, e.g. 0.25 for PBE0) or "
                        "rsh_functional."
                    )
                functional_flags = ["-XsTD",
                                    "-ax", format(float(fock_exchange_ax), "g")]
        else:  # sTDA / sTD-DFT
            if rsh_functional is not None:
                normalized_to_parameters = {
                    _normalize_name(key): parameters for key, parameters
                    in STD_RSH_PARAMETERS.items()}
                normalized = _normalize_name(rsh_functional)
                if normalized not in normalized_to_parameters:
                    raise ValueError(
                        f"No sTDA / sTD-DFT parameters tabulated for RSH "
                        f"functional {rsh_functional!r}. Valid options: "
                        f"{', '.join(STD_RSH_PARAMETERS)} "
                        "(手册第 3.3.2 节表格；其他泛函请显式给出 "
                        "fock_exchange_ax + alpha + beta)。"
                    )
                parameters = normalized_to_parameters[normalized]
                fock_exchange_ax = parameters["ax"]
                alpha = parameters["alpha"]
                beta = parameters["beta"]
            if fock_exchange_ax is None:
                raise ValueError(
                    f"Method {method} requires fock_exchange_ax (the amount "
                    "of Fock exchange of the ground-state functional, e.g. "
                    "0.25 for PBE0) or rsh_functional."
                )
            if (alpha is None) != (beta is None):
                raise ValueError(
                    "alpha (-al) and beta (-be) must be given as a pair "
                    "(both or neither)."
                )
            functional_flags = ["-ax", format(float(fock_exchange_ax), "g")]
            if alpha is not None:
                functional_flags += ["-al", format(float(alpha), "g"),
                                     "-be", format(float(beta), "g")]

        core_flags = (["-f", molden_file] + functional_flags
                      + ["-sty", str(molden_style)])

    flags = core_flags + ["-e", format(energy_threshold_eV, "g")]
    if method in _FULL_RESPONSE_METHODS:
        flags.append("-rpa")
    if triplet:
        flags.append("-t")
    return flags


# =====================================================================
# 可执行文件与环境变量
# =====================================================================

def _resolve_sTD2_tool_executable(tool_name: str) -> str:
    """解析随包分发的 sTD2 工具（``std2`` / ``g_spec`` / ``g2molden``）。

    只有 Linux 构建（全静态、可移植）；Windows 上直接报错——本模块在
    Windows 上只能用于命令行构造（脚本模式）与结果解析，实际运行须在
    Linux（含集群）上进行。找不到文件时同样直接报错（fail early），
    **不回落**到 PATH。
    """
    import platform

    if platform.system() == "Windows":
        raise RuntimeError(
            f"The bundled {tool_name} binary is Linux-only (no Windows "
            "build exists for the std2 package). 在 Windows 上本模块只能"
            "构造命令行（脚本模式）与解析结果；实际运行请在 Linux "
            "（含集群）上进行。"
        )
    packaged = os.path.join(chem_lib_directory(), "Executable_sTD2",
                            "Linux", "bin", tool_name)
    if not os.path.isfile(packaged):
        raise FileNotFoundError(
            f"{tool_name} executable not found: {packaged}\n"
            "该二进制随 Chem_Lib 包分发（不进 git，由 "
            "A0_HPC_Sync_Python_Lib.py 同步上集群，pip / uv 安装经 "
            "pyproject 的 package-data 带入）。它缺失说明本机的 "
            "Executable_sTD2 文件夹还没同步 / 安装到位——先补齐它，"
            "不要改用 PATH 上的同名程序。"
        )
    ensure_executable_bit(packaged)
    return packaged


def resolve_std2_executable() -> str:
    """解析 ``std2`` 主程序的路径（仅 Linux，见 :func:`_resolve_sTD2_tool_executable`）。"""
    return _resolve_sTD2_tool_executable("std2")


def resolve_g_spec_executable() -> str:
    """解析绘谱工具 ``g_spec`` 的路径（仅 Linux）。"""
    return _resolve_sTD2_tool_executable("g_spec")


def resolve_g2molden_executable() -> str:
    """解析 Gaussian 输出 → Molden 转换工具 ``g2molden`` 的路径（仅 Linux）。"""
    return _resolve_sTD2_tool_executable("g2molden")


def resolve_xtb4stda_executable() -> str:
    """解析 sTDA-xTB 基态程序 ``xtb4stda`` 的路径（仅 Linux）。

    官方 GitHub v1.0 发行版的静态二进制，随本包分发于
    ``Executable_sTD2/Linux/bin/``。它写出 ``wfn.xtb`` 供
    ``std2 -xtb`` 读取。
    """
    return _resolve_sTD2_tool_executable("xtb4stda")


def xtb4stda_home_directory() -> str:
    """返回 xtb4stda 的参数目录（``XTB4STDAHOME`` 应指向的路径）。

    目录为 ``Executable_sTD2/xtb4stda_home/``，须含 ``.param_stda1.xtb``
    与 ``.param_stda2.xtb``（2018 扩展参数集）。缺失直接报错（fail
    early）——xtb4stda 在 ``XTB4STDAHOME`` 未设置时会回落到 ``~/``，
    本库绝不依赖这种回落（会静默读到陈旧参数）。
    """
    home_directory = os.path.join(chem_lib_directory(), "Executable_sTD2",
                                  "xtb4stda_home")
    missing = [name for name in (".param_stda1.xtb", ".param_stda2.xtb")
               if not os.path.isfile(os.path.join(home_directory, name))]
    if missing:
        raise FileNotFoundError(
            f"xtb4stda parameter directory is incomplete: {home_directory} "
            f"is missing {', '.join(missing)}. 参数目录随 Chem_Lib 包分发"
            "（同步 / 安装 Executable_sTD2 整文件夹即得），缺失说明还没"
            "同步到位。"
        )
    return home_directory


#: 家目录下指向随包参数目录的短符号链接名（见模块 docstring 的
#: 80 字符路径陷阱）。
XTB4STDA_HOME_SYMLINK_NAME = ".xtb4stda_home"

#: 参数目录里最长的文件名（长度守卫用）。
_LONGEST_PARAMETER_FILE_NAME = ".param_gbsa_acetonitrile"


def ensure_short_xtb4stda_home() -> str:
    """确保 ``~/.xtb4stda_home`` 指向随包参数目录，返回该短路径。

    xtb4stda 的路径缓冲区只有 80 字符（``character*80``），随包参数
    目录的真实路径必超限——运行前把家目录下的短符号链接创建 / 刷新到
    真实目录，环境变量 ``XTB4STDAHOME`` 一律指向短链接。幂等：链接
    已存在且指向正确时不动；指向别处时刷新；同名的**非链接**文件存在
    时报错（不覆盖来历不明的文件）。

    Returns:
        短链接的绝对路径（已验证「链接路径 + 最长参数文件名」仍在
        80 字符限制内）。
    """
    real_directory = xtb4stda_home_directory()
    symlink_path = os.path.join(os.path.expanduser("~"),
                                XTB4STDA_HOME_SYMLINK_NAME)
    if os.path.islink(symlink_path):
        if os.path.realpath(symlink_path) != os.path.realpath(real_directory):
            os.remove(symlink_path)
            os.symlink(real_directory, symlink_path)
    elif os.path.exists(symlink_path):
        raise FileExistsError(
            f"{symlink_path} exists but is not a symlink — refusing to "
            "replace it. Remove it manually, then rerun."
        )
    else:
        os.symlink(real_directory, symlink_path)

    longest_parameter_path = (symlink_path + "/"
                              + _LONGEST_PARAMETER_FILE_NAME)
    if len(longest_parameter_path) >= 80:
        raise RuntimeError(
            f"Even the short symlink path exceeds xtb4stda's 80-character "
            f"buffer: {longest_parameter_path!r} "
            f"({len(longest_parameter_path)} chars). 家目录路径过长，"
            "需要重新编译 xtb4stda（加大路径缓冲区）才能在本机使用。"
        )
    return symlink_path


def resolve_xtb4stda_solvent_flag(solvent: str) -> tuple[str, str]:
    """校验 GBSA 溶剂并返回 ``(规范化名, xtb4stda 命令行参数)``。

    关键参数、无默认值：solvent 必须显式提供；气相必须显式写
    ``"none"``。同时校验对应的 ``.param_gbsa_<名>`` 参数文件确实
    存在于随包参数目录中。
    """
    if solvent is None:
        raise ValueError(
            "solvent is required and has NO default — pass a GBSA solvent "
            f"name ({', '.join(VALID_XTB4STDA_GBSA_SOLVENTS)}), or the "
            'explicit string "none" for a gas-phase calculation.'
        )
    solvent_lower = str(solvent).strip().lower()
    solvent_lower = _XTB4STDA_SOLVENT_ALIASES.get(solvent_lower, solvent_lower)
    if solvent_lower == "none":
        return "none", ""
    if solvent_lower not in VALID_XTB4STDA_GBSA_SOLVENTS:
        raise ValueError(
            f"Unsupported GBSA solvent: {solvent!r}. Valid options: "
            f"{', '.join(VALID_XTB4STDA_GBSA_SOLVENTS)}; "
            'use "none" for gas phase.'
        )
    parameter_file = os.path.join(xtb4stda_home_directory(),
                                  f".param_gbsa_{solvent_lower}")
    if not os.path.isfile(parameter_file):
        raise FileNotFoundError(
            f"GBSA parameter file not found: {parameter_file}"
        )
    return solvent_lower, f"-gbsa {solvent_lower}"


def std2_runtime_environment(cores: int) -> dict:
    """构造运行 std2 所需的环境变量表（subprocess 用）。

    在 ``os.environ`` 的副本上覆盖。std2 用 OpenMP 并行并依赖 MKL
    （手册第 1 节：把 ``OMP_NUM_THREADS`` 与 ``MKL_NUM_THREADS`` 设为
    可用核数可加速计算；随包二进制静态链接 MKL，变量同样生效）；
    ``OMP_STACKSIZE=4G`` 沿用 xTB 家族的约定，防大体系线程栈溢出。
    """
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(cores)
    environment["MKL_NUM_THREADS"] = str(cores)
    environment["OMP_STACKSIZE"] = "4G"
    return environment


def std2_environment_lines(cores: int, *,
                           include_xtb4stda_home: bool = False) -> list[str]:
    """生成运行 std2 / xtb4stda 所需的环境变量 export 行（bash 作业脚本用）。

    *include_xtb4stda_home* = True 时附加两行：把家目录短符号链接
    ``~/.xtb4stda_home`` 刷新到随包参数目录（``ln -sfn``，幂等），并把
    ``XTB4STDAHOME`` 指向短链接——xtb4stda 的路径缓冲区只有 80 字符，
    参数目录的真实路径必超限（见模块 docstring）；任何要运行 xtb4stda
    的作业脚本都必须带上这两行，也绝不能让它回落到 ``~/`` 读陈旧参数。
    """
    lines = [f"export OMP_NUM_THREADS={cores}",
             f"export MKL_NUM_THREADS={cores}",
             "export OMP_STACKSIZE=4G"]
    if include_xtb4stda_home:
        lines.append(f"ln -sfn {xtb4stda_home_directory()} "
                     f"$HOME/{XTB4STDA_HOME_SYMLINK_NAME}")
        lines.append(f"export XTB4STDAHOME=$HOME/{XTB4STDA_HOME_SYMLINK_NAME}")
    return lines


def _execute_sTD2_subprocess(argument_list: list[str], run_folder: str,
                             log_path: str, cores: int, *,
                             stdin_path: str | None = None,
                             extra_environment: dict | None = None) -> None:
    """在 *run_folder* 里启动 sTD2 工具子进程，输出重定向到 *log_path*。

    stdout 与 stderr 合并写入日志；*stdin_path* 给出时接到子进程的标准
    输入（``g_spec < tda.dat`` 的调用形式）。Linux 上在子进程里解除栈
    大小限制（同 xTB 家族，防大分子栈溢出）。退出码非零时抛
    RuntimeError 并附日志末尾内容。

    注意 Fortran 的 ``stop '<消息>'`` 退出码是 0——std2 的很多参数错误
    走这条路径，因此**退出码为零不代表成功**，调用方必须另行检查预期
    产物（如 ``tda.dat``）是否生成。
    """
    import platform
    import subprocess

    environment = std2_runtime_environment(cores)
    if extra_environment:
        environment.update(extra_environment)

    preexec_function = None
    if platform.system() != "Windows":
        def preexec_function():  # noqa: F811 — 仅非 Windows 平台使用
            import resource
            try:
                resource.setrlimit(resource.RLIMIT_STACK,
                                   (resource.RLIM_INFINITY,
                                    resource.RLIM_INFINITY))
            except (ValueError, OSError):
                pass  # 无权限解除时按系统上限运行

    stdin_file = None
    try:
        if stdin_path is not None:
            stdin_file = open(stdin_path, "rb")
        with open(log_path, "w", encoding="utf-8", errors="ignore") as log_file:
            completed = subprocess.run(
                argument_list, cwd=run_folder, env=environment,
                stdin=stdin_file,
                stdout=log_file, stderr=subprocess.STDOUT,
                preexec_fn=preexec_function,
            )
    finally:
        if stdin_file is not None:
            stdin_file.close()

    if completed.returncode != 0:
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as log_file:
                log_tail = "".join(log_file.readlines()[-25:])
        except OSError:
            log_tail = "(log unreadable)"
        raise RuntimeError(
            f"{os.path.basename(argument_list[0])} exited with code "
            f"{completed.returncode} "
            f"(command: {' '.join(argument_list)}; folder: {run_folder}).\n"
            f"Log tail ({log_path}):\n{log_tail}"
        )


# =====================================================================
# 输入文件的定位与生成
# =====================================================================

def resolve_molden_file(std2_folder: str, molden_file: str | None = None) -> str:
    """返回文件夹中作为输入的 Molden 文件名（不含路径）。

    选择顺序：显式指定的 *molden_file*（须存在）＞ 文件夹中**唯一**的
    文件名含 ``molden`` 的文件。零个或多个候选且无法判定时报错。
    """
    contents = os.listdir(std2_folder)
    if molden_file:
        if molden_file not in contents:
            raise FileNotFoundError(
                f"Specified Molden file not found in {std2_folder}: "
                f"{molden_file}"
            )
        return molden_file
    candidates = [f for f in contents if "molden" in f.lower()
                  and os.path.isfile(os.path.join(std2_folder, f))]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No Molden file (filename containing 'molden') found in: "
            f"{std2_folder}"
        )
    raise ValueError(
        f"Multiple Molden candidates in {std2_folder}: {candidates}. "
        "Specify the input explicitly via molden_file."
    )


def _gaussian_output_route_text(gaussian_output_path: str) -> str | None:
    """从 Gaussian 输出文件头部提取路由区段文本（用于关键词检查）。

    路由回显被 Gaussian 折行且可能从关键词中间断开，因此把各行去空白后
    **无缝拼接**再做子串检查（被折断的关键词会重新接上；行边界处两个
    关键词粘连不影响子串存在性判断）。找不到路由回显时返回 ``None``
    （检查方降级为不拦截，让 g2molden / std2 自己报错）。
    """
    header_lines: list[str] = []
    with open(gaussian_output_path, encoding="utf-8", errors="ignore") as f:
        for line_count, line in enumerate(f):
            if line_count > 400:
                break
            header_lines.append(line.rstrip("\n"))

    route_pieces: list[str] = []
    inside_route = False
    for line in header_lines:
        stripped = line.strip()
        if not inside_route:
            if stripped.startswith("#"):
                inside_route = True
                route_pieces.append(stripped)
        else:
            if set(stripped) <= {"-"} or not stripped:
                break  # 路由回显以短横线分隔行结束
            route_pieces.append(stripped)
    if not route_pieces:
        return None
    return "".join(route_pieces).lower()


def check_gaussian_output_route_for_std2(gaussian_output_path: str) -> None:
    """检查 Gaussian 输出的路由是否满足 g2molden / std2 的要求。

    要求（手册第 5 节）：``#P`` 详细打印、``gfinput``（打印基组）、
    ``pop=full`` 或 ``IOp(6/7=3)``（打印 LCAO-MO 系数）、``6D 10F``
    （笛卡尔基函数——std2 只能处理 Cartesian GTO）。缺任何一项都会
    让 g2molden 产出残缺 / 无效的 Molden 文件，直接报错（fail early）。
    在输出里找不到路由回显时不拦截（降级为让下游工具自己报错）。
    """
    route = _gaussian_output_route_text(gaussian_output_path)
    if route is None:
        return
    missing = []
    if not route.startswith("#p"):
        missing.append("#P（详细打印）")
    if "gfinput" not in route:
        missing.append("gfinput")
    if "pop=full" not in route and not re.search(r"pop=\(?\s*full", route) \
            and "6/7=3" not in route:
        missing.append("pop=full（或 IOp(6/7=3)）")
    if "6d" not in route:
        missing.append("6D")
    if "10f" not in route:
        missing.append("10F")
    if missing:
        raise ValueError(
            f"Gaussian output {gaussian_output_path} 的路由缺少 std2 / "
            f"g2molden 所需的关键词：{', '.join(missing)}。请在 Gaussian "
            "输入的路由区加上（手册第 5 节的完整要求）：#P ... gfinput "
            "pop=full 6D 10F，重新计算后再转换。"
        )


def std2_input_from_Gaussian_output(
    gaussian_output_path: str,
    output_folder: str,
    *,
    molden_filename: str | None = None,
) -> str:
    """用随包的 g2molden 把 Gaussian 输出转换成 std2 的 Molden 输入。

    先检查 Gaussian 路由是否满足要求
    （:func:`check_gaussian_output_route_for_std2`），再在 *output_folder*
    里生成 Molden 文件与 ``.std2_config``（provenance 记录：来源文件、
    转换工具、``molden_style = 3``；运行函数不读取它，方法与阈值等
    关键参数必须在运行时显式给出）。仅 Linux（g2molden 无 Windows
    构建）。

    Args:
        gaussian_output_path: Gaussian 输出文件（.out / .log）路径。
                              路由须含 ``#P gfinput pop=full 6D 10F``。
        output_folder:        std2 计算文件夹路径，不存在时自动创建。
        molden_filename:      输出的 Molden 文件名（不含路径）。默认用
                              Gaussian 输出的基名（``-`` 替换成 ``_``）
                              加 ``.molden.inp``。

    Returns:
        生成的 Molden 文件的完整路径。

    Raises:
        ValueError:   路由缺少必需关键词，或指定的文件名含连字符 / 空白。
        RuntimeError: g2molden 转换失败（输出不是 Molden 格式——注意
                      g2molden 把报错信息也写到标准输出且退出码仍为 0，
                      因此以内容判断成败）。
    """
    gaussian_output_path = os.path.abspath(gaussian_output_path)
    if not os.path.isfile(gaussian_output_path):
        raise FileNotFoundError(
            f"Gaussian output file not found: {gaussian_output_path}"
        )
    check_gaussian_output_route_for_std2(gaussian_output_path)

    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    if molden_filename is None:
        base = os.path.splitext(os.path.basename(gaussian_output_path))[0]
        molden_filename = base.replace("-", "_") + ".molden.inp"
    molden_filename = _validate_std2_argument_filename(molden_filename)
    molden_path = os.path.join(output_folder, molden_filename)

    import subprocess

    g2molden_executable = resolve_g2molden_executable()
    g2molden_log_path = os.path.join(output_folder, "g2molden.log")
    with open(molden_path, "w", encoding="utf-8", newline="\n") as molden_file, \
            open(g2molden_log_path, "w", encoding="utf-8",
                 errors="ignore") as log_file:
        completed = subprocess.run(
            [g2molden_executable, gaussian_output_path],
            cwd=output_folder, stdout=molden_file, stderr=log_file,
        )

    def _first_line(path: str) -> str:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.readline().strip()

    if (completed.returncode != 0
            or not os.path.getsize(molden_path)
            or "[molden format]" not in _first_line(molden_path).lower()):
        raise RuntimeError(
            f"g2molden failed to convert {gaussian_output_path} — the "
            f"output does not start with [Molden Format] "
            f"(first line: {_first_line(molden_path)!r}). 检查 Gaussian "
            f"输出是否正常终止、路由是否含 #P gfinput pop=full 6D 10F；"
            f"g2molden 的错误信息（也）在 {molden_path} 与 "
            f"{g2molden_log_path} 里。"
        )

    config = {
        "tool": "g2molden",
        "source_gaussian_output": os.path.basename(gaussian_output_path),
        "molden_file": molden_filename,
        "molden_style": 3,
    }
    with open(os.path.join(output_folder, STD2_CONFIG_FILE_NAME), "w",
              newline="\n") as f:
        json.dump(config, f)

    return molden_path


# =====================================================================
# 命令构造（脚本模式：生成 bash 行，供 HPC 作业脚本使用）
# =====================================================================

def build_std2_command_lines(
    *,
    method: str,
    energy_threshold_eV: float,
    std2_exe: str,
    molden_file: str | None = None,
    molden_style: int | None = None,
    fock_exchange_ax: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    rsh_functional: str | None = None,
    triplet: bool = False,
) -> list[str]:
    """生成一次 std2 激发态计算的 shell 行（std2 命令 + 产物复制）。

    全部使用相对路径，需在计算文件夹内执行（调用方负责 cd；环境变量
    export 行见 :func:`std2_environment_lines`）。参数校验与旗标构造
    同 :func:`build_std2_flags`。运行前删除陈旧的 ``tda.dat``——
    Fortran ``stop`` 的退出码是 0，上一轮的产物会掩盖本轮失败；运行后
    把 ``tda.dat`` 复制成稳定名 ``<名>.std2_tda.dat``。
    """
    flags = build_std2_flags(
        method=method, energy_threshold_eV=energy_threshold_eV,
        molden_file=molden_file, molden_style=molden_style,
        fock_exchange_ax=fock_exchange_ax, alpha=alpha, beta=beta,
        rsh_functional=rsh_functional, triplet=triplet,
    )
    if molden_file is not None:
        log_stem = os.path.splitext(molden_file)[0]
        # ".molden.inp" 双后缀时再剥一层
        if log_stem.lower().endswith(".molden"):
            log_stem = os.path.splitext(log_stem)[0]
    else:
        log_stem = "wfn"
    return [
        "rm -f tda.dat tda.dat?*",
        f"{std2_exe} {' '.join(flags)} &> {log_stem}.std2.log",
        _NORMALIZE_TDA_DAT_SHELL_LINE,
        f"if [ -f tda.dat ]; then cp tda.dat {log_stem}.std2_tda.dat; fi",
    ]


def build_g_spec_command_lines(g_spec_exe: str) -> list[str]:
    """生成用 g_spec 从 ``tda.dat`` 绘谱的 shell 行（需在计算文件夹内执行）。

    产物为 ``spec.dat``（展宽谱）与 ``rots.dat``（棒状谱），选项由
    ``tda.dat`` 头部控制（见 :func:`set_tda_dat_g_spec_options`）。
    """
    return [
        "rm -f spec.dat rots.dat",
        f"if [ -f tda.dat ]; then {g_spec_exe} < tda.dat &> g_spec.log; fi",
    ]


# =====================================================================
# 本地直接运行（直接模式：本机 subprocess；仅 Linux）
# =====================================================================

def _normalize_garbled_tda_dat(run_folder: str) -> None:
    """把上游文件名越界 bug 产出的 ``tda.dat<垃圾后缀>`` 改名回 ``tda.dat``。

    见模块 docstring 已知限制第 4 条：``print_tdadat`` 的定长
    ``character*80`` 文件名哑元越界读取，写出的文件名是 ``tda.dat``
    后跟常量池垃圾字符，内容本身完整。幂等：正名 ``tda.dat`` 已存在、
    或候选不唯一时不动（候选多于一个说明文件夹状态异常，交给调用方的
    存在性检查报错）。
    """
    if os.path.isfile(os.path.join(run_folder, "tda.dat")):
        return
    candidates = [name for name in os.listdir(run_folder)
                  if name.startswith("tda.dat") and name != "tda.dat"]
    if len(candidates) == 1:
        os.replace(os.path.join(run_folder, candidates[0]),
                   os.path.join(run_folder, "tda.dat"))


#: 作业脚本里与 :func:`_normalize_garbled_tda_dat` 等价的 shell 行。
_NORMALIZE_TDA_DAT_SHELL_LINE = (
    'if [ ! -f tda.dat ]; then for f in tda.dat?*; do '
    'if [ -e "$f" ]; then mv "$f" tda.dat; break; fi; done; fi')


def run_std2_molden_check(
    std2_folder: str,
    *,
    molden_file: str | None = None,
    cores: int = 1,
) -> str:
    """对 Molden 文件做 std2 输入检查（``-chk``，手册第 3.1 节）。

    std2 以 Mulliken 布居分析核对 GTO / MO 数据的读入，成功时在日志里
    打出 ``--- S U C C E S S ---`` 横幅与实际计算应使用的 ``-sty``
    取值。检查失败（横幅缺失）时抛 RuntimeError。仅 Linux。

    Returns:
        检查日志（``<名>.std2_check.log``）的完整路径。
    """
    std2_folder = os.path.abspath(std2_folder)
    if not os.path.isdir(std2_folder):
        raise FileNotFoundError(
            f"std2 calculation folder not found: {std2_folder}")
    molden_file = resolve_molden_file(std2_folder, molden_file)
    _validate_std2_argument_filename(molden_file)

    std2_executable = resolve_std2_executable()
    log_path = os.path.join(
        std2_folder, os.path.splitext(molden_file)[0] + ".std2_check.log")
    _execute_sTD2_subprocess(
        [std2_executable, "-f", molden_file, "-chk"],
        std2_folder, log_path, cores)

    with open(log_path, encoding="utf-8", errors="ignore") as log_file:
        log_content = log_file.read()
    if "S U C C E S S" not in log_content:
        raise RuntimeError(
            f"std2 input check FAILED for {molden_file} — the log does not "
            f"contain the success banner. 详情见 {log_path}。"
        )
    return log_path


def run_std2_folder(
    std2_folder: str,
    *,
    method: str,
    energy_threshold_eV: float,
    molden_file: str | None = None,
    molden_style: int | None = None,
    fock_exchange_ax: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    rsh_functional: str | None = None,
    triplet: bool = False,
    cores: int = 1,
) -> str:
    """在本机直接运行一次 std2 激发态计算（仅 Linux，不经队列）。

    参数组合规则见 :func:`build_std2_flags`（关键参数必填、无默认值）。
    基于 Molden 的方法：输入文件按 :func:`resolve_molden_file` 定位；
    基于 xTB 基态的方法（sTDA-xTB / sTD-DFT-xTB）：文件夹中须已有
    xtb4stda 写出的 ``wfn.xtb``（xtb4stda 尚未随本包分发，见模块
    docstring）。

    运行前删除陈旧的 ``tda.dat``（Fortran ``stop`` 的退出码是 0，上一轮
    产物会掩盖本轮失败），运行后要求 ``tda.dat`` 已生成并复制成稳定名
    ``<名>.std2_tda.dat``（供 :class:`sTD2_result` 解析；多次以不同参数
    重跑同一文件夹时注意稳定名副本会被覆盖）。

    Args:
        std2_folder:          计算文件夹。
        method:               **必填**。:data:`VALID_STD2_METHODS` 之一。
        energy_threshold_eV:  **必填**。组态空间能量阈值（``-e``，eV）。
        molden_file:          输入 Molden 文件名；不填时自动定位。
        molden_style:         ``-sty`` 取值（Molden 方法必填，无默认值）。
        fock_exchange_ax:     基态泛函的 Fock 交换比例（``-ax``）。
        alpha, beta:          ``-al`` / ``-be``（成对给出；仅 sTDA /
                              sTD-DFT 配 range-separated hybrid 时用）。
        rsh_functional:       range-separated hybrid 泛函名（与上面三个
                              参数互斥，按方法自动查表）。
        triplet:              计算 singlet-triplet 激发（``-t``）。
        cores:                OpenMP / MKL 线程数。

    Returns:
        日志文件（``<名>.std2.log``）的完整路径。

    Raises:
        RuntimeError: std2 退出码非零，或运行结束但 ``tda.dat`` 未生成
                      （附日志末尾内容）。
    """
    import shutil

    method = resolve_std2_method(method)
    std2_folder = os.path.abspath(std2_folder)
    if not os.path.isdir(std2_folder):
        raise FileNotFoundError(
            f"std2 calculation folder not found: {std2_folder}")

    if method in _XTB_GROUND_STATE_METHODS:
        if not os.path.isfile(os.path.join(std2_folder, "wfn.xtb")):
            raise FileNotFoundError(
                f"Method {method} requires the xtb4stda wavefunction file "
                f"wfn.xtb in {std2_folder}. 先用随包的 xtb4stda 对该结构做"
                "基态计算（完整复合链用 run_sTD_DFT_xTB_folder，一步到位）。"
            )
        log_stem = "wfn"
    else:
        molden_file = resolve_molden_file(std2_folder, molden_file)
        log_stem = os.path.splitext(molden_file)[0]
        if log_stem.lower().endswith(".molden"):
            log_stem = os.path.splitext(log_stem)[0]

    flags = build_std2_flags(
        method=method, energy_threshold_eV=energy_threshold_eV,
        molden_file=molden_file, molden_style=molden_style,
        fock_exchange_ax=fock_exchange_ax, alpha=alpha, beta=beta,
        rsh_functional=rsh_functional, triplet=triplet,
    )

    tda_dat_path = os.path.join(std2_folder, "tda.dat")
    for stale_name in os.listdir(std2_folder):
        if stale_name.startswith("tda.dat"):
            os.remove(os.path.join(std2_folder, stale_name))

    std2_executable = resolve_std2_executable()
    log_path = os.path.join(std2_folder, f"{log_stem}.std2.log")
    _execute_sTD2_subprocess([std2_executable] + flags,
                             std2_folder, log_path, cores)

    _normalize_garbled_tda_dat(std2_folder)
    if not os.path.isfile(tda_dat_path):
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as log_file:
                log_tail = "".join(log_file.readlines()[-25:])
        except OSError:
            log_tail = "(log unreadable)"
        raise RuntimeError(
            f"std2 finished without writing tda.dat — the calculation did "
            f"not succeed (Fortran 'stop' exits with code 0, so a zero exit "
            f"code proves nothing).\nLog tail ({log_path}):\n{log_tail}"
        )
    shutil.copy(tda_dat_path,
                os.path.join(std2_folder, f"{log_stem}.std2_tda.dat"))

    return log_path


def run_g_spec(tda_dat_path: str) -> tuple[str, str]:
    """用随包的 g_spec 从 ``tda.dat`` 生成谱文件（仅 Linux）。

    在 *tda_dat_path* 所在文件夹里运行 ``g_spec < tda.dat``，生成
    ``spec.dat``（Gaussian 展宽谱：摩尔消光系数或摩尔圆二色，单位
    L·mol⁻¹·cm⁻¹）与 ``rots.dat``（未展宽的棒状谱，同单位；默认被
    LFAKTOR = 0.5 缩放）。选项由 ``tda.dat`` 头部控制，先用
    :func:`set_tda_dat_g_spec_options` 修改。

    Returns:
        ``(spec.dat 完整路径, rots.dat 完整路径)``。

    Raises:
        RuntimeError: g_spec 退出码非零，或产物未生成。
    """
    tda_dat_path = os.path.abspath(tda_dat_path)
    if not os.path.isfile(tda_dat_path):
        raise FileNotFoundError(f"tda.dat file not found: {tda_dat_path}")
    run_folder = os.path.dirname(tda_dat_path)

    spec_dat_path = os.path.join(run_folder, "spec.dat")
    rots_dat_path = os.path.join(run_folder, "rots.dat")
    for stale_product in (spec_dat_path, rots_dat_path):
        if os.path.isfile(stale_product):
            os.remove(stale_product)

    g_spec_executable = resolve_g_spec_executable()
    log_path = os.path.join(run_folder, "g_spec.log")
    _execute_sTD2_subprocess([g_spec_executable], run_folder, log_path, 1,
                             stdin_path=tda_dat_path)

    missing = [path for path in (spec_dat_path, rots_dat_path)
               if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(
            f"g_spec finished without writing "
            f"{', '.join(os.path.basename(x) for x in missing)} — "
            f"详情见 {log_path}。"
        )
    return spec_dat_path, rots_dat_path


# =====================================================================
# sTDA-xTB / sTD-DFT-xTB 复合链（xtb4stda 基态 + std2 激发态）
# =====================================================================

#: sTD2 任务文件夹的标记文件：两者有其一即被识别为 sTD2 任务
#: （``.std2_config`` 由输入生成函数写出，``std2_task.toml`` 留给
#: 手工准备的文件夹——与 g-xTB 的标记约定同构）。
STD2_MARKER_FILES = (".std2_config", "std2_task.toml")


def is_strict_sTD2_input_folder(folder: str) -> bool:
    """检查文件夹是否满足 sTDA-xTB / sTD-DFT-xTB 任务的输入约定。

    要求：至少一个 ``.xyz``、``.CHRG``、标记文件
    （:data:`STD2_MARKER_FILES` 之一）。``.UHF`` 可选（不存在即闭壳层）。
    """
    if not os.path.isdir(folder):
        return False
    contents = os.listdir(folder)
    has_xyz = any(f.lower().endswith(".xyz") for f in contents)
    has_marker = any(marker in contents for marker in STD2_MARKER_FILES)
    return has_xyz and ".CHRG" in contents and has_marker


def validate_sTD_DFT_xTB_folder(std2_folder: str,
                                structure: str | None = None) -> str:
    """校验 sTDA-xTB / sTD-DFT-xTB 任务文件夹的输入完备性，返回输入
    ``.xyz`` 文件名。

    要求 ``.CHRG`` 与标记文件存在（``.UHF`` 可选——不存在即闭壳层，
    xtb4stda 自行从这两个文件读电荷与未配对电子数）；同时拒绝混有
    经典 xTB / g-xTB 任务标记的歧义文件夹。任何缺失直接报错
    （fail early）。
    """
    from .Lib_xTB import resolve_structure_xyz

    contents = os.listdir(std2_folder)
    if ".CHRG" not in contents:
        raise FileNotFoundError(
            f"{std2_folder} is missing .CHRG. sTD2 input folders must "
            "contain .xyz + .CHRG (+ .UHF only for open-shell systems)."
        )
    if not any(marker in contents for marker in STD2_MARKER_FILES):
        raise FileNotFoundError(
            f"{std2_folder} has no sTD2 marker file "
            f"({' / '.join(STD2_MARKER_FILES)}). 由 "
            "Chem_Lib.Lib_sTD2.sTD_DFT_xTB_input_from_gjf 生成的文件夹"
            "自带 .std2_config；手工准备的文件夹请放一个 std2_task.toml。"
        )
    conflicting = [f for f in ("xtb_task.toml", ".gxtb_config",
                               "gxtb_task.toml") if f in contents]
    if conflicting:
        raise ValueError(
            f"{std2_folder} contains BOTH an sTD2 marker and other task "
            f"markers ({', '.join(conflicting)}) — ambiguous; remove one."
        )
    xyz_file = resolve_structure_xyz(std2_folder, structure)
    _validate_std2_argument_filename(xyz_file)
    return xyz_file


def sTD_DFT_xTB_input_from_gjf(
    gjf_path: str,
    output_folder: str,
    *,
    xyz_filename: str | None = None,
) -> str:
    """从 Gaussian gjf（含电荷 / 自旋多重度 / 坐标）生成 sTDA-xTB /
    sTD-DFT-xTB 任务文件夹。

    :func:`Chem_Lib.Lib_xTB.xTB_input_from_Gaussian_gjf` 的薄壳（与
    g-xTB 的生成函数同构）：坐标与 ``.CHRG`` 的生成完全复用它，然后
    按 sTD2 链的约定修正三处——

    1. 闭壳层（multiplicity = 1）时**删除** ``.UHF``（只有开壳层才
       需要告诉 xtb4stda 未配对电子数）；
    2. 把 ``.xtb_config`` 记录替换为 ``.std2_config``（provenance +
       任务标记，运行 / 提交函数不读取其中的参数——溶剂与能量阈值
       属于关键参数，必须在运行 / 提交时显式给出）；
    3. XYZ 文件名过 :func:`_validate_std2_argument_filename` 校验
       （xtb4stda 与 std2 的命令行解析都是子串匹配，文件名禁止含
       连字符；默认文件名会把 gjf 基名里的 ``-`` 替换成 ``_``）。

    Returns:
        生成的 XYZ 文件的完整路径。
    """
    from .Lib_Gaussian import Gaussian_Input
    from .Lib_xTB import xTB_input_from_Gaussian_gjf

    if xyz_filename is None:
        base = os.path.splitext(os.path.basename(gjf_path))[0]
        xyz_filename = base.replace("-", "_") + ".xyz"
    _validate_std2_argument_filename(xyz_filename)

    # method 是占位值：xTB 生成函数要求一个合法的 GFN 方法名写进
    # .xtb_config，而该记录文件随即被替换成 .std2_config。
    xyz_path = xTB_input_from_Gaussian_gjf(
        gjf_path, output_folder,
        xyz_filename=xyz_filename, solvent=None, method="GFN2",
    )

    output_folder = os.path.abspath(output_folder)

    uhf_path = os.path.join(output_folder, ".UHF")
    with open(uhf_path) as f:
        unpaired_electrons = int(f.read().strip())
    if unpaired_electrons == 0:
        os.remove(uhf_path)

    xtb_config_path = os.path.join(output_folder, ".xtb_config")
    if os.path.isfile(xtb_config_path):
        os.remove(xtb_config_path)

    with open(os.path.join(output_folder, ".CHRG")) as f:
        charge = int(f.read().strip())

    config = {
        "method_family": "sTDA-xTB / sTD-DFT-xTB",
        "source_gjf": os.path.basename(os.path.abspath(gjf_path)),
        "xyz_file": xyz_filename,
        "charge": charge,
        "unpaired_electrons": unpaired_electrons,
    }
    with open(os.path.join(output_folder, STD2_CONFIG_FILE_NAME), "w",
              newline="\n") as f:
        json.dump(config, f)

    return xyz_path


def build_sTD_DFT_xTB_command_lines(
    xyz_file: str,
    *,
    method: str,
    solvent: str,
    energy_threshold_eV: float,
    xtb4stda_exe: str,
    std2_exe: str,
    triplet: bool = False,
) -> list[str]:
    """生成一个 sTDA-xTB / sTD-DFT-xTB 复合任务的 shell 行
    （xtb4stda 基态 → std2 激发态 → 产物复制）。

    全部使用相对路径，需在任务文件夹内执行（调用方负责 cd）；环境
    变量 export 行用 ``std2_environment_lines(cores,
    include_xtb4stda_home=True)`` 生成（``XTB4STDAHOME`` 必须设置）。
    电荷与未配对电子数由 xtb4stda 自行从 ``.CHRG`` / ``.UHF`` 读取。
    运行前删除陈旧的 ``wfn.xtb`` / ``tda.dat``（Fortran ``stop`` 的
    退出码是 0，旧产物会掩盖本轮失败）；基态一步的产物 ``wfn.xtb``
    存在才继续激发态一步。

    Args:
        xyz_file:             输入结构文件名（须在任务文件夹内；禁止
                              含连字符）。
        method:               ``"sTDA-xTB"`` 或 ``"sTD-DFT-xTB"``
                              （后者即 ``-rpa``，振子强度质量更好，
                              生产建议用它）。
        solvent:              **必填**。GBSA 溶剂名（如 ``"chcl3"``），
                              气相显式写 ``"none"``。
        energy_threshold_eV:  **必填**。std2 的组态能量窗口（eV）。
        xtb4stda_exe:         xtb4stda 可执行文件路径（集群侧由
                              :func:`resolve_xtb4stda_executable` 解析）。
        std2_exe:             std2 可执行文件路径。
        triplet:              计算 singlet-triplet 激发（``-t``）。
    """
    method = resolve_std2_method(method)
    if method not in _XTB_GROUND_STATE_METHODS:
        raise ValueError(
            f"Method {method} does not run on an xtb4stda ground state — "
            f"use one of: {', '.join(_XTB_GROUND_STATE_METHODS)}."
        )
    xyz_file = _validate_std2_argument_filename(xyz_file)
    _, solvent_flag = resolve_xtb4stda_solvent_flag(solvent)
    std2_flags = build_std2_flags(
        method=method, energy_threshold_eV=energy_threshold_eV,
        triplet=triplet)

    xyz_stem = os.path.splitext(xyz_file)[0]
    xtb4stda_parts = [xtb4stda_exe, xyz_file]
    if solvent_flag:
        xtb4stda_parts.append(solvent_flag)
    return [
        "rm -f wfn.xtb tda.dat tda.dat?*",
        f"{' '.join(xtb4stda_parts)} &> {xyz_stem}.xtb4stda.log",
        f"if [ -f wfn.xtb ]; then "
        f"{std2_exe} {' '.join(std2_flags)} &> {xyz_stem}.std2.log; fi",
        _NORMALIZE_TDA_DAT_SHELL_LINE,
        f"if [ -f tda.dat ]; then cp tda.dat {xyz_stem}.std2_tda.dat; fi",
    ]


def run_sTD_DFT_xTB_folder(
    std2_folder: str,
    *,
    method: str,
    solvent: str,
    energy_threshold_eV: float,
    structure: str | None = None,
    triplet: bool = False,
    cores: int = 1,
) -> str:
    """在本机直接运行一个 sTDA-xTB / sTD-DFT-xTB 复合任务（仅 Linux）。

    两步串联：``xtb4stda <结构.xyz> [-gbsa <溶剂>]``（基态，写
    ``wfn.xtb``，电荷与未配对电子数自动读 ``.CHRG`` / ``.UHF``）→
    ``std2 -xtb -e <eV> [-rpa]``（激发态，写 ``tda.dat``）。输入文件夹
    约定与产物命名与 HPC 打包提交完全一致——同一个文件夹既可以本地
    跑，也可以提交集群。

    Args:
        std2_folder:          任务文件夹（``.xyz`` + ``.CHRG`` + 标记
                              文件；开壳层另含 ``.UHF``）。
        method:               **必填**。``"sTDA-xTB"`` 或
                              ``"sTD-DFT-xTB"``（生产建议后者）。
        solvent:              **必填**。GBSA 溶剂名或 ``"none"``。
        energy_threshold_eV:  **必填**。组态能量窗口（eV）。
        structure:            输入结构 ``.xyz`` 文件名；不填时自动选择。
        triplet:              计算 singlet-triplet 激发（``-t``）。
        cores:                OpenMP / MKL 线程数。默认 1（单核）——
                              与 xTB 家族同一裁定，吞吐靠多任务并行。

    Returns:
        std2 日志（``<结构名>.std2.log``）的完整路径；谱数据在
        ``tda.dat`` 及其稳定名副本 ``<结构名>.std2_tda.dat``（供
        :class:`sTD2_result` 解析），基态日志在
        ``<结构名>.xtb4stda.log``。

    Raises:
        RuntimeError: 任一步退出码非零，或运行结束但预期产物
                      （``wfn.xtb`` / ``tda.dat``）未生成。
    """
    import shutil

    method = resolve_std2_method(method)
    if method not in _XTB_GROUND_STATE_METHODS:
        raise ValueError(
            f"Method {method} does not run on an xtb4stda ground state — "
            f"use one of: {', '.join(_XTB_GROUND_STATE_METHODS)}."
        )
    _, solvent_flag = resolve_xtb4stda_solvent_flag(solvent)
    std2_flags = build_std2_flags(
        method=method, energy_threshold_eV=energy_threshold_eV,
        triplet=triplet)

    std2_folder = os.path.abspath(std2_folder)
    if not os.path.isdir(std2_folder):
        raise FileNotFoundError(
            f"sTD2 calculation folder not found: {std2_folder}")
    xyz_file = validate_sTD_DFT_xTB_folder(std2_folder, structure)
    xyz_stem = os.path.splitext(xyz_file)[0]

    for stale_name in os.listdir(std2_folder):
        if stale_name == "wfn.xtb" or stale_name.startswith("tda.dat"):
            os.remove(os.path.join(std2_folder, stale_name))

    xtb4stda_executable = resolve_xtb4stda_executable()
    xtb4stda_environment = {"XTB4STDAHOME": ensure_short_xtb4stda_home()}
    xtb4stda_log_path = os.path.join(std2_folder, f"{xyz_stem}.xtb4stda.log")
    _execute_sTD2_subprocess(
        [xtb4stda_executable, xyz_file] + solvent_flag.split(),
        std2_folder, xtb4stda_log_path, cores,
        extra_environment=xtb4stda_environment)

    if not os.path.isfile(os.path.join(std2_folder, "wfn.xtb")):
        try:
            with open(xtb4stda_log_path, encoding="utf-8",
                      errors="ignore") as log_file:
                log_tail = "".join(log_file.readlines()[-25:])
        except OSError:
            log_tail = "(log unreadable)"
        raise RuntimeError(
            f"xtb4stda finished without writing wfn.xtb — the ground-state "
            f"step did not succeed (Fortran 'stop' exits with code 0).\n"
            f"Log tail ({xtb4stda_log_path}):\n{log_tail}"
        )

    std2_executable = resolve_std2_executable()
    std2_log_path = os.path.join(std2_folder, f"{xyz_stem}.std2.log")
    _execute_sTD2_subprocess([std2_executable] + std2_flags,
                             std2_folder, std2_log_path, cores)

    _normalize_garbled_tda_dat(std2_folder)
    tda_dat_path = os.path.join(std2_folder, "tda.dat")
    if not os.path.isfile(tda_dat_path):
        try:
            with open(std2_log_path, encoding="utf-8",
                      errors="ignore") as log_file:
                log_tail = "".join(log_file.readlines()[-25:])
        except OSError:
            log_tail = "(log unreadable)"
        raise RuntimeError(
            f"std2 finished without writing tda.dat — the excited-state "
            f"step did not succeed.\nLog tail ({std2_log_path}):\n{log_tail}"
        )
    shutil.copy(tda_dat_path,
                os.path.join(std2_folder, f"{xyz_stem}.std2_tda.dat"))

    return std2_log_path


# =====================================================================
# tda.dat：g_spec 选项编辑与结果解析
# =====================================================================

#: tda.dat 头部的开关型关键词（存在即生效；写出的规范顺序）。
_TDA_DAT_FLAG_KEYS = ("NM", "UV", "VELO")

#: tda.dat 头部的带值关键词（下一行是取值；写出的规范顺序）。
_TDA_DAT_VALUE_KEYS = ("MMASS", "LFAKTOR", "RFAKTOR", "WIDTH", "SHIFT")


def _split_tda_dat(tda_dat_path: str) -> tuple[dict, dict, list[str]]:
    """把 ``tda.dat`` 解析成（开关字典, 取值字典, 数据行列表）。

    头部到 ``DATXY`` 为止：:data:`_TDA_DAT_FLAG_KEYS` 是无值开关，
    :data:`_TDA_DAT_VALUE_KEYS` 的取值在其下一行（保留原始字符串）。
    """
    with open(tda_dat_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    flags: dict[str, bool] = {}
    values: dict[str, str] = {}
    data_lines: list[str] = []
    line_count = 0
    seen_data_marker = False
    while line_count < len(lines):
        token = lines[line_count].strip()
        if token == "DATXY":
            data_lines = lines[line_count + 1:]
            seen_data_marker = True
            break
        if token in _TDA_DAT_FLAG_KEYS:
            flags[token] = True
        elif token in _TDA_DAT_VALUE_KEYS:
            if line_count + 1 >= len(lines):
                raise ValueError(
                    f"Malformed tda.dat header: key {token} has no value "
                    f"line ({tda_dat_path})"
                )
            values[token] = lines[line_count + 1].strip()
            line_count += 1
        else:
            raise ValueError(
                f"Malformed tda.dat header: unrecognized line {token!r} "
                f"({tda_dat_path})"
            )
        line_count += 1
    if not seen_data_marker:
        raise ValueError(f"No DATXY marker found in: {tda_dat_path}")
    return flags, values, data_lines


def set_tda_dat_g_spec_options(
    tda_dat_path: str,
    *,
    absorption: bool | None = None,
    velocity_representation: bool | None = None,
    nanometer_scale: bool | None = None,
    gaussian_width_eV: float | None = None,
    energy_shift_eV: float | None = None,
    stick_scaling_factor: float | None = None,
    broadened_scaling_factor: float | None = None,
) -> None:
    """就地修改 ``tda.dat`` 头部的 g_spec 选项（手册第 4 节）。

    每个参数为 ``None`` 时保持原状：

    - ``absorption``:               True = 吸收谱（``UV``）；False = 圆
                                    二色谱（删除 ``UV``）。
    - ``velocity_representation``:  用速度表象的 R 与 f（``VELO``）。
    - ``nanometer_scale``:          谱以 nm 为横轴（``NM``；False 为 eV）。
    - ``gaussian_width_eV``:        Gaussian 展宽的 1/e 半宽（``WIDTH``）。
    - ``energy_shift_eV``:          整谱能量平移（``SHIFT``）。
    - ``stick_scaling_factor``:     棒状谱缩放因子（``LFAKTOR``，std2
                                    默认写 0.5）。
    - ``broadened_scaling_factor``: 展宽谱缩放因子（``RFAKTOR``）。
    """
    flags, values, data_lines = _split_tda_dat(tda_dat_path)

    for flag_key, requested in (("UV", absorption),
                                ("VELO", velocity_representation),
                                ("NM", nanometer_scale)):
        if requested is True:
            flags[flag_key] = True
        elif requested is False:
            flags.pop(flag_key, None)

    for value_key, requested in (("WIDTH", gaussian_width_eV),
                                 ("SHIFT", energy_shift_eV),
                                 ("LFAKTOR", stick_scaling_factor),
                                 ("RFAKTOR", broadened_scaling_factor)):
        if requested is not None:
            values[value_key] = format(float(requested), "g")

    with open(tda_dat_path, "w", encoding="utf-8", newline="\n") as f:
        for flag_key in _TDA_DAT_FLAG_KEYS:
            if flags.get(flag_key):
                f.write(f" {flag_key}\n")
        for value_key in _TDA_DAT_VALUE_KEYS:
            if value_key in values:
                f.write(f" {value_key}\n")
                f.write(f" {values[value_key]}\n")
        f.write(" DATXY\n")
        f.writelines(data_lines)


class sTD2_result:
    """解析 std2 的谱数据文件 ``tda.dat``（或其稳定名副本 ``<名>.std2_tda.dat``）。

    数据行格式（源码 ``print_tdadat`` / ``apbtrafo``，两条路径一致）：
    态编号、激发能（eV）、振子强度（长度表象 fL、速度表象 fV）、旋转
    强度（长度表象 RL、速度表象 RV，单位 10⁻⁴⁰ erg·cm³）。注意 std2
    只把能量阈值以内的态写进 ``tda.dat``。

    Attributes:
        filename:                      被解析的文件路径。
        molecular_weight:              分子量（头部 ``MMASS``）。
        state_numbers:                 各态编号（list[int]，1 起）。
        excitation_energies__eV:       各态激发能（eV）。
        excitation_wavelengths__nm:    各态激发波长（nm，由激发能换算）。
        oscillator_strengths_length:   振子强度，长度表象。
        oscillator_strengths_velocity: 振子强度，速度表象。
        rotatory_strengths_length:     旋转强度，长度表象（10⁻⁴⁰ erg·cm³）。
        rotatory_strengths_velocity:   旋转强度，速度表象（同单位）。
        state_count:                   态数。
    """

    def __init__(self, path: str):
        self.filename = os.path.abspath(path)
        flags, values, data_lines = _split_tda_dat(self.filename)

        self.molecular_weight = (float(values["MMASS"])
                                 if "MMASS" in values else None)

        self.state_numbers: list[int] = []
        self.excitation_energies__eV: list[float] = []
        self.oscillator_strengths_length: list[float] = []
        self.oscillator_strengths_velocity: list[float] = []
        self.rotatory_strengths_length: list[float] = []
        self.rotatory_strengths_velocity: list[float] = []

        for line in data_lines:
            fields = line.split()
            if len(fields) < 6:
                if line.strip():
                    raise ValueError(
                        f"Malformed tda.dat data line {line!r} "
                        f"({self.filename})"
                    )
                continue
            self.state_numbers.append(int(fields[0]))
            self.excitation_energies__eV.append(float(fields[1]))
            self.oscillator_strengths_length.append(float(fields[2]))
            self.oscillator_strengths_velocity.append(float(fields[3]))
            self.rotatory_strengths_length.append(float(fields[4]))
            self.rotatory_strengths_velocity.append(float(fields[5]))

        self.excitation_wavelengths__nm = [
            PHOTON_ENERGY_WAVELENGTH_PRODUCT__eV_nm / energy
            for energy in self.excitation_energies__eV]
        self.state_count = len(self.state_numbers)


# =====================================================================
# 端到端便捷函数
# =====================================================================

def std2_excited_states_from_Gaussian_output(
    gaussian_output_path: str,
    output_folder: str,
    *,
    method: str,
    energy_threshold_eV: float,
    fock_exchange_ax: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    rsh_functional: str | None = None,
    triplet: bool = False,
    cores: int = 1,
    generate_spectrum_files: bool = False,
) -> sTD2_result:
    """从 Gaussian 输出出发，在本机完成一次 std2 激发态计算并返回解析结果。

    三步串联（仅 Linux）：:func:`std2_input_from_Gaussian_output`
    （g2molden 转换，``molden_style`` 因而已知为 3）→
    :func:`run_std2_folder` → 用 :class:`sTD2_result` 解析
    ``<名>.std2_tda.dat``。方法只能是基于 Molden 的四种（sTDA /
    sTD-DFT / XsTDA / XsTD-DFT）；参数组合规则见
    :func:`build_std2_flags`。

    Args:
        gaussian_output_path:     Gaussian 输出文件，路由须含
                                  ``#P gfinput pop=full 6D 10F``。
        output_folder:            计算文件夹路径，不存在时自动创建。
        method:                   **必填**。基于 Molden 的方法之一。
        energy_threshold_eV:      **必填**。组态空间能量阈值（eV）。
        fock_exchange_ax:         基态泛函的 Fock 交换比例（``-ax``）。
        alpha, beta:              ``-al`` / ``-be``（成对；仅 sTDA /
                                  sTD-DFT 配 RSH 泛函时用）。
        rsh_functional:           RSH 泛函名（与上面三个参数互斥）。
        triplet:                  计算 singlet-triplet 激发（``-t``）。
        cores:                    OpenMP / MKL 线程数。
        generate_spectrum_files:  True 时随后运行 g_spec 生成
                                  ``spec.dat`` / ``rots.dat``（选项用
                                  std2 写出的头部默认值；要改先自行调
                                  :func:`set_tda_dat_g_spec_options` 再
                                  :func:`run_g_spec`）。

    Returns:
        :class:`sTD2_result`（各态激发能与振子 / 旋转强度）。
    """
    method = resolve_std2_method(method)
    if method in _XTB_GROUND_STATE_METHODS:
        raise ValueError(
            f"Method {method} runs from an xtb4stda wavefunction, not from "
            "a Gaussian output — prepare wfn.xtb and call run_std2_folder "
            "directly."
        )

    molden_path = std2_input_from_Gaussian_output(
        gaussian_output_path, output_folder)
    run_std2_folder(
        output_folder, method=method,
        energy_threshold_eV=energy_threshold_eV,
        molden_file=os.path.basename(molden_path), molden_style=3,
        fock_exchange_ax=fock_exchange_ax, alpha=alpha, beta=beta,
        rsh_functional=rsh_functional, triplet=triplet, cores=cores,
    )
    if generate_spectrum_files:
        run_g_spec(os.path.join(os.path.abspath(output_folder), "tda.dat"))

    log_stem = os.path.basename(molden_path)
    log_stem = os.path.splitext(log_stem)[0]
    if log_stem.lower().endswith(".molden"):
        log_stem = os.path.splitext(log_stem)[0]
    return sTD2_result(
        os.path.join(os.path.abspath(output_folder),
                     f"{log_stem}.std2_tda.dat"))


if __name__ == "__main__":
    pass
