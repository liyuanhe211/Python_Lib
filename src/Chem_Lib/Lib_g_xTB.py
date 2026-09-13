# -*- coding: utf-8 -*-
"""
Lib_g_xTB — g-xTB 几何优化的输入生成与结果解析
================================================

g-xTB（Grimme 组的通用半经验方法，逼近 ωB97M-V/def2-TZVPPD，支持
Z = 1–103 全部元素）通过一个修改版的 xtb 6.7.1 二进制以 ``--gxtb``
旗标调用，输入与输出和经典 xTB 完全同构，因此本模块整体是
:mod:`Chem_Lib.Lib_xTB` 的薄壳，只处理两者的差异。方法说明与本地
二进制的位置见 ``Chem_Lib_Manuals/g-xTB_20260514/g-xTB_Usage.md``。

支持范围（2026-08-12 用户裁定）：**只支持气相（无溶剂）、带电 / 不带电、
开壳层 / 闭壳层的几何优化**。溶剂化、MD、scan / pull 等其他任务类型一律
不支持——g-xTB 的溶剂模型截至快照 2026-05-14 仍在开发中（只含静电项，
上游明确警告 ``--gbe`` 在几何优化中不稳定、``--cosmo`` 的梯度不自洽）。

与经典 xTB 的两个关键差异：

1. **方法旗标**：``--gxtb``（替代 ``--gfn N`` / ``--gfnff``），g-xTB 自身
   不需要外部参数文件。
2. **开壳层判定**：g-xTB 只要看到 ``--uhf`` 旗标或 ``.UHF`` 文件就使用
   非限制性波函数——**即使未配对电子数为 0**（g-xTB README）。因此本
   模块的输入约定与 xTB 不同：闭壳层体系**不写 ``.UHF`` 文件**（保持
   限制性波函数）；只有开壳层（multiplicity > 1，或显式要求非限制性
   单重态）才写 ``.UHF``。

生成的输入文件夹内容：

- ``<名>.xyz``     — 分子坐标（同 xTB）
- ``.CHRG``        — 电荷（同 xTB）
- ``.UHF``         — **仅开壳层时存在**（见上）
- ``.gxtb_config`` — JSON，生成参数的记录（provenance），同时是 HPC
  分发器识别 g-xTB 输入文件夹的标记文件（区别于经典 xTB 文件夹）。
  提交函数不读取其中的参数。

集群作业提交见 :mod:`HPC_Lib.HPC_g-xTB`（模块文件名含连字符，需经
``importlib.import_module("HPC_Lib.HPC_g-xTB")`` 加载，或用
``HPC_Lib.HPC`` 中的同名薄壳函数）；**本地直接运行**（不经队列，
Windows / Linux 均可）用本模块的 :func:`run_gxtb_opt_folder` /
:func:`gxtb_optimize_gjf`。可执行文件（Windows / Linux 两版）随本包
分发，位于与本文件同目录的 ``Executable_g-xTB``——不进版本控制
（.gitignore 排除），由 ``A0_HPC_Sync_Python_Lib.py`` 同步上集群、
pip / uv 安装经 pyproject 的 package-data 带入；一律相对于 Chem_Lib
包定位，不写死绝对路径。
"""

from __future__ import annotations

__author__ = "LiYuanhe"

import json
import os

from .Lib_xTB import (
    xTB_input_from_Gaussian_gjf,
    xTB_opt_result,
    build_save_converged_gjf_lines,
    chem_lib_directory,
    ensure_executable_bit,
    resolve_opt_level,
    resolve_structure_xyz,
    save_converged_gjf,
    _execute_xtb_subprocess,
)

#: 标记 + provenance 文件名（HPC 分发器凭它识别 g-xTB 输入文件夹）
GXTB_CONFIG_FILE_NAME = ".gxtb_config"

#: g-xTB 输入文件夹的标记文件：两者有其一即被识别为 g-xTB 任务。
GXTB_MARKER_FILES = (".gxtb_config", "gxtb_task.toml")


def gxtb_input_from_Gaussian_gjf(
    gjf_path: str,
    output_folder: str,
    *,
    xyz_filename: str | None = None,
    force_unrestricted: bool = False,
) -> str:
    """从 Gaussian gjf 读取结构、电荷与自旋多重度，生成 g-xTB 输入文件夹。

    :func:`Chem_Lib.Lib_xTB.xTB_input_from_Gaussian_gjf` 的薄壳：坐标与
    ``.CHRG`` 的生成完全复用它，然后按 g-xTB 的约定修正两处——

    1. 闭壳层（multiplicity = 1 且未显式要求非限制性）时**删除** ``.UHF``：
       g-xTB 只要看到 ``.UHF`` 文件就用非限制性波函数（即使内容为 0），
       闭壳层体系应保持限制性。
    2. 把 xTB 的 ``.xtb_config`` 记录替换为 ``.gxtb_config``（方法固定为
       g-xTB、气相），它同时是 HPC 分发器识别 g-xTB 文件夹的标记。

    Args:
        gjf_path:            Gaussian 输入文件路径。
        output_folder:       g-xTB 计算文件夹路径，不存在时自动创建。
        xyz_filename:        输出的 XYZ 文件名（不含路径）。默认使用 gjf
                             文件的基名。
        force_unrestricted:  True 时即使 multiplicity = 1 也保留 ``.UHF``
                             （内容为 0），显式要求非限制性单重态计算
                             （例如双自由基单重态）。默认 False。

    Returns:
        生成的 XYZ 文件的完整路径。
    """
    # method 是占位值：xTB 生成函数要求一个合法的 GFN 方法名写进
    # .xtb_config，而该记录文件随即被替换成 .gxtb_config。
    xyz_path = xTB_input_from_Gaussian_gjf(
        gjf_path, output_folder,
        xyz_filename=xyz_filename, solvent=None, method="GFN2",
    )

    output_folder = os.path.abspath(output_folder)

    uhf_path = os.path.join(output_folder, ".UHF")
    with open(uhf_path) as f:
        unpaired_electrons = int(f.read().strip())

    unrestricted = force_unrestricted or unpaired_electrons > 0
    if not unrestricted:
        # 闭壳层：.UHF 文件的存在本身就会触发 g-xTB 的非限制性波函数，
        # 必须删除（这是与经典 xTB 输入约定的关键差异）。
        os.remove(uhf_path)

    xtb_config_path = os.path.join(output_folder, ".xtb_config")
    if os.path.isfile(xtb_config_path):
        os.remove(xtb_config_path)

    with open(os.path.join(output_folder, ".CHRG")) as f:
        charge = int(f.read().strip())

    config = {
        "method": "g-xTB",
        "task": "opt",
        "solvent": "none",
        "charge": charge,
        "unpaired_electrons": unpaired_electrons,
        "unrestricted": unrestricted,
    }
    with open(os.path.join(output_folder, GXTB_CONFIG_FILE_NAME), "w",
              newline="\n") as f:
        json.dump(config, f)

    return xyz_path


class gxtb_opt_result(xTB_opt_result):
    """g-xTB 几何优化结果。

    g-xTB 作业的产物命名与 xTB 的 opt 任务完全一致
    （``<结构名>.xtbopt_traj.xyz`` + ``<结构名>.xtb.log``，见
    :mod:`HPC_Lib.HPC_g-xTB` 模块文档），日志中的 SUMMARY /
    THERMODYNAMIC 表格式也与经典 xtb 相同，因此解析逻辑整体复用
    :class:`Chem_Lib.Lib_xTB.xTB_opt_result`，只把 method 标注改为 g-xTB。
    """

    def __init__(self, path, one_structure=None):
        super().__init__(path, one_structure)
        self.method = "g-xTB"


# =====================================================================
# 以下为 g-xTB 的"运行层"（2026-08-13 从 HPC_Lib.HPC_g-xTB 迁移进本模块）
# =====================================================================
#
# 与 :mod:`Chem_Lib.Lib_xTB` 的运行层同构：文件夹校验、可执行文件解析、
# 命令构造、本地直接运行都在这里；HPC_Lib.HPC_g-xTB 只保留 SLURM 侧的
# 壳并引用这里的实现。环境变量设置（OMP / XTBPATH / XTBHOME、Linux 的
# 栈限制解除）复用 Lib_xTB 的 :func:`~Chem_Lib.Lib_xTB.xtb_runtime_environment`
# 与 :func:`~Chem_Lib.Lib_xTB._execute_xtb_subprocess`——g-xTB 自身不
# 需要外部参数文件，XTBPATH 指向其发行版的 share/xtb 无害。
# =====================================================================

def is_strict_gxtb_input_folder(folder: str) -> bool:
    """检查文件夹是否满足**运行 / 提交时**的 g-xTB 输入约定。

    要求：至少一个 ``.xyz``、``.CHRG``、标记文件（``.gxtb_config`` 或
    ``gxtb_task.toml``）之一。``.UHF`` 可选——不存在即闭壳层（限制性），
    这一点与经典 xTB 的严格约定（.UHF 必须存在）不同。HPC 分发器的
    宽松检测（``.CHRG`` 可由 TOML 落盘）见 ``HPC.py`` 的
    ``_is_gxtb_input_folder``。
    """
    if not os.path.isdir(folder):
        return False
    contents = os.listdir(folder)
    has_xyz = any(f.lower().endswith(".xyz") for f in contents)
    has_marker = any(marker in contents for marker in GXTB_MARKER_FILES)
    return has_xyz and ".CHRG" in contents and has_marker


def validate_gxtb_folder(gxtb_folder: str,
                         structure: str | None = None) -> tuple[str, bool]:
    """校验 g-xTB 任务文件夹的输入完备性。

    返回 ``(输入 .xyz 文件名, 是否开壳层/非限制性)``。任何缺失直接报错
    （fail early）。同时拒绝混有经典 xTB 任务描述文件的歧义文件夹。
    """
    contents = os.listdir(gxtb_folder)
    if ".CHRG" not in contents:
        raise FileNotFoundError(
            f"{gxtb_folder} is missing .CHRG. g-xTB input folders must "
            "contain .xyz + .CHRG (+ .UHF only for open-shell systems)."
        )
    if not any(marker in contents for marker in GXTB_MARKER_FILES):
        raise FileNotFoundError(
            f"{gxtb_folder} has no g-xTB marker file "
            f"({' / '.join(GXTB_MARKER_FILES)}). 由 "
            "Chem_Lib.Lib_g_xTB.gxtb_input_from_Gaussian_gjf 生成的文件夹"
            "自带 .gxtb_config；手工准备的文件夹请放一个 gxtb_task.toml。"
        )
    if "xtb_task.toml" in contents:
        raise ValueError(
            f"{gxtb_folder} contains BOTH a g-xTB marker and a classic-xTB "
            "task file (xtb_task.toml) — ambiguous; remove one."
        )
    unrestricted = ".UHF" in contents
    xyz_file = resolve_structure_xyz(gxtb_folder, structure)
    return xyz_file, unrestricted


def resolve_gxtb_executable() -> str:
    """解析 g-xTB 二进制（修改版 xtb）的路径（按当前平台选版本）。

    一律相对于 Chem_Lib 包定位：Windows 取
    ``Executable_g-xTB/Windows/bin/xtb.exe``（所需 DLL 在同一目录），
    其他平台取 ``Executable_g-xTB/Linux/bin/xtb``（静态链接单文件，并
    确保带可执行位——SFTP 同步 + harden_permissions 会把不带 x 位的
    文件收成 600）。

    找不到时直接报错（fail early），**不回落**到 PATH 上的经典 ``xtb``
    ——它不认识 ``--gxtb`` 旗标。
    """
    import platform

    if platform.system() == "Windows":
        packaged = os.path.join(chem_lib_directory(), "Executable_g-xTB",
                                "Windows", "bin", "xtb.exe")
    else:
        packaged = os.path.join(chem_lib_directory(), "Executable_g-xTB",
                                "Linux", "bin", "xtb")
    if not os.path.isfile(packaged):
        raise FileNotFoundError(
            f"g-xTB executable not found: {packaged}\n"
            "该二进制随 Chem_Lib 包分发（不进 git，由 "
            "A0_HPC_Sync_Python_Lib.py 同步上集群，pip / uv 安装经 "
            "pyproject 的 package-data 带入）。它缺失说明本机的 "
            "Executable_g-xTB 文件夹还没同步 / 安装到位——先补齐它，"
            "不要改用 PATH 上的经典 xtb（不认识 --gxtb 旗标）。"
        )
    if not packaged.endswith(".exe"):
        ensure_executable_bit(packaged)
    return packaged


def build_gxtb_opt_command_lines(
    xyz_file: str,
    opt_level: str | None,
    unrestricted: bool,
    gxtb_exe: str,
) -> list[str]:
    """生成一个 g-xTB 几何优化任务的 shell 行（xtb 命令 + 产物复制 +
    收敛 gjf 生成）。

    脚本模式：供 HPC_Lib 写进 SLURM 作业脚本，需在任务文件夹内执行
    （调用方负责 cd）。产物命名与经典 xTB 的 ``opt`` 任务逐字一致，
    使结果解析与回收流程无需区分两种方法。闭壳层时不携带 ``--uhf``
    （g-xTB 见到该旗标即用非限制性波函数）。末尾附「收敛时生成
    ``<结构名>.g_xtb_converged.gjf``」的 shell 行（见
    :func:`Chem_Lib.Lib_xTB.build_save_converged_gjf_lines`），使远程
    任务与本地直接运行（:func:`run_gxtb_opt_folder`）留下同一个
    收敛 gjf。
    """
    xyz_stem = os.path.splitext(xyz_file)[0]
    level_suffix = f" {opt_level}" if opt_level else ""
    flags = ["--gxtb", "--chrg $(cat .CHRG)"]
    if unrestricted:
        flags.append("--uhf $(cat .UHF)")
    flags_str = " ".join(flags)
    return [
        f"{gxtb_exe} {xyz_file} --opt{level_suffix} {flags_str} "
        f"&> {xyz_stem}.xtb.log",
        f"if [ -f xtbopt.log ]; then "
        f"cp xtbopt.log {xyz_stem}.xtbopt_traj.xyz; fi",
    ] + build_save_converged_gjf_lines(
        xyz_stem, converged_gjf_suffix=".g_xtb_converged.gjf")


def run_gxtb_opt_folder(
    gxtb_folder: str,
    *,
    opt_level: str | None = None,
    structure: str | None = None,
    cores: int = 1,
) -> str:
    """在本机直接运行一个 g-xTB 气相几何优化（不经 HPC 队列）。

    输入文件夹约定与产物命名与 HPC 提交
    （``HPC_Lib.HPC_g-xTB.submit_gxTB_folder``）完全一致——同一个
    文件夹既可以本地跑，也可以提交集群。开壳层与否由 ``.UHF`` 的
    存在性决定（见模块 docstring）。

    Args:
        gxtb_folder: g-xTB 计算文件夹（``.xyz`` + ``.CHRG`` + 标记文件，
                     开壳层另含 ``.UHF``）。
        opt_level:   优化精度等级；不填时让 xtb 用自身默认（normal）。
        structure:   输入结构 ``.xyz`` 文件名；不填时自动选择。
        cores:       OpenMP 线程数（libgomp 兼容环境：
                     ``OMP_NUM_THREADS=<cores>`` + ``OMP_MAX_ACTIVE_LEVELS=1``）。
                     默认 1（单核）——与经典 xTB 相同，g-xTB 的 OpenMP
                     并行效率低，且单任务本来就运行迅速，一般不必要
                     多核并行；大批量吞吐靠多个单核任务并行（如打包
                     提交），不靠单任务多核（2026-08-13 用户裁定）。

    Returns:
        日志文件（``<结构名>.xtb.log``）的完整路径；优化轨迹产物为
        同名 ``<结构名>.xtbopt_traj.xyz``。运行结束后还会检查优化是否
        收敛，收敛时把最后一帧几何另存为
        ``<结构名>.g_xtb_converged.gjf``（后缀与经典 xTB 的
        ``.xtb_converged.gjf`` 区分方法，见
        :func:`Chem_Lib.Lib_xTB.save_converged_gjf`），未收敛则不生成
        该文件。

    Raises:
        RuntimeError: xtb 退出码非零（附日志末尾内容）。
    """
    import shutil

    opt_level = resolve_opt_level(opt_level)
    gxtb_folder = os.path.abspath(gxtb_folder)
    if not os.path.isdir(gxtb_folder):
        raise FileNotFoundError(f"g-xTB calculation folder not found: {gxtb_folder}")
    xyz_file, unrestricted = validate_gxtb_folder(gxtb_folder, structure)

    with open(os.path.join(gxtb_folder, ".CHRG")) as f:
        charge = int(f.read().strip())

    gxtb_executable = resolve_gxtb_executable()
    xyz_stem = os.path.splitext(xyz_file)[0]
    argument_list = [gxtb_executable, xyz_file, "--opt"]
    if opt_level:
        argument_list.append(opt_level)
    argument_list += ["--gxtb", "--chrg", str(charge)]
    unpaired_electrons = 0
    if unrestricted:
        with open(os.path.join(gxtb_folder, ".UHF")) as f:
            unpaired_electrons = int(f.read().strip())
        argument_list += ["--uhf", str(unpaired_electrons)]

    log_path = os.path.join(gxtb_folder, f"{xyz_stem}.xtb.log")
    # g-xTB 的 Windows 版是 GNU 工具链（libgomp）构建：手册推荐的
    # OMP_NUM_THREADS=<n>,1 嵌套写法会让它直接段错误，OMP_STACKSIZE=4G
    # 也超出其解析上限——启用 libgomp 兼容环境（2026-08-13 本机实测）。
    _execute_xtb_subprocess(argument_list, gxtb_folder, log_path,
                            cores, gxtb_executable,
                            gnu_openmp_workarounds=True)

    optimization_trajectory = os.path.join(gxtb_folder, "xtbopt.log")
    if os.path.isfile(optimization_trajectory):
        shutil.copy(optimization_trajectory,
                    os.path.join(gxtb_folder, f"{xyz_stem}.xtbopt_traj.xyz"))

    save_converged_gjf(log_path, charge=charge,
                       multiplicity=unpaired_electrons + 1,
                       converged_gjf_suffix=".g_xtb_converged.gjf")

    return log_path


def write_converged_gjf_for_folder(gxtb_folder: str) -> str | None:
    """为一个已经跑完的 g-xTB 优化文件夹补写收敛几何 gjf（如已收敛）。

    本地直接运行（:func:`run_gxtb_opt_folder` / :func:`gxtb_optimize_gjf`）
    在运行结束时就地生成 ``<结构名>.g_xtb_converged.gjf``；而 HPC 提交的
    作业脚本是纯 shell（见 :func:`build_gxtb_opt_command_lines`），只产出
    日志与轨迹——产物下载回本地后调用本函数，把文件夹补齐到与本地运行
    完全一致的终态。此后「``.g_xtb_converged.gjf`` 存在」在两种运行方式下
    都是「优化已收敛」的可靠判据，下游一律只看该文件，不必再读日志
    （:func:`~Chem_Lib.Lib_xTB.save_converged_gjf` 是本层的实现细节，
    库外不应直接调用）。

    电荷 / 自旋多重度读取文件夹自身的 ``.CHRG`` / ``.UHF``——与运行时
    实际使用的值同源。幂等：已收敛时重复调用只是重写同一文件；未收敛时
    不生成，且陈旧的同名收敛 gjf 会被顺带删除。

    Args:
        gxtb_folder: 已包含 ``<结构名>.xtb.log`` 的 g-xTB 计算文件夹。

    Returns:
        收敛时返回 ``<结构名>.g_xtb_converged.gjf`` 的完整路径；未收敛
        （含异常终止）返回 ``None``。

    Raises:
        FileNotFoundError: 文件夹里没有 ``*.xtb.log``（任务未跑过 / 产物
                           未下载），或缺 ``.CHRG``。
        ValueError:        文件夹里有多个 ``*.xtb.log``，无法判定归属。
    """
    gxtb_folder = os.path.abspath(gxtb_folder)
    log_files = sorted(f for f in os.listdir(gxtb_folder)
                       if f.endswith(".xtb.log"))
    if not log_files:
        raise FileNotFoundError(
            f"{gxtb_folder} 中没有 *.xtb.log——该 g-xTB 任务尚未运行，"
            "或产物还没有下载回来。")
    if len(log_files) > 1:
        raise ValueError(
            f"{gxtb_folder} 中有多个 *.xtb.log：{log_files}，无法判定"
            "归属；清理文件夹后重试。")
    charge_file = os.path.join(gxtb_folder, ".CHRG")
    if not os.path.isfile(charge_file):
        raise FileNotFoundError(
            f"{gxtb_folder} 缺少 .CHRG——不是完整的 g-xTB 任务文件夹。")
    with open(charge_file) as f:
        charge = int(f.read().strip())
    unpaired_electrons = 0
    uhf_file = os.path.join(gxtb_folder, ".UHF")
    if os.path.isfile(uhf_file):
        with open(uhf_file) as f:
            unpaired_electrons = int(f.read().strip())
    return save_converged_gjf(
        os.path.join(gxtb_folder, log_files[0]),
        charge=charge, multiplicity=unpaired_electrons + 1,
        converged_gjf_suffix=".g_xtb_converged.gjf")


def gxtb_optimize_gjf(
    gjf_path: str,
    output_folder: str,
    *,
    opt_level: str | None = None,
    force_unrestricted: bool = False,
    cores: int = 1,
    xyz_filename: str | None = None,
) -> gxtb_opt_result:
    """从 Gaussian gjf 出发，在本机完成一次 g-xTB 几何优化并返回解析结果。

    三步串联：:func:`gxtb_input_from_Gaussian_gjf` 生成输入文件夹 →
    :func:`run_gxtb_opt_folder` 本地运行 → 用 :class:`gxtb_opt_result`
    解析 ``<结构名>.xtbopt_traj.xyz``（及同名 ``.xtb.log`` 里的能量）。

    Args:
        gjf_path:            Gaussian 输入文件（含电荷 / 自旋多重度 / 坐标）。
        output_folder:       计算文件夹路径，不存在时自动创建。
        opt_level:           优化精度等级；不填用 xtb 自身默认（normal）。
        force_unrestricted:  True 时即使闭壳层也用非限制性波函数
                             （见 :func:`gxtb_input_from_Gaussian_gjf`）。
        cores:               OpenMP 线程数，默认 1（单核）。g-xTB 并行
                             效率低、单任务本来就快，一般不必要多核，
                             见 :func:`run_gxtb_opt_folder`。
        xyz_filename:        输出 XYZ 文件名；默认用 gjf 基名。

    Returns:
        :class:`gxtb_opt_result`（含优化后坐标与能量，单位 kJ/mol）。
        优化收敛时，计算文件夹里还会多一个
        ``<结构名>.g_xtb_converged.gjf``（最后一帧几何，见
        :func:`Chem_Lib.Lib_xTB.save_converged_gjf`）。
    """
    xyz_path = gxtb_input_from_Gaussian_gjf(
        gjf_path, output_folder, xyz_filename=xyz_filename,
        force_unrestricted=force_unrestricted,
    )
    run_gxtb_opt_folder(
        output_folder, opt_level=opt_level,
        structure=os.path.basename(xyz_path), cores=cores,
    )
    trajectory_path = os.path.splitext(xyz_path)[0] + ".xtbopt_traj.xyz"
    if not os.path.isfile(trajectory_path):
        raise RuntimeError(
            f"g-xTB optimization finished but the trajectory product is "
            f"missing: {trajectory_path}"
        )
    return gxtb_opt_result(trajectory_path)


if __name__ == "__main__":
    pass
