"""从 SMILES 生成三维初始结构（RDKit ETKDG 构象嵌入 + 可选 MMFF 力场优化）。

本模块把下游使用方项目中的 Monomer 结构生成方法（原 ``embed_only()``）
提炼为跨项目通用函数（2026-08-14 迁移）。
与原方法保持一致的要点：

- ETKDGv3 嵌入，固定随机种子（默认 0xF00D）、``numThreads=0``、
  先用确定性坐标（``useRandomCoords=False``），一个构象都嵌不出来时
  整体退回随机坐标模式再试一次；
- 每个构象用 MMFF 力场优化（分子缺 MMFF 参数时退回 UFF），按力场
  能量升序排序；
- 输出为最小 Gaussian gjf（``#p`` 路由 + 标题 + 电荷/自旋多重度 + 笛卡尔
  坐标），可直接交给 :func:`Chem_Lib.Lib_xTB.xTB_input_from_Gaussian_gjf`
  或 :func:`Chem_Lib.Lib_g_xTB.gxtb_input_from_Gaussian_gjf` 转成
  xTB / g-xTB 输入。

与原方法不同的扩展（2026-08-14 用户要求）：

- 可以要求任意数量的构象，全部写盘（文件名 ``<主干>_01.gjf``、
  ``<主干>_02.gjf`` …，按力场能量升序编号），而不是只保留最优的一个；
- 力场优化可以关闭（``optimize_with_mmff=False``），此时保留 ETKDG
  原始几何，仅用未优化的力场单点能量排序。

分子构建与 MMFF / UFF 力场部分已于 2026-09-02 抽出为
:mod:`Chem_Lib.Lib_MMFF`（``*`` 连接位的 H 封端、寡聚体头尾拼接、ETKDGv3
嵌入、力场构造与能量排序），本模块只保留「把排好序的构象写成 Gaussian
gjf 文件」这一层，力场相关的行为一律以 :mod:`Chem_Lib.Lib_MMFF` 为准。
"""
from __future__ import annotations

import os

from Chem_Lib.Lib_MMFF import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RANDOM_SEED,
    optimized_conformers_from_smiles,
)

#: 打印信息前缀。历史沿用，改动会让既有日志的 grep 失效。
_LOG_PREFIX = "[structures_from_smiles]"

__all__ = ["DEFAULT_RANDOM_SEED", "gjf_structures_from_smiles"]


def _write_minimal_gjf(gjf_path: str, title: str, charge: int,
                       multiplicity: int, elements: list[str],
                       coordinates: list[tuple[float, float, float]]) -> None:
    """写最小 gjf：``#p`` 路由 + 标题 + 电荷/多重度 + 坐标。

    该格式为最小可解析的 gjf，
    经 :class:`Chem_Lib.Lib_Gaussian.Gaussian_Input` 解析无碍。
    """
    geometry_lines = [
        f"{element:<2} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
        for element, (x, y, z) in zip(elements, coordinates)
    ]
    body = "\n".join(
        ["#p", "", title, "", f"{charge} {multiplicity}", *geometry_lines,
         "", ""])
    with open(gjf_path, "w", encoding="utf-8", newline="\n") as gjf_file:
        gjf_file.write(body)


def gjf_structures_from_smiles(
    smiles: str,
    number_of_structures: int,
    output_folder: str,
    *,
    file_stem: str | None = None,
    optimize_with_mmff: bool = True,
    charge: int = 0,
    multiplicity: int = 1,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> list[str]:
    """由 SMILES 生成若干三维构象并逐个写成 Gaussian gjf 文件。

    构象的生成、力场优化与能量排序全部交给
    :func:`Chem_Lib.Lib_MMFF.optimized_conformers_from_smiles`；本函数只负责
    输出文件夹、文件命名与 gjf 写盘。

    Args:
        smiles:               输入 SMILES。含 ``*`` 连接占位原子时自动以 H
                              封端（与聚合物单体 Monomer 生成约定一致）。
        number_of_structures: 希望生成的构象数量。实际嵌入数量可能少于
                              要求（RDKit 对刚性小分子可能给不满），此时
                              打印明确提示并按实际数量输出，绝不静默。
        output_folder:        输出文件夹，不存在时自动创建。
        file_stem:            输出文件名主干；默认取 output_folder 的
                              文件夹名。文件命名 ``<主干>_01.gjf`` 起，
                              按力场能量升序编号（能量最低的是 ``_01``）。
        optimize_with_mmff:   True（默认）时对每个构象做 MMFF 力场优化
                              （最多 2000 步；缺 MMFF 参数的分子退回 UFF）；
                              False 时保留 ETKDG 原始几何，仅做力场单点
                              能量用于排序。
        charge:               写入 gjf 的电荷（默认 0）。
        multiplicity:         写入 gjf 的自旋多重度（默认 1）。
        random_seed:          ETKDG 随机种子，默认 0xF00D（与既有数据集
                              生成时的历史设定一致，保证可复现）。

    Returns:
        写出的 gjf 文件完整路径列表，按能量升序（与文件编号一致）。

    Raises:
        ValueError:   SMILES 不可解析，或 number_of_structures < 1。
        RuntimeError: 连随机坐标模式也嵌不出任何构象，或某个构象拿不到力场。
    """
    if number_of_structures < 1:
        raise ValueError(
            f"number_of_structures must be >= 1, got {number_of_structures}")

    ensemble = optimized_conformers_from_smiles(
        smiles,
        number_of_structures,
        optimize=optimize_with_mmff,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        random_seed=random_seed,
        log_prefix=_LOG_PREFIX,
    )
    ranked = ensemble.ranked_conformers
    elements = ensemble.elements

    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    if file_stem is None:
        file_stem = os.path.basename(os.path.normpath(output_folder))

    index_width = max(2, len(str(len(ranked))))
    written_paths: list[str] = []
    for rank, conformer in enumerate(ranked, start=1):
        coordinates = ensemble.coordinates(conformer.conformer_id)
        optimization_note = (
            f"{conformer.force_field_name} optimized"
            if conformer.optimized
            else f"unoptimized ({conformer.force_field_name} "
                 f"single-point for ranking)")
        title = (f"{file_stem} conformer {rank}/{len(ranked)}; "
                 f"RDKit ETKDGv3 seed {random_seed}; {optimization_note}; "
                 f"force-field energy = {conformer.energy_kcal_mol:.4f} "
                 f"kcal/mol")
        gjf_path = os.path.join(
            output_folder, f"{file_stem}_{rank:0{index_width}d}.gjf")
        _write_minimal_gjf(gjf_path, title, charge, multiplicity,
                           elements, coordinates)
        written_paths.append(gjf_path)
        print(f"{_LOG_PREFIX} {os.path.basename(gjf_path)}: "
              f"{conformer.energy_kcal_mol:.4f} kcal/mol "
              f"({optimization_note})")

    print(f"{_LOG_PREFIX} done: {len(written_paths)} written / "
          f"{number_of_structures} requested -> {output_folder}")
    return written_paths


if __name__ == "__main__":
    pass
