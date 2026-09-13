# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

# import sys
# import pathlib
# parent_path = str(pathlib.Path(__file__).parent.resolve())
# sys.path.insert(0,parent_path)

from Python_Lib.My_Lib_Stock import *
from .Lib_Coordinates import *
from .Lib_XYZ import *
from .Lib_Constants import *


class xTB_Coord_file():
    def __init__(self, path):
        """形如下面的例子，而且单位是bohr
        $coord
        33.06453805899918	25.545317796282273	20.95517303173355	c
        32.33510377364891	25.70405479102171	23.470398507902487	c
        33.82042851013935	26.053654124674043	24.84233967672191	h
        35.03363268421934	25.832556167715538	20.44872642946963	h
        31.244731797879204	25.035091741762653	19.082454439033764	c
        31.798421553339388	24.98595886243854	17.10769063543005	h
        $end"""
        self.filename = path
        with open(self.filename) as input_file_lines:
            input_file_lines = input_file_lines.readlines()
        coordinates = []
        for line in input_file_lines:
            if line.strip().startswith("$coord"):
                continue
            if line.strip().startswith("$end"):
                break
            line = line.strip().split()
            assert len(line) == 4
            element = line[-1]
            element = element[0].upper() + element[1:].lower()
            line = [float(x) * bohr__A for x in line[:3]]
            std_coordinate_line = "{}\t{}\t{}\t{}".format(element, *line)
            coordinates.append(std_coordinate_line)
        self.coordinate_object = Coordinates(coordinates)
        # open_with_gview(self.coordinate_object.gjf_file())

class xTB_pull_result:
    def __init__(self, path):
        self.filename = path
        self.normal_termination = True
        self.method = 'GFN2-xTB-D4'
        self.xyz_object = XYZ_file(path, last_only=True)
        self.electronic_energy = float(re.findall(r"SCF done {3}}(-*\d+\.\d+)", self.xyz_object.titles[-1])[0]) * Hartree__KJ_mol
        self.coordinates = str(self.xyz_object.last_coordinate) + '\n'
        self.coordinate_object = self.xyz_object.last_coordinate


class xTB_opt_result:
    def __init__(self, path, one_structure=None):
        """
        :param path: .xtbopt_traj.xyz file, but .xtb.log is also needed.
        :param one_structrue: list of lines, one standard xyz file structure,
                              [atomnumber,comment_line(with energy in kJ/mol in the format 1.23454 kJ/mol),coordinates]
        """

        self.filename = path
        self.normal_termination = True
        self.method = 'GFN2-xTB-D4'
        if one_structure:
            self.coordinates = [x.strip()[0].upper() + x.strip()[1:] for x in one_structure[2:]]  # xTB会把元素的第一个字母小写，造成std_coordainte认不出来
            self.coordinates = [std_coordinate(x) for x in self.coordinates]
            self.coordinates = '\n'.join(self.coordinates) + '\n'
            self.coordinate_object = Coordinates(self.coordinates.splitlines())
        else:
            xyz_object = XYZ_file(path, last_only=True, equal_atom_count=True)
            self.coordinate_object = xyz_object.coordinates[-1]
            self.coordinates = str(self.coordinate_object)
            self.last_title = xyz_object.titles[-1]

        self.H = 0
        self.G = 0
        self.electronic_energy = 0

        self.output_filename = None

        if self.filename.endswith('.last.xtbopt_traj.xyz'):
            self.output_filename = rreplace(self.filename, '.last.xtbopt_traj.xyz', '.xtb.log')
        elif self.filename.endswith('.xtbopt_traj.xyz'):
            self.output_filename = rreplace(self.filename, '.xtbopt_traj.xyz', '.xtb.log')
        elif self.filename.endswith('.xtbopt_final.xyz'):
            self.output_filename = rreplace(self.filename, '.xtbopt_final.xyz', '.xtb.log')

        if (self.output_filename is not None) and os.path.isfile(self.output_filename):
            with open(self.output_filename, encoding='utf-8', errors='ignore') as xTB_output_file_object:
                xTB_output_content = xTB_output_file_object.readlines()

            thermo_table_head_line_count = [count for count, x in enumerate(xTB_output_content) if "::                  THERMODYNAMIC                  ::" in x]
            if len(thermo_table_head_line_count) == 1:
                thermo_table_head_line_count = thermo_table_head_line_count[-1]

                free_energy_line = xTB_output_content[thermo_table_head_line_count + 2]
                if "total free energy" in free_energy_line:
                    self.G = re.findall(r'-*\d+\.\d+', free_energy_line)
                    if self.G:
                        # note that this is actually the Gibbs free energy, not the H, This is to make the Gaussian Extract work
                        self.G = float(self.G[0]) * Hartree__KJ_mol

                for H_line in xTB_output_content[thermo_table_head_line_count + 2:]:
                    if "TOTAL ENTHALPY" in H_line:
                        self.H = re.findall(r'-*\d+\.\d+', free_energy_line)
                        if self.H:
                            # note that this is actually the Gibbs free energy, not the H, This is to make the Gaussian Extract work
                            self.H = float(self.H[0]) * Hartree__KJ_mol
                            break

            summary_table_head_line_count = [count for count, x in enumerate(xTB_output_content) if
                                             "::                     SUMMARY                     ::" in x]
            if summary_table_head_line_count:
                summary_table_head_line_count = summary_table_head_line_count[-1]

                electronic_energy_line = xTB_output_content[summary_table_head_line_count + 2]
                if "total energy" in electronic_energy_line:
                    self.electronic_energy = re.findall(r'-*\d+\.\d+', electronic_energy_line)
                    if self.electronic_energy:
                        # note that this is actually the Gibbs free energy, not the H, This is to make the Gaussian Extract work
                        self.electronic_energy = float(self.electronic_energy[0]) * Hartree__KJ_mol
        else:
            # no log file mode, only electronic energy
            re_ret = re.findall(r'energy:\s*(-*\d+\.\d+)', self.last_title)
            if re_ret:
                self.electronic_energy = float(re_ret[0]) * Hartree__KJ_mol


def xTB_input_from_Gaussian_gjf(
    gjf_path: str,
    output_folder: str,
    *,
    xyz_filename: str | None = None,
    solvent: str | None = None,
    method: str = "GFN2",
) -> str:
    """从 Gaussian gjf 文件中读取结构、电荷和自旋多重度，在 output_folder 中创建 xTB 所需的全部输入文件。

    创建的文件：
        - <name>.xyz    — XYZ 格式的分子坐标
        - .CHRG          — 电荷
        - .UHF           — 未配对电子数（= multiplicity - 1）
        - .xtb_config    — JSON 格式的计算参数（method、solvent），仅作
                           生成时参数的**记录**（provenance）。提交函数
                           不读取它：solvent / method 属于关键参数，必须
                           在提交时显式给出（无默认值，fail early），见
                           :mod:`HPC_Lib.HPC_xTB`。

    Args:
        gjf_path:       Gaussian 输入文件路径。
        output_folder:  xTB 计算文件夹的路径，不存在时自动创建。
        xyz_filename:   输出的 XYZ 文件名（不含路径）。默认使用 gjf 文件的基名。
        solvent:        溶剂名称（用于 ALPB 隐式溶剂模型），如 ``"water"``、
                        ``"THF"``、``"CH2Cl2"`` 等。为 ``None`` 时表示气相计算。
        method:         xTB 方法。``"GFN2"``（默认）、``"GFN1"``、``"GFN0"`` 或
                        ``"GFNFF"``。不区分大小写。

    Returns:
        生成的 XYZ 文件的完整路径。
    """
    import json
    from .Lib_Gaussian import Gaussian_Input

    gjf_path = os.path.abspath(gjf_path)
    if not os.path.isfile(gjf_path):
        raise FileNotFoundError(f"Gaussian input file not found: {gjf_path}")

    gaussian_input = Gaussian_Input(gjf_path)
    step = gaussian_input.steps[0]

    if step.coordinate is None or step.coordinate.is_fault:
        raise ValueError(f"Cannot extract valid coordinates from: {gjf_path}")

    charge = step.charge
    uhf = step.multiplet - 1

    os.makedirs(output_folder, exist_ok=True)

    if xyz_filename is None:
        base = os.path.splitext(os.path.basename(gjf_path))[0]
        xyz_filename = base + ".xyz"

    xyz_path = os.path.join(output_folder, xyz_filename)
    step.coordinate.xyz_file(
        title=f"from {os.path.basename(gjf_path)}  charge={charge} uhf={uhf}",
        filename=xyz_path,
    )

    with open(os.path.join(output_folder, ".CHRG"), "w") as f:
        f.write(str(charge))

    with open(os.path.join(output_folder, ".UHF"), "w") as f:
        f.write(str(uhf))

    config = {"method": method.upper()}
    if solvent:
        config["solvent"] = solvent
    with open(os.path.join(output_folder, ".xtb_config"), "w") as f:
        json.dump(config, f)

    return xyz_path


# =====================================================================
# Pull-and-optimize input generator for hard-to-build oligomers
# =====================================================================
#
# 当 RDKit ETKDG 难以一次性嵌入大体积 dimer / trimer 时，我们改用
# xTB 的 distance-restraint scan：把 N 个 monomer 先分别用 RDKit
# 嵌入（单体很小，ETKDG 不会卡），再在空间里把它们沿 x 轴拉远
# （8 Å 起步），然后让 xTB 一边优化一边把每对相邻 monomer 的接合
# 原子从远距离拉到正常成键距离（约 1.55 Å）。该过程产生的最后
# 一帧 (在 HPC 端) 再读出来做一次无约束 opt，得到平衡构型。
#
# 参考：https://xtb-docs.readthedocs.io/en/latest/scan.html
#
# 本函数只负责构造 input 文件夹；后续 sbatch 提交由
# HPC_Lib.HPC_xTB.submit_xTB_pull_and_optimize 完成。
# =====================================================================
def xTB_pull_input_from_oligomer_smiles(
    units_smiles: list[str],
    output_folder: str,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    initial_separation_A: float = 8.0,
    target_bond_A: float = 1.55,
    scan_steps: int = 30,
    force_constant: float = 1.0,
    solvent: str,
    method: str = "GFN2",
) -> str:
    """构建 N-mer oligomer 的 xTB pull-and-opt 输入文件夹。

    每个 monomer 是带两个 ``*`` 通配标记的 SMILES（同 ``oligomer_builder``
    约定）。函数会：

    1. 把每个 monomer 的 ``*`` 临时替换成 ``H``，用 RDKit ETKDG 生成单体
       3D 结构（单体很小，ETKDG 不会卡）。记录每个 monomer 上两个
       attachment 原子（即原来与 ``*`` 相连的 C/N/S 等重原子）的索引。
    2. 把 N 个 monomer 沿 x 轴排列：第 i 个 monomer 的几何中心放在
       ``x = i * initial_separation_A``。
    3. 删除中间 monomer 两端 attach 原子上的 H、两端 monomer 内侧 attach
       原子上的 H，外侧 attach 原子上的 H 保留作为 cap。
    4. 合并所有原子写入 ``oligomer.xyz``。
    5. 计算 N-1 对相邻 monomer 的 attach 原子（1-indexed for xTB），写入
       ``scan.inp``：每对距离从初始（约 8 Å）扫到 ``target_bond_A``
       (1.55 Å)，共 ``scan_steps`` 步。
    6. 同时写 ``.CHRG`` / ``.UHF`` / ``.xtb_config``（与
       :func:`xTB_input_from_Gaussian_gjf` 一致）。

    Args:
        units_smiles:           monomer SMILES 有序列表，每个恰好两个
                                ``*``。例：dimer = ``[A, A]``、
                                heterotrimer = ``[A, B, A]``。
        output_folder:          xTB 计算文件夹路径，不存在时自动创建。
        charge:                 全分子总电荷。Cation 时传 ``+1``。
        multiplicity:           自旋多重度。Cation 时传 ``2``。
        initial_separation_A:   monomer 中心沿 x 轴的初始间距（Å）。
                                默认 8.0，对绝大多数有机片段都足够远以
                                避开 vdW 接触，又不至于太远导致初始
                                pair distance 远超 scan 起点。
        target_bond_A:          scan 终点的成键距离（Å）。默认 1.55，
                                即典型 C-C 单键。
        scan_steps:             从初始距离扫到 ``target_bond_A`` 的步数。
        force_constant:         distance restraint 的力常数（Hartree/Bohr²）。
        solvent:                **必填、无默认值**。ALPB 溶剂名，气相必须
                                显式写 ``"none"``。生成时刻即经
                                :func:`resolve_solvent_flag` 校验（非法值
                                直接报错），规范化名称写进 ``.xtb_config``；
                                本函数不直接调 xTB，提交时仍须显式提供。
        method:                 ``GFN2``（默认）/ ``GFN1`` / ``GFN0`` /
                                ``GFNFF``。写进 ``.xtb_config``。

    Returns:
        生成的 ``oligomer.xyz`` 的完整路径。

    Raises:
        ValueError: SMILES 不可解析、``*`` 数目不为 2、或单体 ETKDG 失败。
    """
    import json
    import math
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if not units_smiles:
        raise ValueError("units_smiles must contain at least one monomer SMILES")

    solvent_name, _ = resolve_solvent_flag(solvent, method.upper())

    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # ---------- 单体 3D 嵌入 ----------
    # per_monomer_atoms: list of (elements:list[str], coords:list[(x,y,z)])
    # per_monomer_attach: list of (head_attach_local_idx, tail_attach_local_idx)
    # per_monomer_remove_h: list of set[int]  — local indices of H atoms to drop later
    per_monomer_atoms: list[tuple[list[str], list[tuple[float, float, float]]]] = []
    per_monomer_attach: list[tuple[int, int]] = []
    per_monomer_remove_h: list[set[int]] = []

    for unit_count, smiles in enumerate(units_smiles):
        m_raw = Chem.MolFromSmiles(smiles)
        if m_raw is None:
            raise ValueError(f"unparsable SMILES (unit {unit_count}): {smiles!r}")

        # 收集 attach atoms (the heavy atom each '*' is bonded to)
        dummies = [a for a in m_raw.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummies) != 2:
            raise ValueError(
                f"monomer must have exactly two '*' (unit {unit_count}): {smiles!r}"
            )
        attach_atomic_indices_in_raw = []
        for dummy_atom in dummies:
            neighbors = dummy_atom.GetNeighbors()
            if len(neighbors) != 1:
                raise ValueError(
                    f"attachment '*' must have exactly one neighbor (unit {unit_count})"
                )
            attach_atomic_indices_in_raw.append(neighbors[0].GetIdx())
        head_attach_raw, tail_attach_raw = attach_atomic_indices_in_raw

        # 把 '*' 改成 H，再加显式 H、嵌入、优化
        rw = Chem.RWMol(m_raw)
        # 注意：把 dummy 改成 H 不改变其它原子的索引（只改类型）
        for dummy_atom in dummies:
            rw.GetAtomWithIdx(dummy_atom.GetIdx()).SetAtomicNum(1)
        m_saturated = rw.GetMol()
        Chem.SanitizeMol(m_saturated)

        # head/tail attach atoms 的索引在 m_saturated 里仍是
        # head_attach_raw / tail_attach_raw（dummy 仍在原位，只是变了类型）。
        head_attach_local = head_attach_raw
        tail_attach_local = tail_attach_raw

        m_with_h = Chem.AddHs(m_saturated)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE + unit_count
        params.numThreads = 1
        ret = AllChem.EmbedMolecule(m_with_h, params)
        if ret != 0:
            params.useRandomCoords = True
            ret = AllChem.EmbedMolecule(m_with_h, params)
        if ret != 0:
            raise ValueError(
                f"RDKit ETKDG failed to embed monomer (unit {unit_count}): {smiles!r}"
            )
        # 单体很小，UFF 优化一次即可
        try:
            AllChem.UFFOptimizeMolecule(m_with_h, maxIters=500)
        except Exception:
            pass

        conf = m_with_h.GetConformer()
        elements = [a.GetSymbol() for a in m_with_h.GetAtoms()]
        coords = [
            (conf.GetAtomPosition(i).x,
             conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z)
            for i in range(m_with_h.GetNumAtoms())
        ]

        # 找到 head/tail attach atom 上要删除的 H 索引：
        #   - 中间 monomer：两端 attach 各删一个 H
        #   - 第一个 monomer：tail attach 删一个 H（接下一个 monomer）
        #   - 最后一个 monomer：head attach 删一个 H（接上一个 monomer）
        is_first = (unit_count == 0)
        is_last  = (unit_count == len(units_smiles) - 1)

        remove_h_local: set[int] = set()

        def _pick_one_H_neighbor(atom_idx: int) -> int | None:
            atom = m_with_h.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    return neighbor.GetIdx()
            return None

        if not is_first:
            h_idx = _pick_one_H_neighbor(head_attach_local)
            if h_idx is None:
                raise ValueError(
                    f"head attach atom has no H to strip (unit {unit_count}): {smiles!r}"
                )
            remove_h_local.add(h_idx)
        if not is_last:
            h_idx = _pick_one_H_neighbor(tail_attach_local)
            if h_idx is None:
                raise ValueError(
                    f"tail attach atom has no H to strip (unit {unit_count}): {smiles!r}"
                )
            remove_h_local.add(h_idx)

        per_monomer_atoms.append((elements, coords))
        per_monomer_attach.append((head_attach_local, tail_attach_local))
        per_monomer_remove_h.append(remove_h_local)

    # ---------- 沿 x 轴排列 + 居中 ----------
    # 把每个 monomer 平移到 x = unit_count * initial_separation_A，y/z 居中。
    # 同时计算保留下来的全局原子索引（被删的 H 不计入）。
    global_elements: list[str] = []
    global_coords: list[tuple[float, float, float]] = []
    global_attach_pairs: list[tuple[int, int]] = []   # (head_global, tail_global)
    global_offsets_for_unit: list[int] = []

    for unit_count, (elements, coords) in enumerate(per_monomer_atoms):
        head_local, tail_local = per_monomer_attach[unit_count]
        remove_h_local = per_monomer_remove_h[unit_count]

        # local index -> global index 映射（被删 H 没有 global）
        local_to_global: dict[int, int] = {}
        # 几何中心 (基于保留下来的原子)
        keep_indices = [i for i in range(len(elements)) if i not in remove_h_local]
        cx = sum(coords[i][0] for i in keep_indices) / len(keep_indices)
        cy = sum(coords[i][1] for i in keep_indices) / len(keep_indices)
        cz = sum(coords[i][2] for i in keep_indices) / len(keep_indices)

        shift_x = unit_count * initial_separation_A
        for i in keep_indices:
            x, y, z = coords[i]
            new_xyz = (x - cx + shift_x, y - cy, z - cz)
            local_to_global[i] = len(global_elements)
            global_elements.append(elements[i])
            global_coords.append(new_xyz)

        global_offsets_for_unit.append(len(global_elements))
        head_global = local_to_global[head_local]
        tail_global = local_to_global[tail_local]
        global_attach_pairs.append((head_global, tail_global))

    # ---------- 写 oligomer.xyz ----------
    xyz_path = os.path.join(output_folder, "oligomer.xyz")
    title = (
        f"pull-and-opt oligomer from {len(units_smiles)} units; "
        f"charge={charge} uhf={multiplicity - 1}; "
        f"initial_separation={initial_separation_A}A target_bond={target_bond_A}A"
    )
    with open(xyz_path, "w", newline="\n") as f:
        f.write(f"{len(global_elements)}\n")
        f.write(title + "\n")
        for el, (x, y, z) in zip(global_elements, global_coords):
            f.write(f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")

    # ---------- 写 scan.inp ----------
    # 相邻 monomer 之间 pair = (上一个 unit 的 tail_global, 下一个 unit 的 head_global)
    inter_pairs: list[tuple[int, int]] = []     # 0-indexed
    initial_distances_A: list[float] = []
    for i in range(len(units_smiles) - 1):
        prev_tail_global = global_attach_pairs[i][1]
        next_head_global = global_attach_pairs[i + 1][0]
        inter_pairs.append((prev_tail_global, next_head_global))
        x1, y1, z1 = global_coords[prev_tail_global]
        x2, y2, z2 = global_coords[next_head_global]
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        initial_distances_A.append(distance)

    scan_inp_path = os.path.join(output_folder, "scan.inp")
    with open(scan_inp_path, "w", newline="\n") as f:
        if inter_pairs:
            f.write("$constrain\n")
            f.write(f"   force constant = {force_constant}\n")
            for (a, b) in inter_pairs:
                # xTB 用 1-indexed
                f.write(f"   distance: {a + 1}, {b + 1}, auto\n")
            f.write("$end\n")
            f.write("$scan\n")
            for idx, (init_d) in enumerate(initial_distances_A, start=1):
                f.write(
                    f"   {idx}: {init_d:.4f}, {target_bond_A:.4f}, {scan_steps}\n"
                )
            f.write("$end\n")
        else:
            # 单 monomer 情况：不写 scan，只写一个空的 constraint 段以保持
            # 文件存在性（让后续 dispatcher 仍能识别为 pull folder）。
            f.write("# single-monomer pull folder — no inter-unit scan required\n")
            f.write("$constrain\n$end\n")

    # ---------- .CHRG / .UHF / .xtb_config ----------
    with open(os.path.join(output_folder, ".CHRG"), "w", newline="\n") as f:
        f.write(str(charge))
    with open(os.path.join(output_folder, ".UHF"), "w", newline="\n") as f:
        f.write(str(multiplicity - 1))

    config = {"method": method.upper()}
    config["solvent"] = solvent_name
    config["pull_and_opt"] = True
    config["target_bond_A"] = target_bond_A
    config["scan_steps"] = scan_steps
    config["n_units"] = len(units_smiles)
    with open(os.path.join(output_folder, ".xtb_config"), "w", newline="\n") as f:
        json.dump(config, f)

    return xyz_path


# =====================================================================
# xTB dimerize
# =====================================================================
#
# User-specified geometric protocol for stitching TWO pre-relaxed fragments
# into one larger fragment via xTB Pull. The classic use case is iterative
# oligomer construction:
#
#   monomer + monomer  ->  dimer
#   dimer   + monomer  ->  trimer
#
# Each fragment is supplied as a Gaussian `.gjf` (carries coords + charge +
# multiplicity) plus the **1-indexed atom number** of the cap H that will
# be replaced by the new C-C bond. The function:
#
#   1. Reads each fragment from its gjf.
#   2-4. Delegates ALL coordinate processing (cap-H validation, alignment,
#      fixed-2-Å-gap layout, cap-H deletion, dihedral-anchor picking) to
#      `Lib_Piece_Molecule.dimer_pull_init_geom_from_monomer` — see its
#      docstring for the geometric protocol.
#   5. Writes `oligomer.xyz` (combined N-2 atoms), `scan.inp` with
#      `$constrain distance: idx_C_A, idx_C_B, auto` + `$scan` from
#      the initial C-C distance down to `target_bond_A` (default 1.54 Å)
#      in `pull_step_A`-sized increments (default 0.2 Å), with
#      `$opt maxcycle=opt_steps_per_pull` (default 30) constraining each
#      scan point's optimization budget. When `target_dihedral_deg` is
#      given, the inter-fragment dihedral is added as a second scanned
#      coordinate in `mode=concerted`, so it rotates into place gradually
#      alongside the approach instead of being pinned from cycle one.
#   6. Writes `.CHRG` / `.UHF` / `.xtb_config` (`pull_and_opt: true`) so
#      `HPC_Lib.HPC_xTB.submit_xTB_pull_and_optimize` can drive the
#      two-stage cluster job (scan + final unconstrained opt; solvent /
#      method / opt levels are given at submission) WITHOUT any
#      modification.
#
# The output folder structure mirrors `xTB_pull_input_from_oligomer_smiles`
# exactly, so the same downstream submitter/puller works for both.
#
def _dihedral_angle_deg(point_A, point_B, point_C, point_D) -> float:
    """Signed A-B-C-D dihedral angle in degrees, in (-180, 180].

    Standard IUPAC sign convention, computed from the three connecting vectors
    so it matches what xtb's ``dihedral:`` constraint measures.
    """
    import math

    import numpy as np

    b1 = np.asarray(point_B, dtype=float) - np.asarray(point_A, dtype=float)
    b2 = np.asarray(point_C, dtype=float) - np.asarray(point_B, dtype=float)
    b3 = np.asarray(point_D, dtype=float) - np.asarray(point_C, dtype=float)

    b2_norm = np.linalg.norm(b2)
    if b2_norm == 0.0:
        raise ValueError("dihedral is undefined: the two central atoms coincide")

    normal_1 = np.cross(b1, b2)
    normal_2 = np.cross(b2, b3)
    if np.linalg.norm(normal_1) == 0.0 or np.linalg.norm(normal_2) == 0.0:
        raise ValueError("dihedral is undefined: three of the atoms are collinear")

    x = float(np.dot(normal_1, normal_2))
    y = float(np.dot(np.cross(normal_1, normal_2), b2 / b2_norm))
    return math.degrees(math.atan2(y, x))


def xTB_dimerize_input_from_gjfs(
    gjf_A: str,
    h_atom_index_A: int,
    gjf_B: str,
    h_atom_index_B: int,
    output_folder: str,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    pull_step_A: float = 0.2,
    opt_steps_per_pull: int = 30,
    target_bond_A: float = 1.54,
    target_dihedral_deg: float | None = None,
    dihedral_atom_A_1based: int | None = None,
    dihedral_atom_B_1based: int | None = None,
    force_constant: float = 1.0,
    solvent: str,
    method: str = "GFN2",
) -> str:
    """Build an xTB Pull-dimerize input folder from two pre-relaxed fragments.

    Args:
        gjf_A, gjf_B:           Paths to Gaussian `.gjf` files of the two
                                fragments (typically a previous xTB opt's
                                converged geometry, e.g. ``monomer.gjf`` or
                                ``dimer.gjf``).
        h_atom_index_A:         1-indexed atom number (within `gjf_A`'s
                                geometry block) of the cap H on fragment A
                                that will be replaced by the new C-C bond
                                to fragment B.
        h_atom_index_B:         1-indexed atom number of the corresponding
                                cap H on fragment B.
        output_folder:          Path where the new xTB Pull folder will be
                                written (created if missing).
        charge:                 Total molecular charge of the combined
                                oligomer.
        multiplicity:           Spin multiplicity of the combined oligomer.
        pull_step_A:            Pull-step size in Å for the constraint scan.
                                Default 0.2 — i.e. each scan point shortens
                                the new C-C distance by 0.2 Å until reaching
                                ``target_bond_A``.
        opt_steps_per_pull:     Maximum xTB optimization cycles per scan
                                point. Default 30 — written to scan.inp as
                                ``$opt; maxcycle=30; $end``. xTB honors
                                this even when `--opt normal` is supplied
                                on the command line (cycle cap, not
                                convergence cap).
        target_bond_A:          Final C-C bond length to scan to, in Å.
                                Default 1.54 (typical C(sp3)-C(sp3) single
                                bond; C(sp2)-C(sp2) inter-ring would be
                                ~1.48 but starting longer + letting the
                                final unconstrained opt relax it is fine).
        target_dihedral_deg:    If given, the A-c_A-c_B-B dihedral is driven
                                to this value. It is
                                RAMPED there concertedly with the distance —
                                scan point k holds both the distance and the
                                dihedral at their k-th interpolated values —
                                rather than being clamped at the target from
                                the first cycle, which would force the entire
                                rotation into one strained optimization while
                                the fragments are still apart. The ramp turns
                                through the short arc; the final unconstrained
                                opt then releases the dihedral entirely.
        dihedral_atom_A_1based: Explicit A-side dihedral anchor (1-indexed in
        dihedral_atom_B_1based: the combined post-deletion list). Left None,
                                the nearest non-H neighbour of each bonding C
                                is picked.
        force_constant:         Distance-restraint force constant in
                                Hartree/Bohr² for the `$constrain` block.
        solvent:                ALPB solvent name; REQUIRED, no default.
                                Gas phase must be an explicit ``"none"``.
                                Validated via :func:`resolve_solvent_flag`
                                at build time (invalid values raise); the
                                normalized name is recorded in
                                ``.xtb_config``. Submission still takes its
                                own explicit solvent argument.
        method:                 ``"GFN2"`` (default) / ``"GFN1"`` / ``"GFN0"``
                                / ``"GFNFF"``.

    Returns:
        Path to ``output_folder/oligomer.xyz``.

    Raises:
        ValueError on: unparseable gjf, atom index out of range, atom at
            the index is not H, no non-H neighbor in C-H range, or the
            initial inter-fragment distance falling below `target_bond_A`
            after layout (suggests caller picked H indices that produce
            overlap — usually a wrong index).

    Notes:
        - Atom-number convention: 1-indexed throughout (matches xTB's
          ``$constrain`` syntax). The function accepts 1-indexed and writes
          1-indexed atom IDs into ``scan.inp``.
        - The output folder is drop-in compatible with
          ``HPC_Lib.HPC_xTB.submit_xTB_pull_and_optimize`` — no separate
          submit helper is needed for dimerize.
        - For trimer construction:
          ``trimer X-Y-Z = dimerize(dimer[X_Y].gjf, dimer_free_end_H_idx,
          monomer_Z.gjf, monomer_Z_free_end_H_idx, ...)``. The dimer must
          have completed its own xTB opt first (its `.gjf` carries that
          relaxed geometry).
    """
    import json
    import math

    solvent_name, _ = resolve_solvent_flag(solvent, method.upper())

    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # ---------- Parse the two fragments ----------
    from .Lib_Gaussian import Gaussian_Input
    from .Lib_Piece_Molecule import dimer_pull_init_geom_from_monomer

    def _read_gjf(path: str):
        gaussian_input = Gaussian_Input(path)
        coordinate = gaussian_input.steps[0].coordinate
        if not coordinate or not getattr(coordinate, "elements", None):
            raise ValueError(f"Cannot read coords from gjf: {path}")
        return coordinate

    molecule_A = _read_gjf(gjf_A)
    molecule_B = _read_gjf(gjf_B)

    # ---------- All coordinate processing (cap-H validation, alignment, layout,
    # cap-H deletion, dihedral-anchor picking) is in Lib_Piece_Molecule ----------
    dimer_geometry = dimer_pull_init_geom_from_monomer(
        molecule_A, h_atom_index_A,
        molecule_B, h_atom_index_B,
        min_initial_CC_A=target_bond_A,
        pick_dihedral_anchors=(target_dihedral_deg is not None),
        dihedral_atom_A_1based=dihedral_atom_A_1based,
        dihedral_atom_B_1based=dihedral_atom_B_1based,
    )
    combined_elements = dimer_geometry["elements"]
    combined_coordinates = dimer_geometry["coordinates_np"]
    initial_CC_distance = dimer_geometry["initial_CC_A"]
    c_A_global_1based = dimer_geometry["c_A_global_1based"]
    c_B_global_1based = dimer_geometry["c_B_global_1based"]
    dihedral_A_global_1based = dimer_geometry["dihedral_A_global_1based"]
    dihedral_B_global_1based = dimer_geometry["dihedral_B_global_1based"]

    # ---------- Write oligomer.xyz ----------
    xyz_path = os.path.join(output_folder, "oligomer.xyz")
    title = (
        f"xTB-dimerize from {os.path.basename(gjf_A)} (H#{h_atom_index_A}) + "
        f"{os.path.basename(gjf_B)} (H#{h_atom_index_B}); "
        f"charge={charge} uhf={multiplicity - 1}; "
        f"initial C-C={initial_CC_distance:.3f}A target={target_bond_A:.3f}A "
        f"pull_step={pull_step_A}A maxcycle={opt_steps_per_pull}"
    )
    with open(xyz_path, "w", newline="\n") as f:
        f.write(f"{len(combined_elements)}\n")
        f.write(title + "\n")
        for element, (x, y, z) in zip(combined_elements, combined_coordinates):
            f.write(f"{element:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")

    # ---------- Write scan.inp ----------
    # Number of scan points: ceil((initial_CC_distance - target_bond_A) / pull_step_A) + 1
    # (include both endpoints).
    n_scan_points = max(2, int(math.ceil((initial_CC_distance - target_bond_A) / pull_step_A)) + 1)

    initial_dihedral_deg = None
    if target_dihedral_deg is not None:
        initial_dihedral_deg = _dihedral_angle_deg(
            combined_coordinates[dihedral_A_global_1based - 1],
            combined_coordinates[c_A_global_1based - 1],
            combined_coordinates[c_B_global_1based - 1],
            combined_coordinates[dihedral_B_global_1based - 1],
        )
        # Both scan endpoints are left inside (-180, 180]. A dihedral is
        # periodic, so the shorter arc between start and target sometimes runs
        # through ±180 and would need an endpoint like 190° or -350° to express
        # — a value xtb is not documented to accept. Sweeping the long way
        # instead costs nothing that matters here: the ramp is divided into the
        # same 30-to-55 scan points as the distance, so each step still turns
        # the bond by only a few degrees, which is the whole point of ramping.

    scan_inp_path = os.path.join(output_folder, "scan.inp")
    with open(scan_inp_path, "w", newline="\n") as f:
        f.write("$constrain\n")
        f.write(f"   force constant = {force_constant}\n")
        f.write(f"   distance: {c_A_global_1based}, {c_B_global_1based}, auto\n")
        if target_dihedral_deg is not None:
            # The A-c_A-c_B-B
            # dihedral is SCANNED to its target alongside the distance, not
            # clamped there from the first cycle. Pinning it at the target
            # while the fragments are still ~2 Å apart would force the whole
            # rotation to happen in one violently strained optimization.
            f.write(f"   dihedral: {dihedral_A_global_1based}, "
                    f"{c_A_global_1based}, {c_B_global_1based}, "
                    f"{dihedral_B_global_1based}, auto\n")
        f.write("$end\n")
        f.write("$scan\n")
        if target_dihedral_deg is not None:
            # Concerted: both coordinates advance together over the same number
            # of points. Sequential (xtb's default) would instead scan the
            # distance fully, then the dihedral, giving n_scan_points^2 points
            # and a fully-formed bond before the rotation ever starts.
            f.write("   mode=concerted\n")
        f.write(f"   1: {initial_CC_distance:.4f}, {target_bond_A:.4f}, {n_scan_points}\n")
        if target_dihedral_deg is not None:
            f.write(f"   2: {initial_dihedral_deg:.4f}, "
                    f"{target_dihedral_deg:.4f}, {n_scan_points}\n")
        f.write("$end\n")
        f.write("$opt\n")
        f.write(f"   maxcycle={opt_steps_per_pull}\n")
        f.write("$end\n")

    # ---------- .CHRG / .UHF / .xtb_config ----------
    with open(os.path.join(output_folder, ".CHRG"), "w", newline="\n") as f:
        f.write(str(charge))
    with open(os.path.join(output_folder, ".UHF"), "w", newline="\n") as f:
        f.write(str(multiplicity - 1))

    config = {"method": method.upper()}
    config["solvent"] = solvent_name
    config["pull_and_opt"] = True
    config["dimerize"] = True
    config["target_bond_A"] = target_bond_A
    config["pull_step_A"] = pull_step_A
    config["opt_steps_per_pull"] = opt_steps_per_pull
    config["initial_C_C_distance_A"] = round(initial_CC_distance, 4)
    config["fragment_A_gjf"] = os.path.basename(gjf_A)
    config["fragment_B_gjf"] = os.path.basename(gjf_B)
    config["fragment_A_H_atom_1based"] = h_atom_index_A
    config["fragment_B_H_atom_1based"] = h_atom_index_B
    config["c_A_global_1based"] = c_A_global_1based
    config["c_B_global_1based"] = c_B_global_1based
    if target_dihedral_deg is not None:
        config["target_dihedral_deg"] = target_dihedral_deg
        config["dihedral_A_global_1based"] = dihedral_A_global_1based
        config["dihedral_B_global_1based"] = dihedral_B_global_1based
        config["initial_dihedral_deg"] = round(initial_dihedral_deg, 4)
        config["dihedral_scan_mode"] = "concerted_with_distance"
        config["dihedral_scan_sweep_deg"] = round(
            target_dihedral_deg - initial_dihedral_deg, 4)
    with open(os.path.join(output_folder, ".xtb_config"), "w", newline="\n") as f:
        json.dump(config, f)

    return xyz_path


# =====================================================================
# 以下为 xTB 的"运行层"（2026-08-13 从 HPC_Lib.HPC_xTB 迁移进本模块）
# =====================================================================
#
# 迁移原则：任务类型 / 方法 / 溶剂 / 优化等级的合法值与解析、输入文件夹
# 校验、可执行文件与环境变量、xtb 命令构造、以及本地直接运行，都是 xTB
# 的领域知识，与队列系统无关，统一放在本模块；HPC_Lib.HPC_xTB 只保留
# SLURM 侧的壳（资源解析、sbatch 头、打包、提交），经同名下划线别名
# 引用这里的实现。
#
# 两种执行模式共用同一张任务表（_single_step_task_specification）：
#
# - **脚本模式**（build_task_command_lines 等）：生成 bash 命令行，写进
#   SLURM 作业脚本，由 HPC_Lib 在集群上提交执行；
# - **直接模式**（run_xTB_task_folder / xTB_optimize_gjf）：在本机用
#   subprocess 启动随本包分发的可执行文件（Windows / Linux 两套都有，
#   见 Executable_xTB），不经任何队列。
#
# 环境变量设置依据手册 Chem_Lib_Manuals/xTB_20260806/01_Quickstart/
# 01_Setup_and_Installation.md 与发行版自带的 share/xtb/config_env.bash：
#
# - ``OMP_NUM_THREADS=<核数>,1`` / ``OMP_STACKSIZE=4G`` /
#   ``MKL_NUM_THREADS=<核数>``（Parallelisation 一节）；
# - ``XTBPATH`` 指向参数文件目录（share/xtb），``XTBHOME`` 指向发行版
#   前缀（config_env.bash 的约定，旧版本 xtb 的回退变量）；
# - Linux 上运行前解除栈限制（``ulimit -s unlimited``，防大分子栈溢出）
#   ——直接模式里由子进程的 preexec_fn 用 resource.setrlimit 完成。
# =====================================================================

#: 单步任务类型（复合流程 pull_and_optimize 不在此列，它只存在于
#: HPC 提交侧，见 HPC_Lib.HPC_xTB.submit_xTB_pull_and_optimize）。
VALID_XTB_TASKS = ("energy", "opt", "ohess", "md", "omd", "pull")

VALID_METHODS = {"GFN2", "GFN1", "GFN0", "GFNFF"}

#: xTB 手册 02_Guides/05_Implicit_Solvation.md 中 ALPB 模型的全部可用
#: 溶剂（--alpb 接受的小写写法）。"none" 表示气相（不加溶剂旗标）。
VALID_ALPB_SOLVENTS = (
    "acetone", "acetonitrile", "aniline", "benzaldehyde", "benzene",
    "ch2cl2", "chcl3", "cs2", "dioxane", "dmf", "dmso", "ethanol",
    "ether", "ethylacetate", "furane", "hexadecane", "hexane",
    "methanol", "nitromethane", "octanol", "woctanol", "phenol",
    "toluene", "thf", "water",
)

#: 常见等价写法 → 规范名
_SOLVENT_ALIASES = {
    "h2o": "water",
    "n-hexane": "hexane",
    "octanol(wet)": "woctanol",
    "octanol (wet)": "woctanol",
}

#: xTB --opt / --ohess 的精度等级（xTB 自身默认 normal）。
VALID_OPT_LEVELS = ("crude", "sloppy", "loose", "lax", "normal",
                    "tight", "vtight", "extreme")


def resolve_task(task: str) -> str:
    """校验并规范化单步任务类型。"""
    task_lower = str(task).strip().lower()
    if task_lower not in VALID_XTB_TASKS:
        raise ValueError(
            f"Unsupported xTB task: {task!r}. "
            f"Valid single-step tasks: {', '.join(VALID_XTB_TASKS)}. "
            f"(复合流程用 HPC_Lib.HPC_xTB.submit_xTB_pull_and_optimize，"
            f"不经此参数。)"
        )
    return task_lower


def resolve_method_flag(method: str) -> tuple[str, str]:
    """校验 method 并返回 ``(规范化大写名, xTB 命令行参数)``。"""
    if method is None:
        raise ValueError(
            "method is required — pass one of: "
            + ", ".join(sorted(VALID_METHODS))
        )
    method_upper = str(method).upper()
    if method_upper not in VALID_METHODS:
        raise ValueError(
            f"Unsupported method: {method!r}. "
            f"Valid options: {', '.join(sorted(VALID_METHODS))}"
        )
    if method_upper == "GFNFF":
        return method_upper, "--gfnff"
    return method_upper, f"--gfn {method_upper.replace('GFN', '')}"


def resolve_solvent_flag(solvent: str, method_upper: str) -> tuple[str, str]:
    """校验 solvent 并返回 ``(规范化小写名, xTB 命令行参数)``。

    关键参数、无默认值：solvent 必须显式提供；气相必须显式写 ``"none"``。
    使用 ALPB 隐式溶剂模型（``--alpb``）；ALPB 没有为 GFN0 参数化
    （xTB 手册 05_Implicit_Solvation），GFN0 配溶剂直接报错。
    """
    if solvent is None:
        raise ValueError(
            "solvent is required and has NO default — pass an ALPB solvent "
            f"name ({', '.join(VALID_ALPB_SOLVENTS)}), or the explicit "
            'string "none" for a gas-phase calculation.'
        )
    solvent_lower = str(solvent).strip().lower()
    solvent_lower = _SOLVENT_ALIASES.get(solvent_lower, solvent_lower)
    if solvent_lower == "none":
        return "none", ""
    if solvent_lower not in VALID_ALPB_SOLVENTS:
        raise ValueError(
            f"Unsupported solvent: {solvent!r}. Valid ALPB solvents: "
            f"{', '.join(VALID_ALPB_SOLVENTS)}; use \"none\" for gas phase."
        )
    if method_upper == "GFN0":
        raise ValueError(
            "ALPB implicit solvation is not parameterized for GFN0 "
            '(xTB manual, Implicit Solvation). Use solvent="none" or '
            "another method."
        )
    return solvent_lower, f"--alpb {solvent_lower}"


def resolve_opt_level(opt_level: str | None) -> str | None:
    """校验优化精度等级。None 表示不传等级、让 xTB 用自身默认（normal）。"""
    if opt_level is None:
        return None
    opt_level_lower = str(opt_level).strip().lower()
    if opt_level_lower not in VALID_OPT_LEVELS:
        raise ValueError(
            f"Unsupported opt level: {opt_level!r}. "
            f"Valid levels: {', '.join(VALID_OPT_LEVELS)}"
        )
    return opt_level_lower


# =====================================================================
# 输入文件夹的结构解析与校验
# =====================================================================

def resolve_structure_xyz(xtb_folder: str, structure: str | None = None) -> str:
    """返回文件夹中作为输入结构的 ``.xyz`` 文件名（不含路径）。

    选择顺序：显式指定的 *structure*（须存在）＞ ``oligomer.xyz``
    （pull 输入生成函数的固定命名）＞ 文件夹中唯一的 ``.xyz``。
    多个 ``.xyz`` 且无法判定时报错（已跑过的文件夹会积累
    ``xtbopt.xyz`` / ``xtbscan_last.xyz`` 等产物，重投时须显式指定）。
    """
    contents = os.listdir(xtb_folder)
    if structure:
        if structure not in contents:
            raise FileNotFoundError(
                f"Specified structure file not found in {xtb_folder}: {structure}"
            )
        return structure
    xyz_files = [f for f in contents if f.lower().endswith(".xyz")]
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz file found in: {xtb_folder}")
    if "oligomer.xyz" in contents:
        return "oligomer.xyz"
    if len(xyz_files) == 1:
        return xyz_files[0]
    raise ValueError(
        f"Multiple .xyz files in {xtb_folder}: {xyz_files}. "
        "Specify the input structure explicitly (xtb_task.toml 的 "
        "structure 键，或清理产物文件)。"
    )


def validate_task_folder(xtb_folder: str, task: str,
                         structure: str | None = None) -> str:
    """校验单步任务文件夹的输入完备性，返回输入 ``.xyz`` 文件名。

    要求：``.CHRG`` 与 ``.UHF`` 必须存在（本库的标准输入约定，由
    :func:`xTB_input_from_Gaussian_gjf` 等生成函数或 HPC 提交流程落盘）；
    ``md`` / ``omd`` 要求 ``md.inp``；``pull`` 要求 ``scan.inp``。
    任何缺失直接报错（fail early），绝不猜测或代答。
    """
    contents = os.listdir(xtb_folder)
    missing = [f for f in (".CHRG", ".UHF") if f not in contents]
    if missing:
        raise FileNotFoundError(
            f"{xtb_folder} is missing required file(s): {', '.join(missing)}. "
            "xTB input folders must contain .xyz + .CHRG + .UHF."
        )
    if task in ("md", "omd") and "md.inp" not in contents:
        raise FileNotFoundError(
            f"Task '{task}' requires md.inp in: {xtb_folder} "
            "(手写，或在 xtb_task.toml 的 [md] 表中给出参数由提交流程生成)。"
        )
    if task == "pull" and "scan.inp" not in contents:
        raise FileNotFoundError(
            f"Task 'pull' requires scan.inp in: {xtb_folder} "
            "(手写，或在 xtb_task.toml 的 [[pull]] 表中给出原子对由提交流程生成)。"
        )
    return resolve_structure_xyz(xtb_folder, structure)


def is_strict_xtb_input_folder(folder: str) -> bool:
    """检查文件夹是否满足**运行 / 提交时**的输入约定（.xyz + .CHRG + .UHF）。

    注意与 HPC.py 分发器的检测（``_is_xtb_input_folder``，额外接受
    「.xyz + xtb_task.toml、待提交流程落盘 .CHRG/.UHF」的变体）不同：
    到达运行层时所有文件必须已经就位。
    """
    if not os.path.isdir(folder):
        return False
    contents = os.listdir(folder)
    has_xyz = any(f.lower().endswith(".xyz") for f in contents)
    return has_xyz and ".CHRG" in contents and ".UHF" in contents


# =====================================================================
# 可执行文件与环境变量
# =====================================================================

def chem_lib_directory() -> str:
    """返回 Chem_Lib 包所在目录（即本文件所在目录）。

    可执行文件一律相对于本目录定位，不写死绝对路径——editable 安装
    （路径即仓库 ``src/Chem_Lib``）与 pip / uv 常规安装（路径在
    site-packages 里，二进制经 package-data 一起带入）下都成立。
    """
    return os.path.dirname(os.path.abspath(__file__))


def ensure_executable_bit(path: str) -> None:
    """确保文件带可执行位（幂等；Windows 上无操作）。

    二进制经 SFTP 同步上集群后默认是 644，而同步脚本随后的
    ``harden_permissions`` 会把不带 x 位的文件一律收成 600——因此每次
    使用前都在这里把可执行位补回来（收紧约定是 700）。
    """
    if not os.access(path, os.X_OK):
        os.chmod(path, 0o700)


def resolve_xtb_executable() -> str:
    """解析经典 xTB 可执行文件的路径（按当前平台选 Windows / Linux 版）。

    一律使用随 Chem_Lib 分发的二进制：Windows 取
    ``Executable_xTB/Windows/bin/xtb.exe``（所需 DLL 在同一目录），
    其他平台取 ``Executable_xTB/Linux/bin/xtb``（静态链接，并确保带
    可执行位）。

    找不到时直接报错（fail early），**不回落**到 PATH 上的 ``xtb``——
    xTB 的运行环境（可执行文件与 ``XTBPATH`` / ``XTBHOME``）完全由
    本库动态提供，任何机器都不需要、也不应该在 ``.bashrc`` 里配置
    xtb 相关的环境变量（2026-08-14 用户裁定）。
    """
    import platform

    if platform.system() == "Windows":
        packaged = os.path.join(chem_lib_directory(), "Executable_xTB",
                                "Windows", "bin", "xtb.exe")
    else:
        packaged = os.path.join(chem_lib_directory(), "Executable_xTB",
                                "Linux", "bin", "xtb")
    if not os.path.isfile(packaged):
        raise FileNotFoundError(
            f"xTB executable not found: {packaged}\n"
            "该二进制随 Chem_Lib 包分发（不进 git，由 "
            "A0_HPC_Sync_Python_Lib.py 同步上集群，pip / uv 安装经 "
            "pyproject 的 package-data 带入）。它缺失说明本机的 "
            "Executable_xTB 文件夹还没同步 / 安装到位——先补齐它，"
            "不要改用 PATH 上的 xtb（本库不依赖任何机器自装的 xTB）。"
        )
    if not packaged.endswith(".exe"):
        ensure_executable_bit(packaged)
    return packaged


def _xtb_share_directory(xtb_executable: str) -> tuple[str, str]:
    """由可执行文件路径推出发行版前缀与 ``share/xtb`` 参数文件目录。

    随包分发的 xTB / g-xTB 发行版固定是 ``<前缀>/bin/xtb`` +
    ``<前缀>/share/xtb`` 的布局；``share/xtb`` 不存在说明发行版没有
    同步 / 安装完整，直接报错（fail early），不静默省略环境变量。
    """
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(xtb_executable)))
    share_directory = os.path.join(prefix, "share", "xtb")
    if not os.path.isdir(share_directory):
        raise FileNotFoundError(
            f"xTB distribution is incomplete: {share_directory} not found "
            f"(executable: {xtb_executable}). 发行版必须整文件夹同步 / "
            "安装（bin + share），不能只带二进制。"
        )
    return prefix, share_directory


def xtb_environment_lines(xtb_executable: str) -> list[str]:
    """生成使用 *xtb_executable* 所需的环境变量 export 行（bash 作业脚本用）。

    仿照发行版自带的 ``share/xtb/config_env.bash``：``XTBHOME`` 指向
    发行版前缀、``XTBPATH`` 指向参数文件目录（``share/xtb``，经典 GFN
    方法的 ``param_gfn*-xtb.txt`` 等都在这里）。g-xTB 自身不需要参数
    文件，但它的发行版同样带 ``share/xtb``，统一导出无害。运行环境
    完全由这些行动态提供，集群的 ``.bashrc`` 里不需要任何 xtb 相关的
    环境变量（作业脚本 source ``.bashrc`` 在先、这些 export 在后，
    即使残留旧配置也会被覆盖）。
    """
    prefix, share_directory = _xtb_share_directory(xtb_executable)
    return [f"export XTBHOME={prefix}",
            f"export XTBPATH={share_directory}"]


def xtb_runtime_environment(xtb_executable: str, cores: int, *,
                            gnu_openmp_workarounds: bool = False) -> dict:
    """构造本机直接运行 xtb 所需的完整环境变量表（subprocess 用）。

    在 ``os.environ`` 的副本上覆盖（依据手册 Setup and Installation 的
    Parallelisation 一节与 config_env.bash）：

    - 默认（Intel 工具链构建，如官方 xtb 发行版）：
      ``OMP_NUM_THREADS=<核数>,1``、``OMP_STACKSIZE=4G``、
      ``MKL_NUM_THREADS=<核数>``；
    - *gnu_openmp_workarounds* = True（GNU 工具链 / libgomp 构建，如
      g-xTB 的 Windows 版，2026-08-13 本机实测）：手册推荐的
      ``<核数>,1`` 嵌套写法会让该构建**直接段错误**，改用
      ``OMP_NUM_THREADS=<核数>`` + ``OMP_MAX_ACTIVE_LEVELS=1``（手册
      提到的等价去嵌套方式）；``OMP_STACKSIZE=4G`` 超出 Windows 上
      libgomp 的解析上限（LLP64 的 32 位 unsigned long）会被拒绝，而
      过大的合法值又会因线程栈全额分配失败（``Thread creation
      failed``），Windows 上改用 ``512M``；
    - ``XTBHOME`` / ``XTBPATH``（同 :func:`xtb_environment_lines`）；
    - Windows 上把可执行文件所在的 ``bin`` 目录加到 ``PATH`` 最前，
      保证同目录的 DLL（``libiomp5md.dll`` / ``libgfortran-5.dll``
      等）可被找到。
    """
    import platform

    environment = dict(os.environ)
    if gnu_openmp_workarounds:
        environment["OMP_NUM_THREADS"] = str(cores)
        environment["OMP_MAX_ACTIVE_LEVELS"] = "1"
        environment["OMP_STACKSIZE"] = (
            "512M" if platform.system() == "Windows" else "4G")
    else:
        environment["OMP_NUM_THREADS"] = f"{cores},1"
        environment["OMP_STACKSIZE"] = "4G"
        environment["MKL_NUM_THREADS"] = str(cores)

    prefix, share_directory = _xtb_share_directory(xtb_executable)
    environment["XTBHOME"] = prefix
    environment["XTBPATH"] = share_directory
    if platform.system() == "Windows":
        binary_directory = os.path.dirname(os.path.abspath(xtb_executable))
        environment["PATH"] = (binary_directory + os.pathsep
                               + environment.get("PATH", ""))
    return environment


# =====================================================================
# 单步任务表：两种执行模式（bash 脚本 / 本机 subprocess）共用
# =====================================================================

def _single_step_task_specification(
    task: str,
    xyz_file: str,
    folder_stem: str,
    opt_level: str | None,
    scan_inp_has_scan_section: bool,
) -> dict:
    """返回一个单步任务的组成要素，供两种执行模式共用。

    Returns:
        ``{"task_flags": [...], "log_name": str, "products": [(源, 目标)]}``。
        ``task_flags`` 是 xyz 文件名之后、``--chrg`` 等公共旗标之前的
        任务旗标；``products`` 是任务结束后的产物复制表（沿用历史
        qsub_xTB_*.py 脚本的命名约定，见 HPC_Comp_Chem_xTB.md）。
    """
    xyz_stem = os.path.splitext(xyz_file)[0]
    level_tokens = [opt_level] if opt_level else []

    if task == "energy":
        return {"task_flags": [],
                "log_name": f"{xyz_stem}.xtb.log",
                "products": []}
    if task in ("opt", "ohess"):
        return {"task_flags": [f"--{task}"] + level_tokens,
                "log_name": f"{xyz_stem}.xtb.log",
                "products": [("xtbopt.log", f"{xyz_stem}.xtbopt_traj.xyz")]}
    if task in ("md", "omd"):
        return {"task_flags": ["--input", "md.inp", f"--{task}"],
                "log_name": f"{folder_stem}.xtb.log",
                "products": [("xtb.trj", f"{folder_stem}.xtb_MD_traj.xyz")]}
    if task == "pull":
        # scan.inp 含 $scan 段 → 产物是扫描轨迹 xtbscan.log；
        # 仅有 $constrain（受约束优化）→ 产物是优化轨迹 xtbopt.log。
        if scan_inp_has_scan_section:
            products = [("xtbscan.log", f"{folder_stem}.xtb_Pull.xyz")]
        else:
            products = [("xtbopt.log", f"{folder_stem}.xtb_RstrnOPT.xyz")]
        return {"task_flags": ["--input", "scan.inp", "--opt"] + level_tokens,
                "log_name": f"{folder_stem}.xtb.log",
                "products": products}
    raise ValueError(f"Unhandled task: {task}")  # resolve_task 已经拦下非法值


# =====================================================================
# 命令构造（脚本模式：生成 bash 行，供 HPC_Lib 写进 SLURM 作业脚本）
# =====================================================================

def build_task_command_lines(
    xtb_folder: str,
    task: str,
    xyz_file: str,
    method_flag: str,
    solvent_flag: str,
    opt_level: str | None,
    xtb_exe: str,
) -> list[str]:
    """生成一个单步任务的 shell 行（xTB 命令 + 产物复制）。

    全部使用相对路径，需在 *xtb_folder* 内执行（调用方负责 cd）。
    产物复制命名沿用历史 qsub_xTB_*.py 脚本的约定（见 HPC_Comp_Chem_xTB.md）。
    ``opt`` / ``ohess`` 任务末尾另附「收敛时生成
    ``<结构名>.xtb_converged.gjf``」的 shell 行（见
    :func:`build_save_converged_gjf_lines`），使远程任务与本地直接运行
    （:func:`run_xTB_task_folder`）留下同一个收敛 gjf。
    """
    folder_stem = os.path.basename(xtb_folder.rstrip("/\\"))
    scan_inp_has_scan_section = False
    if task == "pull":
        scan_inp_content = open(os.path.join(xtb_folder, "scan.inp")).read()
        scan_inp_has_scan_section = "$scan" in scan_inp_content

    specification = _single_step_task_specification(
        task, xyz_file, folder_stem, opt_level, scan_inp_has_scan_section)

    common_flags = [method_flag, "--chrg $(cat .CHRG)", "--uhf $(cat .UHF)"]
    if solvent_flag:
        common_flags.append(solvent_flag)

    command_parts = [xtb_exe, xyz_file] + specification["task_flags"] + common_flags
    lines = [" ".join(command_parts) + f" &> {specification['log_name']}"]
    for source_name, target_name in specification["products"]:
        lines.append(f"if [ -f {source_name} ]; then "
                     f"cp {source_name} {target_name}; fi")
    if task in ("opt", "ohess"):
        lines.extend(build_save_converged_gjf_lines(
            os.path.splitext(xyz_file)[0]))
    return lines


def build_save_converged_gjf_lines(
    xyz_stem: str,
    converged_gjf_suffix: str = ".xtb_converged.gjf",
) -> list[str]:
    """生成「优化收敛时把最后一帧存成 ``<名><收敛后缀>``」的 shell 行。

    :func:`save_converged_gjf` 的脚本模式等价物，供 HPC_Lib 写进 SLURM
    作业脚本——使**远程**运行的优化任务与本地直接运行在任务文件夹里
    留下同一个收敛 gjf（2026-08-15 用户指令：不论远程还是本地，优化
    正常结束才生成 ``*_converged.gjf``，否则不生成）。

    判据与文件内容逐项镜像 :func:`save_converged_gjf`：

    - 收敛判据 = 日志出现 ``GEOMETRY OPTIMIZATION CONVERGED``
      （见 :func:`xtb_optimization_converged`）；
    - 几何 = ``<名>.xtbopt_traj.xyz`` 的最后一帧；
    - 电荷 = ``.CHRG``；多重度 = ``.UHF`` + 1，无 ``.UHF`` 记 1
      （经典 xTB 文件夹恒有 ``.UHF``，g-xTB 只在开壳层有——同一写法
      通吃两种输入约定）；
    - 文件内容 = 空路由 ``#p`` / 空行 / ``Converged geometry from
      <名>.xtb.log`` / 空行 / ``电荷 多重度`` / 坐标；
    - 未收敛（含日志缺失）时删除既有的收敛 gjf——该文件的存在性
      本身就是「这次优化已收敛」的可靠判据。

    全部使用相对路径，需在任务文件夹内执行（调用方负责 cd）。
    """
    log_name = f"{xyz_stem}.xtb.log"
    trajectory_name = f"{xyz_stem}.xtbopt_traj.xyz"
    gjf_name = f"{xyz_stem}{converged_gjf_suffix}"
    awk_print_coordinates = (
        "awk '{printf(\"%s\\t%.14f\\t%.14f\\t%.14f\\n\", $1, $2, $3, $4)}'")
    return [
        f'rm -f "{gjf_name}"',
        f'if grep -q "GEOMETRY OPTIMIZATION CONVERGED" "{log_name}" '
        f'2>/dev/null && [ -f "{trajectory_name}" ]; then',
        f'    NATOMS=$(head -n 1 "{trajectory_name}" | awk \'{{print $1}}\')',
        '    if [ -n "$NATOMS" ]; then',
        '        MULT=1',
        '        if [ -f .UHF ]; then MULT=$(( $(cat .UHF) + 1 )); fi',
        '        {',
        '            echo "#p"',
        '            echo ""',
        f'            echo "Converged geometry from {log_name}"',
        '            echo ""',
        '            echo "$(cat .CHRG) $MULT"',
        f'            tail -n $((NATOMS + 2)) "{trajectory_name}" '
        f'| tail -n +3 | {awk_print_coordinates}',
        f'        }} > "{gjf_name}"',
        '    fi',
        'fi',
    ]


def build_extract_last_frame_lines(fallback_xyz_filename: str) -> list[str]:
    """生成「抽取 ``xtbscan.log`` 最后一帧到 ``xtbscan_last.xyz``」的 shell 行。

    全部使用相对路径，需在 xTB 文件夹内执行。xtbscan.log 是 multi-frame
    xyz：每帧第一行是原子数 N，第二行是 comment，再跟 N 行原子，因此取
    末尾 N+2 行即最后一帧。``xtbscan.log`` 不存在（例如 scan.inp 只有
    约束、没有扫描段）或抽取失败时，逐级回退到 ``xtbopt.xyz``（受约束
    opt 的最终帧）、*fallback_xyz_filename*（原始输入）。
    """
    return [
        "if [ -f xtbscan.log ]; then",
        "    NATOMS=$(head -n 1 xtbscan.log | awk '{print $1}')",
        '    if [ -n "$NATOMS" ]; then',
        "        FRAME_LINES=$((NATOMS + 2))",
        "        tail -n $FRAME_LINES xtbscan.log > xtbscan_last.xyz",
        "    fi",
        "fi",
        "if [ ! -s xtbscan_last.xyz ]; then",
        "    # 退回 xtbopt.xyz（受约束 opt 的最终帧）或原始输入",
        "    if [ -f xtbopt.xyz ]; then",
        "        cp xtbopt.xyz xtbscan_last.xyz",
        "    else",
        f"        cp {fallback_xyz_filename} xtbscan_last.xyz",
        "    fi",
        "fi",
    ]


def build_pull_and_optimize_lines(
    xyz_file: str,
    scan_opt_level: str | None,
    final_opt_level: str | None,
    common_flags_str: str,
    xtb_exe: str,
) -> list[str]:
    """生成一个文件夹的「scan → 抽取最后一帧 → final opt」复合任务 shell 行。

    全部使用相对路径，需在 xTB 文件夹内执行（调用方负责 cd）。HPC 的
    单文件夹提交与打包提交共用本函数，保证两条提交路径的作业行为与
    产物（``scan.log`` / ``xtbscan_last.xyz`` / ``final_opt.log`` /
    ``oligomer.xtb_pull_final.xyz``）逐字一致。
    """
    scan_level_suffix = f" {scan_opt_level}" if scan_opt_level else ""
    final_level_suffix = f" {final_opt_level}" if final_opt_level else ""

    lines: list[str] = []
    lines.append("# --- Stage 1: distance-scan with constraint optimization ---")
    lines.append(
        f"{xtb_exe} {xyz_file} --input scan.inp --opt{scan_level_suffix} "
        f"{common_flags_str} &> scan.log"
    )
    lines.append("")
    lines.append("# --- Extract last frame of the scan ---")
    lines.extend(build_extract_last_frame_lines(xyz_file))
    lines.append("")
    lines.append("# --- Stage 2: unconstrained final opt ---")
    lines.append(
        f"{xtb_exe} xtbscan_last.xyz --opt{final_level_suffix} "
        f"{common_flags_str} &> final_opt.log"
    )
    lines.append("")
    lines.append("# --- Copy final geometry under stable filename ---")
    lines.append(
        "if [ -f xtbopt.xyz ]; then cp xtbopt.xyz oligomer.xtb_pull_final.xyz; fi"
    )
    return lines


# =====================================================================
# 本地直接运行（直接模式：本机 subprocess，不经任何队列）
# =====================================================================

def _execute_xtb_subprocess(argument_list: list[str], run_folder: str,
                            log_path: str, cores: int,
                            xtb_executable: str, *,
                            gnu_openmp_workarounds: bool = False) -> None:
    """在 *run_folder* 里启动 xtb 子进程，输出重定向到 *log_path*。

    环境变量按手册设置（见 :func:`xtb_runtime_environment`，
    *gnu_openmp_workarounds* 原样透传）；Linux 上额外在子进程里解除栈
    大小限制（手册要求的 ``ulimit -s unlimited``，防大分子栈溢出）。
    退出码非零时抛 RuntimeError 并附日志末尾内容。
    """
    import platform
    import subprocess

    environment = xtb_runtime_environment(
        xtb_executable, cores, gnu_openmp_workarounds=gnu_openmp_workarounds)

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

    with open(log_path, "w", encoding="utf-8", errors="ignore") as log_file:
        completed = subprocess.run(
            argument_list, cwd=run_folder, env=environment,
            stdout=log_file, stderr=subprocess.STDOUT,
            preexec_fn=preexec_function,
        )

    if completed.returncode != 0:
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as log_file:
                log_tail = "".join(log_file.readlines()[-25:])
        except OSError:
            log_tail = "(log unreadable)"
        raise RuntimeError(
            f"xtb exited with code {completed.returncode} "
            f"(command: {' '.join(argument_list)}; folder: {run_folder}).\n"
            f"Log tail ({log_path}):\n{log_tail}"
        )


def xtb_optimization_converged(log_path: str) -> bool:
    """判断一个 xtb 优化日志是否报告了几何优化收敛。

    xtb（含 g-xTB 修改版）在优化成功时打出横幅
    ``*** GEOMETRY OPTIMIZATION CONVERGED AFTER N ITERATIONS ***``，
    失败时打出 ``*** FAILED TO CONVERGE GEOMETRY OPTIMIZATION ***``；
    本函数以前者的出现为收敛判据。日志文件不存在时返回 False。
    """
    if not os.path.isfile(log_path):
        return False
    with open(log_path, encoding="utf-8", errors="ignore") as log_file:
        return "GEOMETRY OPTIMIZATION CONVERGED" in log_file.read()


def save_converged_gjf(log_path: str, *, charge: int, multiplicity: int,
                       converged_gjf_suffix: str = ".xtb_converged.gjf",
                       ) -> str | None:
    """优化收敛时，把最后一帧几何存成 ``<名><收敛后缀>.gjf``。

    以 *log_path*（``<名>.xtb.log``）判断收敛
    （见 :func:`xtb_optimization_converged`）：

    - 收敛：从同名优化轨迹 ``<名>.xtbopt_traj.xyz`` 取最后一帧，连同
      电荷与自旋多重度写成 Gaussian gjf，路径为日志名去掉 ``.xtb.log``
      后接 *converged_gjf_suffix*，返回该路径。后缀按方法区分：经典
      xTB 任务用默认值 ``.xtb_converged.gjf``，g-xTB 任务传
      ``.g_xtb_converged.gjf``（见 :func:`Chem_Lib.Lib_g_xTB.run_gxtb_opt_folder`）。
    - 未收敛：不写文件、返回 ``None``；如果之前的运行留下过同名
      收敛 gjf，顺带删除——保证该文件的存在性本身就是
      「这次优化已收敛」的可靠判据。

    Raises:
        ValueError:   *log_path* 不以 ``.xtb.log`` 结尾。
        RuntimeError: 日志显示收敛，但优化轨迹文件缺失。
    """
    if not log_path.endswith(".xtb.log"):
        raise ValueError(
            f"Expecting an xtb optimization log named <name>.xtb.log, "
            f"got: {log_path}"
        )
    stem = log_path[:-len(".xtb.log")]
    gjf_path = stem + converged_gjf_suffix

    if not xtb_optimization_converged(log_path):
        if os.path.isfile(gjf_path):
            os.remove(gjf_path)
        return None

    trajectory_path = stem + ".xtbopt_traj.xyz"
    if not os.path.isfile(trajectory_path):
        raise RuntimeError(
            f"{log_path} reports a converged optimization but the "
            f"trajectory product is missing: {trajectory_path}"
        )

    last_frame = XYZ_file(trajectory_path, last_only=True).last_coordinate
    last_frame.gjf_file(
        title=f"Converged geometry from {os.path.basename(log_path)}",
        filename=gjf_path,
        override_charge=charge,
        override_multiplicity=multiplicity,
    )
    return gjf_path


def run_xTB_task_folder(
    xtb_folder: str,
    *,
    task: str,
    solvent: str,
    method: str,
    opt_level: str | None = None,
    structure: str | None = None,
    cores: int = 1,
) -> str:
    """在本机直接运行一个单步 xTB 任务（不经 HPC 队列）。

    输入文件夹约定、任务类型、产物复制命名都与 HPC 提交
    （:func:`HPC_Lib.HPC_xTB.submit_xTB_folder`）完全一致——同一个
    文件夹既可以本地跑，也可以提交集群。使用随 Chem_Lib 分发的
    可执行文件（按平台自动选 Windows / Linux 版，见
    :func:`resolve_xtb_executable`），环境变量按手册自动设置。

    Args:
        xtb_folder:  xTB 计算文件夹，须含 ``.xyz`` / ``.CHRG`` / ``.UHF``
                     （``md``/``omd`` 另需 ``md.inp``，``pull`` 另需
                     ``scan.inp``）。
        task:        **必填**。``energy`` / ``opt`` / ``ohess`` / ``md`` /
                     ``omd`` / ``pull`` 之一。
        solvent:     **必填**。ALPB 溶剂名，气相必须显式写 ``"none"``。
                     无默认值。
        method:      **必填**。``GFN2`` / ``GFN1`` / ``GFN0`` / ``GFNFF``。
                     无默认值。GFN0 不能配溶剂（ALPB 未参数化）。
        opt_level:   优化精度等级；不填时让 xtb 用自身默认（normal）。
        structure:   输入结构 ``.xyz`` 文件名；不填时自动选择。
        cores:       OpenMP 线程数（``OMP_NUM_THREADS=<cores>,1``）。
                     默认 1（单核）——xTB 的 OpenMP 并行效率低，且
                     单任务本来就运行迅速（分钟级），一般不必要多核
                     并行；大批量吞吐靠多个单核任务并行（如 HPC 打包
                     提交），不靠单任务多核（2026-08-13 用户裁定）。

    Returns:
        日志文件（``<结构名或文件夹名>.xtb.log``）的完整路径。产物
        文件（如 ``<结构名>.xtbopt_traj.xyz``）位置与 HPC 版一致。
        ``opt`` / ``ohess`` 任务运行结束后还会检查优化是否收敛（见
        :func:`xtb_optimization_converged`），收敛时把最后一帧几何
        另存为 ``<结构名>.xtb_converged.gjf``（见
        :func:`save_converged_gjf`），未收敛则不生成该文件。

    Raises:
        RuntimeError: xtb 退出码非零（附日志末尾内容）。
    """
    import shutil

    task = resolve_task(task)
    method_upper, method_flag = resolve_method_flag(method)
    _, solvent_flag = resolve_solvent_flag(solvent, method_upper)
    opt_level = resolve_opt_level(opt_level)

    xtb_folder = os.path.abspath(xtb_folder)
    if not os.path.isdir(xtb_folder):
        raise FileNotFoundError(f"xTB calculation folder not found: {xtb_folder}")
    xyz_file = validate_task_folder(xtb_folder, task, structure)

    with open(os.path.join(xtb_folder, ".CHRG")) as f:
        charge = int(f.read().strip())
    with open(os.path.join(xtb_folder, ".UHF")) as f:
        unpaired_electrons = int(f.read().strip())

    folder_stem = os.path.basename(xtb_folder.rstrip("/\\"))
    scan_inp_has_scan_section = False
    if task == "pull":
        scan_inp_content = open(os.path.join(xtb_folder, "scan.inp")).read()
        scan_inp_has_scan_section = "$scan" in scan_inp_content
    specification = _single_step_task_specification(
        task, xyz_file, folder_stem, opt_level, scan_inp_has_scan_section)

    xtb_executable = resolve_xtb_executable()
    argument_list = ([xtb_executable, xyz_file]
                     + specification["task_flags"]
                     + method_flag.split()
                     + ["--chrg", str(charge), "--uhf", str(unpaired_electrons)]
                     + solvent_flag.split())

    log_path = os.path.join(xtb_folder, specification["log_name"])
    _execute_xtb_subprocess(argument_list, xtb_folder, log_path,
                            cores, xtb_executable)

    for source_name, target_name in specification["products"]:
        source_path = os.path.join(xtb_folder, source_name)
        if os.path.isfile(source_path):
            shutil.copy(source_path, os.path.join(xtb_folder, target_name))

    if task in ("opt", "ohess"):
        save_converged_gjf(log_path, charge=charge,
                           multiplicity=unpaired_electrons + 1)

    return log_path


def xTB_optimize_gjf(
    gjf_path: str,
    output_folder: str,
    *,
    solvent: str,
    method: str,
    opt_level: str | None = None,
    cores: int = 1,
    xyz_filename: str | None = None,
) -> xTB_opt_result:
    """从 Gaussian gjf 出发，在本机完成一次 xTB 几何优化并返回解析结果。

    三步串联：:func:`xTB_input_from_Gaussian_gjf` 生成输入文件夹 →
    :func:`run_xTB_task_folder` 以 ``task="opt"`` 本地运行 →
    用 :class:`xTB_opt_result` 解析 ``<结构名>.xtbopt_traj.xyz``
    （及同名 ``.xtb.log`` 里的能量）。

    Args:
        gjf_path:       Gaussian 输入文件（含电荷 / 自旋多重度 / 坐标）。
        output_folder:  计算文件夹路径，不存在时自动创建。
        solvent:        **必填**。ALPB 溶剂名或 ``"none"``（气相）。
        method:         **必填**。``GFN2`` / ``GFN1`` / ``GFN0`` / ``GFNFF``。
        opt_level:      优化精度等级；不填用 xtb 自身默认（normal）。
        cores:          OpenMP 线程数，默认 1（单核）。xTB 并行效率低、
                        单任务本来就快，一般不必要多核，见
                        :func:`run_xTB_task_folder`。
        xyz_filename:   输出 XYZ 文件名；默认用 gjf 基名。

    Returns:
        :class:`xTB_opt_result`（含优化后坐标与能量，单位 kJ/mol）。
        优化收敛时，计算文件夹里还会多一个
        ``<结构名>.xtb_converged.gjf``（最后一帧几何，见
        :func:`save_converged_gjf`）。
    """
    method_upper, _ = resolve_method_flag(method)
    solvent_name, _ = resolve_solvent_flag(solvent, method_upper)

    xyz_path = xTB_input_from_Gaussian_gjf(
        gjf_path, output_folder, xyz_filename=xyz_filename,
        solvent=(None if solvent_name == "none" else solvent_name),
        method=method_upper,
    )
    run_xTB_task_folder(
        output_folder, task="opt", solvent=solvent, method=method,
        opt_level=opt_level, structure=os.path.basename(xyz_path),
        cores=cores,
    )
    trajectory_path = os.path.splitext(xyz_path)[0] + ".xtbopt_traj.xyz"
    if not os.path.isfile(trajectory_path):
        raise RuntimeError(
            f"xTB optimization finished but the trajectory product is "
            f"missing: {trajectory_path}"
        )
    return xTB_opt_result(trajectory_path)


if __name__ == '__main__':
    pass