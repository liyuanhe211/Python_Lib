"""RDKit 分子构建与 MMFF94 / UFF 力场优化的通用层。

本模块把此前散落在下游使用方项目里的两件事——**分子构建**（从
SMILES 得到一个封端好的、可以嵌入三维坐标的 RDKit 分子）与 **MMFF 优化**
（ETKDGv3 构象嵌入 + MMFF94 力场优化，缺参数时退回 UFF，按力场能量排序）
——提炼成跨项目通用函数（2026-09-02 迁移）。原实现分布在：

- 下游项目的寡聚体图构建模块（图层面的头尾拼接与封端，本模块的
  :func:`build_oligomer` 与 :func:`attachment_point_pairs` 逐行等价，
  该项目模块现已改为从这里 import，不再各留一份）；
- ``Chem_Lib/Lib_RDKit_Structure_From_SMILES.py`` 的
  ``_cap_attachment_points_with_hydrogen`` 与 ``_force_field_for_conformer``
  以及其中的嵌入 / 优化 / 排序循环（该模块现已改为调用本模块，只保留
  「把结果写成 Gaussian gjf」这一层）。

模块分成两组函数：

**分子构建**
  - :func:`attachment_point_pairs` —— 找出 SMILES 里的 ``*`` 连接占位原子
    及其唯一邻位重原子；
  - :func:`cap_attachment_points_with_hydrogen` —— 删掉全部 ``*`` 并以 H
    封端（单个片段用）；
  - :func:`build_oligomer` —— 把若干个各带两个 ``*`` 的双官能单体按头尾
    顺序拼成一条链，两个末端以 H 或 CH3 封端（聚合物寡聚体用）；
  - :func:`molecule_from_smiles` —— 上面两者的入口封装。

**MMFF / UFF 力场**
  - :func:`force_field_for_conformer` —— 为某个构象构造力场对象，优先
    MMFF94，分子缺 MMFF 参数时退回 UFF；
  - :func:`embed_conformers` —— ETKDGv3 嵌入，固定随机种子、
    ``numThreads=0``、先用确定性坐标，一个构象都嵌不出来时整体退回随机
    坐标模式再试一次；
  - :func:`optimize_and_rank_conformers` —— 逐个构象做力场优化（可关闭，
    此时只做单点能量用于排序），按能量升序排序；
  - :func:`optimized_conformers_from_smiles` —— 从 SMILES 一路走到「按
    能量升序排好的构象集合」的一次性入口。

三个约定跨项目保持不变，改动它们会让新数据与既有数据集不可比：

1. ETKDG 随机种子默认 :data:`DEFAULT_RANDOM_SEED`（``0xF00D``），保证可复现；
2. 力场优化上限默认 :data:`DEFAULT_MAX_ITERATIONS`（2000 步）；
3. 力场选择顺序是「MMFF94 优先、缺参数才退 UFF」，而不是反过来。

本模块只依赖 RDKit，在 Windows 与超算计算节点上同样可以直接运行。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import AllChem

#: ETKDG 默认随机种子。与既有数据集生成时的历史设定一致，
#: 换掉它会让同一个 SMILES 生成出不同的构象集合。
DEFAULT_RANDOM_SEED = 0xF00D

#: 力场优化的默认迭代上限。
DEFAULT_MAX_ITERATIONS = 2000

#: 未指定时默认采样的构象数量（沿用旧 ``embed_only`` 的 ``n_confs=8``）。
DEFAULT_CONFORMER_COUNT = 8

#: 允许的封端方式。``"H"`` 把末端连接位补氢，``"CH3"`` 把末端的 ``*``
#: 直接变成碳原子（sanitize 时补成甲基）。
CAP_CHOICES = ("H", "CH3")


# --------------------------------------------------------------------------- #
# 分子构建
# --------------------------------------------------------------------------- #
def attachment_point_pairs(molecule: Chem.Mol) -> list[tuple[int, int]]:
    """返回 ``[(占位原子序号, 邻位重原子序号), ...]``，按原子序号升序。

    对一个头尾相接的双官能单体来说，第一对是**头**、最后一对是**尾**
    （SMILES 里的书写顺序就是原子序号的升序）。

    Raises:
        ValueError: 某个 ``*`` 的邻位原子不是恰好一个。
    """
    pairs: list[tuple[int, int]] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 0:  # 占位原子 '*'
            neighbors = atom.GetNeighbors()
            if len(neighbors) != 1:
                raise ValueError(
                    f"attachment '*' must have exactly one neighbor, "
                    f"got {len(neighbors)}")
            pairs.append((atom.GetIdx(), neighbors[0].GetIdx()))
    return pairs


def _mark_pending_cap_hydrogen(neighbor: Chem.Atom) -> None:
    """给一个 ``NoImplicit`` 邻位原子记一笔「待补的显式封端氢」。

    方括号原子（硅氧烷单体的 ``[Si]``、季铵的 ``[n+]`` 等）RDKit 不会在
    sanitize 时补隐式 H，直接删掉它旁边的 ``*`` 会让分子悄悄少一个 H、
    变成开壳层自由基。这里先打标记，等占位原子全部删完后再统一把显式 H
    计数加上去（2026-08-23 修复；全库回归中恰好 18 个单体受此影响）。
    """
    pending = (neighbor.GetIntProp("_cap_explicit_h_pending")
               if neighbor.HasProp("_cap_explicit_h_pending") else 0)
    neighbor.SetIntProp("_cap_explicit_h_pending", pending + 1)


def _apply_pending_cap_hydrogens(editable: Chem.RWMol) -> None:
    """把 :func:`_mark_pending_cap_hydrogen` 打过标记的显式 H 计数落实。

    ``AddHs`` 仍会把全部 H 原子追加在末尾，因此重原子的排序约定不受影响。
    """
    for atom in editable.GetAtoms():
        if atom.HasProp("_cap_explicit_h_pending"):
            atom.SetNumExplicitHs(atom.GetNumExplicitHs()
                                  + atom.GetIntProp("_cap_explicit_h_pending"))
            atom.ClearProp("_cap_explicit_h_pending")


def cap_attachment_points_with_hydrogen(
        molecule: Chem.Mol) -> tuple[Chem.Mol, int]:
    """删除分子里全部 ``*`` 占位原子，并把空出来的连接位补氢。

    邻位原子是普通原子时，删掉 ``*`` 后 sanitize 会自动补隐式 H；邻位是
    方括号原子（``NoImplicit``）时不会，此处按
    :func:`_mark_pending_cap_hydrogen` 的办法显式补上。

    与 :func:`build_oligomer` 的 ``cap="H"`` 语义一致，区别只在于本函数
    处理的是单个片段、把**所有**占位原子都当末端封掉，而
    :func:`build_oligomer` 会先把相邻单元之间的占位原子成键消耗掉、只封
    整条链的两个末端。

    Returns:
        ``(封端后的分子, 删除的占位原子个数)``。分子里本来就没有 ``*``
        时原样返回，计数为 0。
    """
    dummy_indices = sorted(
        (atom.GetIdx() for atom in molecule.GetAtoms()
         if atom.GetAtomicNum() == 0),
        reverse=True,
    )
    if not dummy_indices:
        return molecule, 0
    editable = Chem.RWMol(molecule)
    for index in dummy_indices:
        for neighbor in editable.GetAtomWithIdx(index).GetNeighbors():
            if neighbor.GetNoImplicit():
                _mark_pending_cap_hydrogen(neighbor)
        editable.RemoveAtom(index)
    _apply_pending_cap_hydrogens(editable)
    capped = editable.GetMol()
    Chem.SanitizeMol(capped)
    return capped, len(dummy_indices)


def build_oligomer(unit_smiles: list[str], cap: str = "H") -> Chem.Mol:
    """把若干个各带两个 ``*`` 的双官能单体按头尾顺序拼成一条封端的链。

    Args:
        unit_smiles: 有序的单体 SMILES 列表，每个恰好带两个 ``*``。
        cap:         ``"H"``（末端连接位补氢）或 ``"CH3"``（末端补甲基）。

    Returns:
        一个 sanitize 过的 RDKit 分子，全部 ``*`` 已经成键或封端。

    原子排序约定（下游的 cap-H 序号记账完全依赖它，不要改）：
    先按 ``unit_smiles`` 的顺序把各单体的原子拼在一起，然后按序号从大到小
    删除占位原子，于是每个存活的重原子序号向下移动「序号比它小的占位原子
    个数」位；之后 ``Chem.AddHs`` 把全部 H 追加在末尾。

    Raises:
        ValueError: ``cap`` 不是 ``"H"`` / ``"CH3"``，某个 SMILES 不可解析，
                    或者某个单体的 ``*`` 个数不是恰好两个。
    """
    if cap not in CAP_CHOICES:
        raise ValueError(f"cap must be one of {CAP_CHOICES}, got {cap!r}")
    molecules = []
    for one_smiles in unit_smiles:
        molecule = Chem.MolFromSmiles(one_smiles)
        if molecule is None:
            raise ValueError(f"unparsable SMILES: {one_smiles!r}")
        if len(attachment_point_pairs(molecule)) != 2:
            raise ValueError(f"monomer must have exactly two '*': "
                             f"{one_smiles!r}")
        molecules.append(molecule)

    # 拼成一张图，同时记下每个单元的占位原子 / 邻位原子序号。
    combined = molecules[0]
    offsets = [0]
    for molecule in molecules[1:]:
        offsets.append(combined.GetNumAtoms())
        combined = Chem.CombineMols(combined, molecule)

    attachments = []
    for index, molecule in enumerate(molecules):
        (head_dummy, head_neighbor), (tail_dummy, tail_neighbor) = \
            attachment_point_pairs(molecule)
        offset = offsets[index]
        attachments.append({
            "head_dummy": head_dummy + offset,
            "head_neighbor": head_neighbor + offset,
            "tail_dummy": tail_dummy + offset,
            "tail_neighbor": tail_neighbor + offset,
        })

    editable = Chem.RWMol(combined)
    # 单元之间的键：第 i 个单元的尾邻位 → 第 i+1 个单元的头邻位。
    for index in range(len(molecules) - 1):
        editable.AddBond(attachments[index]["tail_neighbor"],
                         attachments[index + 1]["head_neighbor"],
                         Chem.BondType.SINGLE)

    # 末端占位原子 = 第一个单元的头 + 最后一个单元的尾；其余都是内部占位
    # 原子，它们所在的单元已经成键，直接删掉即可。
    terminal = {attachments[0]["head_dummy"], attachments[-1]["tail_dummy"]}
    internal = ({one["head_dummy"] for one in attachments}
                | {one["tail_dummy"] for one in attachments})
    internal -= terminal

    if cap == "CH3":
        for dummy_index in terminal:
            # '*' 改成碳原子，sanitize 时补成甲基。
            editable.GetAtomWithIdx(dummy_index).SetAtomicNum(6)
        to_remove = sorted(internal, reverse=True)
    else:  # H 封端：删掉全部占位原子，邻位在 sanitize 时补隐式 H
        # ……但 RDKit 从不给方括号原子补隐式 H（例如 2026-08-23 那批硅氧烷
        # 前体单体的 [Si] 连接位），不补偿的话封端后的分子会悄悄少一个 H、
        # 变成开壳层自由基。这里先给这类邻位打标记，等占位原子删完再补成
        # 显式 H 计数。
        for dummy_index in terminal:
            neighbor = editable.GetAtomWithIdx(dummy_index).GetNeighbors()[0]
            if neighbor.GetNoImplicit():
                _mark_pending_cap_hydrogen(neighbor)
        to_remove = sorted(internal | terminal, reverse=True)

    for index in to_remove:
        editable.RemoveAtom(index)

    _apply_pending_cap_hydrogens(editable)

    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def molecule_from_smiles(
    smiles: str,
    *,
    cap_attachment_points: bool = True,
) -> tuple[Chem.Mol, int]:
    """解析 SMILES 并（默认）把 ``*`` 连接占位原子以 H 封端。

    Args:
        smiles:                输入 SMILES。
        cap_attachment_points: True（默认）时调用
                               :func:`cap_attachment_points_with_hydrogen`；
                               False 时原样返回解析结果（分子里保留 ``*``，
                               通常不能直接嵌入三维坐标）。

    Returns:
        ``(分子, 封端掉的占位原子个数)``。返回的分子的 H 仍是隐式的，
        嵌入构象前需要自行 :func:`Chem.AddHs`（
        :func:`optimized_conformers_from_smiles` 已经代劳）。

    Raises:
        ValueError: SMILES 不可解析。
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Unparsable SMILES: {smiles!r}")
    if not cap_attachment_points:
        return molecule, 0
    return cap_attachment_points_with_hydrogen(molecule)


# --------------------------------------------------------------------------- #
# MMFF / UFF 力场
# --------------------------------------------------------------------------- #
def force_field_for_conformer(molecule_with_hydrogens: Chem.Mol,
                              conformer_id: int,
                              *,
                              allow_uff_fallback: bool = True):
    """为指定构象构造力场对象，返回 ``(力场对象, 力场名)``。

    优先 MMFF94；分子缺 MMFF 参数（或 MMFF 力场构造失败）时，在
    ``allow_uff_fallback`` 为 True 的情况下退回 UFF。两者都拿不到时返回
    ``(None, None)``——调用方必须显式处理这种情形，不要当作零能量。

    Args:
        molecule_with_hydrogens: 已经 ``AddHs`` 且已经嵌入构象的分子。
        conformer_id:            构象编号。
        allow_uff_fallback:      缺 MMFF 参数时是否允许退回 UFF（默认允许）。

    Returns:
        ``(力场对象, "MMFF94" 或 "UFF")``；都构造不出来时 ``(None, None)``。
    """
    if AllChem.MMFFHasAllMoleculeParams(molecule_with_hydrogens):
        properties = AllChem.MMFFGetMoleculeProperties(molecule_with_hydrogens)
        force_field = AllChem.MMFFGetMoleculeForceField(
            molecule_with_hydrogens, properties, confId=conformer_id)
        if force_field is not None:
            return force_field, "MMFF94"
    if not allow_uff_fallback:
        return None, None
    force_field = AllChem.UFFGetMoleculeForceField(
        molecule_with_hydrogens, confId=conformer_id)
    if force_field is not None:
        return force_field, "UFF"
    return None, None


def embed_conformers(
    molecule_with_hydrogens: Chem.Mol,
    number_of_conformers: int = DEFAULT_CONFORMER_COUNT,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    description: str = "the molecule",
) -> list[int]:
    """用 ETKDGv3 嵌入若干个三维构象，返回构象编号列表。

    嵌入参数是固定约定：指定随机种子（可复现）、``numThreads=0``（用满
    可用核心）、先用确定性坐标 ``useRandomCoords=False``；一个构象都嵌不
    出来时整体退回随机坐标模式再试一次。

    实际嵌出的构象数量可能少于要求（RDKit 对刚性小分子经常给不满），此时
    照实返回，**不**补齐、也**不**报错——由调用方决定要不要提示。

    Args:
        molecule_with_hydrogens: 已经 ``AddHs`` 的分子（会被就地写入构象）。
        number_of_conformers:    希望嵌入的构象数量。
        random_seed:             ETKDG 随机种子。
        description:             出错信息里用来指代这个分子的说法。

    Raises:
        ValueError:   ``number_of_conformers`` 小于 1。
        RuntimeError: 连随机坐标模式也嵌不出任何构象。
    """
    if number_of_conformers < 1:
        raise ValueError(f"number_of_conformers must be >= 1, "
                         f"got {number_of_conformers}")
    embedding_parameters = AllChem.ETKDGv3()
    embedding_parameters.randomSeed = random_seed
    embedding_parameters.numThreads = 0
    embedding_parameters.useRandomCoords = False
    conformer_ids = list(AllChem.EmbedMultipleConfs(
        molecule_with_hydrogens, numConfs=number_of_conformers,
        params=embedding_parameters))
    if not conformer_ids:
        embedding_parameters.useRandomCoords = True
        conformer_ids = list(AllChem.EmbedMultipleConfs(
            molecule_with_hydrogens, numConfs=number_of_conformers,
            params=embedding_parameters))
    if not conformer_ids:
        raise RuntimeError(
            f"RDKit failed to embed any conformer for {description}")
    return conformer_ids


@dataclass
class ConformerEnergy:
    """一个构象的力场能量记录。"""

    energy_kcal_mol: float
    conformer_id: int
    force_field_name: str
    optimized: bool


def optimize_and_rank_conformers(
    molecule_with_hydrogens: Chem.Mol,
    conformer_ids: list[int] | None = None,
    *,
    optimize: bool = True,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    allow_uff_fallback: bool = True,
    description: str = "the molecule",
) -> list[ConformerEnergy]:
    """逐个构象做力场优化（可关闭）并按能量升序排序。

    ``optimize=True``（默认）时对每个构象调用 ``Minimize(maxIts=...)``，
    分子的坐标被就地更新；``optimize=False`` 时保留原始几何，只算一次力场
    单点能量用于排序。排序是稳定的：能量相同的构象保持原有先后顺序。

    Args:
        molecule_with_hydrogens: 已经嵌入构象的分子（优化会就地改坐标）。
        conformer_ids:           要处理的构象编号；None 表示分子里的全部构象。
        optimize:                是否真的做力场优化。
        max_iterations:          优化迭代上限。
        allow_uff_fallback:      缺 MMFF 参数时是否允许退回 UFF。
        description:             出错信息里用来指代这个分子的说法。

    Returns:
        按 ``energy_kcal_mol`` 升序排好的 :class:`ConformerEnergy` 列表。

    Raises:
        RuntimeError: 某个构象连 MMFF 带 UFF 都构造不出力场——这种情况必须
                      响亮失败，不能当作能量缺失静默跳过。
    """
    if conformer_ids is None:
        conformer_ids = [conformer.GetId()
                         for conformer in molecule_with_hydrogens.GetConformers()]
    ranked: list[ConformerEnergy] = []
    for conformer_id in conformer_ids:
        force_field, force_field_name = force_field_for_conformer(
            molecule_with_hydrogens, conformer_id,
            allow_uff_fallback=allow_uff_fallback)
        if force_field is None:
            raise RuntimeError(
                f"Neither MMFF nor UFF could be constructed for "
                f"{description} (conformer {conformer_id})")
        if optimize:
            force_field.Minimize(maxIts=max_iterations)
        ranked.append(ConformerEnergy(
            energy_kcal_mol=force_field.CalcEnergy(),
            conformer_id=conformer_id,
            force_field_name=force_field_name,
            optimized=optimize,
        ))
    ranked.sort(key=lambda item: item.energy_kcal_mol)
    return ranked


def conformer_coordinates(
        molecule_with_hydrogens: Chem.Mol,
        conformer_id: int) -> list[tuple[float, float, float]]:
    """取出指定构象的笛卡尔坐标，顺序与分子的原子顺序一致。"""
    conformer = molecule_with_hydrogens.GetConformer(conformer_id)
    return [
        (conformer.GetAtomPosition(atom_index).x,
         conformer.GetAtomPosition(atom_index).y,
         conformer.GetAtomPosition(atom_index).z)
        for atom_index in range(molecule_with_hydrogens.GetNumAtoms())
    ]


def element_symbols(molecule: Chem.Mol) -> list[str]:
    """按原子顺序返回元素符号列表。"""
    return [atom.GetSymbol() for atom in molecule.GetAtoms()]


@dataclass
class ConformerEnsemble:
    """:func:`optimized_conformers_from_smiles` 的返回值。

    Attributes:
        molecule_with_hydrogens:        已 ``AddHs`` 并嵌好构象的分子。
        ranked_conformers:              按力场能量升序排好的构象记录。
        capped_attachment_point_count:  被 H 封端掉的 ``*`` 占位原子个数。
        requested_conformer_count:      调用方要求的构象数量（可能多于实际
                                        嵌出的数量）。
    """

    molecule_with_hydrogens: Chem.Mol
    ranked_conformers: list[ConformerEnergy] = field(default_factory=list)
    capped_attachment_point_count: int = 0
    requested_conformer_count: int = 0

    @property
    def elements(self) -> list[str]:
        """按原子顺序的元素符号列表。"""
        return element_symbols(self.molecule_with_hydrogens)

    def coordinates(self, conformer_id: int) -> list[tuple[float, float, float]]:
        """取出某个构象的笛卡尔坐标。"""
        return conformer_coordinates(self.molecule_with_hydrogens, conformer_id)


def optimized_conformers_from_smiles(
    smiles: str,
    number_of_conformers: int = DEFAULT_CONFORMER_COUNT,
    *,
    optimize: bool = True,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    log_prefix: str = "[Lib_MMFF]",
) -> ConformerEnsemble:
    """从 SMILES 一路走到「按力场能量升序排好的构象集合」。

    等价于依次调用 :func:`molecule_from_smiles` → ``Chem.AddHs`` →
    :func:`embed_conformers` → :func:`optimize_and_rank_conformers`，并把
    封端个数、构象数不足这两件值得知道的事打印出来（绝不静默）。

    Args:
        smiles:               输入 SMILES。含 ``*`` 时自动以 H 封端。
        number_of_conformers: 希望生成的构象数量。
        optimize:             True（默认）时做力场优化；False 时保留 ETKDG
                              原始几何，只做单点能量用于排序。
        max_iterations:       力场优化迭代上限。
        random_seed:          ETKDG 随机种子。
        log_prefix:           打印信息的前缀，方便调用方保持自己的日志口吻。

    Returns:
        :class:`ConformerEnsemble`。

    Raises:
        ValueError:   SMILES 不可解析，或 ``number_of_conformers`` 小于 1。
        RuntimeError: 嵌不出任何构象，或某个构象拿不到力场。
    """
    molecule, capped_count = molecule_from_smiles(smiles)
    if capped_count:
        print(f"{log_prefix} capped {capped_count} '*' attachment "
              f"point(s) with H")

    molecule_with_hydrogens = Chem.AddHs(molecule)
    conformer_ids = embed_conformers(
        molecule_with_hydrogens, number_of_conformers,
        random_seed=random_seed, description=f"SMILES {smiles!r}")
    if len(conformer_ids) < number_of_conformers:
        print(f"{log_prefix} WARNING: requested {number_of_conformers} "
              f"conformer(s) but RDKit only embedded {len(conformer_ids)}")

    ranked = optimize_and_rank_conformers(
        molecule_with_hydrogens, conformer_ids,
        optimize=optimize, max_iterations=max_iterations,
        description=f"SMILES {smiles!r}")
    return ConformerEnsemble(
        molecule_with_hydrogens=molecule_with_hydrogens,
        ranked_conformers=ranked,
        capped_attachment_point_count=capped_count,
        requested_conformer_count=number_of_conformers,
    )


if __name__ == "__main__":
    pass
