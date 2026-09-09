"""
Lib_Connectivity.py
===================

Match a substructure (drawn in ChemDraw and exported via
``Edit > Copy As > Mol Text``, which yields V3000 MOL text) against an
optimized 3D geometry. Supports half/dashed bonds in the ChemDraw drawing,
intended for transition-state matching.

Pipeline
--------
    query  = parse_chemdraw_mol_text(text)        # ChemDraw V3000 → graph
    target = compute_target(elements, coords)     # XYZ → graph w/ full|partial bonds
    result = find_substructure(query, target)     # VF2 subgraph monomorphism

Bond classification on the target side uses the reduced bond length

    rho_ij = d_ij / (r_cov_i + r_cov_j)

with three tiers (defaults configurable):
    rho <= tau_full                  -> 'full'    (definitely bonded)
    tau_full < rho <= tau_partial    -> 'partial' (forming / breaking)
    rho >  tau_partial               -> not a bond

A query bond marked partial in ChemDraw (V3000 type 9, or DISP=COORD/DASH/HASH)
matches both 'full' and 'partial' target bonds. A query bond drawn as solid
matches only 'full' (unless ``allow_partial_for_full`` is enabled).

Hydrogen atoms are kept in the target. Subgraph monomorphism naturally
ignores target H atoms unless the query references them, so H atoms involved
in a reaction are picked up only when the user draws them in ChemDraw.

Bond-order graph toolkit (added 2026-07)
----------------------------------------
Independent of the ChemDraw/TS matching above, this module also provides a
sparse bond-order-graph toolkit built on RDKit:

    BondOrderGraph                    elements + {(i, j): order} sparse storage
    smiles_to_bond_order_graph()      SMILES -> BondOrderGraph (aromatic = 1.5)
    mol_to_bond_order_graph()         RDKit Mol -> BondOrderGraph
    bond_order_graph_to_mol()         inverse; only orders 1/2/3/1.5 accepted
    bond_order_graph_to_smiles()      inverse, straight to (canonical) SMILES
    find_double_bonds()               atom pairs with bond order exactly 2
    find_conjugated_systems()         conjugated atom groups (RDKit conjugation,
                                      incl. lone-pair heteroatoms, e.g. enol-ether O)
    find_aromatic_rings()             aromatic rings via RDKit aromaticity
    perceive_xyz_bonds()              XYZ -> plain connectivity (rdDetermineBonds)
    match_xyz_to_bond_order_graph()   map graph atom numbering onto an XYZ file
    trim_saturated_side_chain()       cut saturated side chains of a conjugated
                                      molecule at acyclic sp3 carbons, cap with H
    trim_saturated_side_chain_smiles()  SMILES-in / SMILES-out convenience wrapper
"""

from __future__ import annotations

import re
import warnings
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    _HAS_RDKIT = True
except ImportError:
    Chem = None
    rdDetermineBonds = None
    _HAS_RDKIT = False

try:
    import networkx as nx
    from networkx.algorithms import isomorphism as _nx_iso
    _HAS_NX = True
except ImportError:
    nx = None
    _nx_iso = None
    _HAS_NX = False


# ---------------------------------------------------------------------------
# Covalent radii table (with overrides)
# ---------------------------------------------------------------------------

# Pyykkö single-bond covalent radii (Å), used only when RDKit is unavailable.
_FALLBACK_COV_RADII: dict[str, float] = {
    "H": 0.32, "He": 0.46,
    "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63,
    "F":  0.64, "Ne": 0.67,
    "Na": 1.55, "Mg": 1.39, "Al": 1.26, "Si": 1.16, "P": 1.11, "S": 1.03,
    "Cl": 0.99, "Ar": 0.96,
    "K":  1.96, "Ca": 1.71,
    "Br": 1.14, "I": 1.33,
}


class RadiusTable:
    """Covalent radii lookup with element- and pair-level user overrides.

    Pair overrides supply the FULL ``r_i + r_j`` sum and take precedence over
    element overrides; element overrides take precedence over the RDKit /
    Pyykkö default.
    """

    def __init__(self) -> None:
        self._elem_overrides: dict[str, float] = {}
        self._pair_overrides: dict[frozenset, float] = {}

    def set_element_radius(self, element: str, radius: float) -> None:
        self._elem_overrides[element] = float(radius)

    def set_pair_sum(self, e1: str, e2: str, r_sum: float) -> None:
        self._pair_overrides[frozenset((e1, e2))] = float(r_sum)

    def element_radius(self, element: str) -> float:
        if element in self._elem_overrides:
            return self._elem_overrides[element]
        if _HAS_RDKIT:
            return Chem.GetPeriodicTable().GetRcovalent(element)
        return _FALLBACK_COV_RADII.get(element, 1.5)

    def pair_sum(self, e1: str, e2: str) -> float:
        key = frozenset((e1, e2))
        if key in self._pair_overrides:
            return self._pair_overrides[key]
        return self.element_radius(e1) + self.element_radius(e2)


# ---------------------------------------------------------------------------
# Query side: parse ChemDraw V3000 MOL text
# ---------------------------------------------------------------------------

@dataclass
class QueryAtom:
    idx: int                 # 1-based index from the MOL block
    element: str
    x: float = 0.0
    y: float = 0.0


@dataclass
class QueryBond:
    idx: int
    a1: int                  # 1-based atom indices
    a2: int
    order: int               # 1, 2, 3, 4 (aromatic). 0 = unspecified / any
    is_partial: bool = False


@dataclass
class QueryStructure:
    atoms: list[QueryAtom] = field(default_factory=list)
    bonds: list[QueryBond] = field(default_factory=list)

    def element(self, atom_idx: int) -> str:
        for a in self.atoms:
            if a.idx == atom_idx:
                return a.element
        raise KeyError(atom_idx)


# ChemDraw maps half/dashed bonds to one of these DISP attribute values.
_PARTIAL_DISP_VALUES = {"COORD", "DASH", "HASH", "HOLLOW", "DOT", "DOTTED"}


def parse_chemdraw_mol_text(text: str) -> QueryStructure:
    """Parse V3000 MOL text from ChemDraw 'Edit > Copy As > Mol Text'.

    Bonds whose V3000 type is 9 or that carry DISP=COORD/DASH/HASH/HOLLOW are
    flagged as partial (half-bond / TS / forming / breaking).
    """
    structure = QueryStructure()
    section: Optional[str] = None  # 'atom' | 'bond' | None

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped.startswith("M  V30"):
            continue
        body = stripped[len("M  V30"):].strip()
        upper = body.upper()

        if upper == "BEGIN ATOM":
            section = "atom"; continue
        if upper == "END ATOM":
            section = None; continue
        if upper == "BEGIN BOND":
            section = "bond"; continue
        if upper == "END BOND":
            section = None; continue

        parts = body.split()
        if section == "atom" and len(parts) >= 5:
            structure.atoms.append(QueryAtom(
                idx=int(parts[0]),
                element=parts[1],
                x=float(parts[2]),
                y=float(parts[3]),
            ))
        elif section == "bond" and len(parts) >= 4:
            bidx = int(parts[0])
            btype = int(parts[1])
            a1, a2 = int(parts[2]), int(parts[3])
            extras = " ".join(parts[4:])

            is_partial = (btype == 9)
            disp_match = re.search(r"DISP\s*=\s*(\w+)", extras, re.IGNORECASE)
            if disp_match and disp_match.group(1).upper() in _PARTIAL_DISP_VALUES:
                is_partial = True

            order = btype if btype in (1, 2, 3, 4) else 0
            structure.bonds.append(QueryBond(
                idx=bidx, a1=a1, a2=a2,
                order=order, is_partial=is_partial,
            ))

    return structure


# ---------------------------------------------------------------------------
# Target side: build connectivity from XYZ
# ---------------------------------------------------------------------------

@dataclass
class TargetBond:
    a1: int                            # 0-based atom index
    a2: int
    distance: float
    rho: float
    classification: str                # 'full' or 'partial'
    rdkit_order: Optional[float] = None


@dataclass
class TargetStructure:
    elements: list[str]
    coords: np.ndarray                 # shape (N, 3)
    bonds: list[TargetBond] = field(default_factory=list)
    radius_table: Optional[RadiusTable] = None
    rho_matrix: Optional[np.ndarray] = None
    rdkit_mol: "Optional[Chem.Mol]" = None
    tau_full: float = 1.15
    tau_partial: float = 1.45


def read_xyz(text_or_path: str) -> tuple[list[str], np.ndarray]:
    """Read XYZ format. Argument may be a file path or the file's text content."""
    if "\n" not in text_or_path and len(text_or_path) < 4096:
        with open(text_or_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = text_or_path
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = int(lines[0].split()[0])
    elements: list[str] = []
    coords = np.zeros((n, 3), dtype=float)
    for i, line in enumerate(lines[2:2 + n]):
        parts = line.split()
        elements.append(parts[0])
        coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return elements, coords


def _make_xyz_block(elements: Iterable[str], coords: np.ndarray) -> str:
    elements = list(elements)
    out = [str(len(elements)), ""]
    for e, (x, y, z) in zip(elements, coords):
        out.append(f"{e} {x:.8f} {y:.8f} {z:.8f}")
    return "\n".join(out)


def compute_target(
    elements: list[str],
    coords: np.ndarray,
    *,
    tau_full: float = 1.15,
    tau_partial: float = 1.45,
    radius_table: Optional[RadiusTable] = None,
    use_rdkit_perception: bool = True,
    charge: int = 0,
) -> TargetStructure:
    """Build a TargetStructure from elements + 3D coordinates.

    For every atom pair the reduced bond length ``rho = d / (r_i + r_j)`` is
    computed; pairs with ``rho <= tau_full`` are 'full' bonds, those with
    ``tau_full < rho <= tau_partial`` are 'partial' bonds.

    If ``use_rdkit_perception`` is True and RDKit is available, bond orders
    are additionally perceived with ``rdDetermineBonds`` and stored on each
    TargetBond as ``rdkit_order`` (used only when ``match_bond_order`` is
    requested at match time).
    """
    if radius_table is None:
        radius_table = RadiusTable()
    coords = np.asarray(coords, dtype=float)
    n = len(elements)

    diff = coords[:, None, :] - coords[None, :, :]
    distance_matrix = np.sqrt((diff * diff).sum(axis=-1))

    rho = np.zeros_like(distance_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            rs = radius_table.pair_sum(elements[i], elements[j])
            r = distance_matrix[i, j] / rs if rs > 0 else np.inf
            rho[i, j] = rho[j, i] = r

    rdkit_orders: dict[frozenset, float] = {}
    rdkit_mol = None
    if use_rdkit_perception and _HAS_RDKIT:
        try:
            xyz_block = _make_xyz_block(elements, coords)
            raw = Chem.MolFromXYZBlock(xyz_block)
            if raw is not None:
                rw = Chem.RWMol(raw)
                rdDetermineBonds.DetermineBonds(rw, charge=charge)
                rdkit_mol = rw.GetMol()
                for b in rdkit_mol.GetBonds():
                    a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    rdkit_orders[frozenset((a, c))] = b.GetBondTypeAsDouble()
        except Exception:
            rdkit_mol = None
            rdkit_orders = {}

    bonds: list[TargetBond] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = rho[i, j]
            if r <= tau_full:
                cls = "full"
            elif r <= tau_partial:
                cls = "partial"
            else:
                continue
            bonds.append(TargetBond(
                a1=i, a2=j,
                distance=float(distance_matrix[i, j]),
                rho=float(r),
                classification=cls,
                rdkit_order=rdkit_orders.get(frozenset((i, j))),
            ))

    return TargetStructure(
        elements=list(elements),
        coords=coords,
        bonds=bonds,
        radius_table=radius_table,
        rho_matrix=rho,
        rdkit_mol=rdkit_mol,
        tau_full=tau_full,
        tau_partial=tau_partial,
    )


def normalize_thresholds_from_structure(
    target: TargetStructure,
    *,
    full_quantile: float = 0.95,
    partial_extra: float = 0.30,
    partial_extra_sigma: float = 2.0,
) -> tuple[float, float]:
    """Suggest (tau_full, tau_partial) from the target's own ρ distribution.

    Pairs with ρ < 1.05 are taken as 'definitely bonded'; tau_full is the
    chosen quantile of that population, and tau_partial is the population
    mean + ``partial_extra_sigma`` × σ + ``partial_extra``. This absorbs
    systematic DFT bond-length bias.
    """
    if target.rho_matrix is None:
        raise ValueError("target has no rho_matrix")
    n = len(target.elements)
    iu = np.triu_indices(n, k=1)
    rhos = target.rho_matrix[iu]
    definite = rhos[rhos < 1.05]
    if definite.size < 3:
        return target.tau_full, target.tau_partial
    mu = float(np.mean(definite))
    sigma = float(np.std(definite))
    tau_full = float(np.quantile(definite, full_quantile))
    tau_partial = mu + partial_extra_sigma * sigma + partial_extra
    return tau_full, max(tau_partial, tau_full + 0.10)


# ---------------------------------------------------------------------------
# Substructure matching (NetworkX VF2 with custom comparators)
# ---------------------------------------------------------------------------

def _query_to_nx(query: QueryStructure):
    if not _HAS_NX:
        raise RuntimeError("networkx is required for substructure matching")
    g = nx.Graph()
    for a in query.atoms:
        g.add_node(a.idx, element=a.element)
    for b in query.bonds:
        g.add_edge(b.a1, b.a2, order=b.order, is_partial=b.is_partial)
    return g


def _target_to_nx(target: TargetStructure):
    if not _HAS_NX:
        raise RuntimeError("networkx is required for substructure matching")
    g = nx.Graph()
    for i, e in enumerate(target.elements):
        g.add_node(i, element=e)
    for b in target.bonds:
        g.add_edge(b.a1, b.a2,
                   classification=b.classification,
                   rdkit_order=b.rdkit_order,
                   rho=b.rho, distance=b.distance)
    return g


def _node_match(qa, ta) -> bool:
    return qa["element"] == ta["element"]


def _make_edge_match(*, match_bond_order: bool, allow_partial_for_full: bool):
    def em(target_edge, query_edge) -> bool:
        # NOTE: NetworkX passes (G1_edge, G2_edge); we always set up the matcher
        # as GraphMatcher(target_graph, query_graph), so the first arg is target.
        cls = target_edge["classification"]
        if query_edge["is_partial"]:
            return cls in ("full", "partial")
        if cls == "partial" and not allow_partial_for_full:
            return False
        if not match_bond_order or query_edge["order"] in (0, 4):
            return True
        tord = target_edge.get("rdkit_order")
        if tord is None:
            return True
        return abs(tord - query_edge["order"]) < 0.01
    return em


@dataclass
class MatchResult:
    """Outcome of a substructure search."""
    mappings: list[dict[int, int]]            # query atom (1-based) → target atom (0-based)
    query: QueryStructure
    target: TargetStructure
    adaptive_tau_partial: Optional[float] = None  # τ_partial used if widened

    def __bool__(self) -> bool:
        return bool(self.mappings)

    def __len__(self) -> int:
        return len(self.mappings)

    def report(self) -> str:
        if not self.mappings:
            return "No substructure match."
        out: list[str] = []
        if self.adaptive_tau_partial is not None:
            out.append(
                f"Match required widening tau_partial to "
                f"{self.adaptive_tau_partial:.3f}"
            )
        for k, m in enumerate(self.mappings):
            out.append(f"--- Mapping {k + 1} ---")
            for qi in sorted(m):
                ti = m[qi]
                out.append(
                    f"  Q{qi} ({self.query.element(qi)}) "
                    f"-> T{ti} ({self.target.elements[ti]})"
                )
            for qb in self.query.bonds:
                if qb.a1 not in m or qb.a2 not in m:
                    continue
                t1, t2 = m[qb.a1], m[qb.a2]
                tb = next(
                    (b for b in self.target.bonds
                     if {b.a1, b.a2} == {t1, t2}),
                    None,
                )
                qkind = "partial" if qb.is_partial else f"order={qb.order}"
                if tb is None:
                    out.append(
                        f"  bond Q{qb.a1}-Q{qb.a2} ({qkind}) -> no target bond"
                    )
                else:
                    out.append(
                        f"  bond Q{qb.a1}-Q{qb.a2} ({qkind}) -> "
                        f"T{t1}-T{t2}: d={tb.distance:.3f} A, "
                        f"rho={tb.rho:.3f}, {tb.classification}"
                    )
        return "\n".join(out)


def find_substructure(
    query: QueryStructure,
    target: TargetStructure,
    *,
    match_bond_order: bool = False,
    allow_partial_for_full: bool = False,
    adaptive_tau: bool = True,
    tau_partial_range: tuple[float, float] = (1.20, 1.70),
    tau_partial_step: float = 0.05,
) -> MatchResult:
    """Find all subgraph monomorphisms of ``query`` into ``target``.

    Parameters
    ----------
    match_bond_order
        If True, compare query bond order to RDKit-perceived target order.
        Aromatic / unspecified queries always pass.
    allow_partial_for_full
        If True, a query 'full' bond may also match a target 'partial' bond.
    adaptive_tau
        If no mapping is found at the target's current τ_partial, sweep
        τ_partial through ``tau_partial_range`` (inclusive) at ``tau_partial_step``
        intervals; the first τ at which the query matches is reported.
    """
    qg = _query_to_nx(query)
    tg = _target_to_nx(target)
    em = _make_edge_match(
        match_bond_order=match_bond_order,
        allow_partial_for_full=allow_partial_for_full,
    )

    def _run(target_graph) -> list[dict[int, int]]:
        gm = _nx_iso.GraphMatcher(
            target_graph, qg, node_match=_node_match, edge_match=em,
        )
        return [
            {q: t for t, q in mapping.items()}
            for mapping in gm.subgraph_monomorphisms_iter()
        ]

    mappings = _run(tg)
    used_tau: Optional[float] = None
    used_target = target

    if not mappings and adaptive_tau:
        tau_lo, tau_hi = tau_partial_range
        tau = tau_lo
        while tau <= tau_hi + 1e-9:
            new_target = compute_target(
                target.elements, target.coords,
                tau_full=target.tau_full,
                tau_partial=tau,
                radius_table=target.radius_table,
                use_rdkit_perception=False,
            )
            mappings = _run(_target_to_nx(new_target))
            if mappings:
                used_tau = tau
                used_target = new_target
                break
            tau += tau_partial_step

    return MatchResult(
        mappings=mappings,
        query=query,
        target=used_target,
        adaptive_tau_partial=used_tau,
    )


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def find_substructure_in_xyz(
    xyz: str,
    chemdraw_mol_text: str,
    *,
    tau_full: float = 1.15,
    tau_partial: float = 1.45,
    element_radii: Optional[dict[str, float]] = None,
    pair_radius_sums: Optional[dict[tuple[str, str], float]] = None,
    charge: int = 0,
    use_rdkit_perception: bool = True,
    auto_normalize: bool = False,
    **match_kwargs,
) -> MatchResult:
    """One-shot helper: XYZ + ChemDraw Mol Text -> MatchResult.

    Parameters
    ----------
    xyz
        Either a path to an XYZ file or the file's text content.
    chemdraw_mol_text
        V3000 MOL text from ChemDraw 'Edit > Copy As > Mol Text'.
    element_radii
        ``{element: r_cov}`` overrides for individual elements.
    pair_radius_sums
        ``{(e1, e2): r_sum}`` overrides for the FULL pair sum
        ``r_e1 + r_e2``; takes precedence over ``element_radii``.
    auto_normalize
        If True, derive ``(tau_full, tau_partial)`` from the structure's own
        ρ distribution before matching.
    **match_kwargs
        Forwarded to :func:`find_substructure`.
    """
    elements, coords = read_xyz(xyz)

    rt = RadiusTable()
    if element_radii:
        for e, r in element_radii.items():
            rt.set_element_radius(e, r)
    if pair_radius_sums:
        for (e1, e2), r in pair_radius_sums.items():
            rt.set_pair_sum(e1, e2, r)

    target = compute_target(
        elements, coords,
        tau_full=tau_full, tau_partial=tau_partial,
        radius_table=rt,
        use_rdkit_perception=use_rdkit_perception,
        charge=charge,
    )
    if auto_normalize:
        tf, tp = normalize_thresholds_from_structure(target)
        target = compute_target(
            elements, coords,
            tau_full=tf, tau_partial=tp,
            radius_table=rt,
            use_rdkit_perception=use_rdkit_perception,
            charge=charge,
        )

    query = parse_chemdraw_mol_text(chemdraw_mol_text)
    return find_substructure(query, target, **match_kwargs)


# ---------------------------------------------------------------------------
# Bond-order graph toolkit
# ---------------------------------------------------------------------------

def _require_rdkit() -> None:
    if not _HAS_RDKIT:
        raise RuntimeError(
            "RDKit is required for this function but is not importable in the "
            "current environment (install it with `pip install rdkit` or add "
            "it to the project's dependencies)."
        )


@dataclass
class BondOrderGraph:
    """Molecular graph stored as an element list + sparse pairwise bond orders.

    ``bond_orders`` maps ``(i, j)`` (0-based atom indices, normalized so that
    ``i < j``) to the bond order. Any pair absent from the dict has order 0,
    so the format supports querying the bond order between two arbitrary
    atoms via :meth:`get_bond_order` while storing only the actual bonds.
    Aromatic bonds are stored with order 1.5.
    """

    elements: list[str]
    bond_orders: dict[tuple[int, int], float] = field(default_factory=dict)
    formal_charges: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.elements)
        if not self.formal_charges:
            self.formal_charges = [0] * n
        if len(self.formal_charges) != n:
            raise ValueError(
                f"formal_charges has {len(self.formal_charges)} entries but "
                f"there are {n} atoms"
            )
        normalized: dict[tuple[int, int], float] = {}
        for (i, j), order in self.bond_orders.items():
            if not (0 <= i < n and 0 <= j < n) or i == j:
                raise ValueError(f"invalid bond atom pair ({i}, {j}) for {n} atoms")
            key = (i, j) if i < j else (j, i)
            order = float(order)
            if order <= 0:
                raise ValueError(f"bond {key} has non-positive order {order}")
            if key in normalized and normalized[key] != order:
                raise ValueError(f"conflicting duplicate orders for bond {key}")
            normalized[key] = order
        self.bond_orders = normalized

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    def get_bond_order(self, i: int, j: int) -> float:
        """Bond order between atoms ``i`` and ``j``; 0.0 if they are not bonded."""
        return self.bond_orders.get((i, j) if i < j else (j, i), 0.0)

    def neighbors(self, i: int) -> list[int]:
        return sorted(
            (b if a == i else a)
            for a, b in self.bond_orders
            if i in (a, b)
        )

    def adjacency(self) -> dict[int, list[int]]:
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_atoms)}
        for a, b in self.bond_orders:
            adj[a].append(b)
            adj[b].append(a)
        return adj

    def to_dense_matrix(self) -> np.ndarray:
        mat = np.zeros((self.n_atoms, self.n_atoms))
        for (i, j), order in self.bond_orders.items():
            mat[i, j] = mat[j, i] = order
        return mat

    @classmethod
    def from_dense_matrix(
        cls,
        elements: Iterable[str],
        matrix: np.ndarray,
        formal_charges: Optional[Iterable[int]] = None,
    ) -> "BondOrderGraph":
        elements = list(elements)
        matrix = np.asarray(matrix, dtype=float)
        n = len(elements)
        if matrix.shape != (n, n):
            raise ValueError(f"matrix shape {matrix.shape} does not match {n} atoms")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("bond-order matrix must be symmetric")
        orders = {
            (i, j): float(matrix[i, j])
            for i in range(n)
            for j in range(i + 1, n)
            if matrix[i, j] != 0
        }
        charges = list(formal_charges) if formal_charges is not None else []
        return cls(elements, orders, charges)


def smiles_to_bond_order_graph(
    smiles: str,
    *,
    add_hydrogens: bool = True,
    kekulize: bool = False,
) -> BondOrderGraph:
    """SMILES -> :class:`BondOrderGraph` via RDKit.

    Aromatic bonds get order 1.5 unless ``kekulize=True`` (then alternating
    1/2). With ``add_hydrogens=True`` (default) hydrogens become explicit
    atoms of the graph, which the XYZ-matching and trimming functions expect.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    if kekulize:
        mol = Chem.Mol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol_to_bond_order_graph(mol)


def mol_to_bond_order_graph(mol: "Chem.Mol") -> BondOrderGraph:
    """RDKit Mol -> :class:`BondOrderGraph` (orders via ``GetBondTypeAsDouble``)."""
    _require_rdkit()
    orders: dict[tuple[int, int], float] = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        orders[(min(i, j), max(i, j))] = bond.GetBondTypeAsDouble()
    return BondOrderGraph(
        elements=[atom.GetSymbol() for atom in mol.GetAtoms()],
        bond_orders=orders,
        formal_charges=[atom.GetFormalCharge() for atom in mol.GetAtoms()],
    )


def bond_order_graph_to_mol(
    graph: BondOrderGraph,
    *,
    explicit_hydrogens: Optional[bool] = None,
    sanitize: bool = True,
) -> "Chem.Mol":
    """Inverse of :func:`smiles_to_bond_order_graph`: graph -> RDKit Mol.

    Only integer orders 1/2/3 plus the aromatic order 1.5 are convertible;
    any other fractional order raises ValueError. Atoms/bonds carrying order
    1.5 are marked aromatic and must form a valid aromatic system, otherwise
    sanitization (kekulization/valence check) raises ValueError.

    ``explicit_hydrogens``: True -> atoms get no implicit hydrogens (the
    graph is taken as complete, H atoms explicit); False -> RDKit fills
    implicit hydrogens on heavy atoms; None (default) -> auto: True iff the
    graph contains at least one H atom.
    """
    _require_rdkit()
    if explicit_hydrogens is None:
        explicit_hydrogens = any(el == "H" for el in graph.elements)
    order_to_type = {
        1.0: Chem.BondType.SINGLE,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        1.5: Chem.BondType.AROMATIC,
    }
    bad = {
        pair: order
        for pair, order in graph.bond_orders.items()
        if order not in order_to_type
    }
    if bad:
        raise ValueError(
            "unsupported bond orders (only 1, 2, 3 and aromatic 1.5 can be "
            f"converted back to a molecule): {bad}"
        )
    rw = Chem.RWMol()
    for element, charge in zip(graph.elements, graph.formal_charges):
        atom = Chem.Atom(element)
        atom.SetFormalCharge(int(charge))
        if explicit_hydrogens:
            atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for (i, j), order in graph.bond_orders.items():
        rw.AddBond(i, j, order_to_type[order])
        if order == 1.5:
            rw.GetBondBetweenAtoms(i, j).SetIsAromatic(True)
            rw.GetAtomWithIdx(i).SetIsAromatic(True)
            rw.GetAtomWithIdx(j).SetIsAromatic(True)
    mol = rw.GetMol()
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise ValueError(
                f"bond-order graph does not describe a valid molecule: {exc}"
            ) from exc
    return mol


def bond_order_graph_to_smiles(graph: BondOrderGraph, *, canonical: bool = True) -> str:
    """Graph -> (canonical) SMILES; explicit hydrogens are folded back in."""
    mol = bond_order_graph_to_mol(graph)
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=canonical)


def find_double_bonds(graph: BondOrderGraph) -> list[tuple[int, int]]:
    """All atom pairs bonded with order exactly 2.

    Aromatic bonds stored as 1.5 are NOT included; if the graph was built
    with ``kekulize=True`` the alternating aromatic bonds show up here.
    """
    return sorted(pair for pair, order in graph.bond_orders.items() if order == 2.0)


def find_conjugated_systems(
    graph: BondOrderGraph,
    *,
    mol: "Optional[Chem.Mol]" = None,
) -> list[set[int]]:
    """Conjugated systems as atom-index sets, largest first.

    A system is a connected component of RDKit-conjugated bonds. RDKit's
    conjugation includes heteroatoms whose lone pair conjugates with an
    adjacent pi system (e.g. the O of a vinyl ether / anisole), but not
    heteroatoms flanked by saturated atoms only.
    """
    _require_rdkit()
    if mol is None:
        mol = bond_order_graph_to_mol(graph)
    parent = list(range(mol.GetNumAtoms()))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    involved: set[int] = set()
    for bond in mol.GetBonds():
        if not bond.GetIsConjugated():
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        involved.update((i, j))
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj
    groups: dict[int, set[int]] = {}
    for a in involved:
        groups.setdefault(_find(a), set()).add(a)
    return sorted(groups.values(), key=lambda s: (-len(s), min(s)))


def find_aromatic_rings(
    graph: BondOrderGraph,
    *,
    mol: "Optional[Chem.Mol]" = None,
) -> list[tuple[int, ...]]:
    """Aromatic rings (per RDKit aromaticity perception) as atom-index tuples."""
    _require_rdkit()
    if mol is None:
        mol = bond_order_graph_to_mol(graph)
    info = mol.GetRingInfo()
    rings: list[tuple[int, ...]] = []
    for atom_ring, bond_ring in zip(info.AtomRings(), info.BondRings()):
        if all(mol.GetBondWithIdx(b).GetIsAromatic() for b in bond_ring):
            rings.append(tuple(atom_ring))
    return rings


def _skeleton_mol(elements: Iterable[str], bond_pairs: Iterable[tuple[int, int]]) -> "Chem.Mol":
    """Element-labelled, all-single-bond mol for pure connectivity matching."""
    rw = Chem.RWMol()
    for element in elements:
        atom = Chem.Atom(element)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for i, j in bond_pairs:
        rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def perceive_xyz_bonds(
    elements: list[str],
    coords: np.ndarray,
    *,
    cov_factor: float = 1.3,
) -> list[tuple[int, int]]:
    """Perceive plain connectivity (no bond orders) from 3D coordinates.

    Thin wrapper around RDKit's ``rdDetermineBonds.DetermineConnectivity``.
    """
    _require_rdkit()
    raw = Chem.MolFromXYZBlock(_make_xyz_block(elements, np.asarray(coords, dtype=float)))
    if raw is None:
        raise ValueError("could not parse the XYZ input")
    rw = Chem.RWMol(raw)
    rdDetermineBonds.DetermineConnectivity(rw, covFactor=cov_factor)
    return sorted(
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in rw.GetBonds()
    )


def match_xyz_to_bond_order_graph(
    xyz: Union[str, tuple[list[str], np.ndarray]],
    graph: BondOrderGraph,
    *,
    return_all: bool = False,
    cov_factors: tuple[float, ...] = (1.3, 1.4, 1.5),
) -> Union[dict[int, int], list[dict[int, int]]]:
    """Map the atom numbering of ``graph`` onto the atoms of an XYZ geometry.

    Parameters
    ----------
    xyz
        Path to an XYZ file, the XYZ text itself, or ``(elements, coords)``.
    graph
        The reference :class:`BondOrderGraph`. If it contains no explicit H
        atoms, hydrogens in the XYZ are ignored and the mapping covers heavy
        atoms only.
    return_all
        If True, return every graph->xyz mapping (symmetry-equivalent atoms
        give several); otherwise return one mapping ``{graph_idx: xyz_idx}``.
    cov_factors
        Covalent-radius scale factors tried in order for the XYZ bond
        perception; the sweep absorbs mildly stretched bonds. A factor other
        than the first that succeeds is reported via ``warnings``.

    Raises ``ValueError`` when the element compositions differ or no
    connectivity-isomorphic mapping exists at any covalent factor.
    """
    _require_rdkit()
    if isinstance(xyz, str):
        xyz_elements, coords = read_xyz(xyz)
    else:
        xyz_elements, coords = xyz
        xyz_elements = list(xyz_elements)
        coords = np.asarray(coords, dtype=float)

    graph_has_h = any(el == "H" for el in graph.elements)
    if graph_has_h:
        kept_xyz_indices = list(range(len(xyz_elements)))
    else:
        kept_xyz_indices = [i for i, el in enumerate(xyz_elements) if el != "H"]
    pos = {old: new for new, old in enumerate(kept_xyz_indices)}
    target_elements = [xyz_elements[i] for i in kept_xyz_indices]

    if Counter(target_elements) != Counter(graph.elements):
        raise ValueError(
            "element composition mismatch between graph and XYZ"
            + (" (XYZ hydrogens ignored because the graph has none)" if not graph_has_h else "")
            + f": graph {dict(Counter(graph.elements))} vs xyz {dict(Counter(target_elements))}"
        )

    query = _skeleton_mol(graph.elements, graph.bond_orders.keys())
    perceived_bond_counts: list[int] = []
    for cov_factor in cov_factors:
        all_pairs = perceive_xyz_bonds(xyz_elements, coords, cov_factor=cov_factor)
        target_pairs = [
            (pos[i], pos[j]) for i, j in all_pairs if i in pos and j in pos
        ]
        perceived_bond_counts.append(len(target_pairs))
        target = _skeleton_mol(target_elements, target_pairs)
        if return_all:
            matches = target.GetSubstructMatches(query, uniquify=False, maxMatches=100000)
        else:
            match = target.GetSubstructMatch(query)
            matches = (match,) if match else ()
        if not matches:
            continue
        if cov_factor != cov_factors[0]:
            warnings.warn(
                f"XYZ bond perception needed covFactor={cov_factor} (default "
                f"{cov_factors[0]} gave no isomorphic connectivity); the "
                "geometry may contain stretched bonds."
            )
        if len(target_pairs) != len(graph.bond_orders):
            warnings.warn(
                f"XYZ bond perception found {len(target_pairs)} bonds vs "
                f"{len(graph.bond_orders)} in the graph (covFactor="
                f"{cov_factor}); the mapping is a best-effort match with the "
                "extra perceived contacts ignored."
            )
        mappings = [
            {q: kept_xyz_indices[t] for q, t in enumerate(m)} for m in matches
        ]
        return mappings if return_all else mappings[0]

    raise ValueError(
        "no atom mapping found: the connectivity perceived from the XYZ "
        f"coordinates is not isomorphic to the bond graph (graph has "
        f"{len(graph.bond_orders)} bonds; perception with covFactor "
        f"{tuple(cov_factors)} found {perceived_bond_counts} bonds). The "
        "geometry may be distorted, or elements/charges may not correspond."
    )


@dataclass
class TrimResult:
    """Result of :func:`trim_saturated_side_chain`."""

    trimmed: BondOrderGraph
    atom_map: dict[int, int]      # old atom index -> new index, for every kept atom
    removed_atoms: set[int]       # old indices that were cut away
    cut_atoms: list[int]          # old indices of the sp3 carbons where cuts happened
    core_atoms: set[int]          # old indices of the conjugated core that was used


def trim_saturated_side_chain(
    graph: BondOrderGraph,
    n_keep: int,
    *,
    core_atoms: Optional[Iterable[int]] = None,
) -> TrimResult:
    """Trim the saturated side chains of a conjugated molecule to ``n_keep`` atoms.

    Walking outward from the conjugated core, every branch is cut at the
    FIRST atom that satisfies all of:

    * graph distance from the core >= ``n_keep`` (heavy atoms count, the
      first non-core atom having distance 1; heteroatoms count too),
    * carbon with formal charge 0,
    * sp3, i.e. every incident bond has order exactly 1,
    * not a member of any ring.

    The cut atom itself is kept and capped with hydrogen (usually becoming a
    terminal CH3); everything beyond it is removed. Positions failing the
    test (heteroatoms, sp2/sp carbons, ring atoms) push the cut further out,
    so an embedded C=C or ether oxygen is never broken and rings are never
    opened — e.g. an alkyl tail on a saturated macrocycle fused to the core
    is cut at its first acyclic carbon even though that sits beyond
    ``n_keep``.

    The core defaults to the largest conjugated system found by
    :func:`find_conjugated_systems` (RDKit conjugation, which includes
    heteroatoms whose lone pair conjugates with an adjacent pi system, e.g.
    a vinyl-ether oxygen). Pass ``core_atoms`` to override.

    Hydrogen caps are added only when the input graph carries explicit H
    atoms; a heavy-atom-only graph relies on implicit hydrogens instead.
    """
    _require_rdkit()
    if n_keep < 1:
        raise ValueError(f"n_keep must be >= 1, got {n_keep}")
    mol = bond_order_graph_to_mol(graph)
    n = graph.n_atoms
    adj = graph.adjacency()

    if core_atoms is None:
        systems = find_conjugated_systems(graph, mol=mol)
        if not systems:
            raise ValueError(
                "no conjugated system found in the molecule; pass core_atoms "
                "explicitly"
            )
        core = set(systems[0])
    else:
        core = set(core_atoms)
        if not core:
            raise ValueError("core_atoms must not be empty")
        out_of_range = [a for a in core if not (0 <= a < n)]
        if out_of_range:
            raise ValueError(f"core_atoms out of range: {out_of_range}")

    ring_atoms = {i for i in range(n) if mol.GetAtomWithIdx(i).IsInRing()}

    infinity = float("inf")
    distance_from_core: list[float] = [infinity] * n
    queue: deque[int] = deque()
    for c in core:
        distance_from_core[c] = 0
        queue.append(c)
    while queue:
        x = queue.popleft()
        for y in adj[x]:
            if distance_from_core[y] == infinity:
                distance_from_core[y] = distance_from_core[x] + 1
                queue.append(y)

    def _component_beyond(start: int, blocked: int) -> set[int]:
        component = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y != blocked and y not in component:
                    component.add(y)
                    stack.append(y)
        return component

    removed: set[int] = set()
    cut_atoms: list[int] = []
    added_h: dict[int, int] = {}

    candidates = sorted(
        (i for i in range(n) if i not in core and distance_from_core[i] != infinity),
        key=lambda i: distance_from_core[i],
    )
    for x in candidates:
        if x in removed or distance_from_core[x] < n_keep:
            continue
        if graph.elements[x] != "C" or x in ring_atoms:
            continue
        if graph.formal_charges[x] != 0:
            continue
        if any(graph.get_bond_order(x, y) != 1.0 for y in adj[x]):
            continue  # not sp3: carries a double/triple/aromatic bond
        outward = [
            y for y in adj[x]
            if distance_from_core[y] > distance_from_core[x] and graph.elements[y] != "H" and y not in removed
        ]
        if not outward:
            continue  # already terminal (only hydrogens beyond)
        n_cut = 0
        for y in outward:
            component = _component_beyond(y, blocked=x)
            if component & core:
                continue  # safety net; impossible for an acyclic cut atom
            removed |= component
            n_cut += 1
        if n_cut:
            cut_atoms.append(x)
            added_h[x] = n_cut

    conjugated_atoms = {
        idx
        for bond in mol.GetBonds() if bond.GetIsConjugated()
        for idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    }
    notable = removed & (ring_atoms | conjugated_atoms)
    if notable:
        warnings.warn(
            f"trim_saturated_side_chain removed {len(notable)} ring/conjugated "
            f"atoms lying beyond a cut point (old atom indices "
            f"{sorted(notable)}); the cut rule only guarantees the cut atom "
            "itself is an acyclic sp3 carbon."
        )

    kept = [i for i in range(n) if i not in removed]
    atom_map = {old: new for new, old in enumerate(kept)}
    elements = [graph.elements[i] for i in kept]
    charges = [graph.formal_charges[i] for i in kept]
    orders: dict[tuple[int, int], float] = {}
    for (i, j), order in graph.bond_orders.items():
        if i in atom_map and j in atom_map:
            orders[(atom_map[i], atom_map[j])] = order
    if any(el == "H" for el in graph.elements):
        for x, count in added_h.items():
            for _ in range(count):
                new_idx = len(elements)
                elements.append("H")
                charges.append(0)
                orders[(atom_map[x], new_idx)] = 1.0

    trimmed = BondOrderGraph(elements, orders, charges)
    bond_order_graph_to_mol(trimmed)  # fail fast if the result is not a valid molecule

    return TrimResult(
        trimmed=trimmed,
        atom_map=atom_map,
        removed_atoms=removed,
        cut_atoms=cut_atoms,
        core_atoms=core,
    )


def trim_saturated_side_chain_smiles(
    smiles: str,
    n_keep: int,
    *,
    core_atoms: Optional[Iterable[int]] = None,
) -> str:
    """SMILES-in / SMILES-out convenience wrapper for :func:`trim_saturated_side_chain`."""
    graph = smiles_to_bond_order_graph(smiles)
    result = trim_saturated_side_chain(graph, n_keep, core_atoms=core_atoms)
    return bond_order_graph_to_smiles(result.trimmed)


__all__ = [
    "RadiusTable",
    "QueryAtom", "QueryBond", "QueryStructure",
    "TargetBond", "TargetStructure",
    "MatchResult",
    "parse_chemdraw_mol_text",
    "read_xyz",
    "compute_target",
    "normalize_thresholds_from_structure",
    "find_substructure",
    "find_substructure_in_xyz",
    # bond-order graph toolkit
    "BondOrderGraph",
    "TrimResult",
    "smiles_to_bond_order_graph",
    "mol_to_bond_order_graph",
    "bond_order_graph_to_mol",
    "bond_order_graph_to_smiles",
    "find_double_bonds",
    "find_conjugated_systems",
    "find_aromatic_rings",
    "perceive_xyz_bonds",
    "match_xyz_to_bond_order_graph",
    "trim_saturated_side_chain",
    "trim_saturated_side_chain_smiles",
]
