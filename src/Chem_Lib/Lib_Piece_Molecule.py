import numpy as np

from Chem_Lib.Lib import *


# 注：rotation_vec1_to_vec2 已移入 Lib_Coordinates（Coordinates.translate_and_align_bond
# 需要调用它），经由 Chem_Lib.Lib 的星号导入链，本模块内仍可直接使用该名字。


def rotate_molecule(molecule: Coordinates, atom_count1, atom_count2, atom_count3, point1, point2, point3):
    '''
    atom 1,2,3 is the atom count start from 1
    Rotate a molecule so that three atom in the atom is defined "to" three points in space
    The first point defines the origin
    The second point defines the direction to align atom1-atom2 bond
    Last, plain of atom1-2-3 is aligned to point1-2-3
    :return: a list of std_coordinate str
    '''

    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    atom_count1, atom_count2, atom_count3 = atom_count1 - 1, atom_count2 - 1, atom_count3 - 1
    coordinate_np = molecule.coordinates_np
    atom1, atom2, atom3 = coordinate_np[atom_count1], coordinate_np[atom_count2], coordinate_np[atom_count3]

    # for i in coordinate_np:
    #     print(i)
    # print("-----")

    # translate
    translate_vector = point1 - atom1
    coordinate_np = [x + translate_vector for x in coordinate_np]
    atom1, atom2, atom3 = coordinate_np[atom_count1], coordinate_np[atom_count2], coordinate_np[atom_count3]

    # for i in coordinate_np:
    #     print(i)
    # print("-----")

    # align_bond
    original_vector = atom2 - atom1
    target_vector = point2 - point1
    # print(original_vector,target_vector)
    rotation_matrix = rotation_vec1_to_vec2(original_vector, target_vector)
    coordinate_np = [x - atom1 for x in coordinate_np]
    coordinate_np = [rotation_matrix.dot(x) for x in coordinate_np]
    coordinate_np = [x + atom1 for x in coordinate_np]
    atom1, atom2, atom3 = coordinate_np[atom_count1], coordinate_np[atom_count2], coordinate_np[atom_count3]

    # align plain
    normal_vector1 = get_normal_vector(atom1, atom2, atom3)
    normal_vector2 = get_normal_vector(point1, point2, point3)

    rotation_matrix1 = rotation_vec1_to_vec2(normal_vector1, normal_vector2)
    rotation_matrix2 = rotation_vec1_to_vec2(normal_vector1, -normal_vector2)

    point3_test1 = rotation_matrix1.dot(atom3 - atom2) + atom2
    point3_test2 = rotation_matrix2.dot(atom3 - atom2) + atom2

    if np.linalg.norm(point3_test1 - point3) < np.linalg.norm(point3_test2 - point3):
        rotation_matrix = rotation_matrix1
    else:
        rotation_matrix = rotation_matrix2

    coordinate_np = [x - atom2 for x in coordinate_np]
    coordinate_np = [rotation_matrix.dot(x) for x in coordinate_np]
    coordinate_np = [x + atom2 for x in coordinate_np]

    ret = []
    for i in range(molecule.atom_count):
        ret.append("{} {:.20f} {:.20f} {:.20f}".format(molecule.elements[i], *coordinate_np[i]))

    return ret


def get_vector_above_plain(molecule: Coordinates,
                           plain_atoms: list,
                           plain_origin: int,
                           plain_direction_atom: int,
                           distance_start_Angs: float,
                           length_of_vector: float,
                           reverse_direction: False):
    '''
    get a vector above plain
    for the purpose of e.g. conjugated addition to O=C-C=C by R-OH, get a initial vector for O->R that perpendicular to the pi-plain
    :param molecule:
    :param plain_atoms: list of 3-atom to define the plain, start form 1
    :param plain_origin: an int for the atom index, start from 1, or a np.array to define the origin point
    :param plain_direction_atom: an atom that is off-plain, so that the positive direction of a plain can be determined.
    :param distance_start: start point distance from the plain, unit angs
    :param length_of_vector: length_of_vector, unit angs
    :param reverse_direction: if True, the origin + normal vector of the plain will be pointing away from the atom, otherwise it will be pointing towards it
    :return: a 2-tuple, with the start and end point of the above-plain atoms
    '''
    coordinate_np = molecule.coordinates_np
    if isinstance(plain_origin, int):
        plain_origin = coordinate_np[plain_origin - 1]

    plain_atoms = [x - 1 for x in plain_atoms]
    dump, normal_vector, dump, dump = get_plane(molecule, plain_atoms)
    plain_direction_atom = coordinate_np[plain_direction_atom - 1]
    dist_1 = np.linalg.norm(plain_origin + normal_vector - plain_direction_atom)
    dist_2 = np.linalg.norm(plain_origin - normal_vector - plain_direction_atom)
    if dist_1 > dist_2:
        normal_vector = - normal_vector

    if reverse_direction:
        normal_vector = -normal_vector

    start_atom = plain_origin + normal_vector * distance_start_Angs
    end_atom = start_atom + normal_vector * length_of_vector

    return start_atom, end_atom


def get_vector_using_bond(molecule: Coordinates,
                          vector_atom1,
                          vector_atom2,
                          distance_start_Angs: float,
                          length_of_vector: float,
                          origin=None):
    '''
    get vector for atom1-atom2, extent this vector to a specifed amount from a specified origin
    :param molecule:
    :param vector_atom1:
    :param vector_atom2: the default origin
    :param distance_start_Angs:
    :param length_of_vector:
    :param origin: a defined origin using an atom index start from 1, or an np.ndarray / list of float;
                   if not defined, atom 2 in regard as the origin
    :return:
    '''

    atom1 = molecule.coordinates_np[vector_atom1 - 1]
    atom2 = molecule.coordinates_np[vector_atom2 - 1]
    if isinstance(origin, int):
        origin = molecule.coordinates_np[origin - 1]
    elif isinstance(origin, np.ndarray):
        pass
    elif isinstance(origin, list):
        origin = np.array(origin)
    else:
        origin = atom2

    start_atom = origin + get_unit_vector(atom2 - atom1) * distance_start_Angs
    end_atom = start_atom + get_unit_vector(atom2 - atom1) * length_of_vector

    return start_atom, end_atom


def best_fit_plane_normal(points, positive_direction_point=(1, 0, 0)):
    """
    Given an Nx3 array of 3D points, determine the plane that minimizes
    the RMS distance to the points. Return the plane's normal vector
    (pointing outward). The returned normal is a unit vector.

    :param positive_direction_point: As the normal vector can have two directions, this point will be as whether to invert the normal vector,
    such that the vector of the center of all the points to the positive_direction_point has a positive dot product with the normal vector
    :param points: A NumPy array of shape (N, 3), representing N points in 3D.
    :return: A NumPy array of shape (3,), representing the plane's normal vector.
    """
    # Ensure input is a NumPy array
    points = np.asarray(points)

    # Edge case: if there are not enough points or they degenerate, handle gracefully
    if len(points) < 3:
        raise ValueError("At least three points are required for a best-fit plane.")

    # 1) Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # 2) Center the points at the origin
    centered_points = points - centroid

    # 3) Compute the covariance matrix
    #    shape: (3, 3). Using the transpose is common, but any standard covariance approach is fine.
    cov_matrix = np.cov(centered_points, rowvar=False)  # rowvar=False => each row is an observation

    # 4) Perform eigen-decomposition (or SVD) to find the smallest eigenvalue/eigenvector
    #    The eigenvector corresponding to the smallest eigenvalue is the plane normal.
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Identify the index of the smallest eigenvalue
    min_eig_idx = np.argmin(eigenvalues)

    # “normal” is the eigenvector corresponding to smallest eigenvalue
    normal = eigenvectors[:, min_eig_idx]

    # Normalize to get a unit normal vector
    normal /= np.linalg.norm(normal)

    positive_direction_point = np.array(positive_direction_point)
    positive_direction_vector = positive_direction_point - center_point(points)
    if np.dot(normal, positive_direction_vector) < 0:
        normal *= -1

    return normal


def center_point(points):
    """
    Calculate the centroid of a given list of 3D points.

    :param points: A list or numpy array of shape (n, 3) where n is the number of points
    :return: A numpy array of shape (3,) representing the centroid of the points
    """
    # Convert the input list to a numpy array if it isn't already
    points = np.array(points)

    # Compute the mean along axis 0 (mean of each dimension)
    centroid = np.mean(points, axis=0)

    return centroid


def find_cap_H_bonded_atom(molecule: Coordinates, h_atom_index: int, fragment_name: str = "molecule") -> int:
    """
    Validate that atom #h_atom_index (start from 1) is an H atom, then find the atom
    it caps: the nearest non-H atom within the X-H bond range of that heavy element
    (≤ 1.3 Å for C/N/O — 1.09 Å is typical C-H, 1.3 covers strained geometries —
    and element-specific ceilings for Si/P/S/Ge/Se/..., whose X-H bonds are longer).

    :param molecule: the fragment
    :param h_atom_index: atom number of the cap H, start from 1
    :param fragment_name: name used in error messages
    :return: atom number (start from 1) of the heavy-atom neighbor of the cap H
    """
    if not (1 <= h_atom_index <= molecule.atom_count):
        raise ValueError(
            f"{fragment_name}: H atom index {h_atom_index} out of range (have {molecule.atom_count} atoms)"
        )
    if molecule.elements[h_atom_index - 1].strip().upper() != "H":
        raise ValueError(
            f"{fragment_name}: atom #{h_atom_index} is {molecule.elements[h_atom_index - 1]!r}, not H"
        )
    # X-H bond-length ceilings by heavy element. 1.3 Å covers C/N/O-H (1.09 /
    # 1.01 / 0.96 Å typical) incl. strained geometries; heavier p-block
    # elements bond H at 1.3-1.5 Å (Si-H 1.48, P-H 1.42, S-H 1.34, Ge-H 1.53,
    # Se-H 1.46), so a flat 1.3 Å rule would wrongly reject a cap H on e.g.
    # a siloxane -Si(CH3)2-H end group.
    X_H_BOND_MAX_BY_ELEMENT = {
        "SI": 1.65, "P": 1.6, "S": 1.5, "GE": 1.7, "SE": 1.65, "AS": 1.7,
        "SN": 1.9, "B": 1.4, "AL": 1.8,
    }
    H_BOND_MAX_DEFAULT = 1.3
    distances_from_H = molecule.distance_matrix[h_atom_index - 1]
    candidates = []
    for atom_index_0based, element in enumerate(molecule.elements):
        element_symbol = element.strip().upper()
        if atom_index_0based == h_atom_index - 1 or element_symbol == "H":
            continue
        bond_max = X_H_BOND_MAX_BY_ELEMENT.get(element_symbol, H_BOND_MAX_DEFAULT)
        if distances_from_H[atom_index_0based] <= bond_max:
            candidates.append((distances_from_H[atom_index_0based], atom_index_0based + 1))
    if not candidates:
        raise ValueError(
            f"{fragment_name}: H atom #{h_atom_index} has no non-H neighbor "
            f"within the X-H bond ceiling ({H_BOND_MAX_DEFAULT} Å for C/N/O, "
            f"element-specific for Si/P/S/...) — wrong index or detached atom?"
        )
    candidates.sort()
    return candidates[0][1]


def _pick_dihedral_anchor(molecule: Coordinates, center_atom_index, exclude_atom_indexes):
    """Pick the nearest non-H neighbor (within 1.8 Å) of the center atom, excluding
    `exclude_atom_indexes`, as a dihedral anchor. All atom indexes start from 1."""
    distances_from_center = molecule.distance_matrix[center_atom_index - 1]
    neighbors = []
    for atom_index_0based, element in enumerate(molecule.elements):
        atom_index = atom_index_0based + 1
        if atom_index == center_atom_index or atom_index in exclude_atom_indexes:
            continue
        if element.strip().upper() == "H":
            continue
        if distances_from_center[atom_index_0based] < 1.8:
            neighbors.append((distances_from_center[atom_index_0based], atom_index))
    if not neighbors:
        raise ValueError(
            "could not find a non-H neighbor of bonding C atom for "
            "dihedral anchor — molecule may be too sparse / wrong index"
        )
    neighbors.sort(key=lambda neighbor: (neighbor[0], neighbor[1]))
    return neighbors[0][1]


def dimer_pull_init_geom_from_monomer(molecule_A: Coordinates,
                                      h_atom_index_A: int,
                                      molecule_B: Coordinates,
                                      h_atom_index_B: int,
                                      *,
                                      min_initial_CC_A: float | None = None,
                                      pick_dihedral_anchors: bool = False,
                                      dihedral_atom_A_1based: int | None = None,
                                      dihedral_atom_B_1based: int | None = None) -> dict:
    """
    Build the initial dimer geometry for an xTB Pull dimerization from two pre-relaxed
    fragments, each supplied with the 1-indexed atom number of the cap H that will be
    replaced by the new C-C bond:

      1. For fragment A: identify the C atom bonded to H[h_atom_index_A]. Translate A so
         that C is at the origin and rotate A so the C->H vector points in +x; the cap H
         is then at (d_CH_A, 0, 0).
      2. For fragment B: same, but C->H in -x. Then translate B in +x so that its leftmost
         atom satisfies  min_x_B - max_x_A == 2 Å — a fixed 2 Å of empty space between
         the fragments along the bonding axis.
      3. Delete the two cap H atoms and concatenate the atom lists (A first, B second).
      4. Optionally pick a dihedral anchor atom on each side (the nearest non-H neighbor
         of each bonding C in the combined post-deletion list) for locking the
         A-c_A-c_B-B dihedral during the pull.

    :param molecule_A: fragment A (typically read from a Gaussian gjf)
    :param h_atom_index_A: 1-indexed atom number of the cap H on fragment A
    :param molecule_B: fragment B
    :param h_atom_index_B: 1-indexed atom number of the cap H on fragment B
    :param min_initial_CC_A: if given, raise ValueError when the laid-out C-C distance is
                             below this (suggests wrong H indices / fragment overlap)
    :param pick_dihedral_anchors: whether to determine dihedral anchor atoms
    :param dihedral_atom_A_1based: explicit anchor (1-indexed, in the combined
                                   post-deletion list) overriding the automatic pick
    :param dihedral_atom_B_1based: same, for the B side
    :return: dict with keys:
             "elements"                 — combined element list (cap H atoms removed)
             "coordinates_np"           — combined list of np.array coordinates
             "c_A_global_1based"        — 1-indexed position of fragment A's bonding C
                                          in the combined list
             "c_B_global_1based"        — same, for fragment B
             "initial_CC_A"             — initial C-C distance in Å
             "dihedral_A_global_1based" — anchor atom (None unless picked / given)
             "dihedral_B_global_1based" — same, for the B side
    """
    bonding_C_A = find_cap_H_bonded_atom(molecule_A, h_atom_index_A, "fragment_A")
    bonding_C_B = find_cap_H_bonded_atom(molecule_B, h_atom_index_B, "fragment_B")

    # Rotate so the C->H bond points in +x (A) / -x (B), then translate so the
    # bonding C sits at the origin (both bonding C atoms must lie on the x axis).
    fragment_A = molecule_A.align_bond(bonding_C_A, h_atom_index_A, (+1.0, 0.0, 0.0)).translate_atom_to_origin(bonding_C_A)
    fragment_B = molecule_B.align_bond(bonding_C_B, h_atom_index_B, (-1.0, 0.0, 0.0)).translate_atom_to_origin(bonding_C_B)

    # Translate B in +x so min_x_B - max_x_A == 2 Å
    rightmost_x_A = max(point[0] for point in fragment_A.coordinates_np)
    leftmost_x_B = min(point[0] for point in fragment_B.coordinates_np)  # currently negative
    fragment_B = fragment_B.translate((rightmost_x_A + 2.0 - leftmost_x_B, 0.0, 0.0))

    initial_CC_distance = float(np.linalg.norm(fragment_B.coordinates_np[bonding_C_B - 1]
                                               - fragment_A.coordinates_np[bonding_C_A - 1]))
    if min_initial_CC_A is not None and initial_CC_distance < min_initial_CC_A:
        raise ValueError(
            f"initial C-C distance {initial_CC_distance:.3f} Å is already below "
            f"target {min_initial_CC_A:.3f} Å — wrong H index or fragment "
            f"overlap. Re-check h_atom_index_A={h_atom_index_A} / "
            f"h_atom_index_B={h_atom_index_B}."
        )

    # Delete the two cap H atoms and concatenate (A first, B second) via the
    # Coordinates - / + operators.
    combined = (fragment_A - h_atom_index_A) + (fragment_B - h_atom_index_B)

    def _index_after_cap_deletion(atom_index, cap_H_index):
        # Deleting one atom shifts every later atom's number down by one.
        return atom_index - (1 if atom_index > cap_H_index else 0)

    c_A_global_1based = _index_after_cap_deletion(bonding_C_A, h_atom_index_A)
    c_B_global_1based = (fragment_A.atom_count - 1) + _index_after_cap_deletion(bonding_C_B, h_atom_index_B)

    dihedral_A_global_1based = None
    dihedral_B_global_1based = None
    if pick_dihedral_anchors:
        if dihedral_atom_A_1based is not None:
            dihedral_A_global_1based = int(dihedral_atom_A_1based)
        else:
            dihedral_A_global_1based = _pick_dihedral_anchor(
                combined, c_A_global_1based, exclude_atom_indexes={c_B_global_1based})
        if dihedral_atom_B_1based is not None:
            dihedral_B_global_1based = int(dihedral_atom_B_1based)
        else:
            dihedral_B_global_1based = _pick_dihedral_anchor(
                combined, c_B_global_1based, exclude_atom_indexes={c_A_global_1based})

    return {
        "elements": combined.elements,
        "coordinates_np": combined.coordinates_np,
        "c_A_global_1based": c_A_global_1based,
        "c_B_global_1based": c_B_global_1based,
        "initial_CC_A": initial_CC_distance,
        "dihedral_A_global_1based": dihedral_A_global_1based,
        "dihedral_B_global_1based": dihedral_B_global_1based,
    }
