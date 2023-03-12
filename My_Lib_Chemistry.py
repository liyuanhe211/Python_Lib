# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import subprocess

import sys
import pathlib

Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Stock import *


def get_canonical_smiles(original_smiles):
    import subprocess
    if not original_smiles:
        return ""

    indigo_path = os.path.join(filename_class(os.path.realpath(__file__)).path, 'indigo-cano.exe').replace('/', '\\')

    print("Calling", ' '.join([indigo_path, '-', original_smiles]))

    try:
        ret = subprocess.check_output([indigo_path, '-', original_smiles])
    except subprocess.CalledProcessError:
        print("Indigo SMILES Bug Detected:", original_smiles)
        try:
            ret = subprocess.check_output(" ".join(['"' + indigo_path + '"', '-', '"' + original_smiles + '"', '-smiles', '-no-arom']), shell=True)
        except subprocess.CalledProcessError:
            return ""

    return ret.decode('utf-8').strip()

    # if not original_smiles:
    #     return ""
    #
    # from indigo.indigo import Indigo
    # indigo = Indigo()
    # try:
    #     mol = indigo.loadMolecule(original_smiles)
    # except:
    #     print("Ineffective SMILES:",original_smiles)
    #     return ""
    #
    # mol.aromatize()
    # try:
    #     ret = mol.canonicalSmiles() #有时有问题
    # except:
    #     ret = original_smiles
    #     print("Indigo SMILES Bug Detected:",original_smiles)
    #
    # # print("Canonicalization time","{:.2f}".format(time.time()-t1))
    #
    # return ret


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_of_np_vectors(v1, v2):
    """ Returns the angle in degree between vectors 'v1' and 'v2'::"""
    import numpy as np
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / math.pi * 180


def smiles_from_xyz(input_file):
    import subprocess
    babel_exe = r"C:\Program Files (x86)\OpenBabel-2.3.2\babel.exe"
    assert os.path.isfile(babel_exe), "OpenBabel not found."
    temp_file = r"Temp\temp_xyz_file_for_smiles.smi"

    subprocess.call([babel_exe, '-ixyz', input_file, "-osmi", temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(temp_file) as temp_file:
        ret = temp_file.read().split()
        if ret:
            return (get_canonical_smiles(ret[0]))
        else:
            print("SMILES ERROR! Original str:", ret)


def smiles_from_mol2(input_file):
    import subprocess
    babel_exe = r"C:\Program Files (x86)\OpenBabel-2.3.2\babel.exe"
    assert os.path.isfile(babel_exe), "OpenBabel not found."
    temp_file = r"Temp\temp_xyz_file_for_smiles.smi"

    subprocess.call([babel_exe, '-imol2', input_file, "-osmi", temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(temp_file) as temp_file:
        ret = temp_file.read().split()
        if ret:
            return (get_canonical_smiles(ret[0]))
        else:
            print("SMILES ERROR! Original str:", ret)


def get_bonds(filename, neighbor_list=False, add_bond="", bond_radii_factor=1.3):
    """

    :param filename: ALL file format supported by mogli
    :param neighbor_list: is neighbor_list=True, a dict of neighbor list of connection are returned, instead of the standard one {1:[3,4,5], 2:[3,6,7]}
    :param add_bond: a str with the format of "1-2,1-5", atom count start from 0, where a (single) bond will be added if they are not bonded before
    :param bond_radii_factor:

    :return: a list of tuple of bonded atoms,no bond order information
            e.g. [(0, 1), (0, 2), (0, 13), (0, 26), (2, 3), (2, 4), (2, 5), (5, 6), (5, 9), (5, 10),
                  (6, 7), (6, 8), (7, 13), (10, 11), (10, 12), (12, 13), (12, 38), (13, 14), (14, 15),
                  (14, 16), (14, 20), (16, 17), (16, 18), (16, 19), (20, 21), (20, 22), (20, 23), (23, 24),
                  (23, 25), (23, 26), (26, 27), (27, 28), (27, 32), (28, 29), (28, 30), (28, 31), (32, 33),
                  (32, 34), (32, 35), (35, 36), (35, 37), (35, 38), (38, 39), (38, 40), (40, 41), (40, 45),
                  (40, 46), (41, 42), (41, 43), (41, 44), (46, 47), (46, 48), (46, 49)]

            if neighbor_list is enabled:
            {0: [1, 3, 4], 1: [0, 2, 10], 2: [1], 3: [0], 4: [0, 5, 6], 5: [4], 6: [4, 7, 8],
             7: [6], 8: [6, 9, 10], 9: [8], 10: [1, 8, 11], 11: [10, 12, 13, 14], 12: [11],
             13: [11], 14: [11, 15], 15: [14, 16, 17, 49], 16: [15], 17: [15, 18, 19, 34],
             18: [17], 19: [17, 20, 21, 22], 20: [19], 21: [19], 22: [19, 23, 24, 25],
             23: [22], 24: [22], 25: [22, 26, 27, 28], 26: [25], 27: [25], 28: [25, 29, 30, 31],
             29: [28], 30: [28], 31: [28, 32, 33, 34], 32: [31], 33: [31], 34: [17, 31, 35, 36],
             35: [34, 49], 36: [34, 37, 38, 39], 37: [36], 38: [36], 39: [36, 40, 41, 42], 40: [39],
             41: [39], 42: [39, 43, 47, 61], 43: [42, 44, 45, 46], 44: [43], 45: [43], 46: [43],
             47: [42, 48, 49, 53], 48: [47], 49: [15, 35, 47, 50], 50: [49, 51], 51: [50, 52, 53],
             52: [51], 53: [47, 51, 54, 55], 54: [53], 55: [53, 56, 60, 61], 56: [55, 57, 58, 59],
             57: [56], 58: [56], 59: [56], 60: [55], 61: [42, 55, 62], 62: [61]}
    """

    # convert "1-2,1-5" to [(1,2),(1,5)]
    if not add_bond:
        add_bond = []
    else:
        add_bond = add_bond.split(",")
        new_add_bond = []
        for bond in add_bond:
            bond = bond.split('-')
            bond = [int(x) - 1 for x in bond]
            bond.sort()
            new_add_bond.append(bond)

        add_bond = new_add_bond

    from mogli.mogli import mogli
    import time
    molecules = mogli.read(filename)
    retry_attempts = 0
    while not molecules:
        print("Mogli reading error, retrying...")
        time.sleep(0.2)
        molecules = mogli.read(filename)
        retry_attempts += 1
        if retry_attempts > 5:
            break

    molecule = molecules[-1]
    molecule.calculate_bonds(param=bond_radii_factor)
    bonds = molecule.bonds
    bonds = [sorted(list(x)) for x in bonds.index_pairs]
    # print(bonds)

    for new_bond in add_bond:
        for existed_bond in bonds:
            if new_bond[0] == existed_bond[0] and new_bond[1] == existed_bond[1]:
                break
        else:
            bonds.append(new_bond)

    bonds = sorted([sorted(list(x)) for x in bonds])
    # print(bonds)

    if neighbor_list:
        atom_count = len(molecule.atomic_numbers)
        ret = {}
        for atom in range(atom_count):
            bonding_to = []
            for bond in bonds:
                if atom in bond:
                    bonding_to.append(bond[0] if bond[0] != atom else bond[1])
            ret[atom] = bonding_to
        return ret

    return bonds


if __name__ == '__main__':
    pass
