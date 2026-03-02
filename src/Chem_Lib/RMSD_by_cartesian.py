"""
Calculate RMSD between two coordinate_object

"""

import pathlib
import sys

import numpy as np
import re
from .Lib import *


def kabsch_rmsd(P, Q):
    """
    Rotate matrix P unto Q and calculate the RMSD
    """
    P = rotate(P, Q)
    return rmsd(P, Q)


def rotate(P, Q):
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm
    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P


def kabsch(P, Q):
    """
    The optimal rotation matrix U is calculated and then used to rotate matrix
    P unto matrix Q so the minimum root-mean-square deviation (RMSD) can be
    calculated.

    Using the Kabsch algorithm with two sets of paired point P and Q,
    centered around the center-of-mass.
    Each vector set is represented as an NxD matrix, where D is
    the dimension of the space.

    The algorithm works in three steps:
    - a translation of P and Q
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U

    https://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters:
    P -- (N, number of points)x(D, dimension) matrix
    Q -- (N, number of points)x(D, dimension) matrix

    Returns:
    U -- Rotation matrix

    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


def centroid(X):
    """
    Calculate the centroid from a vector set X
    """
    C = sum(X) / len(X)
    return C


def rmsd(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.
    """
    D = len(V[0])
    N = len(V)
    ret_rmsd = 0.0
    for v, w in zip(V, W):
        ret_rmsd += sum([(v[i] - w[i]) ** 2.0 for i in range(D)])
    return np.sqrt(ret_rmsd / N)


def get_vectors(coordinate_object, ignore_hydrogens):
    """
    Get coordinates from a filename.xyz and return a vector set with all the
    coordinates.

    This function has been written to parse XYZ files, but can easily be
    written to parse others.

    """
    vectors = coordinate_object.coordinates_np
    if ignore_hydrogens:
        vectors = [x for count, x in enumerate(vectors) if coordinate_object.elements[count] != 'H']
    vectors = np.array(vectors)
    return vectors


def generate_rmsd_list(ref_coordinate: Coordinates, list_of_coordinates: tuple, ignore_hydrogens=True, print_percentage=True):
    # no multithread supported

    if print_percentage:
        print("Generating RMSD Matrix...")

    last_percentage = 0
    if print_percentage:
        print("0%")

    ret_list = [-1 for _ in range(len(list_of_coordinates))]
    for row in range(len(list_of_coordinates)):
        if print_percentage:
            current_percentage = int(row / len(list_of_coordinates) * 100)
            if current_percentage != last_percentage:
                print(current_percentage, '%', sep="")
                last_percentage = current_percentage
        rmsd_values = get_one_rmsd_value(list_of_coordinates[row], ref_coordinate, ignore_hydrogens)
        ret_list[row] = rmsd_values

    return ret_list


def generate_rmsd_matrix(list_of_coordinates: list, ignore_hydrogens=True, multithread=False):
    if multithread:
        from multiprocessing import Pool
        from multiprocessing import cpu_count
        pool = Pool(processes=int(cpu_count() / 2))

    print("Generating RMSD Matrix...")

    last_percentage = 0
    print("0%")

    ret_matrix = [[-1 for _ in range(len(list_of_coordinates))] for _ in range(len(list_of_coordinates))]
    for row in range(len(list_of_coordinates)):
        current_percentage = int(((len(list_of_coordinates) * 2 - row) * row) / len(list_of_coordinates) ** 2 * 100)
        if current_percentage != last_percentage:
            print(current_percentage, '%', sep="")
            last_percentage = current_percentage
        if multithread:
            rmsd_values = pool.map(get_one_rmsd_value, ((list_of_coordinates[row], list_of_coordinates[column], ignore_hydrogens) for column in
                                                        range(row, len(list_of_coordinates))))
        else:
            rmsd_values = [get_one_rmsd_value(list_of_coordinates[row], list_of_coordinates[column], ignore_hydrogens) for column in
                           range(row, len(list_of_coordinates))]
        for column in range(row, len(list_of_coordinates)):
            ret_matrix[row][column] = rmsd_values[column - row]
            ret_matrix[column][row] = rmsd_values[column - row]

    # for i in ret_matrix:
    #     print(i)
    return ret_matrix


def get_one_rmsd_value(coordinate1, coordinate2=None, ignore_hydrogens=None):
    if isinstance(coordinate1, tuple) and len(coordinate1) == 3:
        coordinate1, coordinate2, ignore_hydrogens = coordinate1

    coord1 = get_vectors(coordinate1, ignore_hydrogens)
    coord2 = get_vectors(coordinate2, ignore_hydrogens)

    # Create the centroid of P and Q which is the geometric center of an
    # N-dimensional region and translate P and Q onto that center.
    coord1 -= centroid(coord1)
    coord2 -= centroid(coord2)
    rmsd_value = kabsch_rmsd(coord1, coord2)

    return rmsd_value


if __name__ == "__main__":
    pass
