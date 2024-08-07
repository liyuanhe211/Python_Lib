# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import pathlib
import sys

Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Stock import *
from scipy.interpolate import interp1d, splrep, splev
from statistics import mean
from itertools import groupby


def interpolation_with_grouping(Xs, Ys, kind, smoothing=None):
    """
    Interpolates the given data points. Handles both types supported by interp1d and
    B-splines through splrep based on the kind specified.

    :param Xs: List of X coordinates
    :param Ys: List of Y coordinates
    :param kind: Type of interpolation or spline (e.g., 'linear', 'cubic', 'spline')
    :param s: Smoothing factor for splines; default is 0 (interpolates through all points)
    :return: A callable function that evaluates the interpolation at any given X
    """
    # Process the (X, Y) pairs
    process_XYs = list(zip(Xs, Ys))
    process_XYs.sort(key=lambda x: x[0])
    grouper = groupby(process_XYs, key=lambda x: x[0])
    processed_XYs = [[x, mean(yi[1] for yi in y)] for x, y in grouper]

    # Unzip the processed data
    interp1d_X, interp1d_Y = zip(*processed_XYs)

    # if kind in ['zero', 'slinear', 'quadratic', 'cubic', 'linear', 'nearest', 'nearest-up', 'previous', 'next']:
    if smoothing is None:
        print('using if')
        # Use interp1d for these specific kinds
        interp_function = interp1d(interp1d_X, interp1d_Y, kind=kind)
        return lambda x: interp_function(x)
    else:
        print("using else")
        # Use splrep and splev for B-spline or custom spline types
        tck = splrep(interp1d_X, interp1d_Y, k=kind_to_degree(kind), s=smoothing)
        return lambda x: splev(x, tck)


def kind_to_degree(kind):
    """
    Convert spline type to its corresponding degree, or default to cubic (k=3).
    """
    translation = {
        'linear': 1,
        'quadratic': 2,
        'cubic': 3
    }
    if kind in translation:
        return translation[kind]
    return kind


# def interpolation_with_grouping(Xs, Ys, kind):
#     """
#     When X have duplicate values, interp1d fails.
#     It is averaged so to be as an single value function
#     """

#
#     process_XYs = list(zip(Xs, Ys))
#     process_XYs.sort(key=lambda x: x[0])
#     grouper = groupby(process_XYs, key=lambda x: x[0])
#     process_XYs = [[x, mean(yi[1] for yi in y)] for x, y in grouper]
#     interp1d_X = [x[0] for x in process_XYs]
#     interp1d_Y = [x[1] for x in process_XYs]
#
#     # print(len(interp1d_X))
#     # print(len(set(interp1d_X)))
#
#     return interp1d(interp1d_X, interp1d_Y, kind=kind)


def find_range_for_certain_percentage(data, percentage=80, offset=0):
    """
    try to find the smallest data range, where it can cover certain percentage of the data
    :param data: a list of numbers, or any shape of numpy array, which will be flattened
    :param offset:a number, if it is 0.1, then the color map will be shifted up for 10% of max(Z)-min(Z)
    :param percentage:
    :return: a 2-tuple, the desired smallest range
    """
    import numpy as np
    data = np.array(data)
    data_dist = max(data) - min(data)
    data = np.sort(data, axis=None)
    data_count = data.size
    target_count = math.ceil(data_count * percentage / 100)
    ranges = []
    for i in range(0, data_count - target_count + 1):
        ranges.append((data[i], data[i + target_count - 1]))
    ranges.sort(key=lambda x: x[1] - x[0])
    ret = list(ranges[0])
    print(ret)
    ret[0] = ret[0] + data_dist * offset
    ret[1] = ret[1] + data_dist * offset
    print(ret)
    return ret


if __name__ == '__main__':
    pass
