# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import subprocess

import sys
import pathlib
Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Stock import *


def interpolation_with_grouping(Xs, Ys, kind):
    """
    When X have duplicate values, interp1d fails.
    It is averaged so to be as an single value function
    """
    from scipy.interpolate import interp1d
    from statistics import mean
    from itertools import groupby

    process_XYs = list(zip(Xs, Ys))
    process_XYs.sort(key=lambda x: x[0])
    grouper = groupby(process_XYs, key=lambda x: x[0])
    process_XYs = [[x, mean(yi[1] for yi in y)] for x, y in grouper]
    interp1d_X = [x[0] for x in process_XYs]
    interp1d_Y = [x[1] for x in process_XYs]

    # print(len(interp1d_X))
    # print(len(set(interp1d_X)))

    return interp1d(interp1d_X, interp1d_Y, kind=kind)


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
