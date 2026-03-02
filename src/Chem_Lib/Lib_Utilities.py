# -*- coding: utf-8 -*-
import os
import pathlib
import sys

from Python_Lib.My_Lib_Stock import *

fluctuation_message = ""  # 用于记录震荡的提示，重复的不要输出
opt_flucturation_threshold_shown = False

def count_pass_through(data, threshold):
    ret = 0
    for count in range(len(data) - 1):
        if (data[count] - threshold) * (data[count + 1] - threshold) < 0:
            ret += 1
    return ret

def fluctuation_determine(data=None, atom_count=-1, silent=False):
    """

    Args:
        data:
        atom_count: 对较大的分子，应增加许可的循环数量
        silent:

    Returns:

    """

    if data is None:
        data = []

    # 曲线最小值为第n值，取集合data[n:]的min，max
    # 对任意min,max能量间的阈值，计算折线穿越阈值的次数，超过一定次数报震荡

    # 对data[n:]排序，相邻数区间内的穿越次数是相同的，遍历取最大即可。
    if not data:
        return ("", 0)
    min_index = data.index(min(data))
    test_sub_list = data[min_index:]
    test_sub_list.sort()
    thresholds = [(test_sub_list[i] + test_sub_list[i + 1]) / 2 for i in range(len(test_sub_list) - 1)] + [data[-1]]

    throughs = [count_pass_through(data, x) for x in thresholds]

    max_through = max(throughs)
    max_through_threshold = thresholds[throughs.index(max(throughs))]

    # print(max_through)
    global fluctuation_message
    global opt_flucturation_threshold_shown
    threshold = atom_count / 3 if atom_count > 50 else 16
    if not opt_flucturation_threshold_shown:
        if not silent:
            print("Using flucturation threshold", threshold)
        opt_flucturation_threshold_shown = True
    if max_through > threshold:  # 16 and 1/3 is an arbitrary sensitivity control number
        new_fluctuation_message = get_print_str("Fluctuation detected! Max fluctuation count:", max_through, " Threshold:", max_through_threshold)
        if new_fluctuation_message != fluctuation_message:
            if not silent:
                print("Fluctuation detected! Max fluctuation count:", max_through, " Threshold:", max_through_threshold)
            fluctuation_message = new_fluctuation_message
        return ("Definitive Fluctuation.", max_through_threshold)
    elif max_through > threshold / 2:
        new_fluctuation_message = get_print_str("Possible Fluctuation! Max fluctuation count:", max_through, " Threshold:", max_through_threshold)
        if new_fluctuation_message != fluctuation_message:
            if not silent:
                print("Possible Fluctuation! Max fluctuation count:", max_through, " Threshold:", max_through_threshold)
            fluctuation_message = new_fluctuation_message
        return ("Possible Fluctuation.", max_through_threshold)
    else:
        return ("", max_through_threshold)

