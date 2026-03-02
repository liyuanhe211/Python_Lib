# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import subprocess

import sys
import pathlib
Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Stock import *


def mix_two_colors(color1, color2, percentage, mixing_order=2):
    """
    :param color1: color in hex, e.g. "00FF00"
    :param color2: color in hex, e.g. "00FF00"
    :param mixing_order:
    :param percentage: if =0, color1, if =1, color2, else, return an interpolation of the color
    :return: a mixed color in hex number
    """
    if mixing_order == 'hsv':
        import colorsys
    if percentage == 0:
        return color1
    if percentage == 1:
        return color2

    color1 = [eval('0x' + x) for x in [color1[0:2], color1[2:4], color1[4:6]]]
    color2 = [eval('0x' + x) for x in [color2[0:2], color2[2:4], color2[4:6]]]

    def mixing(value1, value2, mixing_percentage, order):
        return round((value1 ** order + (value2 ** order - value1 ** order) * mixing_percentage) ** (1 / order))

    def hsv_mixing(hsv_color1, hsv_color2, hsv_percentage):
        hsv_color1 = [x / 255 for x in hsv_color1]
        hsv_color2 = [x / 255 for x in hsv_color2]
        hsv_color1 = colorsys.rgb_to_hsv(*hsv_color1)
        hsv_color2 = colorsys.rgb_to_hsv(*hsv_color2)
        hsv_mixing_ret = colorsys.hsv_to_rgb(*[hsv_color1[x] + hsv_percentage * (hsv_color2[x] - hsv_color1[x]) for x in range(3)])
        return (round(x * 255) for x in hsv_mixing_ret)

    if mixing_order == 'hsv':
        ret = hsv_mixing(color1, color2, percentage)
    else:
        ret = [mixing(color1[x], color2[x], percentage, mixing_order) for x in range(3)]

    ret = ''.join('{:02X}'.format(int(round(num))) for num in ret)
    return ret


def color_scale(colors, ref_points, value, mixing_order=2):
    """
    generate a color scale, extract color for a value
    :param colors: list of colors corresponds to list of ref_points, color in Hex like FFFFFF
    :param ref_points:
    :param value:
    :param mixing_order:
    :return:
    """

    assert sorted(ref_points) == ref_points or sorted(ref_points, reverse=True) == ref_points, 'Color Ref Points need to be in sequence.'
    if sorted(ref_points, reverse=True) == ref_points:
        ref_points = list(reversed(ref_points))
        colors = list(reversed(colors))

    if value < ref_points[0]:
        return colors[0]
    if value > ref_points[-1]:
        return colors[-1]

    for value_count in range(len(ref_points) - 1):
        value1, value2 = ref_points[value_count:value_count + 2]
        if value1 <= value <= value2:
            color1, color2 = colors[value_count:value_count + 2]
            return mix_two_colors(color1, color2, (value - value1) / (value2 - value1), mixing_order=mixing_order)


if __name__ == '__main__':
    pass
