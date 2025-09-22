# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import sys
import os
import math
import copy
import shutil
import pickle
import re
import time
import datetime
import csv
import json
from datetime import datetime, timedelta
import random
import subprocess
import collections
from collections import OrderedDict
import traceback
import pathlib
import importlib
import importlib.util
from typing import Optional, Union, Sequence, Tuple, List, Callable, TypeVar, Literal

Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Constants import *
from My_Lib_File import *


def smart_format_float(num, precision=3, scientific_notation_limit=5):
    """
    对precision = 3, scientific_notation_limit = 5
        2.77E13 --> 2.77 × 10^13
        2.778E13 --> 2.78 × 10^13
        2.771E13 --> 2.77 × 10^13
        2.777E-13 --> 2.78 × 10^-13
        0.0000156789 --> 1.57 × 10^-5
        0.00016789 --> 0.000168
        0.13456 --> 0.135
        1.6789 --> 1.68
        15.6789 --> 15.7
        155.6789 --> 156
        1234.6789 --> 1235
        12345.6789 --> 12346
        123456.6789 --> 1.23 × 10^5
        14582622310.678967 --> 1.46 × 10^10
        0.000012345 --> 1.23 × 10^-5
        0.00012345 --> 0.000123
        0.12345 --> 0.123
        1.2345 --> 1.23
        15.2345 --> 15.2
        155.2345 --> 155
        1234.2345 --> 1234
        12345.2345 --> 12345
        123456.2345 --> 1.23 × 10^5
        14582622310.678967 --> 1.46 × 10^10
        2.7E13 --> 2.70 × 10^13
        0.000015 --> 1.50 × 10^-5
        0.00016 --> 0.000160
        0.13 --> 0.130
        1.6 --> 1.60
        15 --> 15.0
        155 --> 155
        1234 --> 1234
        12345 --> 12345
        123456 --> 1.23 × 10^5
        14582622310 --> 1.46 × 10^10

    Args:
        num:
        precision:

    Returns:
    :param precision:
    :param num:
    :param scientific_notation_limit:

    """
    if num == 0:
        return "0." + "0" * (precision - 1)

    if num < 0:
        neg = True
        num *= -1
    else:
        neg = False

    # 123456 -> 1.23 × 10^5
    if num >= 10 ** scientific_notation_limit:
        return ("-" if neg else "") + ("{:." + str(precision - 1) + "e}").format(num).replace("e+0", ' × 10^').replace("e+", ' × 10^')

    # return ("{:."+str(precision-1)+"f}").format(num/(10**expo)) + " × 10^" + str(expo)

    # 155 -> 155
    # 1234 -> 1234
    # 12345 -> 12345
    if num >= 10 ** (precision - 1):
        return ("-" if neg else "") + str(round(num))

    # 0.0001 -> 0.000100
    # 0.1 -> 0.100
    # 1 -> 1.00
    # 15 -> 15.0
    if num >= 10 ** (-scientific_notation_limit + 1):
        exponent = int(math.log10(abs(num)))
        digits_left = precision - 1 - exponent
        if abs(num) < 1:
            digits_left += 1
        rounded = round(num, digits_left)
        return '{:.{}f}'.format(rounded, max(0, digits_left))

    return ("-" if neg else "") + ("{:." + str(precision - 1) + "e}").format(num).replace("e-0", ' × 10^-').replace("e-", ' × 10^-')


# smart_format_float_tests = [(2.77E13, "2.77 × 10^13"),
#                             (2.778E13, "2.78 × 10^13"),
#                             (2.771E13, "2.77 × 10^13"),
#                             (2.777E-13, "2.78 × 10^-13"),
#                             (0.0000156789, "1.57 × 10^-5"),
#                             (0.00016789, "0.000168"),
#                             (0.13456, "0.135"),
#                             (1.6789, "1.68"),
#                             (15.6789, "15.7"),
#                             (155.6789, "156"),
#                             (1234.6789, "1235"),
#                             (12345.6789, "12346"),
#                             (123456.6789, "1.23 × 10^5"),
#                             (14582622310.678967, "1.46 × 10^10"),
#                             (0.000012345, "1.23 × 10^-5"),
#                             (0.00012345, "0.000123"),
#                             (0.12345, "0.123"),
#                             (1.2345, "1.23"),
#                             (15.2345, "15.2"),
#                             (155.2345, "155"),
#                             (1234.2345, "1234"),
#                             (12345.2345, "12345"),
#                             (123456.2345, "1.23 × 10^5"),
#                             (14582622310.678967, "1.46 × 10^10"),
#                             (2.7E13, "2.70 × 10^13"),
#                             (0.000015, "1.50 × 10^-5"),
#                             (0.00016, "0.000160"),
#                             (0.13, "0.130"),
#                             (1.6, "1.60"),
#                             (15, "15.0"),
#                             (155, "155"),
#                             (1234, "1234"),
#                             (12345, "12345"),
#                             (123456, "1.23 × 10^5"),
#                             (14582622310, "1.46 × 10^10")]
#
# for i in smart_format_float_tests:
#     assert smart_format_float(i[0])==i[1],f"{i[0]},{smart_format_float(i[0])},{i[1]}"


def remove_forbidden_char_from_filename(filename):
    chars = r'<>:"/\|?*'
    for char in chars:
        filename = filename.replace(char, '_')
    return filename


def quote_url(url):
    from urllib import parse
    return parse.quote(url, safe=':?/=')


def smart_print_time(time_sec):
    sec = str(int(time_sec % 60))
    minute = str(int((time_sec / 60) % 60))
    hour = str(int((time_sec / 3600) % 24))
    day = str(int(time_sec / 86400))

    if time_sec <= 60:
        return smart_format_float(time_sec) + ' s'
    if time_sec < 60 * 60:
        return minute + ' m ' + sec + " s"
    if time_sec < 86400:
        return hour + ' h ' + minute + ' m'
    return day + " d " + hour + ' h'


# print(smart_print_time(0.01234))
# print(smart_print_time(1.234))
# print(smart_print_time(10.234))
# print(smart_print_time(100.234))
# print(smart_print_time(1023.234))
# print(smart_print_time(10162.234))
# print(smart_print_time(163120.234))
# print(smart_print_time(1631206.234))


def file_is_descendant(file, parent):
    from pathlib import Path

    return Path(parent) in Path(file).resolve().parents


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


def reverse(string):
    return str(string[::-1])


def rreplace(string, old, new, count=None):
    """string right replace"""
    string = str(string)
    r = reverse(string)
    if count is None:
        count = -1
    r = r.replace(reverse(old), reverse(new), count)
    return type(string)(reverse(r))


def open_explorer_and_select(file_path):
    import subprocess
    import platform
    if platform.system() == 'Windows':
        open_explorer_command = r'explorer /select,"' + str(file_path).replace('/', '\\') + '"'
        subprocess.Popen(open_explorer_command)
    else:
        print("Not Windows system. Please check the file by yourself:", file_path)


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def locate_matching_parenthesis(input_str: str, char1="(", char2=")"):
    """
    Recognize the first pairs of top level parenthesis in the input_str
    Args:
        input_str:
        char1:
        char2:

    Returns:
        a tuple of two numbers, start and finish
        if nothing is found, return None
    """
    start = input_str.find(char1)
    status = 0
    for count, i in enumerate(input_str[start:]):
        if i == char1:
            status += 1
        if i == char2:
            status -= 1
        if not status:
            return (start, count + start)


class MyException(Exception):
    def __init__(self, explanation):
        Exception.__init__(self)
        print(explanation)


def list_or(input_list):
    # input [a,b,c] return a or b or c
    any(input_list)


def list_and(input_list):
    # input [a,b,c] return a and b and c
    all(input_list)


def is_in_folder(subfolder_or_file, parent_folder):
    import pathlib
    if not isinstance(subfolder_or_file, pathlib.WindowsPath):
        subfolder_or_file = pathlib.Path(subfolder_or_file)
    if not isinstance(parent_folder, pathlib.WindowsPath):
        parent_folder = pathlib.Path(parent_folder)
    return parent_folder in subfolder_or_file.parents


def get_ipv6_public_addresses_on_windows():
    """
    :return: a list of ipv6 addresses
    """
    # https://stackoverflow.com/questions/53497/regular-expression-that-matches-valid-ipv6-addresses
    ipv6_address_regexp = r'(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,' \
                          r'4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,' \
                          r'4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,' \
                          r'7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,' \
                          r'3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,' \
                          r'3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])) '

    ret = []
    import subprocess
    a = subprocess.check_output('ipconfig').decode('gbk').lower()

    for i in a.splitlines():
        if 'ipv6' in i.lower() and "本地" not in i.lower() and "local" not in i.lower():
            re_ret = re.findall(ipv6_address_regexp, i)
            if re_ret:
                address = re_ret[0][0]
                if not address.startswith('fe'):
                    ret.append(address)
    return ret


def get_ipv6_public_addresses_on_linux():
    """

    :return: a list of ipv6 addresses
    """
    # https://stackoverflow.com/questions/53497/regular-expression-that-matches-valid-ipv6-addresses
    ipv6_address_regexp = r'(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,' \
                          r'4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,' \
                          r'4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,' \
                          r'7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,' \
                          r'3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,' \
                          r'3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])) '

    ret = []
    import subprocess
    a = subprocess.check_output('ifconfig').decode('gbk').lower()
    for i in a.splitlines():
        if 'inet6 addr' in i.lower():
            re_ret = re.findall(ipv6_address_regexp, i)
            if re_ret:
                address = re_ret[0][0]
                if address.startswith('2'):
                    ret.append(address)
    return ret


def get_ipv6_address_from_external():
    import requests
    return requests.get("https://v6.ident.me/", verify=False).text


def remove_key_from_dict(input_dict, key):
    if key in input_dict:
        input_dict.pop(key)
    return input_dict


def safe_get_dict_value(input_dict, key, default_value=""):
    if key in input_dict:
        return input_dict[key]
    else:
        return default_value


def secure_print(*object_to_print):
    # print some character will cause UnicodeEncodeError,
    # if the message is not necessarily printed, use this function will just print nothing and avoid the error

    try:
        print(*object_to_print)
    except Exception as e:
        print("Print function error. Print of information omitted:", e)


def get_print_str(*object_to_print, sep=" "):
    ret = ""
    for current_object in object_to_print:
        try:
            ret += str(current_object) + sep
        except Exception:
            print("get_print_str Error...")

    return ret


def read_last_n_lines_fast(file, n_lines):
    return read_last_n_char_fast(file, '\n', n_lines)


def read_last_n_char_fast(file, char, n):
    """
    a fast method to read the last n appearance of a specific character, and return one multi-line decoded string
    if there is less than n matches, the whole file will be returned
    :param file:
    :param char
    :param n:
    :return:
    """
    char = char.encode('utf-8')
    import mmap
    with open(file, "r+b") as f:
        # memory-map the file, size 0 means whole file
        m = mmap.mmap(f.fileno(), 0)
        # prot argument is *nix only
        current_cut = m.rfind(char)
        count = 1
        while count < n:
            current_cut = m.rfind(char, 0, current_cut)
            if current_cut == -1:
                count = n
                current_cut = 0
            count += 1
        m.seek(current_cut)
        return m.read().decode()


def split_list_by_item(input_list: list, separator, lower_case_match=False, include_separator=False, include_empty=False):
    return split_list(input_list, separator, lower_case_match, include_separator, include_empty)


def split_list(input_list: list,
               separator=None,
               lower_case_match=False,
               include_separator=False,
               include_separator_after=False,
               include_empty=False,
               separator_with_context=None):
    """

    :param input_list:
    :param separator: a separator, either a str or function.
    If it's a function, it should take a str as input, and return
    :param lower_case_match:
    :param include_separator:
    :param include_separator_after:
    :param include_empty:
    :param a function like separator_with_context(input_list, i) with i being the index return a bool
    :return:
    """
    ret = []
    temp = []

    if separator is None and separator_with_context is None:
        raise Exception("You need to either designate separator or separator_with_context")

    if include_separator or include_separator_after:
        assert not (include_separator and include_separator_after), 'include_separator and include_separator_after can not be True at the same time'

    for count, item in enumerate(input_list):
        split_here_bool = False
        if callable(separator_with_context):
            split_here_bool = separator_with_context(input_list, count)
        elif callable(separator):
            split_here_bool = separator(item)
        elif isinstance(item, str) and item == separator:
            split_here_bool = True
        elif lower_case_match and isinstance(item, str) and item.lower() == separator.lower():
            split_here_bool = True

        if split_here_bool:
            if include_separator_after:
                temp.append(item)
            ret.append(temp)
            temp = []
            if include_separator:
                temp.append(item)
        else:
            temp.append(item)
    ret.append(temp)

    if not include_empty:
        ret = [x for x in ret if x]

    return ret


def split_list_by_length(input_list, chunk_length, discard_tail=False):
    """
    [0,1,2,3,4,5,6] split by 2 -->
    if discard_tail: [[0,1],[2,3],[4,5],[6]]
    else: [[0,1],[2,3],[4,5]]

    :param input_list:
    :param chunk_length:
    :param discard_tail:
    :return:
    """
    ret = [input_list[i:i + chunk_length] for i in range(0, len(input_list), chunk_length)]
    if discard_tail and len(input_list) % chunk_length != 0:
        return ret[:-1]
    return ret


def get_appropriate_ticks(ranges, num_tick_limit=(4, 6), accept_closest_out_of_range=True):
    """
    a function to get the desired ticks, e.g. for 1.2342 - 1.58493, with a tick_limit of (4,8),
        the tick should be (1.25,1.30,1.35,1.40,1.45,1.50,1.55)
    :param ranges: a 2-tuple, upper limit and lower limit
    :param num_tick_limit: the maximum and minimum amount of ticks
    :param accept_closest_out_of_range: if closest, out-of-range answer is accepted if in-range answer is not possible
    :return: a (lower-limit, upper-limit, spacing) tuple
    if no appropriate choice is possible, and accept_closest_out_of_range = False, return [ranges[1],ranges[0],ranges[1]-ranges[0]]
    """
    # the ticking should be either ending in 5 or 2 or 0
    if ranges[1] < ranges[0]:
        ranges = reversed(ranges)

    assert all(x > 0 for x in num_tick_limit)
    assert ranges[0] != ranges[1]

    span = abs(ranges[1] - ranges[0])
    mid_limit = sum(num_tick_limit) / 2
    ideal_distance = span / mid_limit
    ideal_distance_log = int(math.log(ideal_distance, 10))
    test_distance = 10 ** ideal_distance_log
    test_distances = [test_distance * x for x in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)]
    tick_start_points = [math.ceil(ranges[0] / x) * x for x in test_distances]
    test_tick_count = []
    for count, test_distance in enumerate(test_distances):
        tick_start_point = tick_start_points[count]
        test_tick_count.append(int((ranges[1] - tick_start_point) / test_distance))
    acceptable_counts = [x for x in test_tick_count if num_tick_limit[0] <= x <= num_tick_limit[1]]
    if not acceptable_counts and not accept_closest_out_of_range:
        return [ranges[0], ranges[1], ranges[1] - ranges[0]]
    elif not acceptable_counts:
        acceptable_count_differences = [abs(x - mid_limit) for x in test_tick_count]
        optimal_index = acceptable_count_differences.index(min(acceptable_count_differences))
        optimal_distance = test_distances[optimal_index]
    else:
        acceptable_count_differences = [abs(x - mid_limit) for x in acceptable_counts]
        optimal_count = acceptable_counts[acceptable_count_differences.index(min(acceptable_count_differences))]
        optimal_index = test_tick_count.index(optimal_count)
        optimal_distance = test_distances[optimal_index]
    optimal_start_point = tick_start_points[optimal_index]
    optimal_end_point = int((ranges[1] - optimal_start_point) / optimal_distance) * optimal_distance + optimal_start_point
    return [optimal_start_point, optimal_end_point, optimal_distance]


def get_input_with_while_cycle(break_condition=lambda x: not x.strip(),
                               input_prompt="",
                               strip_quote=True,
                               backup_file=None,
                               context_break_condition=None) -> list:
    """
    get multiple line of input, terminate with a condition, return the accepted lines
    :param break_condition: give a function, when it is met, the while loop is terminated.
    :param input_prompt: will print this every line
    :param strip_quote:
    :param backup_file: a file-like object (created by "open()") which will store the inputs for backup
    :param context_break_condition: a function accepting two parameters, the first is the lines already get, the second is the current input_line.
    :return: list of accepted lines
    """

    ret = []
    while True:
        input_line = input(input_prompt)
        if backup_file:
            backup_file.write(input_line)
            backup_file.write('\n')
        if strip_quote:
            input_line = input_line.strip().strip('"')
        if context_break_condition is not None:
            if context_break_condition(ret, input_line):
                break
        elif break_condition(input_line):
            break
        ret.append(input_line)
    return ret


def PolygonArea(corners):
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def remove_special_chr_from_str(input_str):
    """
    A function for fuzzy search "3-propyl-N'-ethylcarbodiim"-->"propylnethylcarbodiim"
    :param input_str:
    :return:
    """
    import string
    ret = ''.join(ch for ch in input_str if ch not in string.punctuation + string.whitespace + string.digits).lower()
    if not ret:
        ret = ''.join(ch for ch in input_str if ch not in string.punctuation + string.whitespace).lower()
    else:
        ret = input_str
    return ret


def get_unused_filename(input_filename, replace_hash=True, use_proper_filename=True):
    """
    verify whether the filename is already exist, if it is, a filename like filename_01.append; filename_02.append will be returned.
    maximum 99 files can be generated
    :param input_filename:
    :param replace_hash
    :param use_proper_filename
    :return: a filename
    """

    input_filename = os.path.realpath(input_filename)

    if use_proper_filename:
        input_filename = proper_filename(input_filename, replace_hash=replace_hash)

    if not os.path.isfile(input_filename) and not os.path.isdir(input_filename):
        # 是新的
        return input_filename
    else:
        if os.path.isfile(input_filename):
            no_append = filename_class(input_filename).only_remove_append
            append = filename_class(input_filename).append
        else:
            no_append = input_filename
            append = ""

        number = 1
        ret = no_append + "_" + '{:0>2}'.format(number) + (('.' + append) if append else "")
        while os.path.isfile(ret) or os.path.isdir(ret):
            number += 1
            if number == 9999:
                Qt.QMessageBox.critical(None, "YOU HAVE 9999 INPUT FILE?!", "AND YOU DON'T CLEAN IT?!",
                                        Qt.QMessageBox.Ok)
                break
            ret = no_append + "_" + '{:0>2}'.format(number) + (('.' + append) if append else "")

        return ret


def average(data, convert_to_arith_function=lambda x: x, convert_back_function=lambda x: x):
    """

    :param data:
    :param convert_to_arith_function: a function that convert the data to a number to be subjected to arithmetic average
    :param convert_back_function: a reverse function of convert_to_arith_function
    :return:
    """

    data_to_arith = [convert_to_arith_function(x) for x in data]
    arith_average = sum(data_to_arith) / len(data_to_arith)
    return convert_back_function(arith_average)


optimization_timer_u3yc24t04389y09sryc09384yn098 = 0  # this wired name is to avoid collision with other files


def optimization_timer(position_label=""):
    """
    A simple function to record the time to current operation, and print the elapsed time till then
    """

    # return None # comment this out to activate this function

    global optimization_timer_u3yc24t04389y09sryc09384yn098
    if optimization_timer_u3yc24t04389y09sryc09384yn098 == 0:
        optimization_timer_u3yc24t04389y09sryc09384yn098 = time.time()
    else:
        delta = time.time() - optimization_timer_u3yc24t04389y09sryc09384yn098
        optimization_timer_u3yc24t04389y09sryc09384yn098 = time.time()
        print("————————————", position_label, int(delta * 1000))


def readable_timestamp(timestamp=None):
    from datetime import datetime
    if timestamp is None:
        return datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        return datetime.fromtimestamp(timestamp).strftime("%Y%m%d%H%M%S")


def remove_duplicate(input_list: list, access_function=None):
    ret = []
    index = []
    for i in input_list:
        if callable(access_function):
            if access_function(i) not in index:
                index.append(access_function(i))
                ret.append(i)
        else:
            if i not in index:
                index.append(i)
                ret.append(i)

    return ret


def remove_blank(input_list: list, is_valid_function=None):
    if is_valid_function is None:
        return [x for x in input_list if x]
    else:
        return [x for x in input_list if is_valid_function(x)]


def cas_wrapper(input_str: str, strict=False, correction=False):
    """
    Match or not:
                    Partial     strict
    111-11-5           Yes        Yes     normal match
    111-11-1           No!         No     not match CAS number with wrong check digit
    111-11-5aa         Yes        Yes     match if other non [digit,'-'] concatenate with it

    # partial (will print a warning)
    111-11             Yes                partial match
    111-11-aa          Yes                same for partial match
    111-11aa           Yes                same for partial match
    111-1

    # not match with other number concatenate with it (prevent phone-number match)
    111-11-523          No
    0111-11-5           No

    # wrong format
    111-112-3           No
    111-119             No
    12345678-12-2       No

    :param input_str:
    :param strict: match complete or partial
    :param correction:允许纠正验证位错误
    :return: completed CAS number, if not find or the check digit not match the initial input, return '',
    """

    prefix = r"(^|[^\d-])"  # prevent 0111-11-1
    base = r"([1-9]\d{1,7}-\d{2})"  # matches 111-11
    postfix = r"(\-\d)"
    closure_complete = r"($|[^\d-])"  # matches 111-11-1, prevent 111-11-123
    closure_partial = r"|(\-($|[^\d-]))|($|[^\d-])"  # matches 111-11-, 111-11

    re_complete = ''.join([prefix, "(", base, postfix, ")", closure_complete])
    re_partial = ''.join([prefix, base, '((', postfix, closure_complete, ')', closure_partial, ')'])

    find_complete = re.findall(re_complete, input_str)  # match complete 128-38-2-->128-38
    find_partial = re.findall(re_partial, input_str)  # match the former digits of 128-38-2-->128-38

    if strict:
        if len(find_complete) > 1:  # 找到多个结果
            print('\n\n\nMultiple CAS match.', input_str, '\n\n\n')

        if not find_complete:
            return ""

        find_complete = find_complete[0][1]
        find_partial = find_partial[0][1]

    else:
        if len(find_partial) > 1:  # 找到多个结果
            print('\n\n\nMultiple CAS match.', input_str, '\n\n\n')

        if not find_partial:
            return ""

        find_partial = find_partial[0][1]

        if find_complete:
            find_complete = find_complete[0][1]
        else:
            find_complete = ""

    # 计算验证位
    only_digit = list(reversed([int(dig) for dig in find_partial if dig.isdigit()]))
    check_digit = sum([only_digit[i] * (i + 1) for i in range(len(only_digit))]) % 10

    ret = find_partial + '-' + str(check_digit)

    if find_complete and ret == find_complete:
        return ret

    else:
        if strict:  # 如果 strict，不满足检验直接跳出检测，返回空
            return ""

    if not find_complete:
        print('CAS Wrapper Doubt! Find:', repr(ret), '. Complete wrapper:', repr(find_complete), '. Original:',
              repr(input_str))
        return ret

    if find_complete and ret != find_complete:
        print('CAS Wrapper Disagree! Find:', repr(ret), '. Complete wrapper:', repr(find_complete), '. Original:',
              repr(input_str))
        if correction:  # 允许纠正错误的验证位
            return ret
        else:
            return ""

    if not find_partial:
        return ""


def transpose_2d_list(list_input):
    return list(map(list, zip(*list_input)))


def moving_averages(Ys, n):
    # Ensure n is an odd window size
    assert is_int(n)
    if n % 2 == 0:
        n += 1
    half = (n - 1) // 2  # "radius" on each side

    # Build prefix sum array: prefix_sum[i] = sum of Ys[:i]
    prefix_sum = [0] * (len(Ys) + 1)
    for i, val in enumerate(Ys):
        prefix_sum[i + 1] = prefix_sum[i] + val

    # Compute moving averages
    result = []
    for i in range(len(Ys)):
        start = max(0, i - half)
        end = min(len(Ys), i + half + 1)  # end is exclusive in prefix sums
        window_sum = prefix_sum[end] - prefix_sum[start]
        window_count = end - start
        result.append(window_sum / window_count)

    return result


def is_float(input_str):
    # 确定字符串可以转换为float
    try:
        float(input_str)
        return True
    except (ValueError, TypeError):
        return False


def int_able(input_str):
    try:
        int(input_str)
        return True
    except (ValueError, TypeError):
        return False


def is_int(input_str):
    if not is_float(input_str):
        return False
    num = float(input_str)
    # print(int(input_str))
    # print(num)
    if int(input_str) == num:
        return True
    else:
        return False


def same_length_2d_list(input_2D_list, fill=""):
    """
    read a list of list, and fill the sub_list to the same length
    :return: 
    """
    max_column_count = max([len(x) for x in input_2D_list])
    ret = [x + [fill] * (max_column_count - len(x)) for x in input_2D_list]
    return ret


def find_within_bracket(input_str, get_last_one=False):
    """
    Get all text within bracket
    :param input_str: "123123127941[12313[123]112313]asdf[123]
    :param get_last_one
    :return: [12313[123]112313][123]
    """
    in_bracket = 0
    ret = ""

    last_one_start = []  # 记录每一个最外括号起始位置
    last_one_end = -1  # 记录最后一个最外括号终止位置

    for count, char in enumerate(input_str):
        if char == '[':
            if in_bracket == 0:
                last_one_start.append(count)
            in_bracket += 1
        if char == ']':
            in_bracket -= 1
            if in_bracket == 0:
                last_one_end = count
        if char == ']' or in_bracket > 0:
            ret += char

    if get_last_one:
        if last_one_start and last_one_end != -1:
            qualified_starts = [x for x in last_one_start if x < last_one_end]
            if qualified_starts:
                return input_str[qualified_starts[-1]:last_one_end + 1]
            else:
                return ""

    return ret


def left_strip_sequence_from_str(input_string: str, to_match):
    """
    "abcabca","ab" --> “cabca"
    "abcabcd","abc" --> “d"
    """

    while input_string.startswith(to_match):
        input_string = input_string[len(to_match):]

    return input_string


strip_sequence_from_str = left_strip_sequence_from_str


def right_strip_sequence_from_str(input_string: str, to_match):
    """
    Same as left_strip_sequence_from_str only from the right
    :param input_string:
    :param to_match:
    :return:
    """

    while input_string.endswith(to_match):
        input_string = input_string[:-len(to_match)]

    return input_string


def parse_range_selection(input_str, decrease_by_1=True):
    """
    Input a range like 1,5,7-9; output a list. If by index [0,4,6,7,8]; if not index [1,5,7,8,9]
    :param input_str:
    :param decrease_by_1: if true, the index will be decreased by 1 than what's inputted
    :return:
    """

    if not input_str.strip():
        return []

    input_list = input_str.replace(',', ' ').split(' ')
    choices = copy.deepcopy(input_list)

    for choice in input_list:
        if '-' in choice:
            choices.remove(choice)
            if not re.findall(r'\d+-\d+', choice):
                print("Invalid")
                return None

            start, end = choice.split('-')
            choices += [str(x) for x in range(int(start), int(end) + 1)]
    if decrease_by_1:
        choices = sorted(list(set([int(x) - 1 for x in choices if '-' not in x])))
    else:
        choices = sorted(list(set([int(x) for x in choices if '-' not in x])))
    return choices


phrase_range_selection = parse_range_selection


def filename_from_url(url):
    forbidden_chrs = "<>:\"/\\|?*-"
    if 'http://' in url:
        ret = re.findall(r"http://(.+)", url)[0]
    else:
        ret = url
    for forbidden_chr in forbidden_chrs:
        ret = ret.replace(forbidden_chr, '___')
    ret = 'Download/' + ret
    return ret


elements_dict = {0: "X", 89: 'Ac', 47: 'Ag', 13: 'Al', 95: 'Am', 18: 'Ar', 33: 'As', 85: 'At', 79: 'Au', 5: 'B', 56: 'Ba', 4: 'Be', 107: 'Bh', 83: 'Bi',
                 97: 'Bk', 35: 'Br', 6: 'C', 20: 'Ca', 48: 'Cd', 58: 'Ce', 98: 'Cf', 17: 'Cl', 96: 'Cm', 112: 'Cn', 27: 'Co', 24: 'Cr', 55: 'Cs', 29: 'Cu',
                 105: 'Db', 110: 'Ds', 66: 'Dy', 68: 'Er', 99: 'Es', 63: 'Eu', 9: 'F', 26: 'Fe', 114: 'Fl', 100: 'Fm', 87: 'Fr', 31: 'Ga', 64: 'Gd', 32: 'Ge',
                 1: 'H', 2: 'He', 72: 'Hf', 80: 'Hg', 67: 'Ho', 108: 'Hs', 53: 'I', 49: 'In', 77: 'Ir', 19: 'K', 36: 'Kr', 57: 'La', 3: 'Li', 103: 'Lr',
                 71: 'Lu', 116: 'Lv', 101: 'Md', 12: 'Mg', 25: 'Mn', 42: 'Mo', 109: 'Mt', 7: 'N', 11: 'Na', 41: 'Nb', 60: 'Nd', 10: 'Ne', 28: 'Ni', 102: 'No',
                 93: 'Np', 8: 'O', 76: 'Os', 15: 'P', 91: 'Pa', 82: 'Pb', 46: 'Pd', 61: 'Pm', 84: 'Po', 59: 'Pr', 78: 'Pt', 94: 'Pu', 88: 'Ra', 37: 'Rb',
                 75: 'Re', 104: 'Rf', 111: 'Rg', 45: 'Rh', 86: 'Rn', 44: 'Ru', 16: 'S', 51: 'Sb', 21: 'Sc', 34: 'Se', 106: 'Sg', 14: 'Si', 62: 'Sm', 50: 'Sn',
                 38: 'Sr', 73: 'Ta', 65: 'Tb', 43: 'Tc', 52: 'Te', 90: 'Th', 22: 'Ti', 81: 'Tl', 69: 'Tm', 92: 'U', 118: 'Uuo', 115: 'Uup', 117: 'Uus',
                 113: 'Uut', 23: 'V', 74: 'W', 54: 'Xe', 39: 'Y', 70: 'Yb', 30: 'Zn', 40: 'Zr'}

num_to_element_dict = elements_dict
element_to_num_dict = {}
for key, value in elements_dict.items():
    element_to_num_dict[value] = key
    element_to_num_dict[value.lower()] = key
    element_to_num_dict[value.upper()] = key


def chr_is_chinese(char):
    char = ord(char)
    return (
            0x4e00 <= char <= 0x9fff or  # CJK Unified Ideographs (most common Chinese characters)
            0x3000 <= char <= 0x303f or  # CJK Symbols and Punctuation (。！？「」、 etc.)
            0xff00 <= char <= 0xffef or  # Fullwidth forms (，．！＠＃ etc.)
            0x2e80 <= char <= 0x2eff or  # CJK Radicals Supplement
            0x3400 <= char <= 0x4dbf  # CJK Unified Ideographs Extension A
    )


def has_chinese_char(string):
    for char in string:
        if chr_is_chinese(char):
            return True

    return False


def has_only_alphabet(string):
    for char in string.lower():
        if not ord('a') <= ord(char) <= ord('z'):
            return False
    return True


# had a typo on initial deploy, this is for back compatibility
has_only_alphabat = has_only_alphabet


def mytimeout(timeout):
    from threading import Thread
    import functools

    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                print("Timeout in mytimeout decorator", func)
            return ret

        return wrapper

    return deco


def open_config_file():
    """
    Assuming this folder structure:
    Project_folder
        - Python_Lib
            - My_Lib_Stock.py
        - Config.json
    Returns:

    """
    config_folder = str(pathlib.Path(__file__).parent.parent.resolve())
    config_file_json = os.path.join(config_folder, 'Config.json')
    # for backward compatible
    config_file_ini = os.path.join(config_folder, 'Config.ini')
    if os.path.isfile(config_file_json) and os.path.isfile(config_file_ini):
        os.remove(config_file_ini)

    config_file_failure = False
    if os.path.isfile(config_file_json):
        try:
            config = json.loads(open(config_file_json).read())
        except json.decoder.JSONDecodeError as e:
            traceback.print_exc()
            print(e)
            config_file_failure = True
    # for backward compatible
    elif os.path.isfile(config_file_ini):
        try:
            config = eval(open(config_file_ini).read())
        except Exception as e:
            traceback.print_exc()
            print(e)
            config_file_failure = True
    else:
        config_file_failure = True

    if config_file_failure:
        open(config_file_json, 'w').write('{}')
        config = {}

    return config


def get_config(config, key, absence_return=""):
    if key in config:
        return config[key]
    else:
        config[key] = absence_return
        save_config(config)
        return absence_return


def save_config(config):
    config_file = os.path.join(filename_class(sys.argv[0]).path, 'Config.json')
    #    print(config_file)
    open(config_file, "w").write(json.dumps(config, indent=4))


# def get_response_header_using_cookie(url):
#     import requests
#     r = requests.get(url)
#     header = r.headers
#     # cookie = r.cookies
#     r = requests.get(url,cookies = cookie)
#     print(r.headers)

# def read_csv(file):
#     """
#     Read a csv file, return a same_length_2D_list
#     :return: return a 2D-list, same as the csv file, all sub-list are the same length. If the content is 'floatable', it will be convert to float
#     """
#
#     with open(file) as input_file:
#         input_file_content = input_file.readlines()
#
#     input_file_content = [x.strip().split(',') for x in input_file_content]
#     data = [[(float(y) if is_float(y) else y) for y in x] for x in input_file_content]
#     return same_length_2d_list(data)
#


def read_csv(file):
    data = []
    with open(file, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([(float(x) if is_float(x) else x) for x in row])
    return same_length_2d_list(data)


def read_csv_and_transpose(filename):
    """
    read in an csv file and transpose it
    :param filename:
    :return: a list of lists, each list contain one column of the xlsx file
    """

    data = read_csv(filename)
    ret = transpose_2d_list(data)
    # for i in ret:
    #     print(i[600:])
    return ret


# backward compatibility
read_csv_to_horizontal_lists = read_csv_and_transpose


def read_txt_table(file, separater='\t'):
    data = open(file).readlines()
    data = [x.split(separater) for x in data]
    return same_length_2d_list(data)


def read_txt_and_transpose(file, separater='\t'):
    return transpose_2d_list(read_txt_table(file, separater=separater))


def raise_not_implemented_exception():
    raise Exception("Not Implemented")


def import_from_absolute_path(path_to_py):
    """
    Import a python file with its absolute path, and return the module.
    For example:

    my_module = import_from_absolute_path(r"C:/my_program/script.py")
    my_module.my_function()

    :param path_to_py:
    :return:
    """
    abs_path = os.path.abspath(path_to_py)
    module_name = filename_class(abs_path).name_stem
    spec = importlib.util.spec_from_file_location(module_name, path_to_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def remove_invalid_data(x, y, accept_function=float, first_as_label=False):
    """
    # filter two lists of data (it should be the same length), only when a pair of them are all acceptable, the data point is taken
    :param first_as_label:
    :param x:
    :param y:
    :param accept_function: a function, that returns true when the data is acceptable
    :return:output_x,output_y,label_x,label_y
    """

    if first_as_label:
        label_x = str(x[0])
        label_y = str(y[0])
    else:
        label_x = ""
        label_y = ""

    output_x = []
    output_y = []
    assert len(x) == len(y), 'Length of two input lists are not the same.'

    for i in range(1, len(x)):

        try:
            new_x = accept_function(x[i])
            new_y = accept_function(y[i])
        except Exception:
            continue

        output_x.append(new_x)
        output_y.append(new_y)

    return output_x, output_y, label_x, label_y



def print_float_and_stderr(value, stderr, sig_digits=2):
    # e.g. digit=3
    # 12312.15 ± 12.13 --> 12312 ± 12
    # 12312.15 ± 0.03 --> 12312.15 ± 0.03
    # 0.001234 ± 0.000010 --> 0.00123 ± 0.00001

    # number of the digits after the decimal
    # minimum zero
    # if stderr larger than zero, take that
    # if sig_digit>current sig_digits, take that
    try:
        digits_after_dec = -int(math.log(stderr, 10)) + 1
        digits_after_dec = max(0, digits_after_dec)
        value_accuracy = math.floor(-math.log(abs(value), 10)) + sig_digits
        digits_after_dec = max(digits_after_dec, value_accuracy)
        return "{:.[DIGIT]f} ± {:.[DIGIT]f}".replace("[DIGIT]", str(digits_after_dec)).format(value, stderr)
    except OverflowError as e:
        print(e)
        return str(value) + " ± " + str(stderr)
    except ValueError as e:
        print(e)
        return str(value) + " ± " + str(stderr)