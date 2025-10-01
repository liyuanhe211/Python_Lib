# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import sys
import pathlib

Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Stock import *


def load_xlsx_as_list(filename, sheet=0, column_as_inner_list=False):
    import pyexcel
    if isinstance(sheet, int):
        excel_book_dict = pyexcel.get_book_dict(file_name=filename)
        excel_object = excel_book_dict[list(excel_book_dict.keys())[sheet]]
    else:
        excel_object = pyexcel.get_book_dict(file_name=filename)[sheet]

    if not column_as_inner_list:
        return excel_object

    else:  # 转置
        excel_object = [[r[col] for r in excel_object] for col in range(len(excel_object[0]))]
        for i in excel_object:
            print(i)
        return excel_object


def excel_formula(formula: str, *cells):
    if isinstance(cells[0], list):
        cells = cells[0]
    for count, current_cell in enumerate(cells):
        formula = formula.replace('[cell' + str(count + 1) + ']', current_cell)

    return formula


def cell(column_num, row_num):
    # start from 0
    if column_num < 26:
        return chr(ord('A') + column_num) + str(row_num + 1)
    if column_num >= 26:
        return chr(ord('A') + int(column_num / 26) - 1) + chr(ord('A') + column_num % 26) + str(row_num + 1)


def read_xlsx(file, sheet=0, all_sheets=False):
    """

    :param file: A xlsx file
    :param sheet: which sheet to read, using numbers
    :param all_sheets: read all sheet, return an ordered dict, with sheet name as key
    :return: if not all_sheets, return a 2D-list, all sub-list are the same length
    """
    import pyexcel_xlsx
    data = pyexcel_xlsx.get_data(file, skip_hidden_row_and_column=False)

    if all_sheets:
        import collections
        ret = collections.OrderedDict()
        for key, value in data.items():
            ret[key] = same_length_2d_list(value)
        return ret
    else:
        ret = data[list(data)[sheet]]
        return same_length_2d_list(ret)


def read_xlsx_to_list_of_dicts(filename):
    data = read_xlsx(filename)
    ret = []
    headers = data[0]
    assert len(headers)==len(set(headers)), "You have duplicated header item"
    for line in data[1:]:
        item = {}
        for header,cell in zip(headers,line):
            item[header] = cell
        ret.append(item)
    return ret



def read_xlsx_and_transpose(filename):
    """
    read in an xlsx file and transpose it
    :param filename:
    :return: a list of lists, each list contain one column of the xlsx file
    """

    data = read_xlsx(filename)
    return transpose_2d_list(data)


read_xlsx_to_horizontal_lists = read_xlsx_and_transpose


def write_xlsx(filename, list_2D, transpose=False):
    """
    A simple function for writing a 2D list to a xlsx file
    :param filename:
    :param list_2D:
    :param transpose: if False, the outer-layer of list-2D will be rows
    :return:
    """

    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    for row_count, row in enumerate(list_2D):
        for column_count, current_cell in enumerate(row):
            if not transpose:
                worksheet.write(row_count, column_count, current_cell)
            else:
                worksheet.write(column_count, row_count, current_cell)

    workbook.close()


if __name__ == '__main__':
    pass
