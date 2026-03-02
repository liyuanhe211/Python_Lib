__author__ = 'LiYuanhe'

# -*- coding: utf-8 -*-
import os
import pathlib
import sys
import pyexcel
import numpy as np

parent_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(parent_path)

# from Python_Lib.My_Lib_PyQt6 import *
# from Lib_Coordinates import *
from Lib_Coordinates import Coordinates
from Lib_Gaussian import *
from Lib_CP2K import *
from Lib_Data import *
from Lib_Filetype import Filetype, file_type
from Lib_GUI import *
from Lib_MOPAC import *
from Lib_ORCA import *
from Lib_xTB import *
from Lib_Utilities import *


def unify_basis(input_str: str):
    """
    Unify multiple writing of the same basis, like 6-31G(d,p) and 6-31G*, def-TZVP and TZVP
    :param input_str:
    :return:
    """
    if not isinstance(input_str, str):
        print(input_str)
    input_str = input_str.lower()
    if input_str.endswith("(d,p)"):
        input_str = input_str.replace("(d,p)", '**')
    elif input_str.endswith("(d)"):
        input_str = input_str.replace("(d)", '*')
    elif input_str.startswith('def2-'):
        input_str = input_str.replace('def2-', 'def2')
    elif input_str.startswith('def-'):
        input_str = input_str.replace('def-', '')
    if input_str.startswith('sv(p)'):
        input_str = input_str.replace('sv(p)', 'sv')
    return input_str


def unify_method(input_str: str):
    input_str = input_str.lower()
    input_str = input_str.replace('pbe1pbe', 'pbe0')
    input_str = input_str.replace('pbepbe', 'pbe')
    input_str = ''.join([x for x in input_str if 'a' <= x <= 'z' or "0" <= x <= '9'])
    if input_str.startswith('u') or input_str.startswith('r'):  # 去掉开壳层闭壳层标记
        input_str = input_str[1:]
    return input_str


def load_factor_database(filename="校正因子.xlsx"):

    scaling_factor_database = pyexcel.get_records(file_name=filename)
    for line in scaling_factor_database:
        # 储存一个原始的，一个最后的
        line['Basis'] = [line['Basis'], unify_basis(line['Basis'])]
        line['Method'] = [line['Method'], unify_method(line['Method'])]

    return scaling_factor_database





def read_comp_config():
    """
    read computer configurations from Comp_configs.txt
    :return:a list of dict, each dict contains the information like system, path, etc.
    """

    with open('Comp_configs.txt') as file:
        file = file.readlines()
    file = [x.strip() for x in file][2:]  # 删掉说明行
    file = split_list_by_item(file, "")

    ret = []

    for computer in file:
        temp = collections.OrderedDict()
        temp['title'] = computer[0]
        temp["system"] = computer[1].lstrip("system=")
        temp['path'] = computer[2].lstrip("path=")
        temp['mem'] = float(computer[3].lstrip("mem="))
        temp['proc'] = int(computer[4].lstrip("proc="))

        ret.append(temp)

    return ret



class MECP_output:
    # only one step orca was supported
    def __init__(self, output):
        if isinstance(output, str) and file_type(output) == Filetype.MECP_output_folder:
            report_file_path = os.path.join(os.path.realpath(output), 'ReportFile')
            with open(report_file_path) as file:
                self.lines = file.readlines()

        self.normal_termination = False

        self.is_optimization = True
        self.opt_energies = []
        self.converged = [[], [], [], [], []]
        self.opt_coordinates = []

        self.coordinates = []
        self.electronic_energy = 0

        self.charge = 999
        self.multiplicity = 999

        self.coords = []  # list of Coordinate Class Object
        self.get_coords()

        # self.keywords = []
        # self.level = []
        # self.get_keywords()

        self.geom_steps = []  # contain list of [list of lines] <-- each step of optimization

        self.geom_steps = split_list(self.lines, lambda x: "Geometry at Step" in x, include_separator=True)[1:]

        self.get_opt_energies()
        self.get_opt_coords()
        self.get_converged()

        self.process()

        # self.level_str = '/'.join(self.level)

    def get_opt_coords(self):
        re_pattern1 = "Geometry at Step"
        re_pattern2 = "Initial Geometry"
        for step_content in self.geom_steps:
            for count, line in enumerate(step_content):
                if re_pattern1 in line or re_pattern2 in line:
                    coordinate_lines = []
                    for line2 in step_content[count + 1]:
                        if std_coordinate(line2):
                            coordinate_lines.append(line2)
                        else:
                            break
                    self.opt_coordinates.append(Coordinates(coordinate_lines))
                    break

    def get_opt_energies(self):

        re_pattern = r"Difference in E\:\s+(-*[0-9]+\.[0-9]+)"
        for step_content in self.geom_steps:
            for line in reversed(step_content):
                re_ret = re.findall(re_pattern, line)
                if re_ret:
                    self.opt_energies.append(''.join(re_ret[0]))
                    break

        self.opt_energies = [float(x) for x in self.opt_energies]

    def get_converged(self):
        for step_count, step_content in enumerate(self.geom_steps):
            for count, line in enumerate(step_content):
                if "Convergence Check" in line:

                    for k, line2 in enumerate(step_content[count + 1:count + 6]):
                        re_ret = re.findall(r"-*\d\.\d+", line2)
                        if len(re_ret) == 2:
                            value = abs(float(re_ret[0])) / float(re_ret[1])
                            # print(value)
                            if value < 0:
                                value = -value
                            if value < 0.01:
                                value = 0.01  # 防止log时出现负无穷
                            self.converged[k].append(value)

                    break

        # 调整顺序为
        # ["Max F","RMS F","Max D","RMS D",'Energy']

        self.converged = self.converged[:4] + [[x ** 0.5 for x in self.converged[4]]]

        # 从[[...],[...],[...],[...]] 换成 [[ , , , ]...]
        self.converged = [[self.converged[x][step] for x in range(5)] for step in range(len(self.converged[0]))]

    def process(self):
        for count, line in enumerate(self.lines):
            if "The MECP Optimization has CONVERGED" in line:
                self.normal_termination = True
                energies_for_first_state = []
                for energy_lines in self.lines:
                    re_ret = re.findall(r"Energy of First State\:\s+(-*[0-9]+\.[0-9]+)", energy_lines)
                    if re_ret:
                        energies_for_first_state.append(''.join(re_ret[0]))
                #                       print(energies_for_first_state)
                energies_for_first_state = [float(x) * Hartree__KJ_mol for x in energies_for_first_state]
                self.electronic_energy = energies_for_first_state[-1]

    #                print(self.electronic_energy/Hartree__KJ_mol)

    # def get_keywords(self):
    #     for line in self.input_file_lines:
    #         if line.strip().startswith('!'):
    #             self.keywords+=line.strip().strip('!').split()
    #     self.keywords = [x.lower() for x in self.keywords]
    #     self.method = []
    #     self.basis = []
    #     for keyword in self.keywords:
    #         for functional in functional_keywords_of_orca:
    #             if keyword.lower()==functional.lower() or 'ri-'+keyword.lower()==functional.lower():
    #                 self.method.append(functional)
    #         for basis in basis_set_keywords_of_orca:
    #             if keyword.lower()==basis.lower():
    #                 self.basis.append(basis)
    #     self.method = [x if not x.lower().startswith('ri-') else x[3:] for x in self.method]
    #     self.method = list(set(self.method))
    #     self.basis = list(set(self.basis))
    #
    #     self.level = self.method+self.basis

    # def read_charge_and_multiplet(self):
    #     # acquire changes
    #     for count,line in enumerate(self.lines):
    #         charge_re_result =re.findall(r'''Total Charge +Charge +.... +(\d)+''',line) # match " Total Charge           Charge          ....    0"
    #         multiplet_re_result = re.findall(r'''Multiplicity +Mult +.... +(\d)+''',line) # match "Multiplicity           Mult            ....    1"
    #         input_re_result = re.findall(r'''\* +xyz \+(\d+) +(\d+)''',line) # match "* xyz 0   1"
    #
    #         if len(charge_re_result)==1:
    #             self.charge = int(charge_re_result[0])
    #         elif len(multiplet_re_result)==1:
    #             self.multiplicity = int(multiplet_re_result[0])
    #         elif len(input_re_result)==1:
    #             self.charge,self.multiplicity = input_re_result[0]
    #             self.charge = int(self.charge)
    #             self.multiplicity = int(self.multiplicity)

    def get_coords(self):
        marks = {r"Geometry at Step": 1, r"Initial Geometry": 1}  # ,r"CARTESIAN COORDINATES \(A\.U\.\)":3 需要调单位，暂未实现
        # see the discription in the Gaussian version of this function
        # Numbers are the value till the coordinates starts (coordinate start from the next line is 1)

        for count, line in enumerate(self.lines):
            for mark in marks:
                if re.findall(mark, line):

                    coords = []

                    for coord_line in self.lines[count + marks[mark]:]:
                        if std_coordinate(coord_line):  # 确认这一行中存在坐标
                            coords.append(coord_line)
                        else:
                            break
                    if coords:
                        self.coords.append(Coordinates(coords, self.charge, self.multiplicity))

        if self.coords:
            self.coordinates = self.coords[-1]
        else:
            self.coordinates = Coordinates()







if __name__ == "__main__":
    pass
