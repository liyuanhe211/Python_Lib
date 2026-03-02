# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

# import sys
# import pathlib
# parent_path = str(pathlib.Path(__file__).parent.resolve())
# sys.path.insert(0,parent_path)

from Python_Lib.My_Lib_Stock import *
from .Lib_Coordinates import *


class XYZ_file:
    def __init__(self, path, last_only=False, end_condition=None, equal_atom_count=False, terminal_countdown=100):
        """
        Read an standard xyz file, give the components
        :param path:
        :param equal_atom_count: 目前不支持原子数不等的XYZ, 默认认为原子数不等，会耗时多；如果确认文件中原子数都相等，可以快很多
        :param terminal_countdown: 达到end_condition 之后，再接受多少个结构，防止轨迹突然中止
        """

        self.filename = path
        self.coordinates = []
        self.titles = []
        self.energies = []  # you need to call the method to extract the energy

        is_xTB = (self.filename.endswith("xtbscan.traj.xyz") or
                  self.filename.endswith("xtbopt_traj.xyz") or
                  self.filename.endswith("xtbopt_final.xyz") or
                  self.filename.endswith("xtbscan.log") or
                  self.filename.endswith("xtb_Pull.xyz"))

        if last_only:
            atom_count = int(open(self.filename).readline().strip())
            input_file_lines = read_last_n_lines_fast(self.filename, atom_count + 2).splitlines()
            self.titles.append(input_file_lines[1])
            self.coordinates.append(Coordinates(input_file_lines[2:], is_xtb=is_xTB))
        else:
            with open(self.filename) as input_file_lines:
                input_file_lines = input_file_lines.readlines()

            countdown_now = -1
            lines_processed = -1
            for count, line in enumerate(input_file_lines):
                if countdown_now == 0:
                    break
                if count >= lines_processed:
                    self.atom_count = int(line)
                    self.titles.append(input_file_lines[count + 1])
                    current_coordinate = Coordinates(input_file_lines[count + 2:count + 2 + self.atom_count], is_xtb=is_xTB)
                    self.coordinates.append(current_coordinate)
                    lines_processed = count + self.atom_count + 2
                    if end_condition:
                        if end_condition(current_coordinate) and countdown_now == -1:
                            countdown_now = terminal_countdown
                    if countdown_now != -1:
                        countdown_now -= 1

        if self.coordinates:
            self.last_coordinate = self.coordinates[-1]
        else:
            self.last_coordinate = None

    def read_energies(self):
        """
        Read back energies given in the title from a ConfSearch extract
        Example:
            G01_M001_-638159.00
            G02_M001_7.31
            G03_M001_11.26
        """

        for title in self.titles:
            re_ret = re.findall(r"G\d+_M001_(-*\d+\.\d+)", title)
            assert len(re_ret) == 1
            self.energies.append(float(re_ret[0]))

        self.energies = [(x if count == 0 else x + self.energies[0]) for count, x in enumerate(self.energies)]

        assert len(self.energies) == len(self.coordinates)

    def write(self):
        ret = ""
        assert len(self.titles) == len(self.coordinates)
        for count, title in enumerate(self.titles):
            coordinate = self.coordinates[count]
            ret += str(self.atom_count) + '\n'
            ret += title
            ret += str(coordinate) + '\n'
        return ret

    def gjf_file_for_last(self):
        return self.coordinates[-1].gjf_file(filename=self.filename + '.gjf')
