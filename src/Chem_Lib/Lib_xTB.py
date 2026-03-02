# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

# import sys
# import pathlib
# parent_path = str(pathlib.Path(__file__).parent.resolve())
# sys.path.insert(0,parent_path)

from Python_Lib.My_Lib_Stock import *
from .Lib_Coordinates import *
from .Lib_XYZ import *
from .Lib_Constants import *


class xTB_Coord_file():
    def __init__(self, path):
        """形如下面的例子，而且单位是bohr
        $coord
        33.06453805899918	25.545317796282273	20.95517303173355	c
        32.33510377364891	25.70405479102171	23.470398507902487	c
        33.82042851013935	26.053654124674043	24.84233967672191	h
        35.03363268421934	25.832556167715538	20.44872642946963	h
        31.244731797879204	25.035091741762653	19.082454439033764	c
        31.798421553339388	24.98595886243854	17.10769063543005	h
        $end"""
        self.filename = path
        with open(self.filename) as input_file_lines:
            input_file_lines = input_file_lines.readlines()
        coordinates = []
        for line in input_file_lines:
            if line.strip().startswith("$coord"):
                continue
            if line.strip().startswith("$end"):
                break
            line = line.strip().split()
            assert len(line) == 4
            element = line[-1]
            element = element[0].upper() + element[1:].lower()
            line = [float(x) * bohr__A for x in line[:3]]
            std_coordinate_line = "{}\t{}\t{}\t{}".format(element, *line)
            coordinates.append(std_coordinate_line)
        self.coordinate_object = Coordinates(coordinates)
        # open_with_gview(self.coordinate_object.gjf_file())

class xTB_pull_result:
    def __init__(self, path):
        self.filename = path
        self.normal_termination = True
        self.method = 'GFN2-xTB-D4'
        self.xyz_object = XYZ_file(path, last_only=True)
        self.electronic_energy = float(re.findall(r"SCF done {3}}(-*\d+\.\d+)", self.xyz_object.titles[-1])[0]) * Hartree__KJ_mol
        self.coordinates = str(self.xyz_object.last_coordinate) + '\n'
        self.coordinate_object = self.xyz_object.last_coordinate


class xTB_opt_result:
    def __init__(self, path, one_structure=None):
        """
        :param path: .xtbopt_traj.xyz file, but .xtb.log is also needed.
        :param one_structrue: list of lines, one standard xyz file structure,
                              [atomnumber,comment_line(with energy in kJ/mol in the format 1.23454 kJ/mol),coordinates]
        """

        self.filename = path
        self.normal_termination = True
        self.method = 'GFN2-xTB-D4'
        if one_structure:
            self.coordinates = [x.strip()[0].upper() + x.strip()[1:] for x in one_structure[2:]]  # xTB会把元素的第一个字母小写，造成std_coordainte认不出来
            self.coordinates = [std_coordinate(x) for x in self.coordinates]
            self.coordinates = '\n'.join(self.coordinates) + '\n'
            self.coordinate_object = Coordinates(self.coordinates.splitlines())
        else:
            xyz_object = XYZ_file(path, last_only=True, equal_atom_count=True)
            self.coordinate_object = xyz_object.coordinates[-1]
            self.coordinates = str(self.coordinate_object)
            self.last_title = xyz_object.titles[-1]

        self.H = 0
        self.G = 0
        self.electronic_energy = 0

        self.output_filename = None

        if self.filename.endswith('.last.xtbopt_traj.xyz'):
            self.output_filename = rreplace(self.filename, '.last.xtbopt_traj.xyz', '.xtb.log')
        elif self.filename.endswith('.xtbopt_traj.xyz'):
            self.output_filename = rreplace(self.filename, '.xtbopt_traj.xyz', '.xtb.log')
        elif self.filename.endswith('.xtbopt_final.xyz'):
            self.output_filename = rreplace(self.filename, '.xtbopt_final.xyz', '.xtb.log')

        if (self.output_filename is not None) and os.path.isfile(self.output_filename):
            with open(self.output_filename, encoding='utf-8', errors='ignore') as xTB_output_file_object:
                xTB_output_content = xTB_output_file_object.readlines()

            thermo_table_head_line_count = [count for count, x in enumerate(xTB_output_content) if "::                  THERMODYNAMIC                  ::" in x]
            if len(thermo_table_head_line_count) == 1:
                thermo_table_head_line_count = thermo_table_head_line_count[-1]

                free_energy_line = xTB_output_content[thermo_table_head_line_count + 2]
                if "total free energy" in free_energy_line:
                    self.G = re.findall(r'-*\d+\.\d+', free_energy_line)
                    if self.G:
                        # note that this is actually the Gibbs free energy, not the H, This is to make the Gaussian Extract work
                        self.G = float(self.G[0]) * Hartree__KJ_mol

                for H_line in xTB_output_content[thermo_table_head_line_count + 2:]:
                    if "TOTAL ENTHALPY" in H_line:
                        self.H = re.findall(r'-*\d+\.\d+', free_energy_line)
                        if self.H:
                            # note that this is actually the Gibbs free energy, not the H, This is to make the Gaussian Extract work
                            self.H = float(self.H[0]) * Hartree__KJ_mol
                            break

            summary_table_head_line_count = [count for count, x in enumerate(xTB_output_content) if
                                             "::                     SUMMARY                     ::" in x]
            if summary_table_head_line_count:
                summary_table_head_line_count = summary_table_head_line_count[-1]

                electronic_energy_line = xTB_output_content[summary_table_head_line_count + 2]
                if "total energy" in electronic_energy_line:
                    self.electronic_energy = re.findall(r'-*\d+\.\d+', electronic_energy_line)
                    if self.electronic_energy:
                        # note that this is actually the Gibbs free energy, not the H, This is to make the Gaussian Extract work
                        self.electronic_energy = float(self.electronic_energy[0]) * Hartree__KJ_mol
        else:
            # no log file mode, only electronic energy
            re_ret = re.findall(r'energy:\s*(-*\d+\.\d+)', self.last_title)
            if re_ret:
                self.electronic_energy = float(re_ret[0]) * Hartree__KJ_mol


# print(xTB_opt_result(r"D:\Gaussian\LXT_MukaiyamaAldol\xTB_run_Temp\xtbopt.log").electronic_energy)


if __name__ == '__main__':
    pass