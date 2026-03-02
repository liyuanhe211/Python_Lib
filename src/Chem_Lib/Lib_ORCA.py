# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

# import sys
# import pathlib
# parent_path = str(pathlib.Path(__file__).parent.resolve())
# sys.path.insert(0,parent_path)

from Python_Lib.My_Lib_Stock import *
from .Lib_Filetype import Filetype, file_type


class ORCA_output:
    # only one step orca was supported
    def __init__(self, output):
        if isinstance(output, str) and file_type(output) == Filetype.orca_output:
            with open(output, encoding='utf-8') as file:
                self.lines = file.readlines()

        elif isinstance(output, list):  # input as list
            self.lines = output

        else:
            raise MyException('Not valid file')

        self.normal_termination = False

        self.is_optimization = False
        self.has_finalgrid = False
        self.opt_energies = []
        self.converged = [[], [], [], [], []]
        self.opt_coordinates = []

        self.gCP_correction = 0

        self.has_freq = False
        self.harmonic_freqs = []
        self.G_correction = 0
        self.H_correction = 0
        self.S = 0
        self.G = 0
        self.H = 0
        self.imaginary_count = 0
        self.has_imaginary_freq = False

        self.input_file_lines = []
        self.input_filename = ""
        self.coordinates = []
        self.scf_converged = False
        self.electronic_energy = 0
        self.process()

        self.charge = 999
        self.multiplicity = 999
        self.read_charge_and_multiplet()

        self.coords = []  # list of Coordinate Class Object
        self.get_coords()

        self.keywords = []
        self.level = []
        self.get_keywords()
        if 'opt' in self.keywords:
            self.is_optimization = True

        self.geom_steps = []  # contain list of [list of lines] <-- each step of optimization
        self.finalgrid_calculation = []  # contain list of lines of FINAL ENERGY EVALUATION AT THE STATIONARY POINT
        if self.is_optimization:
            self.geom_steps = split_list(self.lines, lambda x: "GEOMETRY OPTIMIZATION CYCLE" in x, include_separator=True)[1:]

            # ORCA 可能会最后单独算一个高格点单点，把这个过程分离出来。
            split_last_step = split_list(self.geom_steps[-1], lambda x: "FINAL ENERGY EVALUATION AT THE STATIONARY POINT" in x, include_separator=True)
            assert len(split_last_step) in [1, 2], 'Split final grid calculation error'
            if len(split_last_step) == 2:
                self.has_finalgrid = True
                self.geom_steps[-1] = split_last_step[0]
                self.finalgrid_calculation = split_last_step[1]

            self.get_opt_energies()
            self.get_opt_coords()
            self.get_converged()

        self.get_MP2_progress()
        self.get_SCF_progress()
        self.get_freq_result()
        # print(self.keywords)

        self.level_str = '/'.join(self.level)

    def get_opt_coords(self):
        re_pattern = "CARTESIAN COORDINATES (ANGSTROEM)"
        for step_content in self.geom_steps:
            for count, line in enumerate(step_content):
                if re_pattern in line:
                    coordinate_lines = []
                    for line2 in step_content[count + 2]:
                        if std_coordinate(line2):
                            coordinate_lines.append(line2)
                        else:
                            break
                    self.opt_coordinates.append(Coordinates(coordinate_lines))

    def get_opt_energies(self):

        re_pattern = r"FINAL SINGLE POINT ENERGY\s+(-[0-9]+\.[0-9]+)"
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
                if "Geometry convergence" in line:

                    for k, line2 in enumerate(step_content[count + 3:count + 8]):
                        re_ret = re.findall(r"-*\d\.\d+", line2)
                        if len(re_ret) == 2:
                            value = abs(float(re_ret[0])) / float(re_ret[1])
                            # print(value)
                            if value < 0.01:
                                value = 0.01  # 防止log时出现负无穷
                            if step_count == 0:
                                if k == 0:  # 第一步时没有energy difference的输出，应识别
                                    self.converged[0].append(100)
                                self.converged[k + 1].append(value)
                            else:
                                self.converged[k].append(value)

                    break

        # 调整顺序为
        # ["Max F","RMS F","Max D","RMS D",'Energy']

        self.converged = [self.converged[2]] + [self.converged[1]] + [self.converged[4]] + [self.converged[3]] + \
                         [[x ** 0.5 for x in self.converged[0]]]

        # 从[[...],[...],[...],[...]] 换成 [[ , , , ]...]
        self.converged = [[self.converged[x][step] for x in range(5)] for step in range(len(self.converged[0]))]

    def process(self):
        for count, line in enumerate(self.lines):
            if not self.input_file_lines and "INPUT FILE" in line:
                for input_lines in self.lines[count + 1:]:
                    if "****END OF INPUT****" in input_lines:
                        break

                    if self.input_filename == "":
                        match = re.findall(r'NAME\s+=\s+(.+)', input_lines)
                        if match:
                            self.input_filename = match[0]

                    match = re.findall(r"\|\s*\d+>(.+)", input_lines)
                    if match:
                        self.input_file_lines.append(match[0])

            if "****ORCA TERMINATED NORMALLY****" in line:
                self.normal_termination = True

            if "gCP correction" in line:
                match = re.findall(r'gCP correction\s+(-*\d+\.\d+)', line)
                if match:
                    self.gCP_correction = float(match[0]) * 2625.49962

            if "FINAL SINGLE POINT ENERGY" in line:
                match = re.findall(r'FINAL SINGLE POINT ENERGY\s+(-\d+\.\d+)', line)
                if match:
                    self.electronic_energy = float(match[0]) * 2625.49962
        pass

    def get_keywords(self):
        for line in self.input_file_lines:
            if line.strip().startswith('!'):
                self.keywords += line.strip().strip('!').split()
        self.keywords = [x.lower() for x in self.keywords]
        self.method = []
        self.basis = []
        for keyword in self.keywords:
            for functional in functional_keywords_of_orca:
                if keyword.lower() == functional.lower() or 'ri-' + keyword.lower() == functional.lower():
                    self.method.append(functional)
            for basis in basis_set_keywords_of_orca:
                if keyword.lower() == basis.lower():
                    self.basis.append(basis)
        self.method = [x if not x.lower().startswith('ri-') else x[3:] for x in self.method]
        self.method = list(set(self.method))
        self.basis = list(set(self.basis))

        self.level = self.method + self.basis

    def read_charge_and_multiplet(self):
        # acquire changes
        for count, line in enumerate(self.lines):
            charge_re_result = re.findall(r"""Total Charge +Charge +.... +(-*\d)+""", line)  # match " Total Charge           Charge          ....    0"
            multiplet_re_result = re.findall(r"""Multiplicity +Mult +.... +(-*\d)+""", line)  # match "Multiplicity           Mult            ....    1"
            input_re_result = re.findall(r"""\* +xyz \+(\d+) +(\d+)""", line)  # match "* xyz 0   1"

            if len(charge_re_result) == 1:
                self.charge = int(charge_re_result[0])
            elif len(multiplet_re_result) == 1:
                self.multiplicity = int(multiplet_re_result[0])
            elif len(input_re_result) == 1:
                self.charge, self.multiplicity = input_re_result[0]
                self.charge = int(self.charge)
                self.multiplicity = int(self.multiplicity)

    def get_MP2_progress(self):
        self.window = -1
        self.per_batch = -1
        self.processed_MP2 = []
        self.has_MP2 = False
        for count in range(len(self.lines) - 1, -1, -1):
            line = self.lines[count]
            re_ret = re.findall(r'Operator \d+ {2}- window\s+\.\.\.\s+\(\s*\d+\-\s*(\d+)\)', line)
            if re_ret:
                self.window = int(re_ret[0])
                for count2, line2 in enumerate(self.lines[count:]):
                    re_ret = re.findall(r'Operator \d+ {2}- Number of orbitals per batch ...\s+(\d+)', line2)
                    if re_ret:
                        self.per_batch = int(re_ret[0])

                    # Process  5:   Internal MO  65
                    re_ret = re.findall(r'Process\s+\d+:\s+Internal MO\s+(\d+)', line2)
                    if re_ret:
                        self.has_MP2 = True
                        self.processed_MP2.append(int(re_ret[0]))

                break

        # for count,line in enumerate(self.lines):
        #     if "Starting loop over batches of integrals:" in line:
        #         for count2,line2 in enumerate(self.lines[count:]):
        #             #Operator 0  - window                       ... (  0-149)x(150-2839)
        #             re_ret = re.findall(r'Operator 0  - window\s+\.\.\.\s+\(\s+\d+\-\s+(\d+)\)',line2)
        #             if re_ret:
        #                 self.window = int(re_ret[0])
        #
        #         break

    def get_SCF_progress(self):
        self.scf_iter = []
        self.scf_converged = False
        for count, line in enumerate(self.lines):
            if "SCF ITERATIONS" in line:
                self.scf_iter = []
                self.scf_converged = False
                for count2, line2 in enumerate(self.lines[count:]):
                    re_ret = re.findall(r'\s*\d+\s+(\-\d+\.\d+)\s+', line2)
                    if re_ret:
                        self.scf_iter.append(float(re_ret[0]))
                    if 'SCF CONVERGED AFTER ' in line2:
                        self.scf_converged = True
                        break

    def get_coords(self):

        marks = {r"\* +xyz": 1, r"CARTESIAN COORDINATES \(ANGSTROEM\)": 2}  # ,r"CARTESIAN COORDINATES \(A\.U\.\)":3 需要调单位，暂未实现
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

    def get_freq_result(self):

        # these information are NOT sufficient for a Thermo calculation

        for count in range(len(self.lines) - 1, -1, -1):  # read the last one
            if "ORCA SCF HESSIAN" in self.lines[count]:
                Hessian_lines = self.lines[count:]

                self.has_freq = True
                self.has_imaginary_freq = False

                for count2, line in enumerate(Hessian_lines):
                    # get frequencies in cm**-1
                    if "VIBRATIONAL FREQUENCIES" in line:
                        self.harmonic_freqs = []
                        for vib_count, vib_line in enumerate(Hessian_lines[count2 + 3:]):
                            re_ret = re.findall(r'\d+:\s+(-*\d+\.\d+)\s+cm\*\*\-1', vib_line)
                            if re_ret:
                                assert len(re_ret) == 1
                                re_ret = re_ret[0]
                                if vib_count < 6:  # 前六个是投影掉的振动和转动，为0
                                    assert float(re_ret) == 0
                                    continue
                                else:
                                    self.harmonic_freqs.append(float(re_ret))
                            else:
                                break

                        self.imaginary_count = len([x for x in self.harmonic_freqs if x < 0])
                        if self.imaginary_count != 0:
                            self.has_imaginary_freq = True

                    if "THERMOCHEMISTRY AT" in line:
                        self.temp = float(re.findall(r'Temperature\s+\.+\s+(\d+\.\d+)\s+K', Hessian_lines[count2 + 3])[0])
                        self.pressure = float(re.findall(r'Pressure\s+\.+\s+(\d+\.\d+)\s+atm', Hessian_lines[count2 + 4])[0])

                    # ORCA cannot determine the rotation symm number, assume 1 for all molecules

                    # get corrections
                    # enthalpy_corr_pattern =r"Thermal Enthalpy correction\s+\.+\s+(-*\d+\.\d+)\s+Eh"
                    gibbs_corr_lead_pattern = r"For completeness - the Gibbs free enthalpy minus the electronic energy"
                    gibbs_corr_pattern = r"G\-E\(el\)\s+\.+\s+(-*\d+\.\d+)\s+Eh"
                    enthalpy_pattern = r"Total Enthalpy\s+\.+\s+(-*\d+\.\d+)\s+Eh"
                    gibbs_pattern = r"Final Gibbs free enthalpy\s+\.+\s+(-*\d+\.\d+)\s+Eh"
                    entropy_pattern = r'sn\= 1\s+qrot\/sn\=\s+-*\d+\.\d+\s+T\*S\(rot\)\=\s+-*\d+\.\d+\s+kcal\/mol\s+T\*S\(tot\)\=\s+(-*\d+\.\d+)\s+kcal\/mol'

                    # orca的 enthalpy correction不是Gaussian里的ZPE+H(0->T)
                    # re_ret = re.findall(enthalpy_corr_pattern,line)
                    # if re_ret: self.H_correction = float(re_ret[0])*2625.49962

                    re_ret = re.findall(entropy_pattern, line)
                    if re_ret: self.S = float(re_ret[0]) * 4.184 * 1000 / self.temp

                    re_ret = re.findall(enthalpy_pattern, line)
                    if re_ret: self.H = float(re_ret[0]) * 2625.49962

                    re_ret = re.findall(gibbs_pattern, line)
                    if re_ret: self.G = float(re_ret[0]) * 2625.49962

                    if gibbs_corr_lead_pattern in line:
                        re_ret = re.findall(gibbs_corr_pattern, Hessian_lines[count2 + 1])
                        if re_ret:
                            self.G_correction = float(re_ret[0]) * 2625.49962

        self.H_correction = self.H - self.electronic_energy

class ORCA_Input:
    def __init__(self, path):
        with open(path) as input_file:
            input_lines = input_file.readlines()

        input_lines = [x.strip() for x in input_lines]
        for count, x in enumerate(input_lines):
            if '#' in x:
                input_lines[count] = x[:x.find("#")]

        input_lines = remove_blank(input_lines)

        self.step_list = remove_blank(split_list_by_item(input_lines, "$new_job"))
        self.step_count = len(self.step_list)
        self.steps = [ORCA_Step(x) for x in self.step_list]


class ORCA_Step:
    def __init__(self, text_list: list):

        self.charge = 0
        self.multiplet = 1
        self.proc = 1
        self.mem_per_core = 85
        self.mem = 0.1
        self.base = ""
        self.geom = []
        self.xyzfile = ""
        self.read_geom = False

        self.input_lines = text_list

        skip_line_count = []

        for count, line in enumerate(self.input_lines):
            pal_find = re.findall(r'\%pal +nprocs +(\d+) end', line)
            maxcore_find = re.findall(r"\%maxcore (\d+)", line)
            base_find = re.findall(r'''%base "(.+)"''', line)
            geometry_find = re.findall(r'''\* *xyz +(-*\d+) +(-*\d+)''', line)
            read_geometry_find = re.findall(r'''\* *xyzfile +(-*\d+) +(-*\d+) +(.+\.xyz)''', line)

            if len(geometry_find) == 1:
                self.charge, self.multiplet = [int(x) for x in geometry_find[0]]

                for count_geom, geom_line in enumerate(self.input_lines[count + 1:]):
                    if geom_line.strip() == "*":
                        skip_line_count.append(count_geom + count + 1)
                        break
                    self.geom.append(geom_line)
                    skip_line_count.append(count_geom + count + 1)
                self.geom_text = '\n'.join(self.geom)

            if len(read_geometry_find) == 1:
                self.read_geom = True
                self.charge, self.multiplet, self.xyzfile = read_geometry_find[0]

            if len(pal_find) == 1:
                self.proc = int(pal_find[0])

            if len(maxcore_find) == 1:
                self.mem_per_core = int(maxcore_find[0])

            if len(base_find) == 1:
                self.base = base_find[0]

            if sum([len(x) for x in [pal_find, maxcore_find, base_find, read_geometry_find, geometry_find]]) == 1:
                skip_line_count.append(count)

        self.mem = self.proc * self.mem_per_core / 1000 / 0.85

        self.other = [x for count, x in enumerate(self.input_lines) if count not in skip_line_count]

    def __str__(self):
        return "\n".join(self.other)


if __name__ == '__main__':
    pass