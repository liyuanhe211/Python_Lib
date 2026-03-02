# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

# import sys
# import pathlib
# parent_path = str(pathlib.Path(__file__).parent.resolve())
# sys.path.insert(0,parent_path)

from Python_Lib.My_Lib_Stock import *

def build_mopac_route_dict(line: str) -> dict:
    """
    Turn something like this: AUX  PM7 CHARGE=0 EF GNORM=0.100 SHIFT=80
    to:
    {"AUX":"",
     "PM7":"",
     "CHARGE":"0",
     "EF":"",
     "GNORM":"0.100",
     "SHIFT":"80"
    }
    """
    keywords = line.split()
    ret_dict = {}
    for keyword in keywords:
        if "=" in keyword:
            ret_dict[keyword.split()[0]] = keyword.split()[1]
        else:
            ret_dict[keyword] = ""
    return ret_dict


class MOPAC_Input:
    def __init__(self, path):
        """
        传入的输入：
        AUX  PM7 CHARGE=0 EF GNORM=0.100 SHIFT=80
        Untitled-1

        C          -1.2079 1         0.6974 1         0.0000 1
        C          -1.2080 1        -0.6974 1         0.0000 1
        C           0.0000 1        -1.3948 1         0.0001 1
        C           1.2079 1        -0.6974 1         0.0001 1
        C           1.2080 1         0.6974 1         0.0000 1
        C           0.0000 1         1.3948 1         0.0000 1
        H          -2.1606 1         1.2474 1         0.0000 1
        H          -2.1606 1        -1.2474 1        -0.0001 1
        H           0.0000 1        -2.4948 1         0.0001 1
        H           2.1606 1        -1.2474 1         0.0001 1
        H           2.1606 1         1.2474 1        -0.0001 1
        H           0.0000 1         2.4948 1         0.0000 1

        特殊情况，允许用Mul开头的关键字声明Multiplicity，用来在Chem3D里override现成的电荷

        Args:
            path:
        """

        with open(path) as input_file:
            input_data = input_file.readlines()
        input_data = [x.strip() for x in input_data]
        self.route = input_data[0]
        self.coordinate = Coordinates(std_coordinate_mopac_arc(x) for x in input_data[3:] if x.strip())
        self.route_dict = build_mopac_route_dict(self.route)
        self.charge = safe_get_dict_value(self.route_dict, 'CHARGE', 0)
        self.charge = int(self.charge)

        self.multiplet = -1

        for key, value in self.route_dict.items():
            if value:
                if key.startswith("Mul"):
                    self.multiplet = int(value)

        if self.multiplet == -1:
            if "MS" in self.route_dict:
                self.multiplet = round(float(self.route_dict['MS']) * 2 + 1)

        if self.multiplet == -1:
            multi_keywords = ["SINGLET", "DOUBLET", "TRIPLET", "QUARTET", "QUINTET", "SEXTET", "SEPTET", "OCTET", "NONET"]
            for count, i in enumerate(multi_keywords):
                if i in self.route_dict:
                    self.multiplet = count + 1

        if self.multiplet == -1:
            self.multiplet = 1

        pass


class MOPAC_Archieve:
    def __init__(self, path):

        self.filename = path
        self.normal_termination = True
        self.H = 0
        self.method = ""
        self.coordinate_filename = ""
        self.coordinates = ""

        with open(path) as input_file:
            input = input_file.readlines()

        for count, line in enumerate(input):
            if not self.method:
                if "SUMMARY OF" in line:
                    re_ret = re.findall(r'SUMMARY OF\s+(\S+)\s+CALCULATION', line)
                    if re_ret:
                        self.method = re_ret[0]

            if "HEAT OF FORMATION" in line:
                re_ret = re.findall(r'HEAT OF FORMATION\s*=\s*-*\d+\.\d+\s*KCAL\/MOL\s*=\s*(-*\d+\.\d+)\s*KJ\/MOL', line)
                if re_ret:
                    self.H = float(re_ret[0])

            if "FINAL GEOMETRY OBTAINED" in line and not self.coordinates:
                for coordinate in input[count + 4:]:
                    coordinate_output = std_coordinate_mopac_arc(coordinate)
                    if not coordinate_output:
                        break
                    self.coordinates += coordinate_output + '\n'

        if self.normal_termination:
            self.coordinate_filename = filename_class(path).only_remove_append + "_result.xyz"
            with open(self.coordinate_filename, 'w') as output_stucture_file:
                output_stucture_file.write(str(len(self.coordinates.splitlines())) + "\n" + str(self.H) + "\n")
                output_stucture_file.write(self.coordinates)

        self.coordinate_object = Coordinates(self.coordinates.splitlines())



class MOPAC_Output:
    def __init__(self, path):

        arc_file = filename_class(path).replace_append_to('arc')
        if os.path.isfile(arc_file):
            arc_object = MOPAC_Archieve(arc_file)
            self.filename = path
            self.normal_termination = True
            self.H = arc_object.H
            self.method = arc_object.method
            self.coordinate_filename = arc_object.coordinate_filename
            self.coordinates = arc_object.coordinates
            self.coordinate_object = arc_object.coordinate_object

        else:
            self.filename = path
            self.normal_termination = False
            self.H = 0
            self.method = ""
            self.coordinate_filename = ""
            self.coordinates = ""

            with open(path) as input_file:
                input_data = input_file.readlines()

            for count, line in enumerate(input_data):
                if "CALCULATION RESULTS" in line:
                    re_ret = re.findall(r'\s+(\S+)\s+CALCULATION RESULTS', line)
                    if re_ret:
                        self.method = re_ret[0]

                if "CARTESIAN COORDINATES" in line:
                    self.coordinates = ""
                    for coordinate in input_data[count + 2:]:
                        coordinate_output = std_coordinate(coordinate)
                        if not coordinate_output:
                            break
                        self.coordinates += coordinate_output + '\n'

            for line in reversed(input_data):

                if "* JOB ENDED NORMALLY *" in line:
                    self.normal_termination = True

                if "FINAL HEAT OF FORMATION" in line:
                    re_ret = re.findall(r'FINAL HEAT OF FORMATION\s*=\s*-*\d+\.\d+\s*KCAL/MOL\s*=\s*(-*\d+\.\d+)\s*KJ/MOL', line)
                    if re_ret:
                        self.H = float(re_ret[0])
                        break

            if self.normal_termination:
                self.coordinate_filename = filename_class(path).only_remove_append + "_result.xyz"
                with open(self.coordinate_filename, 'w') as output_stucture_file:
                    output_stucture_file.write(str(len(self.coordinates.splitlines())) + "\n" + str(self.H) + "\n")
                    output_stucture_file.write(self.coordinates)

            self.coordinate_object = Coordinates(self.coordinates.splitlines())

if __name__ == '__main__':
    pass