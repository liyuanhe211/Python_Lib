# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

from Python_Lib.My_Lib_Stock import *
from .Lib_Coordinates import *
from .Lib_Filetype import Filetype, file_type

class Gaussian_input:

    # TODO: 实现一些简便的修改文件的函数

    def __init__(self, path):
        with open(path) as input_file:
            input_data = input_file.readlines()

        input_data = [x.strip() for x in input_data]
        self.step_list = remove_blank(split_list(input_data, "--link1--", lower_case_match=True))
        self.step_count = len(self.step_list)
        self.steps = [Gaussian_input_step(x) for x in self.step_list]
        pass


class Gaussian_input_step:
    def __init__(self, input_list: list):  # input 为SplitStep内的一个Step

        self.charge = 999
        self.multiplet = 999
        self.proc = 1
        self.mem = 0.1
        self.chk = ""
        self.rwf = ""

        self.input_list = input_list

        self.phrase_annotates()

        # devide paragraphs
        self.paragraphs = []

        temp = []
        for line in self.input_list:
            if line.strip():  # 如果不是空行
                temp.append(line)
            else:
                self.paragraphs.append(temp)
                temp = []
        if temp:
            self.paragraphs.append(temp)

        # 此时已分成独立的paragraphs

        # read link0 command
        self.link0_list = []
        link0_line_count = 0
        for i, line in enumerate(self.paragraphs[0]):
            line = line.lower()
            if line.strip(" ").startswith("%"):
                self.link0_list.append(line)

                if "%nprocshared=" in line:
                    self.proc = int(re.findall(r"%nprocshared=(.+)", line)[0].strip())
                elif "%mem=" in line and 'mb' in line:
                    self.mem = int(float((re.findall(r"%mem=(.+)mb", line)[0].strip())) / 100) / 10
                elif "%chk=" in line:
                    self.chk = re.findall(r"%chk=(.+)", line)[0].strip()
                elif "%rwf=" in line:
                    self.rwf = re.findall(r"%rwf=(.+)", line)[0].strip()

            else:
                link0_line_count = i
                break

        self.saved = [0, 1, 2]  # 记录访问了多少paragraph，访问当前第一个paragraph使用len

        self.route_list = self.paragraphs[0][link0_line_count:]  # 除了Link0命令外，为route部分
        self.route_str = self.join(self.route_list)
        self.route_dict = Route_dict(self.route_str)
        if 'connectivity' in self.route_str:
            self.connectivity = self.paragraphs.pop(3)

        if not self.route_dict.from_gaussview and 'allcheck' in safe_get_dict_value(self.route_dict, 'geom'):
            self.title = []
            self.geom = []
            self.other = self.paragraphs[1:]

        else:
            self.title = self.paragraphs[1]
            if not [x for x in self.title if x.strip()]:
                self.title = ["Empty Title"]

            self.geom = self.paragraphs[2]

            self.other = self.paragraphs[3:]

            # extract charge and multiplet
            self.charge_and_multiplet = [x for x in self.geom[0].split() if x != '']
            self.charge = int(self.charge_and_multiplet[0])
            self.multiplet = int(self.charge_and_multiplet[1])

            # delete LP, charge & multiplet
            self.geom = [x for x in self.geom[1:] if not x.lower().strip().startswith('lp')]

        self.other_str = ""
        for paragraph in self.other:
            for line in paragraph:
                self.other_str += line + '\n'
            self.other_str += '\n'

        self.route_dict = Route_dict(self.route_str, self.other_str)  # 重新产生一个Route_dict 把其他段落包括进去

        self.geom = [std_coordinate(x) for x in self.geom]
        self.coordinate = Coordinates(self.geom, charge=self.charge, multiplet=self.multiplet)
        self.geom_text = self.join(self.geom)

    def phrase_annotates(self):
        # phrase annotate setting like "!__NAMETAG__=Orbital_Energy"
        self.annotate_lines = []
        for line in self.input_list:
            if line.startswith('!'):
                self.annotate_lines.append(line)
        for line in self.annotate_lines:
            self.input_list.remove(line)

        self.command_lines = []  # a preset of RUN command can be issued and will be phrased by qsubg09.py
        for line in self.annotate_lines:
            # "!__NAMETAG__=Orbital_Energy"
            re_ret = re.findall(r'!RUN (.+)', line)
            if re_ret:
                self.command_lines.append(line)

        for line in self.command_lines:
            if line in self.input_list:
                self.input_list.remove(line)

        self.annotates_dict = {}
        for line in self.annotate_lines:
            # "!__NAMETAG__=Orbital_Energy"
            re_ret = re.findall(r'!__(.+?)__=(.+)', line)
            if re_ret:
                re_ret = re_ret[0]
                self.annotates_dict[re_ret[0]] = re_ret[1]

    def join(self, item):
        ret = ""
        for i in item:
            if not isinstance(i, str):
                return repr(item)
            ret += i.strip() + '\n'

        ret = ret.strip()
        return ret


class Keyword:
    def __init__(self, input_data, slash):
        # accept input_data
        # opt
        # opt = calcfc
        # opt = (calcfc, ts)
        # opt(calcfc,ts)
        # opt(calcfc)

        input_data = input_data.strip()
        self.keyword = ""
        self.option = []

        self.origin_input = input_data

        # identify method
        if slash:
            self.keyword = "level"
            slash_pos = input_data.index('/')
            self.option = [re.sub(" ", "", input_data[:slash_pos]), re.sub(" ", "", input_data[slash_pos + 1:])]
            # remove the R or U identifiers from method
            if self.option[0][0] in ['R', 'r', 'U', 'u']:
                self.option[0] = self.option[0][1:]

        else:
            if '=' not in input_data and '(' not in input_data:
                self.keyword = input_data
                self.option = []
            elif 'iop' in input_data:
                equal_pos = input_data.index('=')
                self.keyword = input_data[:equal_pos]
                self.option = [input_data[equal_pos + 1:]]
            else:

                for i, character in enumerate(input_data):
                    if character == "=" or character == '(':
                        self.keyword = input_data[:i].strip()
                        break

                input_data = input_data[len(self.keyword):]
                if input_data.startswith("=("):
                    input_data = input_data[2:-1]
                elif input_data.startswith('('):
                    input_data = input_data[1:-1]
                elif input_data.startswith("="):
                    input_data = input_data[1:]

                parenthesis = 0
                current_word = ""
                for i, character in enumerate(input_data):

                    current_word += character
                    if character == ')':
                        parenthesis -= 1
                    elif character == '(':
                        parenthesis += 1
                    if parenthesis != 0:
                        continue

                    if character == ',' and parenthesis == 0:
                        self.option.append(current_word[:-1])
                        current_word = ""
                self.option.append(current_word)

        self.keyword = self.keyword.lower()
        if self.keyword != 'external':
            self.option = [x.lower() for x in self.option]


def open_with_gview(filename):
    """
    Use Gview to view a gjf file or a Coordinate object
    :param filename: a filename for a gjf file or a Coordinate object
    :return:
    """
    import subprocess
    gview_exe = r"C:\g16w\gview.exe"
    if os.path.isfile(gview_exe):
        print("Opening File with GView")
        if isinstance(filename, Coordinates):
            filename = filename.gjf_file()
        subprocess.Popen([gview_exe, filename])
        # print("GView opening Finished")
    else:
        print("GView Not Found. Opening of file", filename, "aborted.")


class Route_dict(collections.OrderedDict):
    def __init__(self, route_input: str, other_paragraph="", remove_genchk=True):
        """

        :param route_input:
        :param other_paragraph:
        :param remove_genchk:  产生输入的时候会自动去掉genchk，用于读取输出时应将此项设为False
        :return:
        """
        super(self.__class__, self).__init__()

        self.origin_route_input = route_input

        if isinstance(route_input, Route_dict):  # 用于copy.deepcopy的复制
            for key, value in route_input.items():
                self[key] = value
                self.other_paragraph = route_input.other_paragraph

        else:
            route_input = route_input.replace('\n', ' ').strip()
            other_paragraph = other_paragraph.strip()

            if route_input.startswith('#'):
                route_input = route_input[2:].strip(' ')  # remove #p or #
            parenthesis = 0
            current_word = ''
            slash = False  # identify method/basis

            for i, chr in enumerate(route_input):
                current_word += chr

                if chr == ')':
                    parenthesis -= 1
                elif chr == '(':
                    parenthesis += 1
                if parenthesis != 0:
                    continue
                if parenthesis == 0 and chr == '/':
                    slash = True
                if (chr == ' ' or chr == "\n" or i == len(route_input) - 1) and parenthesis == 0:
                    keyword = Keyword(current_word, slash)
                    self.add_item(keyword.keyword, keyword.option)
                    current_word = ""
                    slash = False

            # 'genchk'和'connectivity'用来防止由GV产生的gjf文件默认带有geom=allchk，其会自动加上genchk，但我们自己永远不会自己写genchk
            self.from_gaussview = False
            if 'connectivity' in list(safe_get_dict_value(self, 'geom')):
                self["geom"].remove('connectivity')
                self.from_gaussview = True

            if 'genchk' in self:
                if remove_genchk:
                    self.pop('genchk')
                self.from_gaussview = True

            if self.from_gaussview and 'allcheck' in safe_get_dict_value(self, 'geom'):
                self['geom'].remove('allcheck')

            if 'geom' in self and (not self['geom']):
                self.pop('geom')

            if 'level' not in self:
                self['level'] = ['Blank_Method', 'Black_Basis']

            # scrf = (smd, dovacuum) 没用，必须重写一个不带scrf的
            if 'dovacuum' in safe_get_dict_value(self, 'scrf'):
                self['scrf'].remove('dovacuum')

            try:
                self.pop("sp")
                self.pop("test")
            except:
                pass

            self.other_paragraph = other_paragraph

    def get_keyword(self, keyword):
        if keyword in self:
            return self[keyword]
        else:
            return []

    def option_exist(self, keyword, option):
        # eg for opt=tight
        # check whether option tight is in opt
        # return false if opt not exist
        # return false if opt=() without tight

        if keyword not in self:
            return False

        return option in self[keyword]

    def add_item(self, key, option):
        key = key.strip().lower()

        if not key:  # key 为空
            return None

        # combine new list with exist list, which is a value of a key in database
        if key == "level":
            self[key] = option
        elif key in self:
            if isinstance(option, str):
                option = [option]
            self[key] = list(set(self[key] + option))
        else:
            if isinstance(option, str):
                option = [option]
            self[key] = list(set(option))

    def remove_item(self, key, option):
        if key in self:
            if isinstance(option, str):
                if option in self[key]:
                    self[key].remove(option)
            if isinstance(option, list):
                for item in option:
                    if item in self[key]:
                        self[key].remove(item)
            if self[key] == [] and key not in ['opt', 'freq', 'scan', 'irc']:
                self.pop(key)

    def remove_key(self, key):
        if key in self:
            self.pop(key)

    def add_and_remove_of_dict(self, key, option, button):

        # combine new list with exist list, which is a value of a key in database
        # true for add, false for remove
        # do not pass key='method' in this

        bool = button.isChecked()

        if bool:  # to add
            self.add_item_to_dict(key, option)
        else:  # to remove
            self.remove_item_from_dict(key, option)

    def print_value(self, value):  # get a output like "(calcfc,ts)" in opt=(calcfc,ts)
        ret = ''
        value = remove_blank(value)
        if value:
            ret += '='
            if len(value) > 1:
                ret += '('
                for i, item in enumerate(value):
                    ret += item
                    if i == len(value) - 1:
                        ret += ')'
                    else:
                        ret += ','
            else:
                ret += value[0]
        return ret

    def __str__(self):
        ret = ""
        if 'level' in self:
            if self['level'] != ['Blank_Method', 'Black_Basis']:
                ret = self['level'][0] + '/' + self['level'][1] + '\n'

        for key, value in self.items():
            if key != 'level' and key != 'iop':
                ret += key
                ret += self.print_value(value)
                ret += '\n'
            elif key == 'iop':
                ret += 'IOp(' + self.print_value(value).lstrip('=').lstrip('(').rstrip(')') + ')' + '\n'

        return ret


class Gaussian_summary:
    def __init__(self, summary_lines: list):
        """
        Return a formatted result for Gaussian summary at the end of each gaussian job

        :param input: 用于接受别的输入

        :return: Something like:
        --------------------------------------------------------------
        [[['1', '1', 'GINC-LIYUANHE-UBUNTU', 'FOpt', 'RB3LYP', '6-31+G(d,p)', 'C19H25N3', 'GAUUSER',
        '12-Oct-2015', '0'], ['#p b3lyp/6-31+g(d,p) opt freq empiricaldispersion=gd3bj'], ['Me2_23_Prod'],
         ['0,1', 'C,0.4732857814,-1.2321049177,-0.7321261073', ............],
          ['Version=ES64L-G09RevD.01', 'State=1-A',
          'HF=-903.4719536', 'RMSD=5.576e-09', 'RMSF=7.235e-06', 'Dipole=-0.4275145,1.6510886,-0.6033573',
           'Quadrupole=8.9935258,-6.1414828,-2.8520429,-2.2448542,-0.1231134,2.7069082', 'PG=C01 [X(C19H25N3)]'],
            ['@']], [['1', '1', 'GINC-LIYUANHE-UBUNTU', 'Freq', 'RB3LYP', '6-31+G(d,p)', 'C19H25N3', 'GAUUSER',
             '12-Oct-2015', '0']...........]............]
        --------------------------------------------------------------
        """

        summary = ""

        for summary_line in summary_lines:
            summary_line = summary_line.strip('\n')
            summary += summary_line[1:] if summary_line[0] == " " else summary_line

        if r'1\1' in summary:  # windows
            summary = summary.replace('\\', '|')
        summary = summary.split('||')
        summary = [x.split("|") for x in summary]

        self.summary = summary

        self.basic_information = self.summary[0]
        self.route = self.summary[1][0]
        self.route_dict = Route_dict(self.route)

        self.name = self.summary[2]
        self.charge, self.multiplet = self.summary[3][0].split(',')

        self.coordinate = Coordinates(self.summary[3][1:], self.charge, self.multiplet)

        self.results = self.summary[4]
        self.results = {x.split("=")[0]: x.split("=")[1] for x in self.results}
        # contains "HF", "ZeroPoint","Thermal","NImag"



class Gaussian_output:
    def __init__(self, output, filename=""):
        if isinstance(output, str) and file_type(output) == Filetype.gaussian_output:
            with open(output, encoding='utf-8') as file:
                self.lines = file.readlines()
            if not filename:
                filename = output

        elif isinstance(output, list):  # input as list
            self.lines = output

        else:
            raise MyException('Not valid file')

        self.filename = filename

        # verify that "#p" was written in route, otherwise Gaussian_output cannot read that
        hash_p_found = False
        for line in self.lines:
            if '#p' in line or "#P" in line:
                hash_p_found = True
                break
        if not hash_p_found:
            raise MyException('Output Files without #P in route are not supported.')

        self.lines = [x.replace('---', '-----------------------------------') for x in self.lines]

        self.steps_list = split_list(self.lines, lambda x: ("l1.exe" in x))
        self.steps = [Gaussian_output_step(x, self.filename) for x in self.steps_list if self.steps_list]

        for count, step in enumerate(self.steps):
            # 确定其是不是前一步溶剂化的的单点
            if count > 0:
                if (not step.has_freq) and (not step.is_opt):
                    last_step = self.steps[count - 1]
                    if last_step.is_solvated:
                        if (last_step.basis_counting == step.basis_counting) or (last_step.basis == step.basis):
                            if last_step.method.upper().lstrip('R').lstrip('U') == step.method.upper().lstrip('R').lstrip('U'):
                                step.is_gas_sp_of_previous_sol_step = True

            if count == len(self.steps) - 1:
                break
            if self.steps[count + 1].is_freq_step_after_opt and self.steps[count + 1].normal_termination:
                step.is_opt_step_before_freq = True

        # for step in self.steps:
        #     print(step.is_opt_step_before_freq,step.is_freq_step_after_opt)

        self.remove_empty_head()

        self.last_opt_pos = [x for x in range(len(self.steps)) if "opt" in self.steps[x].route_dict or "irc" in self.steps[x].route_dict]
        if self.last_opt_pos:
            self.last_opt_pos = self.last_opt_pos[-1]
        else:
            self.last_opt_pos = -1

        for count, step in enumerate(self.steps):
            if count - 1 >= 0:
                if step.mixed_basis_str == 'chk':
                    step.mixed_basis_str = self.steps[count - 1].mixed_basis_str

        # 读取输出的标题（其中含有“[EXTRACT_GEOM]”部分）
        self.title = ""
        self.extract_geoms = []
        for link in self.steps[0].links:
            if link.num == 101:
                for count, line in enumerate(link.lines):
                    if "Symbolic Z-matrix:" in line or "Structure from the checkpoint file" in line:
                        # [1:]是除去每行开头的空格
                        title1 = [x[1:] for x in link.lines[1:count]]  # 高斯有时会把标题写在Structure from the checkpoint file前面
                        title2 = [x[1:] for x in link.lines[count + 1:]]  # 有时会写在后面
                        if True in ['-----' in x for x in title1]:  # 检测有无'--------'行，但这个行会随着标题长度而改变，故而仅支持超过5字符的
                            title = title1
                        else:
                            title = title2
                        self.title = ''.join(split_list(title, lambda x: '-----' in x)[0])
                        break
            if self.title:
                break

        self.title = self.title.replace('\n', "")
        # print("Title:",self.title)
        re_ret = re.findall(r'\[EXTRACT_GEOM\]\:(.+)', self.title)
        if re_ret:
            re_ret = re_ret[0]
            self.extract_geoms = re_ret.split(',')
        self.extract_geoms = [(int(x) - 1 if is_int(x) else x) for x in self.extract_geoms]

        self.frozen_bonds = []
        re_ret = re.findall(r"\[FROZEN\_BONDS\]\:(.+)\[\/FROZEN\_BONDS\]", self.title)
        if re_ret:
            re_ret = re_ret[0]
            self.frozen_bonds = eval(re_ret)

        # 按照相同的结构分成几个部分
        # self.step_groups_by_structure is a list of (list of steps), each step in the same list should have the same structure in l9999
        # only completed steps was included
        # IRC not included
        self.step_groups_by_structure = []
        self.extract_groups = []  # 需要提取的group的编号
        for count, step in enumerate(self.steps):
            if 'irc' in step.route_dict:
                continue

            true_count = count  # 排除opt+freq生成的多余步数
            for i, previous_step in enumerate(self.steps[:count]):
                if 'opt' in previous_step.route_dict and 'freq' in previous_step.route_dict:
                    true_count -= 1

            if step.normal_termination:
                for step_group in self.step_groups_by_structure:
                    if step.summary:
                        if step.summary.coordinate == step_group[0].summary.coordinate:
                            step_group.append(step)

                            # 看这一组的构象要不要提取
                            # 用数字标注
                            if true_count in self.extract_geoms:
                                self.extract_groups.append(self.step_groups_by_structure.index(step_group))
                            # 用Fchk_Tag标注
                            elif any([("[" + x + "]" in step.chk_filename) for x in self.extract_geoms if isinstance(x, str)]):
                                self.extract_groups.append(self.step_groups_by_structure.index(step_group))
                            break
                else:
                    if step.summary:
                        self.step_groups_by_structure.append([step])
                        # 看这一组的构象要不要提取
                        if true_count in self.extract_geoms:
                            self.extract_groups.append(len(self.step_groups_by_structure) - 1)
                        elif True in [("[" + x + "]" in step.chk_filename) for x in self.extract_geoms if isinstance(x, str)]:
                            self.extract_groups.append(len(self.step_groups_by_structure) - 1)

        # 如果没规定，全收
        if not self.extract_geoms:
            self.extract_groups = list(range(len(self.step_groups_by_structure)))

        # print("File\t\t\t\t:",self.filename)
        # print("Defined extract geometry\t:",self.extract_geoms)

        # 包含需要提取的各步骤
        # self.step_groups_require_extract = [self.step_groups_by_structure[count] for count in self.extract_groups]

        self.get_solvation_energy()
        self.get_group_coordinate()

        self.normal_terminated = False not in [step.normal_termination for step in self.steps]

        pass

    def get_group_coordinate(self):
        # 提取组内每一step的坐标，并检查是不是唯一的
        self.coordinate_of_groups = [None for _ in self.step_groups_by_structure]
        self.geom_hash_of_groups = [-1 for _ in self.step_groups_by_structure]

        for group_count, group in enumerate(self.step_groups_by_structure):
            if group_count not in self.extract_groups:  # 不需要提取就滚蛋
                continue

            # 提取组内每一step的坐标，并检查是不是唯一的
            coordinate = [step.summary.coordinate for step in group if step.summary]
            if not coordinate:
                continue

            for count in range(len(coordinate) - 1, 0, -1):
                if coordinate[count] == coordinate[count - 1]:
                    coordinate.pop(count)
            assert len(coordinate) == 1, "group_coordinate not singular"
            self.coordinate_of_groups[group_count] = coordinate[0]

            # 提取geom_hash
            geom_hash = [hash(step.summary.coordinate) for step in group]
            if len(list(set(geom_hash))) != 1:
                print("Warning! Group_coordinate_hash not singular.\nHowever it doesn't necessarily means different stucture.")
            self.geom_hash_of_groups[group_count] = geom_hash[0]

    def remove_empty_head(self):

        # 排除Linux下调用的第一个文件头
        # if self.steps:
        #     non_blank_links = [x for x in self.steps[0].links if x.num!=-1]
        #     if not non_blank_links:
        #         self.steps.pop(0)

        # 排除一个link都没有的情况（Linux下调用，文件有初始指令头）
        pop = []
        for count, step in enumerate(self.steps):
            non_blank_links = [x for x in step.links if x.num != -1]
            if not non_blank_links:
                pop.append(count)
        for count in reversed(pop):
            self.steps.pop(count)

    def get_solvation_energy(self):
        """

        :return: solvation energy (△HF) in kJ/mol
        """

        # 每组structure有一个solvation
        self.solvation_energy = [0 for _ in self.step_groups_by_structure]
        self.solvation_level = ["" for _ in self.step_groups_by_structure]
        self.solvent = ["" for _ in self.step_groups_by_structure]
        self.solvation_steps = [[] for _ in self.step_groups_by_structure]  # 直接存储step对象

        for count, step_group in enumerate(self.step_groups_by_structure):
            if len(step_group) >= 2:
                for step_count, step1 in enumerate(step_group):
                    for step2 in step_group[:step_count]:
                        if step1.summary and step2.summary:  # 确认已经算完了

                            # verify that some 2 routes' only difference is the scrf command
                            route1 = Route_dict(step1.route_dict.origin_route_input)
                            route2 = Route_dict(step2.route_dict.origin_route_input)

                            for route in [route1, route2]:
                                remove_key_from_dict(route, 'geom')
                                remove_key_from_dict(route, 'sp')
                                remove_key_from_dict(route, 'guess')

                            keys_to_remove = []

                            for key in route1:
                                if key != 'scrf' and key in route2:
                                    if set(route1[key]) == set(route2[key]):
                                        keys_to_remove.append(key)

                            for key in keys_to_remove:
                                remove_key_from_dict(route1, key)
                                remove_key_from_dict(route2, key)

                            if list(route1.keys()) == ['scrf'] and 'smd' in route1['scrf'] and list(route2.keys()) == []:

                                self.solvation_steps[count] = [step1, step2]

                                HF_sol = float(step1.summary.results['HF'])
                                HF_gas = float(step2.summary.results['HF'])
                                self.solvation_energy[count] = (HF_sol - HF_gas) * 2625.49962

                                level = step1.summary.route_dict['level']
                                if level[1] == "genecp" or level[1] == "gen":
                                    if len(step1.mixed_basis_str) > 100:
                                        level = (level[0] + '/' + step1.mixed_basis_str[:17] + '......]').upper()
                                    else:
                                        level = (level[0] + '/' + step1.mixed_basis_str).upper()
                                else:
                                    level = (level[0] + '/' + level[1]).upper()

                                self.solvation_level[count] = level

                                self.solvent[count] = "NOT FOUND"
                                for scrf_setup in step1.summary.route_dict['scrf']:
                                    match = re.findall(r'solvent\s*\=\s*(.+)', scrf_setup)
                                    if match:
                                        self.solvent[count] = match[0]
                                        # 第一个字母大写
                                        self.solvent[count] = self.solvent[count][0].upper() + self.solvent[count][1:].lower()

            #                 return None
            #
            # self.solvation_energy = 0
            # self.solvation_level = ""
            # return None


class Gaussian_output_step:
    def __init__(self, step_list, original_filename=""):

        self.links = []
        self.route = ""
        self.route_dict = {}

        self.original_filename = original_filename

        self.has_freq = False
        self.is_freq_step_after_opt = False  # 是单独计算的freq还是opt freq的第二步
        self.is_opt_step_before_freq = False  # 是freq前面的opt步骤，这样opt步骤的单点就不用取了
        self.is_gas_sp_of_previous_sol_step = False  # 是液相优化之后跑的一步气相

        self.harmonic_freqs = []
        self.G_correction = 0
        self.H_correction = 0
        self.S = 0
        self.G = 0
        self.H = 0
        self.imaginary_count = 0
        self.has_imaginary_freq = False
        self.is_IRC = False
        self.IRC_coords = []

        self.lines = step_list
        self.split_by_link()

        try:
            # print([x.leave_time for x in self.links])
            self.last_leave_time = max([x.leave_time for x in self.links])
            self.last_leave_time = Chronyk(datetime.strptime(self.last_leave_time, "%a %b %d %H:%M:%S %Y"))
            # print(self.last_leave_time.timestring("%Y-%m-%d",timezone = 8*3600))
        except:
            self.last_leave_time = ""
            pass

        self.normal_termination = 'Normal termination' in "".join(self.links[-1].lines)
        self.error_termination = 'Error termination' in "".join(self.links[-1].lines)
        self.summary = self.find_summary()
        self.get_last_coords()
        self.get_routes()

        self.method = ""
        self.basis = ""
        self.mixed_basis_str = ""
        self.mixed_basis_list = []
        self.basis_counting = random.random()
        self.get_level()

        self.opt_energies = []
        self.get_opt_energies()

        self.converged = [[], [], [], []]
        self.get_converged()

        self.last_scf_iteration = []
        for link in reversed(self.links):
            if link.scf_iteration:
                self.last_scf_iteration = link.scf_iteration
                break

        if 'irc' in self.route_dict:
            self.is_IRC = True
            self.get_irc_coords()

        self.frozen_bonds = []  # a list of 2-tuples represents the frozen bonds
        self.get_freeze()

        self.get_freq_result()

        self.chk_filename = ""
        for count, line in enumerate(self.lines):
            re_ret = re.findall(r"\%chk\=(.+)", line)
            if re_ret:
                self.chk_filename = re_ret[0]
                while '.chk' not in self.chk_filename:  # Gaussian一行显示不完chk的文件名
                    count += 1
                    self.chk_filename += self.lines[count][1:].strip('\n')
                break

        # Warning, this method only applies to single determinant methods.
        self.is_relaxed_scan = False
        self.converged_relaxed_scan_structures = []
        import collections
        self.converged_relaxed_scan_structures_dict = collections.OrderedDict()  # dict, key is the structure, value is a tuple of step and energy
        self.get_relaxed_scan()

        self.is_solvated = False  # SCF能量里带没带溶剂化
        self.solvent = ""
        self.read_step_solvent()

        self.is_opt = 'opt' in self.route_dict

    def read_step_solvent(self):
        for count, link in enumerate(self.links):
            if link.num == 301:
                # print(link.lines)
                for count, line in enumerate(link.lines):
                    if "Polarizable Continuum Model (PCM)" in line:
                        for line2 in link.lines[count:]:
                            # print(line2)
                            re_ret = re.findall(r"Solvent\s+:\s*(.+?),", line2)
                            if re_ret:
                                self.is_solvated = True
                                self.solvent = re_ret[0]
                                break
                        break
        # print(self.solvent)

    def get_irc_coords(self):
        for count, link in enumerate(self.links):
            for line in link.lines:
                if "Calculating another point on the path." in line:
                    for coord_link in reversed(self.links[:count]):
                        if coord_link.coords:
                            assert len(coord_link.coords) == 1, 'No. of IRC coords not 1'
                            self.IRC_coords.append(coord_link.coords[0])
                            break
                    break

        # for i in self.IRC_coords:
        #     print(i)
        # print('---------------',len(self.IRC_coords))

    def get_freq_result(self):
        if "freq" in self.route_dict and 'opt' not in self.route_dict:
            self.has_freq = True

            if 'genchk' in self.route_dict:
                self.is_freq_step_after_opt = True

            for link in reversed(self.links):
                if link.num == 716:

                    # get T, P, rotation symm
                    for count, line in enumerate(link.lines):
                        re_ret = re.findall(r"Temperature\s+(\d+\.\d+)\s+Kelvin. {2}Pressure\s+(\d+\.\d+)\s+Atm.", line)
                        if re_ret:
                            self.temp, self.pressure = [float(x) for x in re_ret[0]]

                        re_ret = re.findall(r'Rotational symmetry number\s+(\d+)\.', line)
                        if re_ret:
                            self.rotation_symm_number = int(re_ret[0][0])

                        re_ret = re.findall(r'Rotational constants \(GHZ\)\:\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)', line)
                        if re_ret:
                            self.rotation_constants = [float(x) for x in re_ret[0]]

                        re_ret = re.findall(r'Rotational constant \(GHZ\)\:\s+(-*\d+\.\d+)', line)
                        if re_ret:
                            self.rotation_constants = [float(x) for x in re_ret] * 3

                        # Gaussian 有时会因为惯性矩太大而显示为*****************，所以不用它了
                        # if 'Principal axes and moments of inertia in atomic units' in line:
                        #     re_ret=re.findall(r'Eigenvalues --\s+(\d+\.\d{5})\s*(\d+\.\d{5})\s*(\d+\.\d+)',
                        #                       link.lines[count+2])
                        #     self.moment_of_inertia = [float(x) for x in re_ret[0]]

                    # moment_of_inertia in SI
                    if hasattr(self, 'rotation_constants'):
                        self.moment_of_inertia = [h / (B * 1E9) / 8 / pi ** 2 for B in self.rotation_constants]
                    else:
                        self.moment_of_inertia = [0, 0, 0]

                    # get isotopes
                    # "Atom     1 has atomic number  6 and mass  12.00000"
                    self.isotopes = []
                    for line in link.lines:
                        re_ret = re.findall(r"Atom\s+\d+\s+has atomic number\s+\d+\s+and mass\s+(\d+\.\d+)", line)
                        if re_ret:
                            self.isotopes.append(float(re_ret[0]))

                    # get frequencies
                    self.harmonic_freqs = []
                    freq_reg = r"Frequencies\s+--\s+(-*\d+\.\d+)\s*(-*\d+\.\d+)*\s*(-*\d+\.\d+)*"
                    for line in link.lines:
                        match = re.findall(freq_reg, line)
                        if match:
                            self.harmonic_freqs += match[0]

                    self.harmonic_freqs = [float(x) for x in remove_blank(self.harmonic_freqs)]

                    # get corrections
                    zero_point_corr_reg = r'Zero-point correction\=\s+(\-*\d+\.\d+)'
                    enthalpy_corr_reg = r"Thermal correction to Enthalpy\=\s+(\-*\d+\.\d+)"
                    gibbs_corr_reg = r"Thermal correction to Gibbs Free Energy\=\s+(\-*\d+\.\d+)"
                    enthalpy_reg = r"Sum of electronic and thermal Enthalpies\=\s+(\-*\d+\.\d+)"
                    gibbs_reg = r"Sum of electronic and thermal Free Energies\=\s+(\-*\d+\.\d+)"

                    for line in link.lines:
                        match = re.findall(zero_point_corr_reg, line)
                        if match:
                            self.zero_point_correction = float(match[0]) * 2625.49962

                        match = re.findall(enthalpy_corr_reg, line)
                        if match:
                            self.H_correction = float(match[0]) * 2625.49962

                        match = re.findall(gibbs_corr_reg, line)
                        if match:
                            self.G_correction = float(match[0]) * 2625.49962

                        match = re.findall(enthalpy_reg, line)
                        if match:
                            self.H = float(match[0]) * 2625.49962

                        match = re.findall(gibbs_reg, line)
                        if match:
                            self.G = float(match[0]) * 2625.49962

                    entropy_lead_reg = r"\s+E\s+\(Thermal\)\s+CV\s+S"
                    entropy_reg = r'Total\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)'

                    for i, line in enumerate(link.lines):
                        if re.findall(entropy_lead_reg, line):
                            match = re.findall(entropy_reg, link.lines[i + 2])
                            if match:
                                self.S = float(match[0][2]) * 4.184
                            break
                    break

            self.imaginaries = [x for x in self.harmonic_freqs if x < 0]
            self.imaginary_count = len(self.imaginaries)

            if self.imaginary_count != 0:
                self.has_imaginary_freq = True

    def split_by_link(self):
        self.links = []
        # elements are something like [502,["Output","Lines", 'of','L502.exe']]
        # the resting (if not terminated), is -1
        # "Normal termination" is 9999

        leave_time = ""

        current_link = []
        for line in self.lines:
            if "Leave Link" not in line:
                match = []
            else:
                match = re.findall(r' Leave Link +(\d+) at ([A-Za-z]{3} [A-Za-z]{3} +\d+ \d{2}:\d{2}:\d{2} \d{4}).+cpu\:\s+(\d+\.\d+)', line)
            if match and match[0][0].isnumeric() and current_link:
                match = match[0]
                link_num = int(match[0])
                leave_time = match[1]
                cpu_time = match[2]
                self.links.append(Gaussian_output_link(link_num, current_link, leave_time, cpu_time))
                # if match and match[1]:
                #     self.last_leave_time = max(self.last_leave_time,match[1])
                current_link = []
            else:
                current_link.append(line)

        if current_link:  # 最后剩下的，有时不输出leave link 9999
            if "Enter" in current_link[0]:
                re_ret = re.findall(r"\(Enter .+l(\d{1,4}).exe\)", current_link[0])
            else:
                re_ret = []

            if re_ret and re_ret[0].isnumeric():
                self.links.append(Gaussian_output_link(int(re_ret[0]), current_link, leave_time))
            else:
                self.links.append(Gaussian_output_link(-1, current_link, leave_time))

    def find_summary(self):

        l9999_link = [link for link in self.links if link.num == 9999]

        if len(l9999_link) != 1:
            if "Entering Gaussian System" not in self.lines[0]:
                # print("No summary find")
                return ""
        else:
            ret = ""
            sum_lines = l9999_link[0].lines
            for count, line in enumerate(sum_lines):
                if r'1\1' in line or '1|1' in line:
                    summary = []
                    for summary_line in sum_lines[count:]:
                        summary.append(summary_line)
                        if "@" in summary_line:
                            break
                    if summary:
                        ret = Gaussian_summary(summary)
                        l9999_link[0].coords.append(ret.coordinate)
                        return ret

    def get_last_coords(self):

        self.last_coord = Coordinates()

        self.all_coords = sum([link.coords for link in self.links if link.coords], [])
        if self.all_coords:
            self.last_coord = self.all_coords[-1]

        correct_c_and_m = [coord for coord in self.all_coords if coord.charge != 999]
        if correct_c_and_m:
            self.last_coord.charge = correct_c_and_m[-1].charge
            self.last_coord.multiplicity = correct_c_and_m[-1].multiplicity

    def get_freeze(self):

        if 'opt' in self.route_dict and 'modredundant' in self.route_dict['opt']:
            for link in self.links:
                if link.num == 101:
                    for count, line in enumerate(link.lines):
                        if 'The following ModRedundant input section has been read:' in line:
                            for line2 in link.lines[count + 1:]:
                                re_ret = re.findall(r"B\s+(\d+)\s+(\d+)\s+F", line2)
                                if re_ret:
                                    self.frozen_bonds.append([int(x) - 1 for x in re_ret[0]])
                                else:
                                    break
                            break

    def get_routes(self):

        self.route = ""
        for link in self.links:
            if link.num == 1:

                for i, line in enumerate(link.lines):
                    match = re.findall(r'^\ #', line)
                    if match:
                        for route_line in link.lines[i:]:
                            if '---' in route_line:
                                break
                            if route_line[0] != " ":
                                print("Route Line Process Error!")
                            self.route += route_line[1:].strip('\n')
                break
        self.route_dict = Route_dict(self.route, remove_genchk=False)
        pass

    def get_level(self):
        basis_count = -1
        primitive_count = -1
        cartesian_count = -1
        alpha_count = -1
        beta_count = -1

        if "level" in self.route_dict and ("genecp" in self.route_dict['level'] or 'gen' in self.route_dict['level']):
            self.mixed_basis_list = []
            for link in self.links:
                if link.num == 301:
                    basis_list = []
                    current_basis = []
                    for i, line in enumerate(link.lines):

                        if "Basis read from chk" in line:
                            self.mixed_basis_list.append("Check")
                            break

                        if "General basis read from cards:" in line:  # 从这一行开始读
                            for basis_line in link.lines[i + 1:]:
                                if "Ernie" in basis_line:
                                    break
                                if "****" in basis_line:
                                    current_basis.append(basis_line)
                                    basis_list.append(current_basis)
                                    current_basis = []
                                else:
                                    current_basis.append(basis_line)
                            break

                        #   394 basis functions,   659 primitive gaussians,   407 cartesian basis functions
                        #   55 alpha electrons       55 beta electrons

                    for basis in basis_list:
                        if "****" in basis[-1]:
                            for basis_line in basis:
                                if "Centers:" not in basis_line and "****" not in basis_line:
                                    self.mixed_basis_list.append(basis_line.strip())

                    if -1 in (basis_count, primitive_count, cartesian_count, alpha_count, beta_count):
                        for i, line in enumerate(link.lines):
                            re_ret = re.findall(r'''(\d+)\s*basis functions,\s*(\d+)\s*primitive gaussians,\s*(\d+)\s*cartesian basis functions''', line)
                            if re_ret:
                                (basis_count, primitive_count, cartesian_count) = [int(x) for x in re_ret[0]]
                                re_ret = re.findall(r'''(\d+)\s*alpha electrons\s*(\d+)\s*beta electrons''', link.lines[i + 1])
                                (alpha_count, beta_count) = [int(x) for x in re_ret[0]]
                                break

            if self.mixed_basis_list:
                self.mixed_basis_str = '[' + ' + '.join(self.mixed_basis_list) + ']'
            else:
                self.mixed_basis_str = ""
                self.mixed_basis_list = []

            self.method = self.route_dict['level'][0]
            self.basis = self.mixed_basis_list
            if len(self.basis) == 1:
                self.basis = self.basis[0]
            else:
                self.basis = str(self.basis)
        else:
            self.method, self.basis = self.route_dict['level']

        if -1 not in (basis_count, primitive_count, cartesian_count, alpha_count, beta_count):
            self.basis_counting = (basis_count, primitive_count, cartesian_count, alpha_count, beta_count)

    def get_opt_energies(self):
        for link in self.links:
            if link.num == 502:
                re_pattern = r"SCF Done:  E\([0-9A-Za-z\-]+\) =\s+(-*\d+\.\d+E*-*\d*)"
                for line in link.lines:
                    match = re.findall(re_pattern, line)
                    if len(match) > 0:
                        self.opt_energies.append(''.join(match[0]))
                        break

        if not self.opt_energies:
            # External 读入能量的时候在L402
            for link in self.links:
                if link.num == 402:
                    re_pattern = r"Energy\s*=\s+(-*\d+\.\d+)"
                    for line in link.lines:
                        match = re.findall(re_pattern, line)
                        if len(match) > 0:
                            self.opt_energies.append(''.join(match[0]))
                            break

    def get_relaxed_scan(self):

        # Warning, this method only applies to single determinant methods.
        last_step = 0
        if 'opt' in self.route_dict and 'modredundant' in self.route_dict['opt']:
            for link in self.links:
                if link.num == 103:
                    for count, line in enumerate(link.lines):
                        if "Number of optimizations in scan" in line:
                            self.is_relaxed_scan = True
                            for count, link103 in enumerate(self.links):
                                if link103.num == 103:
                                    if any(['Optimization completed.' in x for x in link103.lines]):
                                        relaxed_scan_step = None
                                        relaxed_scan_energy = None
                                        relaxed_scan_structure = None

                                        relaxed_scan_step = self.links.index(link103)
                                        links_range_to_find_202 = self.links[last_step + 1:count]
                                        last_step = relaxed_scan_step
                                        for link202 in reversed(links_range_to_find_202):
                                            if link202.num == 202:
                                                relaxed_scan_structure = link202.coords[-1]
                                                self.converged_relaxed_scan_structures.append(relaxed_scan_structure)
                                                break
                                        for link502 in reversed(links_range_to_find_202):
                                            if link502.num == 502:
                                                if hasattr(link502, 'scf_final_energy'):
                                                    relaxed_scan_energy = link502.scf_final_energy
                                                    break
                                        if relaxed_scan_step and relaxed_scan_structure and relaxed_scan_energy:
                                            self.converged_relaxed_scan_structures_dict[relaxed_scan_structure] = (relaxed_scan_step, relaxed_scan_energy)

                            break
                    break

    def get_converged(self):
        for link in self.links:
            if link.num == 103:
                for count, line in enumerate(link.lines):
                    match = re.findall(r"Converged\?", line)
                    if match:
                        for k, line in enumerate(link.lines[count + 1:count + 5]):
                            re_ret = re.findall(r" \d\.\d+", line)
                            if len(re_ret) == 2:
                                value = float(re_ret[0]) / float(re_ret[1])
                                if value < 0.01:
                                    value = 0.01  # 防止log时出现负无穷
                                self.converged[k].append(value)

                            # 数值超过9.999999时会将其显示为*******
                            else:
                                re_ret = re.findall(r" \*{4,}\s+(\d\.\d+)", line)
                                if re_ret:
                                    self.converged[k].append(10 / float(re_ret[0]))
                                else:
                                    print("Converged Energy Finding Error.")

                # 从[[...],[...],[...],[...]] 换成 [[ , , , ]...]
        self.converged = [[self.converged[x][step] for x in range(4)] for step in range(len(self.converged[0]))]


class Gaussian_output_link:
    def __init__(self, link_num, link_lines, leave_time=0, cpu_time=-1):
        self.num = link_num
        self.lines = link_lines
        self.coords = []
        self.get_coords()

        self.leave_time = leave_time
        self.cpu_time = float(cpu_time)
        self.date_class = Date_Class(str(link_num), leave_time)

        self.scf_iteration = []
        self.get_scf_iteration()
        self.get_scf_final_result()

    def get_scf_final_result(self):
        pre_run_pattern = "SCF Done:"
        re_pattern = r"SCF Done:  E\([0-9A-Za-z\-]+\) =\s+(-*\d+\.\d+E*-*\d*)"
        for line in self.lines:
            if pre_run_pattern not in line:
                continue
            match = re.findall(re_pattern, line)
            if len(match) > 0:
                self.scf_final_energy = float(''.join(match[0]))
                break

    def get_scf_iteration(self):
        if self.num == 502:
            for i, line in reversed(list(enumerate(self.lines))):
                pre_run_pattern = "Cycle"
                if pre_run_pattern not in line:
                    continue
                match = re.findall("Cycle +([0-9]+) ", line)
                if match:
                    for j in range(1, 100):
                        if i + j >= len(self.lines):
                            break
                        if len(re.findall("Cycle +([0-9]+) ", self.lines[i + j])) > 0:
                            break

                        findEnergy = re.findall(r"E= (-\d+\.\d+)", self.lines[i + j])

                        if len(findEnergy) > 0:
                            energy = float(findEnergy[0])
                            self.scf_iteration.append(energy)
                            break

                    if int(match[0]) == 1:
                        break

        # QC
        if self.num == 508:
            for i, line in reversed(list(enumerate(self.lines))):
                pre_run_pattern = "Iteration"
                if pre_run_pattern not in line:
                    continue
                match = re.findall(r"Iteration +(\d+) +EE=", line)
                if match:
                    findEnergy = re.findall(r"Iteration +\d+ +EE= (-\d+\.\d+)", line)
                    if len(findEnergy) > 0:
                        energy = float(findEnergy[0])
                        self.scf_iteration.append(energy)

        self.scf_iteration.reverse()

    def get_coords(self):
        if self.num != 9999:

            marks = {"Symbolic Z-matrix:": 0,
                     "Input orientation:": 4,
                     "Standard orientation:": 4,
                     "Redundant internal coordinates found in file": 0,
                     "CURRENT STRUCTURE": 5}
            # key is a title mark
            # value is a number indicating how many lines should be skipped after the title
            #
            ###############################################################################
            # Symbolic Z-matrix:
            # Charge =  0 Multiplicity = 1
            # C                     0.42047  -1.18677  -0.70467

            # key is "Symbolic Z-matrix:", value should be 0
            ###############################################################################
            ###############################################################################
            #                           Input orientation:
            # ---------------------------------------------------------------------
            # Center     Atomic      Atomic             Coordinates (Angstroms)
            # Number     Number       Type             X           Y           Z
            # ---------------------------------------------------------------------
            #      1          6           0        0.420470   -1.186773   -0.704665

            # key is "Input orientation:" value should be 4
            ###############################################################################

            self.charge = 999
            self.multiplicity = 999

            for count, line in enumerate(self.lines):
                for mark in marks:
                    if mark in line:
                        coords = []
                        pre_run_pattern = "Charge"
                        if pre_run_pattern in self.lines[count + 1] + '\n' + self.lines[count - 1]:
                            re_result = re.findall(r'Charge = +(-*\d+) Multiplicity = +(\d+)',
                                                   self.lines[count + 1] + '\n' + self.lines[count - 1])  # 在某些标题的前面一行有，有的后面一行有
                            if re_result:
                                self.charge, self.multiplicity = re_result[0]  # 如果找到了，就使用新的；如果找不到，沿用上一个
                                count += 1

                        for coord_line in self.lines[count + marks[mark] + 1:]:
                            # print(link.num,mark)
                            # print(coord_line)
                            if std_coordinate(coord_line):  # 确认这一行中存在坐标
                                coords.append(coord_line)
                            else:
                                break
                        if coords:
                            self.coords.append(Coordinates(coords, self.charge, self.multiplicity))


def split_gaussian_output_file_steps(filename):
    # split Gaussian output file to each step (except opt+freq and vacuum-sol SMD calc. pair were viewed as single step)
    # return False if no split was required (already "single step")

    header = ''' Entering Gaussian System, Link 0=/home/xx/g09/g09
 Input=/home/xx.gjf
 Output=/home/xx.out
 Initial command:
 /home/xx/g09/l1.exe "/home/xx/g09/scratch/Gau-6482.inp" -scrdir="/home/xx/g09/scratch/"
 Entering Link 1 = /home/xx/g09/l1.exe PID=      6483.

 Copyright (c) 1988,1990,1992,1993,1995,1998,2003,2009,2013,
            Gaussian, Inc.  All Rights Reserved.

 This is part of the Gaussian(R) 09 program.  It is based on
 the Gaussian(R) 03 system (copyright 2003, Gaussian, Inc.),
 END OF MAN MADE HEADER

 ******************************************
 Gaussian 09:  ES64L-G09RevD.01 24-Apr-2013
                24-Dec-2015
 ******************************************
 '''

    if isinstance(filename, Gaussian_output):
        output_object = filename
        filename = output_object.filename
    elif os.path.isfile(filename):
        output_object = Gaussian_output(filename)
    else:
        return None

    if len(output_object.steps) == 1:
        return False

    if len(output_object.steps) == 2:
        if 'opt' in output_object.steps[0].route_dict and \
                'freq' in output_object.steps[0].route_dict:
            return False
        if output_object.solvation_energy:
            return False

    processed = []

    if output_object.solvation_energy:
        processed += output_object.solvation_steps
        solvation_gas = output_object.steps[output_object.solvation_steps[0]]
        solvation_sol = output_object.steps[output_object.solvation_steps[1]]

        ret = ""

        if output_object.solvation_steps[1] != 0:
            ret += header

        ret += "".join(solvation_sol.lines) + '\n'
        ret += "".join(solvation_gas.lines) + '\n'

        output_filename = filename_class(filename).only_remove_append + '__solvation.out'
        if not os.path.isfile(output_filename):
            with open(output_filename, 'w') as output_file:
                output_file.write(ret)
        # else:
        #     print("File",output_filename,"already exist!")

    for count, step in enumerate(output_object.steps):
        ret = ""
        if count in processed:
            continue

        if count != 0:
            ret += header

        ret += "".join(step.lines) + '\n'
        processed.append(count)
        if 'opt' in step.route_dict and 'freq' in step.route_dict and count + 1 < len(output_object.steps):
            ret += "".join(output_object.steps[count + 1].lines) + '\n'
            processed.append(count + 1)

        output_filename = filename_class(filename).only_remove_append + '__STEP_' + str(count + 1) + '.out'
        if not os.path.isfile(output_filename):
            with open(output_filename, 'w') as output_file:
                output_file.write(ret)
        else:
            print("File", output_filename, "already exist!")

    return True


def split_gaussian_file(filename, file_content=None, only_get_file_names=False, required_step=None):
    """
    split gaussian file by step
    :param filename:
    :param file_content: you can provide the file content as a list of lines,if you already have it, to save time
    :param only_get_file_names: 只要chk的名字,用于monitor的显示
    :return: a list of filename; if required_step is only one single number, return a str of filename

    Args:
        required_step:
    """

    if file_content is None:
        file_content = []
    if required_step is None:
        required_step = []

    if isinstance(required_step, int):
        only_one_step = True
        required_step = [required_step]
    else:
        only_one_step = False

    if file_content:
        output_lines = file_content
    else:
        with open(filename) as output_file_object:
            output_lines = output_file_object.readlines()

    output_steps = split_list(output_lines, lambda x: 'Normal termination of Gaussian ' in x,
                              include_separator_after=True)

    output_steps_process = []
    for step in output_steps:
        if output_steps_process and True in ['Proceeding to internal job step' in x for x in step[:20]]:
            output_steps_process[-1] = output_steps_process[-1] + step
        else:
            output_steps_process.append(step)

    output_steps = output_steps_process

    splitted_output_filenames = []

    for step_count, step in enumerate(output_steps):
        chkfile_filename = ""
        for count, line in enumerate(step):
            if "%chk" in line:
                chkfile_filename += (line.strip().lstrip('%chk='))
                for chk_lines in step[count + 1:]:
                    if '.chk' in chkfile_filename:
                        break
                    chkfile_filename += chk_lines.strip()
                break
        chkfile_filename = chkfile_filename.strip()
        if chkfile_filename:
            splitted_output_filenames.append(
                os.path.join(filename_class(filename).path, filename_class(chkfile_filename).name_stem + '.log'))
        else:
            splitted_output_filenames.append(filename_class(filename).only_remove_append + '[Split' + str(step_count) + '].log')

    splitted_file_names = []

    for step_count, step in enumerate(output_steps):
        if splitted_output_filenames.count(splitted_output_filenames[step_count]) > 1:
            output_filename = filename_class(splitted_output_filenames[step_count]).only_remove_append + '[Split' + str(
                step_count) + '].log'
        else:
            output_filename = splitted_output_filenames[step_count]
        if filename_class(filename).name == filename_class(output_filename).name:
            output_filename = filename_class(splitted_output_filenames[step_count]).only_remove_append + '[Split' + str(
                step_count) + '].log'
        if step_count in required_step or required_step == []:
            splitted_file_names.append(output_filename)
            if not only_get_file_names:
                with open(output_filename, 'w') as output_file:
                    if "\n" in step[0]:
                        output_file.write("".join(step))
                    else:
                        output_file.write("\n".join(step))

    if only_one_step:
        assert len(splitted_file_names) == 1
        return splitted_file_names[0]
    return splitted_file_names


def print_link_List(data=None, running=False, modify_time=datetime.now()):
    if data is None:
        data = []

    returnStr = ""
    Format = ["", "", "", "%H:", "%M:", "%S", " %m.%d"]

    ave_502 = []
    ave_703 = []

    for i, item in enumerate(data):
        returnStr += "L [" + "{:>4}".format(item.link) + "] End at "
        if i == 0:
            returnStr += datetime.strftime(item.datetime, ''.join(Format))
        else:
            last = data[i - 1]
            delta = item.datetime - last.datetime

            if item.link == "502":
                ave_502.append(delta)
            if item.link == "703":
                ave_703.append(delta)
            # if item.link=='1002' or item.link=='1110':
            # print(item.link,"\t",delta.total_seconds()/60)

            if i == len(data) - 16:  # 在倒数第16个显示完整时间
                returnStr += "{:>8}".format(datetime.strftime(item.datetime, ''.join(Format[:6])))
            else:
                for j in range(3, 6):
                    if last.datetime.timetuple()[j] != item.datetime.timetuple()[j]:
                        returnStr += "{:>8}".format(datetime.strftime(item.datetime, ''.join(Format[j:6])))
                        break
                else:
                    returnStr = returnStr[:-4]
                    returnStr += "{:>12}".format('-')

            if delta.seconds != 0:
                returnStr += " in "

                if delta.days > 0:
                    returnStr += "{:>5.1}".format(delta.days + delta.seconds / 86400) + "day"

                else:
                    try:
                        delta_datetime = datetime.strptime(str(delta), "%H:%M:%S")
                    except Exception:
                        delta_datetime = datetime.strptime("23:59:59", "%H:%M:%S")

                    if delta.seconds >= 3600:
                        returnStr = returnStr[:-1]
                        returnStr += "{:>6}".format(delta_datetime.strftime('[%H:%M]'))

                    elif delta.seconds > 60:
                        returnStr += "{:<6}".format(delta_datetime.strftime('%M\'%Ss'))
                    else:
                        returnStr += "{:>6}".format(str(int(delta_datetime.strftime('%S'))) + 's')

        returnStr += '\n'

    # if ave_502:
    #     print("L502:\t",sum(ave_502,timedelta(0)).total_seconds()/len(ave_502)/60)
    # if ave_703:
    #     print("L703:\t",sum(ave_703,timedelta(0)).total_seconds()/len(ave_703)/60)

    if running:
        # print('Running...')
        if len(data) > 1:  # current link running time
            current_delta = modify_time - data[-2].datetime
        else:
            current_delta = 0

        if current_delta:
            returnStr += "\nCurrent " + "{:>5}".format("L" + data[-1].link) + " : "
            if current_delta.days > 0:
                returnStr += str(current_delta.days) + " day "
            current_delta_datetime = datetime.strptime(re.findall(r"\d+:\d{2}:\d{2}", str(current_delta))[0], "%H:%M:%S")
            returnStr += current_delta_datetime.strftime('%H:%M:%S')

    if running:
        returnStr += '\n\n   '
        total_delta = modify_time - data[0].datetime
    else:
        returnStr += '\n'
        total_delta = data[-1].datetime - data[0].datetime

    returnStr += "Total time : "
    if total_delta.days > 0:
        returnStr += str(total_delta.days) + " day "
    total_delta_datetime = datetime.strptime(re.findall(r"\d+:\d{2}:\d{2}", str(total_delta))[0], "%H:%M:%S")
    returnStr += total_delta_datetime.strftime('%H:%M:%S')

    # print("Total wall time:\t",total_delta.total_seconds()/60)

    returnStr += '\n'
    return (returnStr)

if __name__ == '__main__':
    pass