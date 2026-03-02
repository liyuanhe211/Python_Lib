# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

from Python_Lib.My_Lib_Stock import *


class Filetype:
    gaussian_input = "Gaussian Input *##*rvtaefcaefsz(@#"
    orca_input = "ORCA INPUT *##srbgsrgzvzsefv*(@#"
    gaussian_output = "Gaussian Output *##*(zesvestetzevs@#"
    orca_output = "ORCA Output *##*zsvetetvt(@#"
    mopac_input = 'MOPAC INPUT *##*(zvestetv@#'
    mopac_output = 'MOPAC Output *#zvestetvset#*(@#'
    MECP_output = "MECP Output %#zvtetez&*(^%&#"
    MECP_output_folder = "MECP Output Folder %#&agasdgsd*(^%&#"
    xTB_coordinate_indicator = 'xTB_coordinate_indicator#$*)@($&@)(*%^)@('
    input = [gaussian_input, orca_input]
    output = [gaussian_output, orca_output]
    valid = input + output


def file_type(filename):
    if os.path.isdir(filename):
        files = list(os.listdir(filename))
        if "ReportFile" in files:
            return Filetype.MECP_output_folder
    if filename_class(filename).append.lower() in ['gjf', 'com']:
        return Filetype.gaussian_input

    if not filename.lower().endswith('.xtb.log') and filename_class(filename).append.lower() == 'log':
        return Filetype.gaussian_output

    if filename_class(filename).append.lower() == 'inp':
        return Filetype.orca_input

    if filename_class(filename).append.lower() == 'orca':
        return Filetype.orca_output

    if filename_class(filename).append.lower() in ['mopac', 'arc']:
        return Filetype.mopac_output

    if filename_class(filename).append.lower() == 'out':
        # fast determine required for MOPAC files
        with open(filename, encoding='utf-8', errors='ignore') as file:
            for count, line in enumerate(file):
                if count > 20:
                    break
                if "**                                MOPAC2012                                  **" in line:
                    return Filetype.mopac_output
                if "**                                MOPAC2016                                  **" in line:
                    return Filetype.mopac_output

        with open(filename, encoding='utf-8', errors='ignore') as file:
            for line in file:
                line = line.strip().lower()
                # remove filename lines like %rwf, %chk, %base to prevent "!" or "#" appear in filename
                if True in [line.startswith(key) for key in ['%rwf', "%chk", '%base', "%oldchk"]]:
                    continue
                if line.startswith('Entering Gaussian System'.lower()):
                    return Filetype.gaussian_output
                if line.startswith("Gaussian 09, Revision ".lower()):
                    return Filetype.gaussian_output
                if line.startswith("* O   R   C   A *".lower()):
                    return Filetype.orca_output
                if line.startswith('#'):
                    return Filetype.gaussian_output

    # MECP 输出，形如下面的一个单几何结构文件
    #   8    0.66744144    2.46069271   -0.31465362
    #   6   -0.37688619    1.56742458    0.03329066#
    if os.path.isfile(filename):
        with open(filename, encoding='utf-8', errors='ignore') as file_content:
            file_content = file_content.readlines()
            if len(file_content) < 10000:
                file_content = remove_blank([x.strip() for x in file_content])
                if all([re.findall(r"^\s*(\d+)\s+(-*\d+\.\d*)\s+(-*\d+\.\d*)\s+(-*\d+\.\d*)\s*$", line) for line in file_content]):
                    return Filetype.MECP_output

    # if filename_class(filename).append.lower() == 'out':
    #     with open(filename,encoding='utf-8',errors='ignore') as file:
    #         for count,line in enumerate(file):
    #             line = line.strip().lower()
    #             #remove filename lines like %rwf, %chk, %base to prevent "!" or "#" appear in filename
    #             if True in [line.startswith(key) for key in ['%rwf',"%chk",'%base',"%oldchk"]]:
    #                 continue
    #             if line.startswith('Entering Gaussian System'.lower()):
    #                 return Filetype.gaussian_output
    #             if line.startswith("Gaussian 09, Revision ".lower()):
    #                 return Filetype.gaussian_output
    #             if line.startswith("* O   R   C   A *".lower()):
    #                 return Filetype.orca_output
    #             if line.startswith('#'):
    #                 return Filetype.gaussian_output
    #             if "**                                MOPAC2012                                  **".lower() in line:
    #                 return Filetype.mopac_output
    #             if "**                                MOPAC2016                                  **".lower() in line:
    #                 return Filetype.mopac_output
