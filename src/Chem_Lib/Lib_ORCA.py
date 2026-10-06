# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

# import sys
# import pathlib
# parent_path = str(pathlib.Path(__file__).parent.resolve())
# sys.path.insert(0,parent_path)

import os
import re
import random
import shutil
import subprocess
import sys

from Python_Lib.My_Lib_Stock import *
from Python_Lib.My_Lib_File import filename_class, get_unused_filename
from Chem_Lib.Lib_Constants import *
from Chem_Lib.Lib_Coordinates import *
from Chem_Lib.Lib_Filetype import Filetype, file_type


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
                    self.gCP_correction = float(match[0]) * Hartree__KJ_mol

            if "FINAL SINGLE POINT ENERGY" in line:
                match = re.findall(r'FINAL SINGLE POINT ENERGY\s+(-\d+\.\d+)', line)
                if match:
                    self.electronic_energy = float(match[0]) * Hartree__KJ_mol
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
                    # if re_ret: self.H_correction = float(re_ret[0])*Hartree__KJ_mol

                    re_ret = re.findall(entropy_pattern, line)
                    if re_ret: self.S = float(re_ret[0]) * kcal__kJ * 1000 / self.temp

                    re_ret = re.findall(enthalpy_pattern, line)
                    if re_ret: self.H = float(re_ret[0]) * Hartree__KJ_mol

                    re_ret = re.findall(gibbs_pattern, line)
                    if re_ret: self.G = float(re_ret[0]) * Hartree__KJ_mol

                    if gibbs_corr_lead_pattern in line:
                        re_ret = re.findall(gibbs_corr_pattern, Hessian_lines[count2 + 1])
                        if re_ret:
                            self.G_correction = float(re_ret[0]) * Hartree__KJ_mol

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


# ---------------------------------------------------------------------------
# build_ORCA_input_from_template
# ---------------------------------------------------------------------------

# Geometry sections of an ORCA input.  The ``*file`` forms occupy a single line
# (``* xyzfile 0 1 geometry.xyz``); the block forms run until a line holding a
# lone ``*``.  ``xyzfile`` must be tested before ``xyz`` (prefix).
_ORCA_GEOMETRY_FILE_FORM_PATTERN = re.compile(r"^\*\s*(xyzfile|gzmtfile|intfile)\b", re.IGNORECASE)
_ORCA_GEOMETRY_BLOCK_FORM_PATTERN = re.compile(r"^\*\s*(xyz|gzmt|int|internal)\b", re.IGNORECASE)

# 资源与 %base 只认 QM_Creater 一直生成的单行写法；ORCA 允许的多行块写法
# （``%pal\n nprocs 8\nend``）不解析，碰到时明确报错而不是悄悄放过。
_ORCA_PAL_LINE_PATTERN = re.compile(r"^%pal\s+nprocs\s+(\d+)\s+end$", re.IGNORECASE)
_ORCA_MAXCORE_LINE_PATTERN = re.compile(r"^%maxcore\s+(\d+)$", re.IGNORECASE)

# ``# __NAMETAG__=[tag]`` —— ORCA 模板里的名牌标注。Gaussian 模板用 ``!`` 开头的
# 标注行，但 ``!`` 在 ORCA 里是路由行，所以 ORCA 模板改用 ``#`` 注释承载同一约定。
# ``#`` 与 ``__NAMETAG__`` 之间、``=`` 两侧都允许空格（实际模板两种写法都存在）。
_ORCA_NAMETAG_PATTERN = re.compile(r"^#\s*__NAMETAG__\s*=\s*(.*)$", re.IGNORECASE)

# QM_Creater / qsuborca 时代的资源占位注释（由旧提交脚本在集群上替换）。本函数用
# 显式的 %pal / %maxcore 处理资源，这些标记行一律丢弃、不进生成的文件。
_ORCA_LEGACY_RESOURCE_MARKERS = ("#__resource_portion__", "#__auto_mem__", "!__auto_mem__")


def build_ORCA_input_from_template(
    method_template,
    coordinates,
    output_path=None,
    *,
    charge: int | None = None,
    multiplet: int | None = None,
    solvent: str | None = None,
    nprocs: int | None = None,
    maxcore_mb: int | None = None,
) -> str:
    """
    Merge an ORCA method/route template with a geometry to produce a ready-to-run,
    single-step ORCA input (``.inp``), written to disk.

    ORCA-side counterpart of :func:`Chem_Lib.Lib_Gaussian.build_Gaussian_input_from_template`
    — the same *usage* pattern (template supplies *how* to compute, geometry supplies
    *what*; NAMETAG stamping, name-clash avoidance, charge/multiplicity validation,
    EDITTHIS refusal), but built around ORCA's own input conventions, which share
    nothing with Gaussian's:

    * **Single step only.**  A template containing ``$new_job`` raises
      :class:`ValueError` — multi-step ORCA workflows are separate submissions.
    * The template's ``!`` route lines and ``%`` blocks are kept verbatim and in
      order, EXCEPT ``%pal`` / ``%maxcore`` / ``%base``, which this function owns
      (see below).  The three owned resource lines are written after any leading
      ``#`` comment lines of the template (``#`` starts a comment in ORCA), so a
      header comment stays the first thing in the file.  Any geometry the template
      carries (``* xyz ... *`` / ``* xyzfile ...`` and their gzmt/int siblings) is
      discarded and replaced by *coordinates*.
    * ``%base`` is always written, always as the final output filename's stem
      (basename only, no directories) — HPC_Lib runs the job in an isolated
      per-submission run directory and copies the outputs back next to the input
      afterwards (see :func:`finalize_ORCA_run`), so every auxiliary file
      (``.gbw`` / ``.xyz`` / ``.trj`` / ``.hess`` ...) ends up next to the input
      under the same stem.  A ``%base`` in the template is ignored: it would go
      stale the moment the output name shifts on a clash.
    * ``%pal nprocs N end`` is required information: the value comes from *nprocs*,
      falling back to the template's declaration; neither present → ValueError.
      (Besides controlling parallelism, the ``%pal`` line is how HPC_Lib recognises
      a ``.inp`` file as an ORCA input.)  The single-line form is the only one
      parsed; a multi-line ``%pal`` block raises with instructions.  The value is
      rewritten to the actual allocation by ``submit_ORCA_file`` at submission, so
      a nominal template value is fine.
    * ``%maxcore`` (memory per core, MB) follows the same override order
      (*maxcore_mb* > template) but is optional: declared nowhere → no line
      written, and the submission machinery fills it from the SLURM allocation.

    Template placeholders, mirroring the Gaussian builder where ORCA allows:

    * ``CPCM(EDITTHIS)`` in a route line, and ``smdsolvent "EDITTHIS"`` in a
      ``%cpcm`` block, are filled from *solvent* — and only the placeholder: a
      concrete solvent already in the template is never replaced, and a *solvent*
      argument with no placeholder to fill is silently ignored.  ORCA's solvent
      names are version-dependent, so no validation list is applied — a name ORCA
      does not recognise fails at ORCA startup.
    * ``#__NAMETAG__=[tag]`` (``#`` comment, since ``!`` is ORCA's route marker;
      whitespace after ``#`` and around ``=`` is tolerated) is consumed and stamped
      into the output filename by the same rule as the Gaussian builder.  Because
      QM_Creater's ORCA convention keeps ``[`` / ``]`` out of ORCA filenames, the
      stamped stem then has every bracket replaced by ``___`` (a tag
      ``[Opt_r2SCAN3c]`` appears as ``___Opt_r2SCAN3c___``).
    * ``[RANDOM_NUMBER]`` anywhere in the kept template lines is replaced by one
      random integer (the same value for every occurrence), as QM_Creater.
    * ``#__RESOURCE_PORTION__=...`` / ``#__AUTO_MEM__`` legacy resource markers
      (the old qsuborca replacement protocol) are dropped.
    * Any ``EDITTHIS`` still unresolved when the file is about to be written
      raises :class:`ValueError`.

    The output file never overwrites an existing one: a name clash shifts to
    ``<name>_01``, ``_02``, ... (``get_unused_filename`` with
    ``continue_number_suffix=True``), and ``%base`` is derived from the name
    actually used.

    Args:
        method_template:  Path to an ORCA ``.inp`` template, or raw template text.
        coordinates:      Where the geometry comes from — a
                          :class:`Chem_Lib.Lib_Coordinates.Coordinates` object; a path to a
                          geometry-carrying file (``.gjf`` / ``.com`` are read through
                          :class:`Gaussian_Input` and contribute charge / multiplicity;
                          any other file yields its last Cartesian block with charge /
                          multiplicity unknown); or raw coordinate lines
                          (list / tuple / multi-line str).
        output_path:      Where to write.  The template NAMETAG is stamped into the
                          filename first, then brackets are sanitised to ``_``.
                          Default: derived from the geometry's source file — same
                          directory, stamped source stem + ``.inp``; a source ``.gjf``
                          carrying its own ``!__NAMETAG__`` gets exactly that segment
                          replaced (as the Gaussian builder).  Unlike the Gaussian
                          builder there is no in-memory mode: no *output_path* and no
                          known source → :class:`ValueError`.
        charge, multiplet:  Override the charge / spin multiplicity (default: from
                          *coordinates*).  The pair is validated with
                          :meth:`Coordinates.charge_and_multiplicity_problem` before
                          anything is written; there is no way to switch this off.
        solvent:          Fills the ``CPCM(EDITTHIS)`` / ``smdsolvent "EDITTHIS"``
                          placeholders (see above).
        nprocs:           Override the ``%pal nprocs N end`` core count.
        maxcore_mb:       Override the ``%maxcore`` per-core memory (MB).  Note this
                          is ORCA's native per-core quantity — total job memory is
                          roughly ``maxcore_mb × nprocs``.

    Returns:
        The absolute path of the ``.inp`` file written.

    Raises:
        ValueError:  Multi-step template; missing / malformed ``%pal``; duplicate
            resource lines; no ``!`` route line; unresolvable output location; a
            space in the output filename (ORCA aux-file handling and HPC submission
            both refuse spaces); unusable charge / multiplicity; unresolved
            ``EDITTHIS``.
        TypeError:   *method_template* / *coordinates* of an unsupported type.

    Example::

        from Chem_Lib.Lib_Gaussian import Gaussian_Output
        coordinate = Gaussian_Output("ts_freq.out").steps[-1].summary.coordinate
        inp_path = build_ORCA_input_from_template(
            "method_templates/ORCA_wB97M-V_def2TZVPP_SP.inp",
            coordinate,
            solvent="DMSO",
        )
    """
    from Chem_Lib.Lib_Gaussian import Gaussian_Input, _stamp_nametag_into_filename

    # --- 1. template text -----------------------------------------------------------
    if isinstance(method_template, (str, os.PathLike)) and os.path.isfile(str(method_template)):
        with open(method_template, encoding="utf-8") as template_file:
            template_text = template_file.read()
    elif isinstance(method_template, str):
        template_text = method_template
    else:
        raise TypeError(
            "method_template must be a file path or raw ORCA input text, "
            f"got {type(method_template).__name__}"
        )

    if re.search(r"^\s*\$new_job\b", template_text, re.IGNORECASE | re.MULTILINE):
        raise ValueError(
            "The template contains '$new_job' — build_ORCA_input_from_template only "
            "supports single-step ORCA jobs. Split the workflow into one input per step."
        )

    # --- 2. classify the template lines ---------------------------------------------
    nametag = ""
    template_nprocs: int | None = None
    template_maxcore: int | None = None
    kept_lines: list[str] = []

    lines = template_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        lowered = stripped.lower()
        index += 1

        nametag_match = _ORCA_NAMETAG_PATTERN.match(stripped)
        if nametag_match:
            nametag = nametag_match.group(1).strip()
            continue
        if any(lowered.startswith(marker) for marker in _ORCA_LEGACY_RESOURCE_MARKERS):
            continue

        pal_match = _ORCA_PAL_LINE_PATTERN.match(stripped)
        if pal_match:
            if template_nprocs is not None:
                raise ValueError("The template declares %pal more than once.")
            template_nprocs = int(pal_match.group(1))
            continue
        if lowered.startswith("%pal"):
            raise ValueError(
                f"Unrecognised %pal line: {stripped!r}. Only the single-line "
                f"'%pal nprocs N end' form is supported — rewrite a multi-line "
                f"%pal block as a single line."
            )

        maxcore_match = _ORCA_MAXCORE_LINE_PATTERN.match(stripped)
        if maxcore_match:
            if template_maxcore is not None:
                raise ValueError("The template declares %maxcore more than once.")
            template_maxcore = int(maxcore_match.group(1))
            continue
        if lowered.startswith("%maxcore"):
            raise ValueError(
                f"Unrecognised %maxcore line: {stripped!r}. Expected '%maxcore <MB>'."
            )

        if lowered.startswith("%base"):
            continue  # regenerated from the output filename below, whatever it says

        if _ORCA_GEOMETRY_FILE_FORM_PATTERN.match(stripped):
            continue
        if _ORCA_GEOMETRY_BLOCK_FORM_PATTERN.match(stripped):
            while index < len(lines) and lines[index].strip() != "*":
                index += 1
            index += 1  # also skip the closing "*"
            continue

        kept_lines.append(line.rstrip())

    while kept_lines and not kept_lines[0].strip():
        kept_lines.pop(0)
    while kept_lines and not kept_lines[-1].strip():
        kept_lines.pop()

    if not any(line.lstrip().startswith("!") for line in kept_lines):
        raise ValueError("The template contains no '!' route line — not a usable ORCA method template.")

    resolved_nprocs = int(nprocs) if nprocs is not None else template_nprocs
    if resolved_nprocs is None:
        raise ValueError(
            "Neither the template nor the nprocs argument declares a core count — an ORCA "
            "input needs a '%pal nprocs N end' line (it is also how HPC_Lib recognises ORCA "
            "inputs). The value is rewritten to the actual allocation at submission, so a "
            "nominal value is acceptable."
        )
    if resolved_nprocs < 1:
        raise ValueError(f"nprocs must be a positive integer, got {resolved_nprocs}.")
    resolved_maxcore = int(maxcore_mb) if maxcore_mb is not None else template_maxcore

    # --- 3. resolve the geometry ------------------------------------------------------
    geometry_source = None
    if isinstance(coordinates, Coordinates):
        geometry = coordinates
        geometry_source = coordinates.source_path
    elif isinstance(coordinates, (str, os.PathLike)) and os.path.isfile(str(coordinates)):
        coordinates_path = str(coordinates)
        geometry_source = coordinates_path
        if os.path.splitext(coordinates_path)[1].lower() in (".gjf", ".com"):
            gaussian_steps = Gaussian_Input(coordinates_path).steps
            geometry = gaussian_steps[0].coordinate if gaussian_steps else None
            if geometry is None:
                raise ValueError(
                    f"No explicit geometry found in the first step of {coordinates_path}."
                )
        else:
            # last Cartesian block of the file; charge / multiplicity stay unknown
            geometry = Coordinates(coordinates_path)
    elif isinstance(coordinates, (list, tuple, str)):
        geometry = Coordinates(coordinates)
    else:
        raise TypeError(
            "coordinates must be a Coordinates object, raw coordinate lines, or a path "
            f"to a geometry-carrying file, got {type(coordinates).__name__}"
        )

    if geometry.is_fault or not geometry.coordinates:
        raise ValueError("The geometry could not be parsed into Cartesian coordinate lines.")

    resolved_charge = int(charge) if charge is not None else geometry.charge
    resolved_multiplicity = int(multiplet) if multiplet is not None else geometry.multiplicity
    validation_geometry = Coordinates(
        list(geometry.coordinates), charge=resolved_charge, multiplet=resolved_multiplicity
    )
    problem = validation_geometry.charge_and_multiplicity_problem()
    if problem is not None:
        raise ValueError(
            f"Refusing to build the ORCA input: {problem}. Pass charge=... / multiplet=... "
            f"to set them explicitly, or take the geometry from a source that carries the "
            f"intended values."
        )

    # --- 4. placeholders in the kept template lines ----------------------------------
    if solvent is not None:
        solvent_value = str(solvent).strip()
        cpcm_pattern = re.compile(r"cpcm\(\s*editthis\s*\)", re.IGNORECASE)
        smd_pattern = re.compile(r'(smdsolvent\s+")editthis(")', re.IGNORECASE)
        kept_lines = [
            smd_pattern.sub(
                lambda match: match.group(1) + solvent_value + match.group(2),
                cpcm_pattern.sub("CPCM(" + solvent_value + ")", line),
            )
            for line in kept_lines
        ]

    random_number = str(random.randint(0, 100000000000))
    kept_lines = [line.replace("[RANDOM_NUMBER]", random_number) for line in kept_lines]

    # --- 5. resolve the output path ---------------------------------------------------
    if output_path is not None:
        directory, basename = os.path.split(str(output_path))
        stem, extension = os.path.splitext(basename)
        stem = _stamp_nametag_into_filename(stem, nametag)
        extension = extension or ".inp"
    else:
        if not geometry_source:
            raise ValueError(
                "No output_path was given and the geometry does not remember its source "
                "file — the ORCA builder always writes a file, so pass output_path "
                "explicitly."
            )
        directory, basename = os.path.split(str(geometry_source))
        stem, source_extension = os.path.splitext(basename)
        source_nametag = ""
        if source_extension.lower() in (".gjf", ".com") and os.path.isfile(str(geometry_source)):
            source_nametag = (Gaussian_Input(str(geometry_source)).nametag or "").strip()
        stem = _stamp_nametag_into_filename(stem, nametag, source_nametag)
        extension = ".inp"

    # QM_Creater 的 ORCA 文件名约定：方括号不进 ORCA 文件名（%base 与辅助文件名
    # 都由它派生），一律换成下划线。
    stem = stem.replace("[", "___").replace("]", "___")
    if " " in stem + extension:
        raise ValueError(
            f"The ORCA input filename must not contain spaces (it names %base and every "
            f"auxiliary file, and HPC submission refuses spaces): {stem + extension!r}"
        )
    output_path = get_unused_filename(os.path.join(directory, stem + extension),
                                      continue_number_suffix=True)
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    # --- 6. compose and write ---------------------------------------------------------
    geometry_lines = [
        f"{element:<3} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
        for element, (x, y, z) in zip(validation_geometry.elements,
                                      validation_geometry.coordinates_numer)
    ]

    # 模板开头的 "#" 注释块（ORCA 手册 3.2：注释以 "#" 起始）保持在文件最前面，
    # 资源三行插在注释块之后——注释是给人看的文件头，不该被挤到资源行下面。
    leading_comment_count = 0
    while (leading_comment_count < len(kept_lines)
           and kept_lines[leading_comment_count].lstrip().startswith("#")):
        leading_comment_count += 1

    output_lines = list(kept_lines[:leading_comment_count])
    output_lines.append(f"%pal nprocs {resolved_nprocs} end")
    if resolved_maxcore is not None:
        output_lines.append(f"%maxcore {resolved_maxcore}")
    output_lines.append(f'%base "{base_name}"')
    output_lines.extend(kept_lines[leading_comment_count:])
    output_lines.append("")
    output_lines.append(f"* xyz {resolved_charge} {resolved_multiplicity}")
    output_lines.extend(geometry_lines)
    output_lines.append("*")
    final_text = "\n".join(output_lines).strip() + "\n"

    if "editthis" in final_text.lower():
        position = final_text.lower().find("editthis")
        context = final_text[max(0, position - 30):position + 38]
        raise ValueError(
            f"The template still contains an unresolved 'EDITTHIS' placeholder near: "
            f"...{context!r}... Pass solvent=... for a solvent placeholder, or edit the "
            f"template."
        )

    with open(output_path, "w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(final_text)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# 集群侧的运行收尾：成败判定、molden 生成、保留文件移回、运行目录回收
# ---------------------------------------------------------------------------

#: HPC_ORCA 在提交时写进运行目录的归属标记文件名，内容是本次计算的
#: 输入文件路径。:func:`finalize_ORCA_run` 触碰运行目录前靠它核对归属。
ORCA_RUN_DIRECTORY_MARKER_FILENAME = "INPUT_FilePath.txt"

#: 收尾成功分支里移动到 ``<输入目录>/ORCA_Temp_Files/`` 的文件后缀白名单
#: （穷尽列表）。旧式 ``.trj`` 单列处理——移动时改名
#: ``<主干>.trj.xyz``（ORCA 5/6 的轨迹叫 ``<主干>_trj.xyz``，被 ``.xyz``
#: 一项覆盖）。``.TDDFTGuess.gbw`` 虽然后缀命中 ``.gbw``，但明确排除。
#: ``<主干>.orca`` 与 molden 产物不在此表内——它们直接移动到输入文件旁边。
ORCA_TEMP_FILE_SUFFIXES = (
    ".property.txt", ".bibtex", ".gbw", ".densities", ".densitiesinfo",
    ".cpcm", ".cpcm_corr", ".smd", ".xyz", ".engrad", ".opt",
)

#: 存放保留辅助文件的子目录名（建在输入文件所在的目录下）。
ORCA_TEMP_FILES_DIRECTORY_NAME = "ORCA_Temp_Files"


def is_ORCA_output_terminated_normally(output_file: str, tail_bytes: int = 65536) -> bool:
    """只读尾部，判定一个 ORCA 输出文件是否正常结束。

    ORCA 正常结束时输出末尾有 ``****ORCA TERMINATED NORMALLY****``（其后只有
    ``TOTAL RUN TIME`` 一行）；失败时末尾是 ``ORCA finished by error
    termination in ...`` 之类。本函数 ``seek`` 到距文件末尾 *tail_bytes* 处、
    只读这一段找标记子串——内存占用即 *tail_bytes*，数 GB 的输出文件同样瞬时
    完成。与 :class:`ORCA_output` 的 ``normal_termination`` 属性（``readlines()``
    全文件读入内存）不同，收尾流程一律用本函数。

    Args:
        output_file: ORCA 输出文件路径（``.orca``）。不存在时返回 False。
        tail_bytes:  只读末尾多少字节。默认 64 KB，远大于结束标记到文件末尾
                     的距离。

    Returns:
        True 表示正常结束；文件不存在、读取失败或找不到标记都返回 False。
    """
    try:
        file_size = os.path.getsize(output_file)
        with open(output_file, "rb") as output:
            output.seek(max(0, file_size - tail_bytes))
            tail_text = output.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return "ORCA TERMINATED NORMALLY" in tail_text


#: 激发态段落里的单条轨道贡献行，如
#: ``    82a -> 238a  :     0.000003 (c=  0.00170435)``——权重列在前、
#: 系数 c 在括号里（TDA 下权重 = c² 严格成立）。
#: ``[ab]?`` 吸收 UKS 输出的自旋标记（闭壳层输出同样带 ``a``）；``<-`` 分支
#: 防御性保留（与 Gaussian 版对 de-excitation 行的处理对齐）。捕获组是
#: 系数 c——阈值作用在它上面，不是权重列上（阈值定义见下方函数 docstring）。
_ORCA_TDDFT_CONTRIBUTION_PATTERN = re.compile(
    r"^\s*\d+[abAB]?\s*(?:->|<-)\s*\d+[abAB]?\s*:\s*-?\d+\.\d+\s*"
    r"\(c=\s*(-?\d+\.\d+)\)")


def clean_ORCA_tddft_TPrint_small_contributions(filename, threshold=0.05, output_filename=None):
    """从 ORCA TDDFT/TDA 输出里删除小贡献的激发组分行。

    ``%tddft`` 的 ``TPrint`` 调低（如 ``TPrint 1e-8``，对应 Gaussian 的
    ``IOp(9/40=4)`` 打印深度）时，激发态段落（``TD-DFT/TDA EXCITED
    STATES`` 等标题之后）会为每个激发态打印每一对轨道贡献，文件因此极为
    庞大。本函数复制文件，丢弃系数绝对值 |c| 低于 *threshold* 的贡献行；
    段落之前的内容与所有非贡献行（含 ``STATE`` 头、吸收谱表）逐字保留，
    识别不了的行一律保留、绝不误删。Multiwfn 等下游解析器照常读取产物。

    **阈值定义：统一采用 Gaussian 的定义——|c| 下限**，
    与 :func:`Chem_Lib.Lib_Gaussian.clean_gaussian_tddft_9_40_small_contributions`
    的 *threshold* 完全同义：``clean_ORCA(t)`` 等价于 ``clean_Gaussian(t)``。
    两个程序对「打印哪些组分行」的原生门槛定义不同——Gaussian 的
    ``IOp(9/40=N)`` 直接切系数（打印 |c| > 1E-N，默认 0.1），ORCA 的
    ``TPrint x`` 切**权重**（打印贡献 = c² > x，输出头原话 "the weight of
    the individual excitations are printed if larger than ..."，默认 0.01
    = |c| 0.1），因此两把刀之间差一个平方：``TPrint x`` ⇔ |c| ≥ √x。
    Multiwfn 手册 §3.21.A.3 与 Sobereva 博文 758 给出的官方等价正是
    ``TPrint 1E-8`` ≡ ``IOp(9/40=4)``（|c| ≥ 1E-4）。本函数内部**不需要**
    做这个平方换算：它比较的是每行括号里打印的系数 c 本身、而不是行首的
    权重列，平方关系只存在于「本阈值 ↔ TPrint 关键词」之间——
    ``clean_ORCA(t)`` 的产物逐行等于用 ``TPrint = t*t`` 重跑打印的输出
    （例：``clean_ORCA(0.0001)`` ≡ ``TPrint 1e-8`` 重打印）。

    脚注（归一化差异，与平方无关）：闭壳层下 Gaussian 打印的系数满足
    Σc² = 0.5、ORCA 满足 Σc² = 1（同一物理组态 ORCA 的 c 是 Gaussian 的
    √2 倍），故按「系数数值等同」的官方约定，闭壳层时 ORCA 侧同阈值实际
    多保留 √2 倍深度（偏保守、只多不少）；开壳层（UKS）两家都归一到 1，
    等价严格成立。Multiwfn 读取时按各家约定内部换算，两种输出喂给它得到
    同一套归一化振幅，此差异不影响分析正确性。

    Args:
        filename:        ORCA TDDFT/TDA 输出文件路径（``.orca`` / ``.out``）。
        threshold:       保留所需的最小 |c|（Gaussian 定义，默认 0.05）。
        output_filename: 目标路径；默认把扩展名换成
                         ``clean_<threshold>.<原扩展名>``
                         （``X.orca`` → ``X.clean_0.05.orca``）。

    Returns:
        输出文件路径。
    """
    if output_filename is None:
        input_filename_parts = filename_class(filename)
        extension = input_filename_parts.append or "out"
        output_filename = input_filename_parts.replace_append_to(
            f"clean_{threshold}.{extension}")

    started = False
    with open(output_filename, 'w', encoding="utf-8") as clean_output_file:
        with open(filename, encoding="utf-8", errors="ignore") as orca_output_file:
            for line in orca_output_file:
                if "EXCITED STATES" in line:
                    started = True
                if started:
                    coefficient_match = _ORCA_TDDFT_CONTRIBUTION_PATTERN.match(line)
                    if coefficient_match and \
                            abs(float(coefficient_match.group(1))) < threshold:
                        continue
                clean_output_file.write(line)

    return output_filename


def _clean_finalized_tddft_output(orca_output_path: str,
                                  thresholds: list[float]) -> None:
    """成功收尾后按 *thresholds* 各档生成一份精简掉小组分的 ``.orca`` 副本。

    与 :func:`HPC_Lib.HPC_Gaussian_Postprocess.clean_tddft_output` 同一套路：
    阈值从小到大依次处理，后一档在前一档的产物上继续精简（结果与直接在原始
    文件上精简完全相同，但少读一遍大文件）；某一档失败时只警告、下一档退回
    原始文件重新开始。产物名始终按最终的 ``.orca`` 路径命名
    （``X.orca`` → ``X.clean_0.0001.orca`` / ``X.clean_0.05.orca``）。

    清理的任何失败都**不允许**影响收尾（进而作业）的成败——本函数绝不抛出。
    """
    source_path = orca_output_path
    for threshold in sorted(set(thresholds)):
        target_path = filename_class(orca_output_path).replace_append_to(
            f"clean_{threshold}.orca")
        print(f"[Lib_ORCA] cleaning TDDFT contributions below {threshold:g}: "
              f"{os.path.basename(source_path)} -> {os.path.basename(target_path)}")
        try:
            clean_ORCA_tddft_TPrint_small_contributions(
                source_path, threshold, output_filename=target_path)
        except Exception:
            import traceback
            print(f"[Lib_ORCA] WARNING: TDDFT cleaning at threshold {threshold:g} "
                  f"failed (the finalisation itself is unaffected):", file=sys.stderr)
            traceback.print_exc()
            source_path = orca_output_path
            continue
        try:
            size_mb = os.path.getsize(target_path) / 1024 / 1024
            print(f"[Lib_ORCA] wrote {target_path} ({size_mb:.1f} MB)")
        except OSError:
            pass
        source_path = target_path


def generate_ORCA_molden_file(
    run_directory: str,
    orca_input_file: str,
    orca_executable_directory: str | None = None,
) -> list[str]:
    """在运行目录里生成 molden 文件，返回生成的文件路径列表（都在运行目录内）。

    只转换两类波函数文件（不对目录里的每个 ``.gbw`` 都转，以免把
    ``.TDDFTGuess.gbw`` 之类的中间波函数也转出来）：

    * 主波函数 ``<主干>.gbw``（``<主干>`` = 输入文件主干，与 ``%base`` 一致）
      → ``<主干>.molden``；
    * 每个 MP2 自然轨道文件 ``*.mp2nat``：改名加 ``.gbw`` 后缀（``orca_2mkl``
      只认这个后缀）后转换 → ``<主干>.MP2_NO.molden``（多个时顺延编号）。

    生成的 molden 留在运行目录里，由 :func:`finalize_ORCA_run` 统一移动到
    输入文件旁边。任何一个文件的转换失败（``orca_2mkl`` 调不起来、非零退出、
    没有产物）只打印警告并跳过——molden 只是附加产物，绝不能挡住输出文件的
    移回。

    Args:
        run_directory:  ORCA 的运行目录（``.gbw`` 等辅助文件所在处）。
        orca_input_file:  本次计算的 ``.inp`` 输入文件路径；molden 文件以它
                          的主干命名。
        orca_executable_directory:  ORCA 安装目录（``orca_2mkl`` 在其中，用
                          完整路径调用）。None 时用裸 ``orca_2mkl``，依赖
                          PATH——只适合交互环境，作业脚本一律显式传入。

    Returns:
        生成的 molden 文件路径列表（位于 *run_directory* 内）。
    """
    run_directory = os.path.abspath(run_directory)
    orca_input_file = os.path.abspath(orca_input_file)
    input_stem = os.path.splitext(os.path.basename(orca_input_file))[0]
    if orca_executable_directory:
        orca_2mkl_command = os.path.join(orca_executable_directory, "orca_2mkl")
    else:
        orca_2mkl_command = "orca_2mkl"

    def convert_gbw_to_molden(gbw_stem: str) -> str | None:
        """调 ``orca_2mkl`` 把运行目录里的 ``<gbw_stem>.gbw`` 转成 molden，
        返回产物 ``.molden.input`` 的路径；失败打印警告并返回 None。"""
        try:
            conversion = subprocess.run(
                [orca_2mkl_command, gbw_stem, "-molden"],
                cwd=run_directory, capture_output=True, text=True,
            )
        except OSError as call_error:
            print(f"[Lib_ORCA] WARNING: cannot run {orca_2mkl_command} for "
                  f"{gbw_stem}.gbw: {call_error}")
            return None
        molden_input_path = os.path.join(run_directory, gbw_stem + ".molden.input")
        if conversion.returncode != 0 or not os.path.isfile(molden_input_path):
            details = (conversion.stderr or conversion.stdout or "").strip()
            print(f"[Lib_ORCA] WARNING: orca_2mkl failed for {gbw_stem}.gbw "
                  f"(exit code {conversion.returncode}): {details}")
            return None
        return molden_input_path

    generated_files: list[str] = []

    if os.path.isfile(os.path.join(run_directory, input_stem + ".gbw")):
        molden_input_path = convert_gbw_to_molden(input_stem)
        if molden_input_path:
            target_path = get_unused_filename(
                os.path.join(run_directory, input_stem + ".molden"))
            os.replace(molden_input_path, target_path)
            print(f"[Lib_ORCA] Generated molden file: {target_path}")
            generated_files.append(target_path)

    for filename in sorted(os.listdir(run_directory)):
        if not filename.lower().endswith(".mp2nat"):
            continue
        mp2nat_path = os.path.join(run_directory, filename)
        os.replace(mp2nat_path, mp2nat_path + ".gbw")
        molden_input_path = convert_gbw_to_molden(filename)
        if molden_input_path:
            target_path = get_unused_filename(
                os.path.join(run_directory, input_stem + ".MP2_NO.molden"))
            os.replace(molden_input_path, target_path)
            print(f"[Lib_ORCA] Generated MP2 natural-orbital molden file: {target_path}")
            generated_files.append(target_path)
    return generated_files


def finalize_ORCA_run(
    run_directory: str,
    orca_input_file: str,
    orca_executable_directory: str | None = None,
    tddft_clean_thresholds: list[float] | None = None,
) -> list[str]:
    """ORCA 作业结束后的收尾：按成败分流，只把保留文件移回数据目录。

    与 :mod:`HPC_Lib.HPC_ORCA` 生成的作业脚本配套——脚本让 ORCA 在按路径
    映射规则确定的独立运行（RWF）目录里执行、输出写为运行目录里的
    ``<主干>.orca``，ORCA 退出后作业脚本调用本函数的命令行入口收尾。步骤：

    1. 归属核对：运行目录里的 ``INPUT_FilePath.txt``（提交时写下）内容必须
       等于 *orca_input_file*，对不上就整体拒绝——拿错目录时宁可什么都不动。
    2. 成败判定：**只看** ``<主干>.orca`` 的结尾
       （:func:`is_ORCA_output_terminated_normally`，只读尾部），不看退出码。
    3. 成功分支：
       (a) :func:`generate_ORCA_molden_file` —— 主波函数 ``<主干>.gbw`` 转
       ``<主干>.molden``、``*.mp2nat`` 转 ``<主干>.MP2_NO.molden``（转换失败
       只警告不中断）；(b) ``<主干>.orca`` 与 molden 产物**移动**到输入文件
       旁边，随后按 *tddft_clean_thresholds* 各档生成精简副本
       （:func:`_clean_finalized_tddft_output`，清理失败只警告）；(c) 后缀白名单（:data:`ORCA_TEMP_FILE_SUFFIXES`，另加旧式
       ``.trj``——移动时改名 ``<主干>.trj.xyz``；``.TDDFTGuess.gbw`` 明确
       排除）内的辅助文件移动到 ``<输入目录>/ORCA_Temp_Files/``；(d) 每一次
       移动都用 ``get_unused_filename`` 防覆盖（同名顺延 ``_01`` / ``_02``
       ...，绝不覆盖已有文件）；(e) 以上全部成功后删除运行目录连同其中剩余
       的一切（迭代向量、``.cis`` / ``.cis1``、``.bas0``-``.bas5``、
       ``.hostnames``、``.TDDFTGuess.gbw``、``*.tmp``、归属标记）——这是
       全流程**唯一**允许清空运行目录的时点：任务确认成功、且所有保留文件
       都已成功移到输入旁及其子目录之后；任何移动失败则不删除、保留现场
       并报告。
    4. 失败分支：不生成 molden、不做白名单移动，仅把 ``<主干>.orca``
       **复制**（不是移动）到输入文件旁边（同样防覆盖）；运行目录原样保留
       供排查——不论其中有何内容，都不清空、不删除。``<主干>.orca`` 不存在
       （ORCA 启动即崩）也按失败处理并如实报告。

    Args:
        run_directory:  提交时创建的独立运行（RWF）目录。
        orca_input_file:  本次计算的 ``.inp`` 输入文件路径（保留文件移回它
                          所在的目录及其 ``ORCA_Temp_Files/`` 子目录）。
        orca_executable_directory:  ORCA 安装目录，透传给
                          :func:`generate_ORCA_molden_file`。
        tddft_clean_thresholds:  成功分支里 ``<主干>.orca`` 移回输入旁之后，
                          按这些 |c| 阈值各生成一份精简掉小组分的副本
                          （:func:`_clean_finalized_tddft_output`；哪几档由
                          :func:`HPC_Lib.HPC_ORCA._resolve_orca_tddft_clean_thresholds`
                          在**提交时**根据输入里的 ``TPrint`` 声明定下）。
                          清理失败只警告，绝不影响收尾成败；失败分支不清理。

    Returns:
        问题描述列表；空列表表示任务成功且收尾完整完成、运行目录已删除。
        任务失败时列表非空（命令行入口据此以退出码 1 结束）。
    """
    run_directory = os.path.abspath(run_directory)
    orca_input_file = os.path.abspath(orca_input_file)
    input_stem = os.path.splitext(os.path.basename(orca_input_file))[0]

    if not os.path.isdir(run_directory):
        problem = f"run directory does not exist: {run_directory}"
        print(f"[Lib_ORCA] WARNING: {problem}")
        return [problem]

    destination_directory = os.path.dirname(orca_input_file)
    if os.path.realpath(run_directory) == os.path.realpath(destination_directory):
        problem = (f"run directory IS the input file's own directory, refusing to "
                   f"finalize: {run_directory}")
        print(f"[Lib_ORCA] WARNING: {problem}")
        return [problem]

    marker_path = os.path.join(run_directory, ORCA_RUN_DIRECTORY_MARKER_FILENAME)
    marker_content = ""
    if os.path.isfile(marker_path):
        with open(marker_path, encoding="utf-8", errors="replace") as marker_file:
            marker_content = marker_file.read().strip()
    if marker_content != orca_input_file:
        problem = (f"run-directory marker mismatch: {ORCA_RUN_DIRECTORY_MARKER_FILENAME} "
                   f"says {marker_content!r}, expected {orca_input_file!r} — refusing to "
                   f"touch {run_directory}")
        print(f"[Lib_ORCA] WARNING: {problem}")
        return [problem]

    orca_output_path = os.path.join(run_directory, input_stem + ".orca")

    if not is_ORCA_output_terminated_normally(orca_output_path):
        # 失败分支：只把 .orca 复制回输入旁供排查；运行目录不论有何内容，
        # 一律原样保留（不清空、不删除——清空只允许发生在成功收尾的最后
        # 一步）。
        problems: list[str] = []
        if os.path.isfile(orca_output_path):
            problems.append(
                f"ORCA did not terminate normally — no 'ORCA TERMINATED NORMALLY' "
                f"near the end of {orca_output_path}")
            copied_path = get_unused_filename(
                os.path.join(destination_directory, input_stem + ".orca"))
            try:
                shutil.copy2(orca_output_path, copied_path)
                print(f"[Lib_ORCA] Copied the output back for inspection: {copied_path}")
            except OSError as copy_error:
                problems.append(f"failed to copy {orca_output_path} back to "
                                f"{copied_path}: {copy_error}")
        else:
            problems.append(
                f"ORCA produced no output file at {orca_output_path} — it likely "
                f"crashed on startup")
        print(f"[Lib_ORCA] WARNING: {problems[0]}")
        print(f"[Lib_ORCA] The run FAILED; the run directory is kept untouched for "
              f"inspection (nothing is ever cleaned on failure): {run_directory}")
        return problems

    # 成功分支。
    molden_files = generate_ORCA_molden_file(
        run_directory, orca_input_file, orca_executable_directory)

    problems: list[str] = []
    moved_count = 0

    def move_without_overwriting(source_path: str, wanted_target_path: str) -> str | None:
        """把 *source_path* 移动为 *wanted_target_path*（已存在时顺延编号）。
        成功返回实际落位的路径；失败记入 problems、返回 None，不抛出。"""
        nonlocal moved_count
        target_path = get_unused_filename(wanted_target_path)
        try:
            shutil.move(source_path, target_path)
        except OSError as move_error:
            problems.append(f"failed to move {source_path} to {target_path}: {move_error}")
            return None
        print(f"[Lib_ORCA] Retrieved: {target_path}")
        moved_count += 1
        return target_path

    # (b) 输出与 molden 产物直接移动到输入文件旁边。
    retrieved_orca_path = move_without_overwriting(
        orca_output_path, os.path.join(destination_directory, input_stem + ".orca"))
    for molden_path in molden_files:
        move_without_overwriting(
            molden_path,
            os.path.join(destination_directory, os.path.basename(molden_path)))

    # TDDFT 精简副本：任务确认成功、.orca 已在最终位置之后立即生成（产物名
    # 按实际落位路径命名，防覆盖顺延过也跟着顺延）。清理失败绝不影响收尾。
    if retrieved_orca_path and tddft_clean_thresholds:
        _clean_finalized_tddft_output(retrieved_orca_path, tddft_clean_thresholds)

    # (c) 后缀白名单内的辅助文件进 <输入目录>/ORCA_Temp_Files/。先分类再
    # 移动：确有文件要移时才创建子目录，不留空目录。
    temp_files_directory = os.path.join(destination_directory,
                                        ORCA_TEMP_FILES_DIRECTORY_NAME)
    planned_temp_moves: list[tuple[str, str]] = []
    for entry_name in sorted(os.listdir(run_directory)):
        source_path = os.path.join(run_directory, entry_name)
        if not os.path.isfile(source_path):
            continue
        lowered_name = entry_name.lower()
        if lowered_name.endswith(".tddftguess.gbw"):
            continue  # 与主波函数同尺寸的 TDDFT 初猜中间文件，明确不保留
        if lowered_name.endswith(".trj"):
            planned_temp_moves.append(
                (source_path, os.path.join(temp_files_directory,
                                           input_stem + ".trj.xyz")))
        elif any(lowered_name.endswith(suffix) for suffix in ORCA_TEMP_FILE_SUFFIXES):
            planned_temp_moves.append(
                (source_path, os.path.join(temp_files_directory, entry_name)))
    if planned_temp_moves:
        try:
            os.makedirs(temp_files_directory, exist_ok=True)
        except OSError as mkdir_error:
            problems.append(f"cannot create {temp_files_directory}: {mkdir_error}")
        else:
            for source_path, wanted_target_path in planned_temp_moves:
                move_without_overwriting(source_path, wanted_target_path)

    if problems:
        print(f"[Lib_ORCA] WARNING: retrieval incomplete, run directory kept for "
              f"inspection (nothing is cleaned unless every move succeeded): "
              f"{run_directory}")
        for problem in problems:
            print(f"[Lib_ORCA]     {problem}")
        return problems

    # (e) 唯一允许清空/删除运行目录的时点：任务成功、所有保留文件都已移回。
    shutil.rmtree(run_directory)
    print(f"[Lib_ORCA] Finalized: {moved_count} file(s) moved back beside "
          f"{orca_input_file}; run directory removed: {run_directory}")
    return []


_FINALIZE_RUN_USAGE = (
    "Usage: python Lib_ORCA.py finalize_run "
    "<run_directory> <orca_input_file> [<orca_executable_directory>] "
    "[--tddft-clean-threshold X ...]"
)

if __name__ == '__main__':
    # 命令行入口，供 HPC_Lib.HPC_ORCA 生成的作业脚本在 ORCA 退出后调用。
    if len(sys.argv) >= 2 and sys.argv[1] == "finalize_run":
        finalize_positional_arguments: list[str] = []
        finalize_tddft_clean_thresholds: list[float] = []
        finalize_argument_index = 2
        while finalize_argument_index < len(sys.argv):
            finalize_argument = sys.argv[finalize_argument_index]
            if finalize_argument == "--tddft-clean-threshold":
                if finalize_argument_index + 1 >= len(sys.argv):
                    print(_FINALIZE_RUN_USAGE)
                    sys.exit(2)
                try:
                    finalize_tddft_clean_thresholds.append(
                        float(sys.argv[finalize_argument_index + 1]))
                except ValueError:
                    print(_FINALIZE_RUN_USAGE)
                    sys.exit(2)
                finalize_argument_index += 2
                continue
            finalize_positional_arguments.append(finalize_argument)
            finalize_argument_index += 1
        if len(finalize_positional_arguments) not in (2, 3):
            print(_FINALIZE_RUN_USAGE)
            sys.exit(2)
        finalize_problems = finalize_ORCA_run(
            finalize_positional_arguments[0], finalize_positional_arguments[1],
            finalize_positional_arguments[2]
            if len(finalize_positional_arguments) == 3 else None,
            tddft_clean_thresholds=finalize_tddft_clean_thresholds,
        )
        sys.exit(1 if finalize_problems else 0)
    print(_FINALIZE_RUN_USAGE)
    sys.exit(2)