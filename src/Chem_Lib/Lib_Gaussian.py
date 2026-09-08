# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

from Python_Lib.My_Lib_Stock import *
from Python_Lib.My_Lib_File import (get_unused_filename, local_path_prefix_pattern,
                                    local_to_remote_path_mappings, map_local_path_to_remote)
from Chem_Lib.Lib_Coordinates import *
from Chem_Lib.Lib_Filetype import Filetype, file_type

import io
import math
import hashlib
from dataclasses import dataclass

GAUSSIAN_SCRF_SOLVENTS = (
    "Water", "Acetonitrile", "Methanol", "Ethanol", "IsoQuinoline", "Quinoline",
    "Chloroform", "DiethylEther", "Dichloromethane", "DiChloroEthane",
    "CarbonTetraChloride", "Benzene", "Toluene", "ChloroBenzene", "NitroMethane",
    "Heptane", "CycloHexane", "Aniline", "Acetone", "TetraHydroFuran",
    "DiMethylSulfoxide", "Argon", "Krypton", "Xenon", "n-Octanol",
    "1,1,1-TriChloroEthane", "1,1,2-TriChloroEthane", "1,2,4-TriMethylBenzene",
    "1,2-DiBromoEthane", "1,2-EthaneDiol", "1,4-Dioxane", "1-Bromo-2-MethylPropane",
    "1-BromoOctane", "1-BromoPentane", "1-BromoPropane", "1-Butanol", "1-ChloroHexane",
    "1-ChloroPentane", "1-ChloroPropane", "1-Decanol", "1-FluoroOctane", "1-Heptanol",
    "1-Hexanol", "1-Hexene", "1-Hexyne", "1-IodoButane", "1-IodoHexaDecane",
    "1-IodoPentane", "1-IodoPropane", "1-NitroPropane", "1-Nonanol", "1-Pentanol",
    "1-Pentene", "1-Propanol", "2,2,2-TriFluoroEthanol", "2,2,4-TriMethylPentane",
    "2,4-DiMethylPentane", "2,4-DiMethylPyridine", "2,6-DiMethylPyridine",
    "2-BromoPropane", "2-Butanol", "2-ChloroButane", "2-Heptanone", "2-Hexanone",
    "2-MethoxyEthanol", "2-Methyl-1-Propanol", "2-Methyl-2-Propanol", "2-MethylPentane",
    "2-MethylPyridine", "2-NitroPropane", "2-Octanone", "2-Pentanone", "2-Propanol",
    "2-Propen-1-ol", "3-MethylPyridine", "3-Pentanone", "4-Heptanone",
    "4-Methyl-2-Pentanone", "4-MethylPyridine", "5-Nonanone", "AceticAcid",
    "AcetoPhenone", "a-ChloroToluene", "Anisole", "Benzaldehyde", "BenzoNitrile",
    "BenzylAlcohol", "BromoBenzene", "BromoEthane", "Bromoform", "Butanal",
    "ButanoicAcid", "Butanone", "ButanoNitrile", "ButylAmine", "ButylEthanoate",
    "CarbonDiSulfide", "Cis-1,2-DiMethylCycloHexane", "Cis-Decalin", "CycloHexanone",
    "CycloPentane", "CycloPentanol", "CycloPentanone", "Decalin-mixture",
    "DiBromomEthane", "DiButylEther", "DiEthylAmine", "DiEthylSulfide", "DiIodoMethane",
    "DiIsoPropylEther", "DiMethylDiSulfide", "DiPhenylEther", "DiPropylAmine",
    "e-1,2-DiChloroEthene", "e-2-Pentene", "EthaneThiol", "EthylBenzene",
    "EthylEthanoate", "EthylMethanoate", "EthylPhenylEther", "FluoroBenzene",
    "Formamide", "FormicAcid", "HexanoicAcid", "IodoBenzene", "IodoEthane",
    "IodoMethane", "IsoPropylBenzene", "m-Cresol", "Mesitylene", "MethylBenzoate",
    "MethylButanoate", "MethylCycloHexane", "MethylEthanoate", "MethylMethanoate",
    "MethylPropanoate", "m-Xylene", "n-ButylBenzene", "n-Decane", "n-Dodecane",
    "n-Hexadecane", "n-Hexane", "NitroBenzene", "NitroEthane", "n-MethylAniline",
    "n-MethylFormamide-mixture", "n,n-DiMethylAcetamide", "n,n-DiMethylFormamide",
    "n-Nonane", "n-Octane", "n-Pentadecane", "n-Pentane", "n-Undecane",
    "o-ChloroToluene", "o-Cresol", "o-DiChloroBenzene", "o-NitroToluene", "o-Xylene",
    "Pentanal", "PentanoicAcid", "PentylAmine", "PentylEthanoate", "PerFluoroBenzene",
    "p-IsoPropylToluene", "Propanal", "PropanoicAcid", "PropanoNitrile", "PropylAmine",
    "PropylEthanoate", "p-Xylene", "Pyridine", "sec-ButylBenzene", "tert-ButylBenzene",
    "TetraChloroEthene", "TetraHydroThiophene-s,s-dioxide", "Tetralin", "Thiophene",
    "Thiophenol", "trans-Decalin", "TriButylPhosphate", "TriChloroEthene",
    "TriEthylAmine", "Xylene-mixture", "z-1,2-DiChloroEthene",
)

_GAUSSIAN_SCRF_SOLVENTS_LOWER = frozenset(s.lower() for s in GAUSSIAN_SCRF_SOLVENTS)

# Per-step ``!__KEY__=VALUE`` annotations that build_Gaussian_input_from_template()
# *consumes* (to build %chk names, %oldchk chaining and the [EXTRACT_GEOM] title
# directive) and therefore strips from the generated file rather than echoing verbatim.
# NAMETAG and !RUN commands are kept.
_CONSUMED_ANNOTATION_KEYS = frozenset({
    "FCHK_TAG", "EXTRACT_GEOM", "CHK_READ", "RESOURCE_PORTION", "AUTO_MEM",
    "FCHK_TITLE_TAG",
})



def _template_directive_lines(value) -> list[str]:
    """Normalise a str / list-of-str directive block into a clean list of lines."""
    if value is None:
        return []
    raw = value.splitlines() if isinstance(value, str) else [str(item) for item in value]
    return [line.rstrip() for line in raw if line.strip()]


def _stamp_nametag_into_filename(filename_stem: str, new_nametag: str, source_nametag: str = "") -> str:
    """
    Stamp a ``!__NAMETAG__`` tag into a filename stem (the basename without its
    extension — never a full path; directories are the caller's business) and return
    the new stem.

    Rules, in order:

    1. When *source_nametag* is given (the ``!__NAMETAG__`` carried by the geometry
       source file itself) and it appears verbatim in the stem, exactly that segment
       is replaced by *new_nametag* — the stem is a previous product of this
       workflow, so no guessing is needed.
    2. Otherwise the **last** ``[...]`` segment whose content contains at least one
       letter (a-z / A-Z) is replaced by *new_nametag*.  Letter-less segments such as
       a structure numbering ``[000001_000002]`` never qualify.  This heuristic only
       runs when *new_nametag* is non-empty — an empty new tag must not delete a
       segment it merely guessed to be a tag.
    3. No qualifying segment: *new_nametag* is appended to the stem.

    Module-internal: this is the naming rule of
    :func:`build_Gaussian_input_from_template` (the same one QM_Creater applies), not
    a helper for callers to run themselves.  Whoever needs the resulting name reads it
    from the built object's ``.path``.
    """
    new_nametag = (new_nametag or "").strip()
    source_nametag = (source_nametag or "").strip()
    if source_nametag and source_nametag in filename_stem:
        return filename_stem.replace(source_nametag, new_nametag, 1)
    if new_nametag:
        letter_segments = list(re.finditer(r"\[[^\[\]]*[A-Za-z][^\[\]]*\]", filename_stem))
        if letter_segments:
            last_segment = letter_segments[-1]
            return filename_stem[:last_segment.start()] + new_nametag + filename_stem[last_segment.end():]
    return filename_stem + new_nametag



class Date_Class:
    def __init__(self, link='', datetime_str='', cycle=0, energy=0):
        self.link = link
        self.cycle = cycle
        self.energy = energy

        try:
            self.datetime = datetime.strptime(datetime_str, "%a %b %d %H:%M:%S %Y")
        except Exception:
            pass

class Gaussian_Input:
    """
    Parse and optionally modify a Gaussian input file.

    The file may contain multiple steps separated by ``--Link1--``.
    Each step is represented by a :class:`Gaussian_Input_Step` instance stored in
    ``self.steps``.

    Modification workflow:
        1. Create a ``Gaussian_Input(path)`` object.
        2. Call modification methods (they accept a ``step`` parameter:
           ``"ALL"`` to apply to every step, or an ``int`` index starting from 0).
        3. Call :meth:`save` to write the modified file back to disk
           (or :meth:`to_string` to get the text without writing).

    Attributes:
        path:        Original file path (str or None if constructed from text).
        steps:       List of :class:`Gaussian_Input_Step`.
        step_count:  Number of steps.
        annotate_lines:  Annotation lines mirrored from the first step.
        annotates_dict:  Dict of annotation key→value mirrored from the first
                 step's ``!__KEY__=VALUE`` lines.
        run_commands:    List of first-step ``!RUN ...`` command strings.
    """

    def __init__(self, path_or_text, *, is_text: bool = False):
        """
        Args:
            path_or_text:  File path (str), or raw text if ``is_text=True``.
            is_text:       If True, treat *path_or_text* as file content rather
                           than a file path.
        """
        if is_text:
            self.path = None
            raw_text = path_or_text
        else:
            self.path = str(path_or_text)
            with open(self.path, encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

        self.annotate_lines: list[str] = []
        self.annotates_dict: dict[str, str] = {}
        self.run_commands: list[str] = []

        # --- split steps by --Link1-- --------------------------------------
        # Split while preserving the separator so we can reconstruct later.
        parts = re.split(r"(?i)(--link1--)", raw_text)

        step_texts: list[str] = []
        for part in parts:
            if re.match(r"(?i)--link1--", part.strip()):
                continue  # skip separator
            step_texts.append(part)

        self.steps: list[Gaussian_Input_Step] = [
            Gaussian_Input_Step(t) for t in step_texts
        ]
        self.step_count: int = len(self.steps)
        if self.path:
            for step in self.steps:
                if step.coordinate is not None:
                    step.coordinate.source_path = self.path
        self._sync_global_annotations_from_first_step()

    def _sync_global_annotations_from_first_step(self):
        self.annotate_lines = []
        self.annotates_dict = {}
        self.run_commands = []
        if not self.steps:
            return

        first_step = self.steps[0]
        self.annotate_lines = list(first_step.annotate_lines)
        self.annotates_dict = dict(first_step.annotates_dict)
        for line in first_step.command_lines:
            m_run = re.findall(r"!RUN (.+)", line)
            if m_run:
                self.run_commands.append(m_run[0].strip())

    def _mirror_global_annotations_to_first_step(self):
        if not self.steps or not self.annotate_lines:
            return

        first_step = self.steps[0]
        prefix_lines = [line for line in self.annotate_lines if line not in first_step.annotate_lines]
        prefix_commands = [line for line in prefix_lines if line.startswith("!RUN ")]

        first_step.annotate_lines = prefix_lines + list(first_step.annotate_lines)
        first_step.command_lines = prefix_commands + list(first_step.command_lines)
        for key, value in self.annotates_dict.items():
            first_step.annotates_dict.setdefault(key, value)

    # ------------------------------------------------------------------
    # Step-dispatching helpers
    # ------------------------------------------------------------------
    def _resolve_steps(self, step) -> list[int]:
        """Return list of 0-based step indices from a *step* argument."""
        if isinstance(step, str) and step.upper() == "ALL":
            return list(range(self.step_count))
        if isinstance(step, int):
            if step < 0 or step >= self.step_count:
                raise IndexError(f"Step {step} out of range (0..{self.step_count - 1})")
            return [step]
        raise TypeError(f"step must be 'ALL' or int, got {type(step).__name__}")

    # ------------------------------------------------------------------
    # Modification methods  (step="ALL" | int)
    # ------------------------------------------------------------------
    def set_nprocshared(self, step, value: int):
        """Set ``%nprocshared`` for the given step(s)."""
        for i in self._resolve_steps(step):
            self.steps[i].set_nprocshared(value)

    def set_mem(self, step, mem_mb: int):
        """Set ``%mem`` (in MB) for the given step(s)."""
        for i in self._resolve_steps(step):
            self.steps[i].set_mem(mem_mb)

    def set_chk(self, step, chk_path: str):
        """Set ``%chk`` for the given step(s)."""
        for i in self._resolve_steps(step):
            self.steps[i].set_chk(chk_path)

    def set_oldchk(self, step, oldchk_path: str):
        """Set ``%oldchk`` for the given step(s)."""
        for i in self._resolve_steps(step):
            self.steps[i].set_oldchk(oldchk_path)

    def set_rwf(self, step, rwf_path: str):
        """Set ``%rwf`` for the given step(s)."""
        for i in self._resolve_steps(step):
            self.steps[i].set_rwf(rwf_path)

    def set_route_keyword(self, step, keyword: str, options: list[str] | str | None = None):
        """Replace (set) the options of a keyword for the given step(s).

        If the keyword already exists, its options are overwritten (not merged
        as :meth:`set_keyword_option` would).  If absent, it is added.
        Pass ``options=None`` or ``[]`` to keep the keyword present without
        options (e.g. ``freq`` alone).
        """
        for i in self._resolve_steps(step):
            self.steps[i].set_route_keyword(keyword, options)

    def set_keyword_option(self, step, keyword: str, options: list[str] | str | None = None):
        """Value-aware create-or-replace of route options for the given step(s).

        Options of the ``name=value`` form are matched by name, so e.g.
        setting ``maxstep=10`` replaces an existing ``maxstep=5``.  See
        :meth:`Gaussian_Input_Step.set_keyword_option` for details/examples.
        """
        for i in self._resolve_steps(step):
            self.steps[i].set_keyword_option(keyword, options)

    def remove_keyword_option(self, step, keyword: str, options: list[str] | str | None = None):
        """Value-aware removal of route options for the given step(s).

        An option given without ``=`` also removes any ``name=value`` option
        of the same name; ``options=None`` drops the whole keyword.  See
        :meth:`Gaussian_Input_Step.remove_keyword_option`.
        """
        for i in self._resolve_steps(step):
            self.steps[i].remove_keyword_option(keyword, options)

    def clear_keyword(self, step, keyword: str):
        """Drop a route keyword with all its options for the given step(s).

        E.g. ``gaussian_input.clear_keyword(0, "scrf")`` removes ``scrf=(...)`` entirely.
        ``level`` is reset to the blank placeholder instead of removed.
        """
        for i in self._resolve_steps(step):
            self.steps[i].clear_keyword(keyword)

    def set_route_dict(self, step, route_mapping: dict):
        """Replace the entire route section for the given step(s) with a dict.

        Example::

            gaussian_input.set_route_dict(0, {
                "level": ["b3lyp", "6-31g(d)"],
                "opt":   ["calcfc", "ts"],
                "freq":  [],
            })
        """
        for i in self._resolve_steps(step):
            self.steps[i].set_route_dict(route_mapping)

    def set_geometry(self, step, geometry, *, charge: int | None = None, multiplet: int | None = None):
        """Replace the geometry of the given step(s).

        Args:
            geometry:   One of:
                        * a :class:`Coordinates` object,
                        * a list of raw coordinate lines (strings), or
                        * a list of :class:`std_coordinate` objects.
            charge:     Optional new charge (kept if None).
            multiplet:  Optional new multiplicity (kept if None).
        """
        for i in self._resolve_steps(step):
            self.steps[i].set_geometry(geometry, charge=charge, multiplet=multiplet)

    def extract_steps(self, indices) -> "Gaussian_Input":
        """Return a **new** :class:`Gaussian_Input` containing only the selected steps.

        Args:
            indices: An int, a list/tuple of ints, a slice, or "ALL".
                     Indices may be negative (Python-style).

        The returned object shares no mutable state with *self* (deep copy).
        """
        if isinstance(indices, str) and indices.upper() == "ALL":
            picked = list(range(self.step_count))
        elif isinstance(indices, slice):
            picked = list(range(*indices.indices(self.step_count)))
        elif isinstance(indices, int):
            picked = [indices]
        else:
            picked = list(indices)

        picked = [i if i >= 0 else self.step_count + i for i in picked]
        for i in picked:
            if i < 0 or i >= self.step_count:
                raise IndexError(f"Step {i} out of range (0..{self.step_count - 1})")

        new_obj = Gaussian_Input.__new__(Gaussian_Input)
        new_obj.path = self.path
        new_obj.annotate_lines = list(self.annotate_lines)
        new_obj.annotates_dict = dict(self.annotates_dict)
        new_obj.run_commands = list(self.run_commands)
        new_obj.steps = [Gaussian_Input_Step(self.steps[i].to_string()) for i in picked]
        new_obj.step_count = len(new_obj.steps)
        return new_obj

    @classmethod
    def from_scratch(cls, steps: list[dict], *, annotates: dict | None = None,
                     run_commands: list[str] | None = None) -> "Gaussian_Input":
        """Build a new :class:`Gaussian_Input` from a pure-Python spec.

        Each entry in *steps* is a dict describing one step.  Supported keys::

            {
                # link0 (all optional)
                "nprocshared": 16,           # or "proc"
                "mem_mb":      32768,        # or "mem" in GB (float)
                "chk":         "job.chk",
                "oldchk":      "prev.chk",
                "rwf":         "job.rwf",
                "nosave":      True,
                "extra_link0": ["%subst ..."],

                # route (either of these — "route_dict" wins if both given)
                "route_dict":  {"level": ["b3lyp", "6-31g(d)"], "opt": [], ...},
                "route_str":   "#p b3lyp/6-31g(d) opt",

                # molecule (skipped if geom=allcheck)
                "title":       "My Job",          # or list[str]
                "charge":      0,
                "multiplet":   1,
                "geometry":    Coordinates | list[str] | list[std_coordinate],

                # trailing paragraphs (basis, ECP, modredundant, ...)
                "other":       [["line1", "line2"], ["para2 line1"]],

                # step-level annotations / !RUN
                "annotates":   {"TAG": "value"},
                "commands":    ["echo hello"],
            }

        Args:
            steps:         List of step-spec dicts (must be non-empty).
            annotates:     Global ``!__KEY__=VALUE`` annotations for the file.
            run_commands:  Global ``!RUN ...`` commands for the file.
        """
        if not steps:
            raise ValueError("steps must not be empty")

        obj = cls.__new__(cls)
        obj.path = None
        obj.annotate_lines = []
        obj.annotates_dict = dict(annotates or {})
        obj.run_commands = list(run_commands or [])

        for k, v in obj.annotates_dict.items():
            obj.annotate_lines.append(f"!__{k}__={v}")
        for cmd in obj.run_commands:
            obj.annotate_lines.append(f"!RUN {cmd}")

        obj.steps = [Gaussian_Input_Step.from_spec(spec) for spec in steps]
        obj.step_count = len(obj.steps)
        obj._mirror_global_annotations_to_first_step()
        obj._sync_global_annotations_from_first_step()
        return obj

    # ------------------------------------------------------------------
    # Aggregate info accessors
    # ------------------------------------------------------------------
    @property
    def nametag(self) -> str | None:
        return self.annotates_dict.get("NAMETAG")

    @property
    def chk_files(self) -> list[str]:
        return [s.chk for s in self.steps if s.chk]

    @property
    def rwf_files(self) -> list[str]:
        return [s.rwf for s in self.steps if s.rwf]

    @property
    def iop_9_40_values(self) -> list[int | None]:
        """各步骤路由里 ``IOp(9/40=N)`` 的 N（该步骤没写就是 ``None``）。"""
        return [step.iop_9_40_value for step in self.steps]

    @property
    def iop_9_40_thresholds(self) -> list[float | None]:
        """各步骤的 CI 展开系数打印阈值（该步骤没写 ``IOp(9/40)`` 就是 ``None``）。"""
        return [step.iop_9_40_threshold for step in self.steps]

    @property
    def has_iop_9_40(self) -> bool:
        """是否有任何一个步骤写了 ``IOp(9/40=...)``。"""
        return any(threshold is not None for threshold in self.iop_9_40_thresholds)

    @property
    def iop_9_40_threshold(self) -> float | None:
        """整个输入文件的 CI 展开系数打印阈值。

        多步任务取各步骤里**最小**的那一个——阈值越小打印得越详细，输出文件
        的体积由打印得最详细的那一步决定。没有任何步骤写 ``IOp(9/40)`` 时返回
        ``None``。
        """
        return Route_Dict.smallest_iop_9_40_threshold(self.iop_9_40_thresholds)

    @property
    def is_excited_state(self) -> bool:
        """是否有任何一个步骤是激发态任务（``TD`` / ``TDA`` / ``CIS``）。"""
        return any(step.is_excited_state for step in self.steps)

    def generate_job_name(self, filepath: str | None = None) -> str:
        """
        Generate a SLURM-safe job name.

        Uses NAMETAG annotation if present, otherwise the filename stem.
        Sanitised to ``[a-zA-Z0-9_\\-.[\\]]`` and truncated to 60 chars.
        """
        if filepath is None:
            filepath = self.path or "unnamed"
        stem = os.path.splitext(os.path.basename(filepath))[0]
        if self.nametag:
            name = f"{stem}_{self.nametag.strip()}"
        else:
            name = stem
        name = re.sub(r"[^a-zA-Z0-9_\-.\[\]]", "_", name)
        return name[:60]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_string(self) -> str:
        """Reconstruct the full input file text from the parsed data."""
        step_strings = [s.to_string() for s in self.steps]

        if not step_strings:
            return ""

        parts: list[str] = [step_strings[0]]

        for s_str in step_strings[1:]:
            parts.append("\n--Link1--\n" + s_str)

        return "".join(parts)

    def save(self, filepath: str | None = None):
        """
        Write the (possibly modified) input file to disk.

        Args:
            filepath:  Destination path.  Defaults to the original path.
        """
        if filepath is None:
            filepath = self.path
        if filepath is None:
            raise ValueError("No file path specified for save().")
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.to_string())



class Gaussian_Input_Step:
    """
    One step of a Gaussian input file (separated by ``--Link1--``).

    Stores both structured data and raw text so the file can be faithfully
    reconstructed after modification.

    Attributes (link0 commands — directly settable):
        proc (int):        ``%nprocshared`` value (0 = not declared; no line is emitted).
        mem_mb (int):      ``%mem`` value in MB (0 if unset).
        mem (float):       ``%mem`` value in GB (legacy, kept for compat).
        chk (str):         ``%chk`` path (case-preserved).
        oldchk (str):      ``%oldchk`` path.
        rwf (str):         ``%rwf`` path.
        nosave (bool):     Whether ``%nosave`` is present.
        extra_link0 (list[str]):  Other link0 lines we don't parse
                                   specially (e.g. ``%subst``).

    Attributes (route / body — from original parsing):
        route_str (str):            Raw route text.
        route_dict (Route_Dict):    Parsed route.
        title (list[str]):          Title paragraph lines.
        charge (int):               Molecular charge.
        multiplet (int):            Spin multiplicity.
        geom (list):                Geometry lines (std_coordinate objects).
        geom_raw (list[str]):       Geometry lines as raw text (for reconstruction).
        other (list[list[str]]):    Remaining paragraphs (basis, ECP, …).
        annotate_lines (list[str]): Step-level annotation lines.
        annotates_dict (dict):      Step-level annotations.
        command_lines (list[str]):  ``!RUN`` commands in this step.
        is_allcheck (bool):         True if ``geom=allcheck``.
    """

    def __init__(self, raw_text_or_list):
        """
        Args:
            raw_text_or_list:  Either a string of the step's text (for the new
                               ``Gaussian_Input`` path) or a list of already-
                               stripped lines (legacy ``Gaussian_input`` path).
        """
        if isinstance(raw_text_or_list, str):
            # New path: store raw text, derive line list
            self._raw_text = raw_text_or_list
            input_list = [x.strip() for x in raw_text_or_list.splitlines()]
            # Remove leading/trailing blank lines that are splitting artefacts
            while input_list and not input_list[0].strip():
                input_list.pop(0)
        else:
            input_list = list(raw_text_or_list)
            self._raw_text = "\n".join(input_list)

        self.charge: int = 999
        self.multiplet: int = 999
        self.proc: int = 0  # 0 = not declared; a file without %nprocshared must round-trip without one
        self.mem: float = 0.1
        self.mem_mb: int = 0
        self.chk: str = ""
        self.oldchk: str = ""
        self.rwf: str = ""
        self.nosave: bool = False
        self.extra_link0: list[str] = []

        self.input_list = input_list

        self._phrase_annotates()

        # --- divide paragraphs ----
        self.paragraphs: list[list[str]] = []
        temp: list[str] = []
        for line in self.input_list:
            if line.strip():
                temp.append(line)
            else:
                self.paragraphs.append(temp)
                temp = []
        if temp:
            self.paragraphs.append(temp)

        # --- read link0 commands  ----
        self.link0_list: list[str] = []
        link0_line_count = 0
        for i, line in enumerate(self.paragraphs[0]):
            line_lower = line.lower().strip()
            if line_lower.startswith("%"):
                self.link0_list.append(line)  # preserve original case

                if "%nprocshared=" in line_lower:
                    self.proc = int(re.findall(r"(?i)%nprocshared=(.+)", line)[0].strip())
                elif "%mem=" in line_lower and 'mb' in line_lower:
                    mem_val = float(re.findall(r"(?i)%mem=(.+?)mb", line)[0].strip())
                    self.mem_mb = int(mem_val)
                    self.mem = int(mem_val / 100) / 10
                elif "%mem=" in line_lower and 'gb' in line_lower:
                    mem_val = float(re.findall(r"(?i)%mem=(.+?)gb", line)[0].strip())
                    self.mem_mb = int(mem_val * 1024)
                    self.mem = mem_val
                elif "%chk=" in line_lower:
                    self.chk = re.findall(r"(?i)%chk=(.+)", line)[0].strip()
                elif "%oldchk=" in line_lower:
                    self.oldchk = re.findall(r"(?i)%oldchk=(.+)", line)[0].strip()
                elif "%rwf=" in line_lower:
                    self.rwf = re.findall(r"(?i)%rwf=(.+)", line)[0].strip()
                elif "%nosave" in line_lower:
                    self.nosave = True
                else:
                    self.extra_link0.append(line)
            else:
                link0_line_count = i
                break

        self.saved = [0, 1, 2]

        # --- route section ----
        self.route_list = self.paragraphs[0][link0_line_count:]
        self.route_str = self._join(self.route_list)
        self.route_dict = Route_Dict(self.route_str)

        self.connectivity: list[str] | None = None
        if 'connectivity' in self.route_str:
            self.connectivity = self.paragraphs.pop(3)

        # --- detect allcheck ----
        self.is_allcheck = (
            not self.route_dict.from_gaussview
            and 'allcheck' in safe_get_dict_value(self.route_dict, 'geom')
        )

        if self.is_allcheck:
            self.title: list[str] = []
            self.geom: list = []
            self.geom_raw: list[str] = []
            self.other: list[list[str]] = self.paragraphs[1:]
        else:
            self.title = self.paragraphs[1]
            if not [x for x in self.title if x.strip()]:
                self.title = ["Empty Title"]

            self.geom_paragraph = self.paragraphs[2]
            self.other = self.paragraphs[3:]

            # charge and multiplicity
            self.charge_and_multiplet = [x for x in self.geom_paragraph[0].split() if x != '']
            self.charge = int(self.charge_and_multiplet[0])
            self.multiplet = int(self.charge_and_multiplet[1])

            # geometry lines (without LP and charge line)
            self.geom_raw = [
                x for x in self.geom_paragraph[1:]
                if not x.lower().strip().startswith('lp')
            ]
            self.geom = [std_coordinate(x) for x in self.geom_raw]

        self.other_str = ""
        for paragraph in self.other:
            for line in paragraph:
                self.other_str += line + '\n'
            self.other_str += '\n'

        self.route_dict = Route_Dict(self.route_str, self.other_str)

        self.coordinate = Coordinates(self.geom, charge=self.charge, multiplet=self.multiplet) if self.geom else None
        self.geom_text = self._join(self.geom) if self.geom else ""

    # ------------------------------------------------------------------
    # IOp(9/40) —— TDDFT 激发组分的打印阈值
    # ------------------------------------------------------------------
    @property
    def iop_9_40_value(self) -> int | None:
        """本步骤路由里 ``IOp(9/40=N)`` 的 N；没写这个 IOp 时为 ``None``。"""
        return self.route_dict.iop_9_40_value

    @property
    def iop_9_40_threshold(self) -> float | None:
        """本步骤的 CI 展开系数打印阈值 ``10^-N``；没写这个 IOp 时为 ``None``。"""
        return self.route_dict.iop_9_40_threshold

    @property
    def has_iop_9_40(self) -> bool:
        """本步骤是否写了 ``IOp(9/40=...)``。"""
        return self.iop_9_40_value is not None

    @property
    def is_excited_state(self) -> bool:
        """本步骤是不是激发态任务（``TD`` / ``TDA`` / ``CIS``）。"""
        return self.route_dict.is_excited_state

    # ------------------------------------------------------------------
    # Annotation parsing (kept from original, slightly renamed)
    # ------------------------------------------------------------------
    def _phrase_annotates(self):
        self.annotate_lines: list[str] = []
        for line in self.input_list:
            if line.startswith('!'):
                self.annotate_lines.append(line)
        for line in self.annotate_lines:
            if line in self.input_list:
                self.input_list.remove(line)

        self.command_lines: list[str] = []
        for line in self.annotate_lines:
            m = re.findall(r'!RUN (.+)', line)
            if m:
                self.command_lines.append(line)

        for line in self.command_lines:
            if line in self.input_list:
                self.input_list.remove(line)

        self.annotates_dict: dict[str, str] = {}
        for line in self.annotate_lines:
            m = re.findall(r'!__(.+?)__=(.+)', line)
            if m:
                self.annotates_dict[m[0][0]] = m[0][1]

    # ------------------------------------------------------------------
    # Modification methods
    # ------------------------------------------------------------------
    def set_nprocshared(self, value: int):
        """Set ``%nprocshared``."""
        self.proc = value

    def set_mem(self, mem_mb: int):
        """Set ``%mem`` in MB."""
        self.mem_mb = mem_mb
        self.mem = round(mem_mb / 1024, 2)

    def set_chk(self, chk_path: str):
        """Set ``%chk``."""
        self.chk = chk_path

    def set_oldchk(self, oldchk_path: str):
        """Set ``%oldchk``."""
        self.oldchk = oldchk_path

    def set_rwf(self, rwf_path: str):
        """Set ``%rwf``."""
        self.rwf = rwf_path

    def set_route_keyword(self, keyword: str, options: list[str] | str | None = None):
        """Replace (not merge) the options for *keyword* in the route section.

        Examples::

            step.set_route_keyword("opt", ["ts", "calcfc"])  # overwrite
            step.set_route_keyword("freq")                    # keyword-only
            step.set_route_keyword("scrf", "smd")             # single option
        """
        keyword = keyword.strip().lower()
        if not keyword:
            return
        if options is None:
            options = []
        elif isinstance(options, str):
            options = [options]
        self.route_dict[keyword] = list(options)

    def set_keyword_option(self, keyword: str, options: list[str] | str | None = None):
        """Value-aware create-or-replace for a route keyword / its options.

        The keyword is created if absent.  Options of the ``name=value`` form
        are matched by *name*, so writing a new value replaces the old one
        instead of accumulating both.  See :meth:`Route_Dict.set_keyword_option`
        for the full semantics.

        Examples::

            step.set_keyword_option("opt", "tight")          # opt -> opt=tight
            step.set_keyword_option("opt", "maxstep=10")     # opt=maxstep=5 -> opt=maxstep=10
            step.set_keyword_option("freq")                  # ensure bare freq exists
            step.set_keyword_option("scrf", "solvent=water")
            step.set_keyword_option("iop", "3/76=1000002000")
            step.set_keyword_option("level", "m062x/def2svp")
        """
        self.route_dict.set_keyword_option(keyword, options)

    def remove_keyword_option(self, keyword: str, options: list[str] | str | None = None):
        """Value-aware removal of some options of a route keyword.

        An option given without ``=`` also removes any ``name=value`` option
        of the same name.  ``options=None`` drops the whole keyword (same as
        :meth:`clear_keyword`).  See :meth:`Route_Dict.remove_keyword_option`.

        Examples::

            step.remove_keyword_option("freq", "noraman")   # freq=noraman -> freq
            step.remove_keyword_option("opt", "maxstep")    # drops maxstep=5
            step.remove_keyword_option("iop", "5/13")       # drops the 5/13=... IOp entry
        """
        self.route_dict.remove_keyword_option(keyword, options)

    def clear_keyword(self, keyword: str):
        """Drop a route keyword together with all of its options.

        E.g. ``step.clear_keyword("scrf")`` removes ``scrf=(solvent=water)``
        entirely, no matter how many options it carries.  ``level`` is reset
        to the blank placeholder instead of removed.
        """
        self.route_dict.clear_keyword(keyword)

    def set_route_dict(self, route_mapping: dict):
        """Replace the entire route section using an ordinary dict.

        The ``level`` entry (method/basis) is preserved from *route_mapping* if
        present; otherwise the current method/basis is kept.  All other route
        keywords are overwritten.
        """
        preserved_level = self.route_dict.get("level", ["Blank_Method", "Black_Basis"])
        new_route = Route_Dict("", remove_genchk=False)
        new_route.clear()
        new_route["level"] = route_mapping.get("level", preserved_level)
        for k, v in route_mapping.items():
            if k == "level":
                continue
            if v is None:
                v = []
            elif isinstance(v, str):
                v = [v]
            else:
                v = list(v)
            new_route[k.strip().lower()] = v
        new_route.other_paragraph = self.route_dict.other_paragraph
        self.route_dict = new_route

    def set_geometry(self, geometry, *, charge: int | None = None, multiplet: int | None = None):
        """Replace the geometry (atomic coordinates) of this step.

        Args:
            geometry: Accepts:
                      * a :class:`Coordinates` object — charge/multiplet
                        are taken from it if not given explicitly,
                      * a list of raw coordinate-line strings (e.g.
                        ``"C   0.0  0.0  0.0"``),
                      * a list of :class:`std_coordinate` objects.
            charge, multiplet:  Override the current values (kept if None).
        """
        if isinstance(geometry, Coordinates):
            coords_obj = geometry
            std_list = list(getattr(coords_obj, "coordinates", []))
            raw_lines = [str(c) for c in std_list]
            if charge is None and getattr(coords_obj, "charge", 999) != 999:
                charge = int(coords_obj.charge)
            mult_attr = getattr(coords_obj, "multiplet", getattr(coords_obj, "multiplicity", 999))
            if multiplet is None and mult_attr != 999:
                multiplet = int(mult_attr)
        elif isinstance(geometry, list):
            if geometry and isinstance(geometry[0], str):
                raw_lines = [line.rstrip("\n") for line in geometry]
                std_list = [std_coordinate(line) for line in raw_lines]
            else:
                std_list = list(geometry)
                raw_lines = [str(c) for c in std_list]
        else:
            raise TypeError(f"Unsupported geometry type: {type(geometry).__name__}")

        if charge is not None:
            self.charge = int(charge)
        if multiplet is not None:
            self.multiplet = int(multiplet)

        self.geom_raw = raw_lines
        self.geom = std_list
        self.coordinate = Coordinates(self.geom, charge=self.charge, multiplet=self.multiplet) if self.geom else None
        self.geom_text = self._join(self.geom) if self.geom else ""
        self.is_allcheck = False
        if "geom" in self.route_dict and "allcheck" in self.route_dict["geom"]:
            self.route_dict["geom"].remove("allcheck")
            if not self.route_dict["geom"]:
                self.route_dict.pop("geom")

    @classmethod
    def from_spec(cls, spec: dict) -> "Gaussian_Input_Step":
        """Build a step from a dict spec (see :meth:`Gaussian_Input.from_scratch`)."""
        step = cls.__new__(cls)
        step.input_list = []
        step._raw_text = ""

        step.proc = int(spec.get("nprocshared", spec.get("proc", 0)))  # 0 = not declared
        if "mem_mb" in spec:
            step.mem_mb = int(spec["mem_mb"])
            step.mem = round(step.mem_mb / 1024, 2)
        elif "mem" in spec:
            step.mem = float(spec["mem"])
            step.mem_mb = int(step.mem * 1024)
        else:
            step.mem_mb = 0
            step.mem = 0.1
        step.chk = str(spec.get("chk", ""))
        step.oldchk = str(spec.get("oldchk", ""))
        step.rwf = str(spec.get("rwf", ""))
        step.nosave = bool(spec.get("nosave", False))
        step.extra_link0 = list(spec.get("extra_link0", []))

        if "route_dict" in spec and spec["route_dict"] is not None:
            rd_in = spec["route_dict"]
            step.route_dict = Route_Dict("", remove_genchk=False)
            step.route_dict.clear()
            step.route_dict["level"] = list(rd_in.get("level", ["Blank_Method", "Black_Basis"]))
            for k, v in rd_in.items():
                if k == "level":
                    continue
                if v is None:
                    v = []
                elif isinstance(v, str):
                    v = [v]
                else:
                    v = list(v)
                step.route_dict[k.strip().lower()] = v
            step.route_str = str(step.route_dict)
        else:
            step.route_str = spec.get("route_str", "")
            step.route_dict = Route_Dict(step.route_str)

        # Title
        title = spec.get("title", "")
        if isinstance(title, str):
            step.title = [title] if title else ["Empty Title"]
        else:
            step.title = list(title) if title else ["Empty Title"]

        step.charge = int(spec.get("charge", 0))
        step.multiplet = int(spec.get("multiplet", 1))

        # Geometry
        geometry = spec.get("geometry")
        step.geom_raw = []
        step.geom = []
        step.coordinate = None
        step.geom_text = ""
        step.is_allcheck = bool("geom" in step.route_dict and "allcheck" in step.route_dict["geom"])
        if geometry is not None:
            step.set_geometry(geometry)

        # Other paragraphs
        step.other = [list(p) for p in spec.get("other", [])]
        step.other_str = ""
        for paragraph in step.other:
            for line in paragraph:
                step.other_str += line + "\n"
            step.other_str += "\n"

        step.connectivity = None

        # Step-level annotations / run commands
        step.annotate_lines = []
        step.annotates_dict = dict(spec.get("annotates", {}))
        for k, v in step.annotates_dict.items():
            step.annotate_lines.append(f"!__{k}__={v}")
        step.command_lines = [f"!RUN {c}" for c in spec.get("commands", [])]
        step.annotate_lines.extend(step.command_lines)

        step.saved = [0, 1, 2]
        return step

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def _build_link0_lines(self) -> list[str]:
        """Reconstruct link0 command lines from current attribute values."""
        lines: list[str] = []
        if self.proc and self.proc > 0:
            lines.append(f"%nprocshared={self.proc}")
        if self.mem_mb and self.mem_mb > 0:
            lines.append(f"%mem={self.mem_mb}MB")
        if self.rwf:
            lines.append(f"%rwf={self.rwf}")
        if self.nosave:
            lines.append("%nosave")
        if self.oldchk:
            lines.append(f"%oldchk={self.oldchk}")
        if self.chk:
            lines.append(f"%chk={self.chk}")
        for extra in self.extra_link0:
            lines.append(extra)
        return lines

    def to_string(self) -> str:
        """Reconstruct this step's text from the parsed/modified data."""
        parts: list[str] = []

        # Annotation lines for this step
        for line in self.annotate_lines:
            parts.append(line)

        # Link0
        parts.extend(self._build_link0_lines())

        # Route (use route_dict's __str__ which reconstructs the route)
        route_body = str(self.route_dict).strip()
        parts.append(f"#p")
        if route_body:
            # Route_Dict __str__ already produces each keyword on its own line
            parts.append(route_body)

        if self.is_allcheck:
            # allcheck steps: just the "other" paragraphs follow, separated
            # by blank lines after the route section.
            parts.append("")  # blank line after route
            for paragraph in self.other:
                for line in paragraph:
                    parts.append(line)
                parts.append("")  # blank between paragraphs
        else:
            # Normal step with title + geometry + other
            parts.append("")  # blank line after route
            for line in self.title:
                parts.append(line)
            parts.append("")  # blank after title
            parts.append(f"{self.charge} {self.multiplet}")
            for line in self.geom_raw:
                parts.append(line)
            parts.append("")  # blank after geometry
            for paragraph in self.other:
                for line in paragraph:
                    parts.append(line)
                parts.append("")  # blank between paragraphs

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Utilities (kept from original)
    # ------------------------------------------------------------------
    @staticmethod
    def _join(item) -> str:
        ret = ""
        for i in item:
            if not isinstance(i, str):
                return repr(item)
            ret += i.strip() + '\n'
        return ret.strip()

    # Legacy alias
    def join(self, item):
        return self._join(item)

    # Legacy alias
    def phrase_annotates(self):
        return self._phrase_annotates()



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



class Route_Dict(collections.OrderedDict):
    def __init__(self, route_input: str, other_paragraph="", remove_genchk=True):
        """

        :param route_input:
        :param other_paragraph:
        :param remove_genchk:  产生输入的时候会自动去掉genchk，用于读取输出时应将此项设为False
        :return:
        """
        super(self.__class__, self).__init__()

        self.origin_route_input = route_input

        if isinstance(route_input, Route_Dict):  # 用于copy.deepcopy的复制
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
            # dict.fromkeys, not set(): de-duplicate while keeping the order the
            # options were written in.  set() ordering depends on the hash seed, so
            # it made re-reading and re-writing a file reshuffle "opt=(calcfc,ts)".
            self[key] = list(dict.fromkeys(self[key] + option))
        else:
            if isinstance(option, str):
                option = [option]
            self[key] = list(dict.fromkeys(option))

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

    @staticmethod
    def _option_name(option):
        # "maxstep=5" -> "maxstep"; "3/76=1000002000" -> "3/76"; "tight" -> "tight"
        return option.split('=', 1)[0].strip()

    @staticmethod
    def _normalize_options(key, options):
        if options is None:
            options = []
        elif isinstance(options, str):
            options = [options]
        else:
            options = list(options)
        options = [str(x).strip() for x in options]
        if key != 'external':
            options = [x.lower() for x in options]
        return [x for x in options if x]

    def set_keyword_option(self, key, options=None):
        """Value-aware create-or-replace for one route keyword.

        The keyword is created if absent.  Unlike :meth:`add_item` (which
        merges by exact string), options of the form ``name=value`` are
        matched by *name*: setting ``maxstep=10`` when the route has
        ``opt=maxstep=5`` yields ``opt=maxstep=10``, not both.  Plain options
        (``tight``) are appended only if not already present.  Existing option
        order is preserved; new options are appended.  IOp entries
        (``3/76=...``) follow the same name-matching rule.

        ``options=None``/``[]`` just ensures the bare keyword exists.
        ``level`` is special: it is replaced as a whole and accepts either
        ``[method, basis]`` or a single ``"method/basis"`` string.
        """
        key = key.strip().lower()
        if not key:
            return
        options = self._normalize_options(key, options)

        if key == "level":
            if len(options) == 1 and '/' in options[0]:
                options = [x.strip() for x in options[0].split('/', 1)]
            if len(options) != 2:
                raise ValueError("'level' requires [method, basis] or a 'method/basis' string")
            self["level"] = options
            return

        current = list(self[key]) if key in self else []
        for option in options:
            if '=' in option:
                name = self._option_name(option)
                current = [x for x in current if self._option_name(x) != name]
                current.append(option)
            elif option not in current:
                current.append(option)
        self[key] = current

    def clear_keyword(self, key):
        """Drop *key* together with all of its options, however many there are.

        E.g. ``clear_keyword('scrf')`` removes ``scrf=(solvent=water, smd)``
        entirely.  ``level`` is special: it is reset to the blank placeholder
        instead of removed, since much code indexes ``route_dict['level']``
        directly.
        """
        key = key.strip().lower()
        if not key:
            return
        if key == "level":
            self["level"] = ['Blank_Method', 'Black_Basis']
            return
        if key in self:
            self.pop(key)

    def remove_keyword_option(self, key, options=None):
        """Value-aware removal of some options of a route keyword.

        An option given without ``=`` also removes any ``name=value`` option
        of the same name (e.g. removing ``maxstep`` drops ``maxstep=5``,
        removing ``5/13`` drops ``5/13=1``).  A keyword left with no options
        is dropped, except the job-type keywords ``opt``/``freq``/``scan``/
        ``irc`` which stay as bare keywords.

        With ``options=None`` this delegates to :meth:`clear_keyword` (whole
        keyword dropped; ``level`` reset to the blank placeholder).
        """
        key = key.strip().lower()
        if not key:
            return
        options = self._normalize_options(key, options)

        if key == "level" or not options:
            self.clear_keyword(key)
            return
        if key not in self:
            return
        names_to_remove = {self._option_name(x) for x in options if '=' not in x}
        remaining = [
            x for x in self[key]
            if x not in options and self._option_name(x) not in names_to_remove
        ]
        if remaining or key in ['opt', 'freq', 'scan', 'irc']:
            self[key] = remaining
        else:
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

    # ------------------------------------------------------------------
    # IOp(9/40) —— TDDFT 输出里激发组分（CI 展开系数）的打印阈值
    # ------------------------------------------------------------------
    # 依据：Gaussian 16 Rev. C.01 的 IOps Reference，L913 / L914 条目
    # （见 Manual_Gaussian_Full.md 的 "##### IOp(9/40)" 一节）：
    #
    #     L913, L914: Threshold for printing eigenvector components.
    #         0  ->  ITHR = 1
    #         N  ->  ITHR = N
    #     Where threshold = GFLOAT(10)^-ITHR
    #
    # 也就是阈值 = 10^-ITHR：IOp(9/40=5) 对应 1E-5，IOp(9/40=2) 对应 0.01。
    # N 取 0 时 ITHR 取 1，阈值是 0.1——与不写这个 IOp 时的打印行为相同。手册
    # 正文（TD / CIS 输出一节）的说法与之一致："use IOp(9/40=N) to request more
    # coefficients: all that are greater than 10^-N"。
    #
    # 【重载警告】同一个 IOp(9/40) 在 **L906（MP2）** 里含义完全不同——那里它选
    # 的是 MP2 的参考波函数（0=默认 HF，1=CASSCF，2=HF），与打印阈值无关。下面
    # 这些属性一律按「打印阈值」解读，因为它们服务的是 TDDFT 输出瘦身这一个用
    # 途。给一个纯 MP2 任务读出来的值没有意义（实践中也没人给 MP2 写这个 IOp，
    # 默认就是 HF）。

    #: 不写 IOp(9/40) 时 Gaussian 自己使用的 CI 系数打印阈值（= IOp(9/40=0) 的
    #: ITHR=1 所对应的 10^-1）。
    IOP_9_40_DEFAULT_THRESHOLD = 0.1

    #: 匹配 IOp 关键词的一条选项，例如 ``9/40=5``（本类解析时把选项统一转成
    #: 小写并保留原有的空格写法，所以这里对空格宽松处理）。
    _IOP_9_40_OPTION_PATTERN = re.compile(r"^\s*9\s*/\s*40\s*=\s*(-?\d+)\s*$")

    @property
    def iop_9_40_value(self) -> int | None:
        """本路由里 ``IOp(9/40=N)`` 的 N；没写这个 IOp 时为 ``None``。

        同一条路由里写了多个 9/40 时取最后一个（与 Gaussian 「后写的覆盖先
        写的」一致）。
        """
        iop_options = self.get("iop") or []
        if isinstance(iop_options, str):
            iop_options = [iop_options]
        found = None
        for option in iop_options:
            match = self._IOP_9_40_OPTION_PATTERN.match(str(option))
            if match:
                found = int(match.group(1))
        return found

    @property
    def iop_9_40_threshold(self) -> float | None:
        """本路由 ``IOp(9/40=N)`` 对应的 CI 展开系数打印阈值 ``10^-N``。

        没写这个 IOp 时为 ``None``。``IOp(9/40=0)`` 按手册取 ITHR=1，返回
        :data:`IOP_9_40_DEFAULT_THRESHOLD`（0.1，与不写这个 IOp 时相同）。

        注意这个 IOp 在 L906（MP2）里是另一个含义（参考波函数选择），本属性
        一律按 L913 / L914 的打印阈值解读——详见本节开头的注释。
        """
        value = self.iop_9_40_value
        if value is None:
            return None
        if value == 0:
            return self.IOP_9_40_DEFAULT_THRESHOLD
        return 10.0 ** (-value)

    @property
    def has_iop_9_40(self) -> bool:
        """本路由是否写了 ``IOp(9/40=...)``。"""
        return self.iop_9_40_value is not None

    @staticmethod
    def smallest_iop_9_40_threshold(thresholds) -> float | None:
        """多步任务的整体阈值：取各步骤里最小的那一个（打印得最详细的那一步）。

        供 ``Gaussian_Input`` / ``Gaussian_Output`` 聚合各步骤的
        ``iop_9_40_threshold`` 用。
        """
        present = [threshold for threshold in thresholds if threshold is not None]
        return min(present) if present else None

    # ------------------------------------------------------------------
    # 激发态任务的识别
    # ------------------------------------------------------------------
    #: 会让 Gaussian 打印激发态 CI 展开系数的路由关键词——也就是上面 IOp(9/40)
    #: 控制打印阈值的那些组分行。只有这类任务的输出才谈得上「TDDFT 输出瘦身」。
    EXCITED_STATE_KEYWORDS = ("td", "tda", "cis")

    #: 出现在**方法位置**时同样意味着激发态任务的方法名（小写、不带括号里的
    #: 选项）。``CIS`` 与 ``TD`` 不一样：它是一个方法而不是独立的关键词，写成
    #: ``CIS/6-31G*`` 时会被归进 ``level``（``[方法, 基组]``），在 keyword 里
    #: 根本找不到，所以必须单独看方法那一半。
    EXCITED_STATE_METHODS = ("cis",)

    #: 方法名可以带的自旋前缀（``UCIS`` / ``RCIS`` / ``ROCIS``）。比对方法名
    #: 之前先剥掉。
    _SPIN_METHOD_PREFIXES = ("ro", "u", "r")

    def _bare_method_name(self) -> str:
        """取出路由里方法那一半的裸名字：去掉括号选项与自旋前缀，转成小写。

        ``CIS(NStates=10)/6-31G*`` → ``cis``；``UCIS/6-31G*`` → ``cis``；
        路由里没有 ``方法/基组`` 写法时返回空字符串。
        """
        level = self.get("level") or []
        if isinstance(level, str):
            level = [level]
        if not level:
            return ""
        method = str(level[0]).split("(", 1)[0].strip().lower()
        for prefix in self._SPIN_METHOD_PREFIXES:
            if method.startswith(prefix) and len(method) > len(prefix):
                return method[len(prefix):]
        return method

    @property
    def is_excited_state(self) -> bool:
        """本路由是不是激发态任务（``TD`` / ``TDA`` / ``CIS``）。

        两处都要看：``TD`` / ``TDA`` 是独立的路由关键词，而 ``CIS`` 通常写在
        方法位置（``CIS/6-31G*``），解析后落在 ``level`` 里而不在 keyword 里。
        """
        if any(keyword in self for keyword in self.EXCITED_STATE_KEYWORDS):
            return True
        return self._bare_method_name() in self.EXCITED_STATE_METHODS


# Solvents recognised by Gaussian's SCRF ``solvent=`` option, verbatim from
# https://gaussian.com/scrf/ .  Matching (see build_Gaussian_input_from_template) is
# case-insensitive; Gaussian itself accepts these names in any case.


def build_Gaussian_input_from_template(
    method_template,
    coordinates: Coordinates,
    output_path=None,
    *,
    charge: int | None = None,
    multiplet: int | None = None,
    solvent: str | None = None,
    modredundant=None,
    remove_atoms: str | list[int] | None = None,
    include_atoms: str | list[int] | None = None,
    change_elements: dict[int, str] | None = None,
    nprocshared: int | None = None,
    mem_gb: float | None = None,
    mem_mb: int | None = None,
    title=None,
    first_step_oldchk: str | None = None,
    delete_rwf_after: bool = True,
    path_mappings: list[tuple[str, str, str]] | None = None,
    remove_first_geom_allcheck: bool = True,
    validate_solvent: bool = True,
) -> "Gaussian_Input":
    """
    Merge a Gaussian method/route template with a geometry to produce a ready-to-run
    Gaussian input (``.gjf``).

    Standalone, GUI-free reproduction of the QM_Creater "load a route/method template,
    drop in a geometry, tweak a few settings, write a runnable gjf" workflow.

    The *method_template* supplies *how* to compute — the multi-step ``--Link1--``
    structure, per-step routes/keywords, basis-set / ECP paragraphs and checkpoint
    chaining.  The *coordinates* supplies *what* to compute — inserted as the explicit
    geometry of the **first** step only; later steps keep their template route and read
    the geometry from the checkpoint (``geom=allcheck``), as in a normal multi-step job.

    Template placeholders and per-step ``!__KEY__=VALUE`` annotations are honoured the
    same way QM_Creater does:

    * ``solvent=EDITTHIS`` in any ``scrf`` keyword is filled from *solvent* — and ONLY
      the placeholder: a concrete solvent already in the template is never replaced.
    * A ``B EDITTHIS F`` (modredundant) section is replaced by *modredundant*; any
      ``B i j F`` bond freezes are additionally recorded in the title's ``[FROZEN_BONDS]``.
    * A ``Blank_Method`` / ``Black_Basis`` placeholder half in the method/basis slot is
      dropped (``PM6D3/Black_Basis`` serialises as ``pm6d3`` alone); the fully blank
      pair — what keyword-only routes parse into — is omitted entirely.
    * ``!__FCHK_TAG__=[label]`` on a step is baked into that step's ``%chk`` filename.
      A tag used as an ``[EXTRACT_GEOM]`` label must not be pure numeric and must not
      contain a comma (validated, as QM_Creater).
    * ``!__EXTRACT_GEOM__=TRUE`` steps are collected into the title's ``[EXTRACT_GEOM]``
      directive (by FCHK_TAG label, or by step number for the first step).
    * ``!__CHK_READ__=<offset>`` makes a step's ``%oldchk`` point *offset* steps back
      (e.g. an IRC step reading the TS force constants) instead of the previous step.
    * ``!__NAMETAG__=`` is preserved; the consumed annotations above are stripped.
      The tag is also stamped into the output filename (see *output_path* below),
      and the default *title* and the ``%chk`` basename derive from the final filename.

    Any ``editthis`` placeholder left unresolved raises :class:`ValueError`, mirroring the
    GUI's refusal to save such a file.

    The ``%chk`` names are always derived from the file actually written — there is no
    argument to name them independently, so the checkpoints always sit next to their
    input.  Numbering follows QM_Creater exactly: the first step is
    ``<base>_Step0[FCHK_TAG].chk`` and every later step with 0-based index *i* is
    ``<base>_Step<i+1>[FCHK_TAG].chk`` — so a multi-step job is named ``_Step0``,
    ``_Step2``, ``_Step3``, ... and ``_Step1`` never occurs.  ``!__CHK_READ__``
    offsets keep resolving by step index, independent of these display numbers.  The
    base is the resolved *output_path* without its extension (directories included),
    falling back to the NAMETAG and then to ``"Gaussian_job"`` when no output location
    was available.

    The function never modifies or overwrites an existing file: whether *output_path*
    was given explicitly or derived from the geometry source, a name clash shifts the
    output to ``<name>_01``, ``<name>_02``, ... (two-digit zero padding; a stem
    already ending in ``_<number>`` counts up from there) until an unused name is
    found — see :func:`Python_Lib.My_Lib_File.get_unused_filename` with
    ``continue_number_suffix=True``.  The written file ends with seven blank lines,
    as QM_Creater (Gaussian requires at least one after the last input section).

    Args:
        method_template:  A ``.gjf`` file path (``str`` / ``os.PathLike``), an already
                          parsed :class:`Gaussian_Input` (deep-copied, never mutated), or
                          raw gjf text.
        coordinates:      A :class:`Coordinates` object (charge / multiplicity taken from
                          it unless overridden) or a list of raw coordinate-line strings /
                          :class:`std_coordinate` objects.
        output_path:      Where to write the finished input.  The template ``!__NAMETAG__``
                          is stamped into this filename first, by the same rule
                          QM_Creater applies: the last ``[...]`` segment of the name
                          containing at
                          least one letter is replaced by the tag, or the tag is appended
                          when no such segment exists — letter-less segments like a
                          structure numbering ``[000001_000002]`` are never touched (e.g.
                          template tag ``[Complete_PBE0_defTZVP]`` turns
                          ``3300_3325_Cont.gjf`` into
                          ``3300_3325_Cont[Complete_PBE0_defTZVP].gjf`` and
                          ``3300_3325[old]_Cont.gjf`` into
                          ``3300_3325[Complete_PBE0_defTZVP]_Cont.gjf``).
                          Default: derived from the geometry source — when *coordinates*
                          carries the file it was read from (``Coordinates.source_path``,
                          stamped by :class:`Gaussian_Input` / :class:`Gaussian_Output`),
                          the input is written next to that file with the stamped source
                          stem plus ``.gjf``.  Here a source ``.gjf`` that itself carries
                          a ``!__NAMETAG__=`` annotation whose tag appears in its
                          filename — i.e. a previous product of this function — gets
                          exactly that segment replaced (geometry from
                          ``Mol[000001_000002]_opt[Opt_wB97XD_def2SV_Crude]_01.gjf``
                          with the same template goes to
                          ``..._opt[Opt_wB97XD_def2SV_Crude]_02.gjf`` via the name
                          clash rule below); without such an annotation the
                          last-letter-segment / append rule above applies (geometry from
                          ``Mol[000001_000002]_opt.gjf`` is written to
                          ``Mol[000001_000002]_opt[Opt_wB97XD_def2SV_Crude].gjf``).
                          When the source file is unknown, nothing is written and the
                          merged object is only returned.  The path actually written is
                          recorded in the returned object's ``.path`` (``None`` when
                          nothing was written).
        charge, multiplet:  Override the charge / spin multiplicity (default: from
                          *coordinates*).  Whatever the two end up being, they are
                          checked before anything is written (see *Raises* below) —
                          this function never emits a file whose charge/multiplicity
                          line is a leftover placeholder or contradicts the electron
                          count of the geometry it ships with.
        solvent:          Fills every ``solvent=EDITTHIS`` placeholder in the template's
                          ``scrf`` keywords (see :data:`GAUSSIAN_SCRF_SOLVENTS`;
                          validated case-insensitively unless *validate_solvent* is
                          False).  QM_Creater semantics: only the placeholder is ever
                          filled — a concrete solvent already written in the template
                          is never replaced, and a *solvent* argument with no
                          placeholder to fill is silently ignored.
        modredundant:     Modredundant directive lines — a multi-line ``str`` or list of
                          ``str``, e.g. ``["B 1 2 F", "B 2 3 F", "D 1 2 4 5 S 10 0.1"]``.
                          Replaces the ``... EDITTHIS ...`` section of the first step (or is
                          inserted right after the geometry when there is none).  See the
                          ModRedundant section of https://gaussian.com/opt/ .
        remove_atoms:     Atoms to delete from the inserted geometry — 1-based atom
                          numbers as a list of ints, or a selection string like
                          ``"1,5,7-9"``.  Mutually exclusive with *include_atoms*.
        include_atoms:    Keep only these atoms of the inserted geometry (same formats
                          as *remove_atoms*; mutually exclusive with it).
        change_elements:  ``{atom number (1-based): new element symbol}`` — replace the
                          element of the given atoms, e.g. ``{12: "Si"}``.  Numbering
                          refers to the original geometry, and the change applies
                          before *remove_atoms* / *include_atoms* filtering (as
                          QM_Creater).  None of the three editing arguments adjusts
                          charge or spin multiplicity — pass *charge* / *multiplet*
                          explicitly when the edit changes the electron count.
        nprocshared:      Override ``%nprocshared`` for every step.  When neither the
                          template nor this argument declares it, **no** ``%nprocshared``
                          line is written — the core count is left to the submitting
                          machinery, never invented here.
        mem_gb, mem_mb:   Override ``%mem`` for every step (MB wins over GB).  Same as
                          *nprocshared*: nothing declared → no ``%mem`` line.
        title:            Override the molecule-name title line (``str`` or list of lines);
                          the ``[FROZEN_BONDS]`` / ``[EXTRACT_GEOM]`` directive line is
                          appended automatically.  Default: output filename stem, else the
                          NAMETAG, else ``"structure"``.
        first_step_oldchk:  If given, the ``%oldchk`` path of the first step (e.g. to read
                          an initial guess / geometry from a pre-existing checkpoint) —
                          written verbatim into the ``%oldchk`` line, then subject to the
                          *path_mappings* conversion like every other path in the file.
                          The ``%chk`` names themselves are never taken from an argument;
                          they always derive from the output filename (see above).
        delete_rwf_after: Every step gets ``%rwf=`` pointing to one shared read-write
                          file per job: the ``%chk`` basename relocated through
                          *path_mappings* onto the matching remote RWF prefix (e.g.
                          ``D:\\Gaussian\\proj\\job`` →
                          ``%HOME%/Gaussian_RWF/proj/job``, ``E:\\proj\\job`` →
                          ``%HOME%/Gaussian_RWF/proj/job``) plus ``.rwf``.
                          When that basename matches no mapping (or *path_mappings*
                          is None), the rwf simply sits next to the chk files.
                          When True (default), a ``%nosave`` line is written
                          immediately below the ``%rwf`` line and above ``%oldchk`` /
                          ``%chk`` — ``%nosave`` only affects the files declared
                          **above** it, so exactly the rwf is deleted when the job
                          ends and the chk files are kept.
        path_mappings:    List of ``(local_prefix, remote_work_prefix,
                          remote_rwf_prefix)`` triples.  Default ``None`` loads the
                          user-level configuration — the variable
                          ``LOCAL_TO_REMOTE_PATH_MAPPINGS`` in
                          ``<My_Program>/My_Lib_Configuration_Private.py``, via
                          :func:`Python_Lib.My_Lib_File.local_to_remote_path_mappings`
                          (a missing configuration raises immediately rather than
                          silently keeping local paths).  Serves two jobs:
                          (1) the shared ``%rwf`` is placed under the matching remote
                          RWF prefix (see *delete_rwf_after*); (2) the finished input
                          is converted to cluster-side paths — every occurrence of a
                          local prefix (either slash direction, case-insensitive,
                          followed by a path separator) becomes its remote **work**
                          prefix, then every remaining backslash becomes a forward
                          slash.  Earlier entries win, so keep more specific prefixes
                          first.  The remote prefixes use the ``%HOME%`` home-path
                          placeholder that HPC_Lib replaces at submission (legacy
                          files carrying ``/home/gauuser`` are replaced there too).
                          Applied to the in-memory object (``%chk`` / ``%oldchk`` /
                          ``%rwf`` lines, annotations, trailing sections), so the
                          returned object matches the written file.  Pass an empty
                          list to disable and keep local Windows paths.
        remove_first_geom_allcheck:  When True (default) also drop ``guess=tcheck`` from
                          the first step (``geom=allcheck`` is always removed there once an
                          explicit geometry is present).
        validate_solvent:  When True (default) reject a *solvent* that is not a recognised
                          Gaussian SCRF solvent.

    Returns:
        The merged :class:`Gaussian_Input`, already written to the explicit or derived
        *output_path*; its ``.path`` records the file actually written (``None`` when
        no output location was available).

    Raises:
        ValueError: When the charge / spin multiplicity of any step carrying an explicit
            geometry cannot be written as it stands — an unresolved 999 / 99 placeholder,
            a multiplicity that is not a positive integer, or a multiplicity whose parity
            contradicts the electron count (see
            :meth:`Coordinates.charge_and_multiplicity_problem`).  The check runs after
            the *remove_atoms* / *include_atoms* / *change_elements* editing, so it sees
            the electron count the file will actually declare, and it cannot be switched
            off.  Also raised for an unrecognised *solvent*, an unresolved ``EDITTHIS``
            placeholder, and the other template problems described above.

    Example::

        coord = Gaussian_Output("ts_freq.out").steps[-1].summary.coordinate
        gaussian_input = build_Gaussian_input_from_template(
            r"D:\\Gaussian\\0StdRoute\\Complete_PBE0_defTZVP_LongIRC.gjf",
            coord,
            output_path="ts_run.gjf",
            solvent="DiMethylSulfoxide",
            modredundant=["B 12 13 F"],
            nprocshared=16, mem_gb=32,
        )
    """
    # --- 0. validate solvent up front ----------------------------------------------
    if solvent is not None and validate_solvent:
        if solvent.strip().lower() not in _GAUSSIAN_SCRF_SOLVENTS_LOWER:
            raise ValueError(
                f"{solvent!r} is not a recognised Gaussian SCRF solvent (see "
                "GAUSSIAN_SCRF_SOLVENTS / https://gaussian.com/scrf/). Pass "
                "validate_solvent=False to allow a custom user-defined SMD solvent."
            )

    # --- 1. obtain a private, mutable Gaussian_Input from the template --------------
    if isinstance(method_template, Gaussian_Input):
        gaussian_input = method_template.extract_steps("ALL")           # deep copy, never mutate caller
    elif isinstance(method_template, (str, os.PathLike)) and os.path.isfile(method_template):
        gaussian_input = Gaussian_Input(str(method_template))
    elif isinstance(method_template, str):
        gaussian_input = Gaussian_Input(method_template, is_text=True)   # treat as raw gjf text
    else:
        raise TypeError(
            "method_template must be a file path, a Gaussian_Input, or gjf text, "
            f"got {type(method_template).__name__}"
        )
    if not gaussian_input.steps:
        raise ValueError("The method template contains no steps.")

    # --- 1b. resolve the output path -------------------------------------------------
    # The template NAMETAG is stamped into the filename via _stamp_nametag_into_filename
    # (basename only, never the directories).  Explicit output_path: no source tag is
    # consulted — the last letter-containing [...] segment of the given name is
    # replaced, else the tag is appended.  No output_path: derive from the file the
    # geometry was read from, when known — same directory, stamped source stem +
    # ".gjf"; a source .gjf carrying its own !__NAMETAG__ gets exactly that segment
    # replaced.  Either way a name clash shifts the output to <name>_01 / _02 / ... —
    # this function never overwrites an existing file.  title / chk_basename below
    # derive from the final name.
    gaussian_input.path = None  # the merged input is a new document; .save() must never hit the template
    _nametag = (gaussian_input.nametag or "").strip()
    if output_path is not None:
        _directory, _basename = os.path.split(str(output_path))
        _stem, _ext = os.path.splitext(_basename)
        _stem = _stamp_nametag_into_filename(_stem, _nametag)
        output_path = os.path.join(_directory, _stem + _ext)
    else:
        _geometry_source = getattr(coordinates, "source_path", None)
        if _geometry_source:
            _geometry_source = str(_geometry_source)
            _directory, _basename = os.path.split(_geometry_source)
            _stem, _source_extension = os.path.splitext(_basename)
            _source_nametag = ""
            if _source_extension.lower() in (".gjf", ".com") and os.path.isfile(_geometry_source):
                _source_nametag = (Gaussian_Input(_geometry_source).nametag or "").strip()
            _stem = _stamp_nametag_into_filename(_stem, _nametag, _source_nametag)
            output_path = os.path.join(_directory, _stem + ".gjf")
    if output_path is not None:
        output_path = get_unused_filename(str(output_path), continue_number_suffix=True)

    # --- 2. insert the geometry into the first step --------------------------------
    # set_geometry pulls charge/multiplicity from a Coordinates object when not given
    # explicitly, and always strips geom=allcheck from the step it writes into.
    gaussian_input.set_geometry(0, coordinates, charge=charge, multiplet=multiplet)
    if remove_first_geom_allcheck:
        # an explicit geometry makes a checkpoint-based initial guess invalid
        gaussian_input.steps[0].remove_keyword_option("guess", "tcheck")

    # --- 2b. optional geometry editing (QM_Creater's save-time atom tools) ----------
    # Element changes apply by the ORIGINAL 1-based numbering, before any filtering,
    # exactly as QM_Creater iterates the untouched geometry lines.
    if remove_atoms is not None and include_atoms is not None:
        raise ValueError("remove_atoms and include_atoms are mutually exclusive.")
    if remove_atoms is not None or include_atoms is not None or change_elements:
        step0 = gaussian_input.steps[0]
        atom_count = len(step0.geom_raw)

        def _atom_indices(selection) -> set[int]:
            """1-based atom numbers (or a '1,5,7-9' selection string) -> 0-based indices."""
            if isinstance(selection, str):
                indices = set(phrase_range_selection(selection))
            else:
                indices = {int(number) - 1 for number in selection}
            out_of_range = sorted(index + 1 for index in indices if index < 0 or index >= atom_count)
            if out_of_range:
                raise ValueError(f"Atom number(s) {out_of_range} out of range (1..{atom_count}).")
            return indices

        removed_indices = _atom_indices(remove_atoms) if remove_atoms is not None else None
        included_indices = _atom_indices(include_atoms) if include_atoms is not None else None
        element_by_index = {}
        if change_elements:
            _atom_indices(list(change_elements.keys()))
            element_by_index = {int(number) - 1: str(element).strip()
                                for number, element in change_elements.items()}

        new_geometry_lines = []
        for index, line in enumerate(step0.geom_raw):
            if removed_indices is not None and index in removed_indices:
                continue
            if included_indices is not None and index not in included_indices:
                continue
            if index in element_by_index:
                component = line.split()
                line = "\t".join([element_by_index[index]] + component[1:])
            new_geometry_lines.append(line)
        if not new_geometry_lines:
            raise ValueError("Geometry editing removed every atom.")
        step0.set_geometry(new_geometry_lines)

    # --- 2c. charge / spin multiplicity must be known and physically possible --------
    # Placed after the geometry editing above, because removing atoms or swapping
    # elements changes the electron count and only the final geometry can be judged.
    # Every step that writes its own charge/multiplicity line is checked; a geom=allcheck
    # step writes none and has no Coordinates of its own.  There is no way to switch this
    # off: an unresolved 999 placeholder or a multiplicity contradicting the electron
    # count always yields a file Gaussian refuses to run, so it is reported here rather
    # than after the job has been queued on a cluster.
    for index, step in enumerate(gaussian_input.steps):
        if step.is_allcheck or step.coordinate is None:
            continue
        problem = step.coordinate.charge_and_multiplicity_problem()
        if problem is not None:
            raise ValueError(
                f"Refusing to build the Gaussian input: in step {index}, {problem}. "
                f"Pass charge=... / multiplet=... to set them explicitly, or take the "
                f"geometry from a source that carries the intended values."
            )

    # --- 3. solvent: fill solvent=EDITTHIS placeholders only (QM_Creater semantics) --
    # A concrete solvent already in the template is never replaced; a solvent argument
    # with no placeholder to fill is silently ignored, exactly as QM_Creater.
    if solvent is not None:
        for step in gaussian_input.steps:
            scrf_options = step.route_dict.get("scrf")
            if scrf_options is None:
                continue  # only touch steps that already request solvation
            step.route_dict["scrf"] = [
                "solvent=" + solvent if str(option).strip().lower() == "solvent=editthis" else option
                for option in scrf_options
            ]

    # --- 3b. Blank_Method / Black_Basis placeholder halves (QM_Creater cleanup) ------
    # A template route like "PM6D3/Black_Basis" means "method only, no basis slot":
    # the placeholder half is dropped and the real half survives as a bare route
    # keyword.  The fully blank pair (what keyword-only routes parse into) is
    # normalised to the canonical spelling that Route_Dict.__str__ omits entirely.
    # Route parsing lowercases options, so the placeholders are matched insensitively.
    for step in gaussian_input.steps:
        level = step.route_dict.get("level")
        if not level or len(level) != 2:
            continue
        method_is_blank = str(level[0]).strip().lower() == "blank_method"
        basis_is_blank = str(level[1]).strip().lower() in ("black_basis", "blank_basis")
        if method_is_blank and basis_is_blank:
            step.route_dict["level"] = ["Blank_Method", "Black_Basis"]
        elif basis_is_blank or method_is_blank:
            surviving_half = str(level[0] if basis_is_blank else level[1]).strip().lower()
            step.route_dict.pop("level")
            step.route_dict[surviving_half] = []
            step.route_dict.move_to_end(surviving_half, last=False)

    # --- 4. modredundant section + [FROZEN_BONDS] metadata --------------------------
    modredundant_lines = _template_directive_lines(modredundant)
    frozen_bonds: list[list[int]] = []
    for line in modredundant_lines:
        m = re.findall(r"\bB\s+(\d+)\s+(\d+)\s+F\b", line, re.IGNORECASE)
        if m:
            frozen_bonds.append([int(m[0][0]) - 1, int(m[0][1]) - 1])  # 0-indexed, as QM_Creater
    if modredundant_lines:
        step0 = gaussian_input.steps[0]
        for i, paragraph in enumerate(step0.other):
            if any("editthis" in ln.lower() for ln in paragraph):
                step0.other[i] = list(modredundant_lines)   # replace the placeholder block
                break
        else:
            step0.other.insert(0, list(modredundant_lines))  # no placeholder: add after geometry
        step0.other_str = "".join(
            "".join(ln + "\n" for ln in para) + "\n" for para in step0.other
        )

    # --- 5. resources --------------------------------------------------------------
    if nprocshared is not None:
        gaussian_input.set_nprocshared("ALL", int(nprocshared))
    if mem_mb is not None:
        gaussian_input.set_mem("ALL", int(mem_mb))
    elif mem_gb is not None:
        gaussian_input.set_mem("ALL", int(round(float(mem_gb) * 1024)))

    # --- 6. %chk names (carry the FCHK_TAG) and %oldchk chaining --------------------
    # The chk basename is always the name of the file actually written — no argument can
    # decouple the two, so the checkpoints always sit next to their input.
    # %oldchk follows a CHK_READ annotation (a relative step offset, e.g. an IRC step
    # reading the TS force constants) when present, otherwise the immediately prior step;
    # only the FIRST step's %oldchk can be set from outside, via first_step_oldchk.
    if output_path is not None:
        chk_basename = os.path.splitext(str(output_path))[0]
    else:
        chk_basename = (gaussian_input.nametag or "").strip().strip("[]") or "Gaussian_job"
    chk_names: dict[int, str] = {}
    # One shared %rwf per job: chk_basename relocated through path_mappings onto the
    # matching remote RWF prefix (D:\Gaussian\... and E:\... both land under
    # %HOME%/Gaussian_RWF/...); outside every mapping the rwf sits next to the
    # chk files.  %nosave only affects files declared above it, so serialisation keeps
    # it directly below %rwf and above %oldchk/%chk: the rwf dies with the job, the
    # chks survive.
    if path_mappings is None:
        # 未显式给出映射时读用户配置（配置缺失直接报错，不静默保持本地路径）。
        # 明确不要做路径转换时传空列表。
        path_mappings = local_to_remote_path_mappings()
    mapped_remote_paths = (
        map_local_path_to_remote(chk_basename, path_mappings)
        if path_mappings else None
    )
    rwf_name = (mapped_remote_paths[1] if mapped_remote_paths else chk_basename) + ".rwf"
    for index, step in enumerate(gaussian_input.steps):
        fchk_tag = step.annotates_dict.get("FCHK_TAG", "") or ""
        # QM_Creater's step numbering: the first step is _Step0, every later step with
        # 0-based index i is _Step<i+1> — _Step1 never occurs.  Kept identical so
        # downstream tooling that matches chk names by _StepN sees the same names.
        step_number = 0 if index == 0 else index + 1
        chk_names[index] = f"{chk_basename}_Step{step_number}{fchk_tag}.chk"
        step.set_chk(chk_names[index])
        step.set_rwf(rwf_name)
        step.nosave = delete_rwf_after
    for index, step in enumerate(gaussian_input.steps):
        if index == 0:
            if first_step_oldchk:
                step.set_oldchk(str(first_step_oldchk))
            continue
        chk_read = step.annotates_dict.get("CHK_READ")
        if chk_read is not None and re.fullmatch(r"-?\d+", str(chk_read).strip()):
            target = index + int(chk_read)
            if target not in chk_names:
                raise ValueError(
                    f"Step {index}: CHK_READ={chk_read} points to non-existent step {target}."
                )
            step.set_oldchk(chk_names[target])
        else:
            step.set_oldchk(chk_names[index - 1])

    # --- 7. first-step title: molecule name + [FROZEN_BONDS]/[EXTRACT_GEOM] ----------
    if title is not None:
        name_line = title if isinstance(title, str) else " ".join(str(t) for t in title)
    elif output_path is not None:
        name_line = os.path.splitext(os.path.basename(str(output_path)))[0]
    else:
        name_line = (gaussian_input.nametag or "").strip().strip("[]") or "structure"

    extract_labels: list[str] = []
    for index, step in enumerate(gaussian_input.steps):
        if str(step.annotates_dict.get("EXTRACT_GEOM", "")).strip().lower() == "true":
            tag = step.annotates_dict.get("FCHK_TAG", "") if index > 0 else ""
            label = tag.strip().strip("[]") if tag else str(index + 1)
            if tag:
                # as QM_Creater: [EXTRACT_GEOM] labels are comma-separated, and numeric
                # labels are reserved for step numbers
                if re.fullmatch(r"-?\d+", label):
                    raise ValueError(f"FCHK_TAG cannot be pure numeric: {label!r}.")
                if "," in label:
                    raise ValueError(f"FCHK_TAG cannot contain a comma: {label!r}.")
            extract_labels.append(label)

    directive_parts = []
    if frozen_bonds:
        directive_parts.append("[FROZEN_BONDS]:" + repr(frozen_bonds) + "[/FROZEN_BONDS]")
    if extract_labels:
        directive_parts.append("[EXTRACT_GEOM]:" + ",".join(extract_labels))
    gaussian_input.steps[0].title = [name_line] + ([" ".join(directive_parts)] if directive_parts else [])

    # --- 8. drop consumed annotations (keep NAMETAG and !RUN commands) --------------
    for step in gaussian_input.steps:
        step.annotate_lines = [
            line for line in step.annotate_lines
            if not (
                re.match(r"!__(.+?)=", line)
                and re.match(r"!__(.+?)=", line).group(1).rstrip("_") in _CONSUMED_ANNOTATION_KEYS
            )
        ]
        for key in _CONSUMED_ANNOTATION_KEYS:
            step.annotates_dict.pop(key, None)

    # --- 8b. Windows -> Linux path conversion (QM_Creater's Linux mode) --------------
    # Applied to the in-memory object rather than the serialised text, so the returned
    # object always matches the written file.  Each local prefix in path_mappings is
    # rewritten to its remote WORK prefix (the rwf was already placed under the remote
    # RWF prefix above and is untouched here).
    if path_mappings:
        prefix_patterns = [
            (local_path_prefix_pattern(local_prefix), remote_work_prefix)
            for local_prefix, remote_work_prefix, _remote_rwf_prefix in path_mappings
        ]

        def _to_linux(text: str) -> str:
            for pattern, remote_work_prefix in prefix_patterns:
                text = pattern.sub(remote_work_prefix.replace("\\", r"\\"), text)
            return text.replace("\\", "/")

        for step in gaussian_input.steps:
            step.chk = _to_linux(step.chk)
            step.oldchk = _to_linux(step.oldchk)
            step.rwf = _to_linux(step.rwf)
            step.extra_link0 = [_to_linux(line) for line in step.extra_link0]
            step.annotate_lines = [_to_linux(line) for line in step.annotate_lines]
            step.command_lines = [_to_linux(line) for line in step.command_lines]
            step.other = [[_to_linux(line) for line in paragraph] for paragraph in step.other]
            step.other_str = "".join(
                "".join(line + "\n" for line in paragraph) + "\n" for paragraph in step.other
            )

    # --- 9. refuse to emit an unresolved 'editthis' placeholder (as the GUI does) ---
    final_text = gaussian_input.to_string()
    if "editthis" in final_text.lower():
        lowered = final_text.lower()
        if "solvent=editthis" in lowered:
            raise ValueError(
                "The template still contains 'solvent=editthis'. Pass solvent=... to fill "
                "it in (see GAUSSIAN_SCRF_SOLVENTS)."
            )
        pos = lowered.find("editthis")
        context = final_text[max(0, pos - 30):pos + 38]
        raise ValueError(
            "The template still contains an unresolved 'editthis' placeholder near:\n"
            f"    ...{context}...\n"
            "e.g. pass modredundant=[...] to fill a 'B EDITTHIS F' section."
        )

    # --- 10. write out (explicit or derived output path, resolved in step 1b) --------
    if output_path is not None:
        with open(str(output_path), "w", encoding="utf-8", newline="\n") as output_file:
            # last-line terminator + 7 blank lines, as QM_Creater (Gaussian requires
            # at least one blank line after the last input section)
            output_file.write(final_text.rstrip("\n") + "\n" * 8)
        gaussian_input.path = str(output_path)  # record the actual path written

    return gaussian_input


class Gaussian_Summary:
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
        self.route_dict = Route_Dict(self.route)

        self.name = self.summary[2]
        self.charge, self.multiplet = self.summary[3][0].split(',')

        self.coordinate = Coordinates(self.summary[3][1:], self.charge, self.multiplet)

        self.results = self.summary[4]
        self.results = {x.split("=")[0]: x.split("=")[1] for x in self.results}
        # contains "HF", "ZeroPoint","Thermal","NImag"


def read_output_tail_lines(filename, line_count, encoding_errors='strict'):
    """读取输出文件末尾的 line_count 行（从文件尾部倒着读，不整份读文件）。

    ``Gaussian_Output.tail_lines`` 只常驻 ``retained_tail_line_count`` 行；界面上
    要看更多行时用这个函数直接按文件路径取。
    """
    return Gaussian_Output._read_tail_lines(filename, line_count, encoding_errors)


@dataclass
class Excitation:
    """一个激发态：能量（eV）、振子强度 f、多重度 2⟨S²⟩+1。"""
    energy: float = None
    amplitude: float = None
    multiplicity: float = None


class Gaussian_Output:
    """一份 Gaussian 输出文件的解析结果。

    构造参数：

    * ``output``     —— 输出文件路径，或（兼容旧用法）一个行列表。
    * ``filename``   —— 记录用的文件名；传路径时默认取该路径。
    * ``retained_head_line_count`` / ``retained_tail_line_count``
      —— 常驻的头尾行数，对应 ``head_lines`` / ``tail_lines``。
    * ``encoding_errors`` —— 解码保留行时的错误处理方式，默认 ``'strict'``。
      未被任何解析函数读到的行从不解码，因此非保留区域里的乱码不会引发异常。

    本类不再提供 ``lines`` / ``steps_list``：整份输出从不进内存。需要看原文时用
    ``head_lines`` / ``tail_lines``，或者按 ``step_text()`` 取某一步骤的原文。
    """

    def __init__(self, output, filename="", *, retained_head_line_count=10000,
                 retained_tail_line_count=5000, encoding_errors='strict'):
        self._retained_head_line_count = int(retained_head_line_count)
        self._retained_tail_line_count = int(retained_tail_line_count)
        self._encoding_errors = encoding_errors
        self._source_path = None
        self._source_bytes = None
        self._scan_steps = []
        self._step_objects = []
        self._file_size = 0
        self._file_mtime = None
        self._fingerprint = None
        self._incomplete_tail_start = None
        self._resume_offset = 0
        self._hash_p_found = False

        self.head_lines = []
        self.tail_lines = []

        if isinstance(output, str) and file_type(output) == Filetype.gaussian_output:
            self._source_path = output
            if not filename:
                filename = output

        elif isinstance(output, list):
            # 显式复制：调用方拿到的列表在构造之后可能还会被改动
            source_lines = list(output)
            self._source_bytes = self._encode_line_list(source_lines, self._encoding_errors)
            self.head_lines = source_lines[:self._retained_head_line_count]
            self.tail_lines = (source_lines[-self._retained_tail_line_count:]
                               if self._retained_tail_line_count else [])

        else:
            raise MyException('Not valid file')

        self.filename = filename
        self._parse()

    # ------------------------------------------------------------------
    # 扫描与解析
    # ------------------------------------------------------------------
    def _open_source(self, for_scan=True):
        if self._source_path is not None:
            return open(self._source_path, 'rb'), os.path.getsize(self._source_path)
        if self._source_bytes is None:
            raise MyException('Source content is no longer available')
        return io.BytesIO(self._source_bytes), len(self._source_bytes)

    def _parse(self, scan_start=0, carried=None, existing_step_objects=()):
        file_object, file_size = self._open_source()
        try:
            scan_steps, incomplete_tail_start = self._scan_structure(
                file_object, file_size, scan_start, carried, self._encoding_errors)

            window = self._Byte_Window(file_object, file_size, self._encoding_errors, scan_start)
            hash_p_found = bool(carried and carried.get("hash_p_found"))
            for scan_step in scan_steps:
                last_position = len(scan_step.regions) - 1
                superseded_candidate = {}
                # 本步骤里每个「只读最后一个」的 link 号最后出现在第几个区域。收尾链
                # 的编号要扫过首行才知道（region.num 还是 None），那种区域一律照扫，
                # 由后面的 _release_superseded_retained 收拾。
                last_position_of_number = {}
                for position, region in enumerate(scan_step.regions):
                    if region.num in self._SUPERSEDED_RETAINED_FIELDS:
                        last_position_of_number[region.num] = position
                for position, region in enumerate(scan_step.regions):
                    if not region.scanned:
                        self._scan_link_region(
                            window, region, position == last_position,
                            last_position_of_number.get(region.num, position) == position)
                        if region.retained.hash_p_found:
                            hash_p_found = True
                        # 边扫边建 link，建完立刻放掉只有 link 会读的保留行：峰值内存
                        # 出现在扫描过程中，攒到最后再建对象等于把它们全压在峰值上
                        region.link = Gaussian_Output_Link(region.num, region.retained,
                                                           region.leave_time, region.cpu_time)
                        self._release_link_only_retained(region.retained)
                    if region.num in self._SUPERSEDED_RETAINED_FIELDS and region.retained is not None:
                        previous = superseded_candidate.get(region.num)
                        if previous is not None:
                            self._release_superseded_retained(previous.retained, region.num)
                        superseded_candidate[region.num] = region
            del window
        finally:
            file_object.close()

        # 校验路由里写了 #p，否则本类读不出任何东西
        if not hash_p_found:
            raise MyException('Output Files without #P in route are not supported.')

        step_objects = list(existing_step_objects)
        for index in range(len(step_objects), len(scan_steps)):
            scan_step = scan_steps[index]
            step_object = Gaussian_Output_Step(scan_step.regions, self.filename)
            step_object._byte_range = (scan_step.start, scan_step.end)
            step_objects.append(step_object)
            if scan_step.closed:
                # 闭合步骤的保留行用完即弃；未闭合步骤要留着，refresh() 重建它时还要用
                for region in scan_step.regions:
                    region.retained = None

        self._scan_steps = scan_steps
        self._step_objects = step_objects
        self._file_size = file_size
        self._incomplete_tail_start = incomplete_tail_start
        self._hash_p_found = hash_p_found
        self._record_resume_state(scan_steps, file_size)

        if self._source_path is not None:
            self._file_mtime = os.stat(self._source_path).st_mtime
            self._fingerprint = self._file_fingerprint(self._source_path, file_size)
            self.head_lines = self._read_head_lines(self._source_path,
                                                    self._retained_head_line_count,
                                                    self._encoding_errors)
            self.tail_lines = self._read_tail_lines(self._source_path,
                                                    self._retained_tail_line_count,
                                                    self._encoding_errors, file_size)
        else:
            # 行列表输入：原文只在构造时用一次，用完就放掉
            self._source_bytes = None

        self._build_file_level_state()

    def _record_resume_state(self, scan_steps, file_size):
        closed_steps = []
        open_step_start = file_size
        open_step_regions = []
        resume_offset = file_size
        last_leave_time = ""
        for scan_step in scan_steps:
            if scan_step.closed:
                closed_steps.append(scan_step)
            else:
                open_step_start = scan_step.start
                open_step_regions = scan_step.regions[:-1]
                resume_offset = scan_step.regions[-1].start
                last_leave_time = (open_step_regions[-1].leave_time
                                   if open_step_regions else "")
        self._resume_offset = resume_offset
        self._resume_carried = {"closed_steps": closed_steps,
                                "open_step_start": open_step_start,
                                "open_step_regions": open_step_regions,
                                "resume_offset": resume_offset,
                                "last_leave_time": last_leave_time,
                                "hash_p_found": self._hash_p_found}

    def refresh(self) -> bool:
        """重新读取源文件里新增的部分，返回「解析结果有没有变化」。

        文件只是被追加时走增量路径：已闭合的步骤原样保留，只把最后一个未结束的
        link 区域整段重扫，再重建最后一个未完结的步骤对象与文件级汇总。文件被
        重写（变短、开头变了、旧末尾样本对不上）时退回完整重解析。行列表构造出
        来的对象没有可追踪的源文件，一律返回 ``False``。
        """
        if self._source_path is None or not os.path.isfile(self._source_path):
            return False
        stat_result = os.stat(self._source_path)
        if stat_result.st_size == self._file_size and stat_result.st_mtime == self._file_mtime:
            return False

        rewritten = stat_result.st_size < self._file_size or not self._fingerprint_matches()
        # 上一轮扫描时文件末尾停在半行上，而那半行还被当成了步骤 / link 的边界行：
        # 它续写之后边界会挪位，只能完整重解析
        boundary_in_partial_line = (self._incomplete_tail_start is not None
                                    and self._incomplete_tail_start < self._resume_offset)
        if rewritten or boundary_in_partial_line:
            self._parse()
            return True

        carried = self._resume_carried
        self._parse(scan_start=self._resume_offset, carried=carried,
                    existing_step_objects=self._step_objects[:len(carried["closed_steps"])])
        return True

    def _fingerprint_matches(self):
        if self._fingerprint is None:
            return False
        head_hash, sample_start, sample_hash = self._fingerprint
        try:
            with open(self._source_path, 'rb') as file_object:
                if hashlib.sha256(file_object.read(self._FINGERPRINT_HEAD_BYTES)).hexdigest() != head_hash:
                    return False
                file_object.seek(sample_start)
                if hashlib.sha256(file_object.read(self._FINGERPRINT_TAIL_BYTES)).hexdigest() != sample_hash:
                    return False
        except OSError:
            return False
        return True

    def step_text(self, step_index):
        """按记录的字节区间取回某一步骤的原文（不整份读文件）。"""
        start, end = self.steps[step_index]._byte_range
        path = self._source_path
        if path is None:
            path = self.filename if self.filename and os.path.isfile(self.filename) else None
        if path is None:
            raise MyException('Source file is not available for step_text()')
        pieces = []
        with open(path, 'rb') as file_object:
            file_object.seek(start)
            remaining = end - start
            while remaining > 0:
                chunk = file_object.read(min(remaining, self._SCAN_CHUNK_BYTES))
                if not chunk:
                    break
                pieces.append(chunk)
                remaining -= len(chunk)
        return b"".join(pieces).decode("utf-8", self._encoding_errors).replace("\r\n", "\n")

    # ------------------------------------------------------------------
    # 文件级汇总（跨步骤的判断、标题、分组、溶剂化能）
    # ------------------------------------------------------------------
    def _build_file_level_state(self):
        self.steps = list(self._step_objects)

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
                l101_lines = link.region_lines or []
                for count, line in enumerate(l101_lines):
                    if "Symbolic Z-matrix:" in line or "Structure from the checkpoint file" in line:
                        # [1:]是除去每行开头的空格
                        title1 = [x[1:] for x in l101_lines[1:count]]  # 高斯有时会把标题写在Structure from the checkpoint file前面
                        title2 = [x[1:] for x in l101_lines[count + 1:]]  # 有时会写在后面
                        # 标题上下的横线长度随标题长度而变，短标题时不足 5 个连字符，
                        # 所以判据用 '---'（原来是先把全文的 '---' 换成 35 连字符再判
                        # '-----'，两者严格等价，见迁移说明文档）
                        if True in ['---' in x for x in title1]:
                            title = title1
                        else:
                            title = title2
                        self.title = ''.join(split_list(title, lambda x: '---' in x)[0])
                        break
            if self.title:
                break

        self.title = self.title.replace('\n', "")
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

        self.get_solvation_energy()
        self.get_group_coordinate()

        self.normal_terminated = False not in [step.normal_termination for step in self.steps]

    # ------------------------------------------------------------------
    # IOp(9/40) —— TDDFT 激发组分的打印阈值
    # ------------------------------------------------------------------
    @property
    def iop_9_40_values(self) -> list[int | None]:
        """各步骤路由里 ``IOp(9/40=N)`` 的 N（该步骤没写就是 ``None``）。"""
        return [step.iop_9_40_value for step in self.steps]

    @property
    def iop_9_40_thresholds(self) -> list[float | None]:
        """各步骤的 CI 展开系数打印阈值（该步骤没写 ``IOp(9/40)`` 就是 ``None``）。"""
        return [step.iop_9_40_threshold for step in self.steps]

    @property
    def has_iop_9_40(self) -> bool:
        """是否有任何一个步骤写了 ``IOp(9/40=...)``。"""
        return any(threshold is not None for threshold in self.iop_9_40_thresholds)

    @property
    def iop_9_40_threshold(self) -> float | None:
        """整个输出文件的 CI 展开系数打印阈值。

        多步任务取各步骤里**最小**的那一个——阈值越小打印得越详细，输出文件
        的体积由打印得最详细的那一步决定。没有任何步骤写 ``IOp(9/40)`` 时返回
        ``None``。
        """
        return Route_Dict.smallest_iop_9_40_threshold(self.iop_9_40_thresholds)

    @property
    def is_excited_state(self) -> bool:
        """是否有任何一个步骤是激发态任务（``TD`` / ``TDA`` / ``CIS``）。"""
        return any(step.is_excited_state for step in self.steps)

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
                            route1 = Route_Dict(step1.route_dict.origin_route_input)
                            route2 = Route_Dict(step2.route_dict.origin_route_input)

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

    # =======================================================================
    # 分块扫描器（本类的内部机制，Gaussian_Output_Step 亦经由本类使用）
    # -----------------------------------------------------------------------
    # 整份输出**从不**整体读进内存。文件按 8 MB 大块流式读入，只有「解析函数真正会
    # 读到的行」才被解码、保留下来；其余字节读过即弃（因此也天然免疫非保留区域里的
    # 编码错误）。
    #
    # 两级扫描：
    #
    #   第一级（全局，_scan_structure）只找两个结构性标记—— ``l1.exe``（步骤分隔
    #   行）与 `` Leave Link``（link 区域的结束行），由此得到每个步骤 / 每个 link
    #   的**字节区间**。这一级不解码任何行，只对 Leave Link 命中行套用原有正则。
    #
    #   第二级（限界，_scan_link_region）对每个 link 区域，只搜索「该 link 号可能
    #   打印、且确实会被某个解析函数读到」的标记，见 _MARKERS_BY_LINK_NUMBER。
    #
    # 每条保留规则的等价性论证都是同一个模式：**该解析正则 / 判断必须包含字面量
    # X，X 只可能出现在区域 R（G09 D.01 源码逐 l*.F 核对所得），而保留窗口覆盖了原
    # 代码对命中行前后邻域的全部访问**——因此「只喂保留下来的行」与「喂全文」得到
    # 的解析结果相同。逐条论证见 _MARKERS_BY_LINK_NUMBER 上方的注释。
    # =======================================================================

    _SCAN_CHUNK_BYTES = 8 * 1024 * 1024          # 流式读入的大块尺寸
    _SCAN_HISTORY_BYTES = 1 << 16                # 命中行回看前一行所需的历史字节
    _LAST_LINE_LOOKBACK_BYTES = 1 << 12          # 取区域最后一行时的回看上限

    _STEP_SEPARATOR_MARKER = b"l1.exe"
    _LEAVE_LINK_MARKER = b" Leave Link"

    _LEAVE_LINK_PATTERN = re.compile(
        r' Leave Link +(\d+) at ([A-Za-z]{3} [A-Za-z]{3} +\d+ \d{2}:\d{2}:\d{2} \d{4}).+cpu\:\s+(\d+\.\d+)')
    _ENTER_LINK_PATTERN = re.compile(r"\(Enter .+l(\d{1,4}).exe\)")

    # 坐标块的五种标题标记；值是标题行之后需要跳过的行数。
    _COORDINATE_MARKS = {"Symbolic Z-matrix:": 0,
                         "Input orientation:": 4,
                         "Standard orientation:": 4,
                         "Redundant internal coordinates found in file": 0,
                         "CURRENT STRUCTURE": 5}

    # -----------------------------------------------------------------------
    # 保留规则表
    # -----------------------------------------------------------------------
    # 表里每一项是 (标记字节串, 用途)。用途决定「命中之后还要额外保留哪些行」，
    # 见 _scan_link_region 里的分发。
    #
    #  1    路由回显与 Link0 回显都在 L1 区域：
    #       ' #' 行首标记对应 get_routes 的 ``re.findall(r'^\ #', line)``；
    #       '#p' / '#P' 对应构造函数里的 "#P in route" 校验；
    #       '%chk=' 对应 chk_filename 的 ``re.findall(r"\%chk\=(.+)", line)``。
    #       偏差：原来这两项扫全文，现在只扫 L1 区域（回显所在），标题里写了 '#p'
    #       之类的病态文件不再通过校验——已接受。
    #  101  区域小，整段保留（_WHOLE_REGION_LINK_NUMBERS）：标题、
    #       "Structure from the checkpoint file" 提示与 ModRedundant 回显都在其中。
    #       坐标标记另行命中，以便和别的区域共用同一套坐标窗口逻辑。
    #  103  'Converged?' → get_converged（命中行 + 后 4 行）；
    #       'Number of optimizations in scan' / 'Optimization completed.'
    #       → get_relaxed_scan（只做存在性判断）。
    #  123  'CURRENT STRUCTURE' → 坐标；
    #       'Calculating another point on the path.' → get_irc_coords（存在性）；
    #       'Point Number:' → 新属性 irc_point_number / irc_path_number。
    #  202  'Input orientation:' / 'Standard orientation:' → 坐标。
    #  301  'Basis read from chk' / 'General basis read from cards:' /
    #       'basis functions,' → get_level；
    #       'Polarizable Continuum Model (PCM)' / 'Solvent' → read_step_solvent。
    #  402  'Energy' → get_opt_energies 的 External 回退分支。
    #  502  'SCF Done:' → get_scf_final_result / get_opt_energies；
    #       'Cycle' / 'E= ' → get_scf_iteration（保留原文件行距，见下）。
    #  503 / 718  'SCF Done:' → get_scf_final_result。
    #  508  'SCF Done:'；'Iteration' → get_scf_iteration 的 QC 分支。
    #  601  'eigenvalues' → 新属性：轨道能量。
    #  716  热化学：每个正则的字面量都在标记集内。
    #  914  'Excited State' → 新属性：激发态。
    #  9999 '1\1' / '1|1' → find_summary。
    #
    # 终止判断（'Normal termination' / 'Error termination'）与 'Job cpu time' 由
    # _markers_for_link 按「是不是本步骤最后一个 link」动态追加。
    _MARKERS_BY_LINK_NUMBER = {
        1: [(b" #", "route"), (b"#p", "hash_p"), (b"#P", "hash_p"), (b"%chk=", "chk")],
        101: [(b"Symbolic Z-matrix:", "coordinate"),
              (b"Redundant internal coordinates found in file", "coordinate")],
        103: [(b"Converged?", "converged"),
              (b"Number of optimizations in scan", "scan_count"),
              (b"Optimization completed.", "optimization_completed")],
        123: [(b"CURRENT STRUCTURE", "coordinate"),
              (b"Calculating another point on the path.", "irc_calculating"),
              (b"Point Number:", "irc_point_number")],
        202: [(b"Input orientation:", "coordinate"),
              (b"Standard orientation:", "coordinate")],
        301: [(b"Basis read from chk", "basis_from_chk"),
              (b"General basis read from cards:", "general_basis"),
              (b"basis functions,", "basis_counting"),
              (b"Polarizable Continuum Model (PCM)", "pcm"),
              (b"Solvent", "solvent")],
        402: [(b"Energy", "external_energy")],
        # 'E= -'：get_scf_iteration 的正则是 r"E= (-\d+\.\d+)"，本来就要求紧跟一个负号，
        # 所以带负号的标记与正则等价，而且不会被同一行里的 "Delta-E=   -0.000" 重复命中
        502: [(b"SCF Done:", "scf_done"), (b"Cycle ", "scf_cycle"), (b"E= -", "scf_energy")],
        503: [(b"SCF Done:", "scf_done")],
        508: [(b"SCF Done:", "scf_done"), (b"Iteration", "scf_iteration_508")],
        601: [(b"eigenvalues", "eigenvalues")],
        716: [(b"Temperature", "freq"),
              (b"Rotational symmetry number", "freq"),
              (b"Rotational constant", "freq"),
              (b"has atomic number", "freq"),
              (b"Frequencies", "freq"),
              (b"Zero-point correction=", "freq"),
              (b"Thermal correction to", "freq"),
              (b"Sum of electronic and thermal", "freq"),
              (b"(Thermal)", "entropy_header")],
        718: [(b"SCF Done:", "scf_done")],
        914: [(b"Excited State", "excited_state")],
        9999: [(b"1\\1", "summary"), (b"1|1", "summary")],
    }

    # 「不按 link 号筛选的消费者」会读到的标记：get_coords（除 9999 外每个 link 都
    # 跑）、get_scf_final_result（每个 link 都跑）、get_irc_coords（每个 link 都
    # 跑），外加构造函数的 '#P' 校验。link 号未知（-1）或不在归属表里的区域用这一
    # 组扫描：按 link 号筛选的规则对这些区域根本不会触发，扫了也没有消费者，所以这
    # 与「全部标记的并集」逐字段等价，只是不做无用功。
    _ALWAYS_CONSUMED_MARKERS = [
        (b"Symbolic Z-matrix:", "coordinate"),
        (b"Input orientation:", "coordinate"),
        (b"Standard orientation:", "coordinate"),
        (b"Redundant internal coordinates found in file", "coordinate"),
        (b"CURRENT STRUCTURE", "coordinate"),
        (b"SCF Done:", "scf_done"),
        (b"Calculating another point on the path.", "irc_calculating"),
        (b"#p", "hash_p"),
        (b"#P", "hash_p"),
    ]

    _TERMINATION_MARKERS = [(b"Normal termination", "normal_termination"),
                            (b"Error termination", "error_termination")]
    _JOB_CPU_TIME_MARKER = (b"Job cpu time", "job_cpu_time")

    # 需要整段保留的 link 区域（区域小，且标题 / ModRedundant 回显没有稳定的行标记）
    _WHOLE_REGION_LINK_NUMBERS = frozenset({101})

    # 这三条规则只读「本步骤最后一个该号 link 区域」（对应的解析函数都是
    # ``for region in reversed(regions)`` 找到第一个就 break），所以同一步骤里再出现
    # 一个同号区域时，前一个保留的行就再也没有消费者了——几何优化每一轮都会跑一遍
    # L601，一份过渡态搜索输出里光轨道能量行就有十几万行。
    _SUPERSEDED_RETAINED_FIELDS = {
        601: ("eigenvalue_lines",),
        716: ("freq_lines", "entropy_blocks"),
        914: ("excited_state_lines",),
    }
    _SUPERSEDED_PURPOSES = {
        601: frozenset({"eigenvalues"}),
        716: frozenset({"freq", "entropy_header"}),
        914: frozenset({"excited_state"}),
    }

    # 需要知道命中行「在区域里的第几行」的用途。只有这些用途在场时才逐块数换行符——
    # 数换行符是对整段字节的一次扫描，对 L914 那种几百 MB 的区域是纯浪费。
    _LINE_INDEXED_PURPOSES = frozenset({"scf_cycle", "scf_energy", "basis_from_chk",
                                        "general_basis", "pcm", "solvent"})

    _MARKER_SET_CACHE = {}

    _FINGERPRINT_HEAD_BYTES = 1 << 16
    _FINGERPRINT_TAIL_BYTES = 1 << 12

    class _Retained_Link_Lines:
        """一个 link 区域里按用途保留下来的行。

        字段一律按「消费它的解析函数」命名；没有任何解析函数会读到的行从不解码，
        也从不进入这个对象。
        """

        __slots__ = ("first_line", "last_line", "region_lines", "hash_p_found",
                     "coordinate_slabs", "scf_done_lines", "external_energy_lines",
                     "scf_iteration_hits", "scf_iteration_508_lines",
                     "route_blocks", "chk_blocks",
                     "basis_from_chk_index", "general_basis_index", "general_basis_lines",
                     "basis_counting_pairs", "pcm_indices", "solvent_hits",
                     "freq_lines", "entropy_blocks", "converged_blocks",
                     "scan_count_found", "optimization_completed_found",
                     "irc_calculating_found", "irc_point_number_blocks",
                     "eigenvalue_lines", "excited_state_lines", "summary_blocks",
                     "normal_termination", "error_termination", "job_cpu_time_lines")

        def __init__(self):
            self.first_line = ""
            self.last_line = ""
            self.region_lines = None                # 只有 L101 整段保留
            self.hash_p_found = False
            self.coordinate_slabs = []              # [(标记, 跳过行数, 前一行, 命中行, 后续行)]
            self.scf_done_lines = []
            self.external_energy_lines = []
            self.scf_iteration_hits = []            # [(区域内行号, 行文本)]
            self.scf_iteration_508_lines = []
            self.route_blocks = []
            self.chk_blocks = []
            self.basis_from_chk_index = None
            self.general_basis_index = None
            self.general_basis_lines = []
            self.basis_counting_pairs = []          # [(命中行, 下一行)]
            self.pcm_indices = []
            self.solvent_hits = []                  # [(区域内行号, 行文本)]
            self.freq_lines = []
            self.entropy_blocks = []                # [[表头行, 后 1 行, 后 2 行]]
            self.converged_blocks = []              # [[命中行, 后 1..4 行]]
            self.scan_count_found = False
            self.optimization_completed_found = False
            self.irc_calculating_found = False
            self.irc_point_number_blocks = []
            self.eigenvalue_lines = []
            self.excited_state_lines = []
            self.summary_blocks = []
            self.normal_termination = False
            self.error_termination = False
            self.job_cpu_time_lines = []

    class _Region_Scan:
        """一个 link 区域的字节区间与扫描结果。"""

        __slots__ = ("start", "end", "num", "leave_time", "cpu_time", "retained", "scanned",
                     "link")

        def __init__(self, start, end, num, leave_time, cpu_time):
            self.start = start
            self.end = end
            self.num = num                          # None 表示待由区域首行推导（规则 3）
            self.leave_time = leave_time
            self.cpu_time = cpu_time
            self.retained = None
            self.scanned = False
            self.link = None                        # 已建好的 Gaussian_Output_Link，refresh() 复用

    class _Step_Scan:
        """一个步骤的字节区间与它的各 link 区域。"""

        __slots__ = ("start", "end", "regions", "closed")

        def __init__(self, start, end, regions, closed):
            self.start = start
            self.end = end
            self.regions = regions
            self.closed = closed                    # 后面还有 l1.exe 分隔行吗

    class _Byte_Window:
        """文件字节流上的滑动窗口：只顺序向前推进，用完的前段随即丢弃。"""

        def __init__(self, file_object, file_size, encoding_errors, start_offset=0):
            self.file_object = file_object
            self.file_size = file_size
            self.encoding_errors = encoding_errors
            self.buffer = bytearray()
            self.start = start_offset
            self.file_object.seek(start_offset)

        def end(self):
            return self.start + len(self.buffer)

        def ensure(self, end_offset):
            """保证缓冲区覆盖到 end_offset（不超过文件末尾）。"""
            end_offset = min(end_offset, self.file_size)
            missing = end_offset - self.end()
            while missing > 0:
                data = self.file_object.read(max(missing, Gaussian_Output._SCAN_CHUNK_BYTES))
                if not data:
                    break
                self.buffer += data
                missing = end_offset - self.end()

        def release(self, offset):
            """丢弃 offset 之前的缓冲内容。"""
            if offset > self.start:
                del self.buffer[:offset - self.start]
                self.start = offset

        def find(self, marker, begin, end):
            result = self.buffer.find(marker, begin - self.start, end - self.start)
            return -1 if result < 0 else result + self.start

        def rfind_newline(self, begin, end):
            result = self.buffer.rfind(b"\n", begin - self.start, end - self.start)
            return -1 if result < 0 else result + self.start

        def count_newlines(self, begin, end):
            return self.buffer.count(b"\n", begin - self.start, end - self.start)

        def line_text(self, begin, end):
            return Gaussian_Output._decode_output_line(
                bytes(self.buffer[begin - self.start:end - self.start]),
                self.encoding_errors)

        def line_start_at(self, offset, lower_bound):
            newline = self.rfind_newline(lower_bound, offset)
            return lower_bound if newline < 0 else newline + 1

        def line_end_at(self, offset, upper_bound):
            """offset 所在行的结束位置（含换行符），必要时继续读入数据。"""
            while True:
                limit = min(self.end(), upper_bound)
                found = self.find(b"\n", offset, limit)
                if found >= 0:
                    return found + 1
                if limit >= upper_bound:
                    return upper_bound
                self.ensure(min(limit + Gaussian_Output._SCAN_CHUNK_BYTES, upper_bound))
                if min(self.end(), upper_bound) <= limit:
                    return limit

    @classmethod
    def _markers_for_link(cls, link_number, is_last_in_step, is_last_of_its_number=True):
        """某个 link 区域需要限界扫描的标记集，以及它要不要行号。

        结果按 (link 号, 是不是本步骤最后一个 link, 是不是本步骤最后一个同号 link)
        缓存——一份输出可能有上万个 link 区域，不必每个都重新拼一遍列表。
        """
        key = (link_number, is_last_in_step, is_last_of_its_number)
        cached = cls._MARKER_SET_CACHE.get(key)
        if cached is not None:
            return cached

        if link_number in cls._MARKERS_BY_LINK_NUMBER:
            markers = list(cls._MARKERS_BY_LINK_NUMBER[link_number])
        else:
            markers = list(cls._ALWAYS_CONSUMED_MARKERS)
        if not is_last_of_its_number and link_number in cls._SUPERSEDED_PURPOSES:
            # 轨道能量 / 热化学 / 激发态这三条规则只读「本步骤最后一个该号区域」，
            # 前面的同号区域连扫都不用扫（几何优化每轮都跑一遍 L601，光轨道能量行
            # 就有十几万行，扫出来解码完又立刻丢掉是纯浪费）
            superseded = cls._SUPERSEDED_PURPOSES[link_number]
            markers = [x for x in markers if x[1] not in superseded]
        if is_last_in_step:
            # 终止判断只看每步最后一个 link 区域（原来是 'Normal termination' in
            # "".join(links[-1].lines)；行内字符串不跨行，逐区域判断等价）
            markers = markers + cls._TERMINATION_MARKERS
        if is_last_in_step or link_number == 9999:
            markers = markers + [cls._JOB_CPU_TIME_MARKER]

        needs_line_index = any(purpose in cls._LINE_INDEXED_PURPOSES for _, purpose in markers)
        cached = (tuple(markers), needs_line_index)
        cls._MARKER_SET_CACHE[key] = cached
        return cached

    @staticmethod
    def _decode_output_line(raw, encoding_errors):
        """把一行原始字节解码成字符串，并把行尾 ``\\r\\n`` 归一成 ``\\n``。

        复现 ``open(..., encoding='utf-8').readlines()`` 的通用换行行为。用孤立
        ``\\r`` 当换行的古老格式不支持——Gaussian 不产生这种输出。
        """
        text = raw.decode("utf-8", encoding_errors)
        if text.endswith("\r\n"):
            return text[:-2] + "\n"
        return text

    @classmethod
    def _split_raw_lines(cls, data, encoding_errors):
        """把一段字节按 readlines() 的语义切成行（行尾换行符保留在行内）。"""
        if not data:
            return []
        pieces = data.split(b"\n")
        last = pieces.pop()
        lines = [cls._decode_output_line(piece + b"\n", encoding_errors) for piece in pieces]
        if last:
            lines.append(cls._decode_output_line(last, encoding_errors))
        return lines

    @classmethod
    def _iterate_lines(cls, window, start, region_end):
        """从 start 起逐行产出 (行起点, 行终点, 行文本)，到 region_end 为止。

        坐标块动辄几十万行，这里把取行的算术全部内联，避免每行几次方法调用。
        """
        offset = start
        encoding_errors = window.encoding_errors
        while offset < region_end:
            if window.end() <= offset:
                window.ensure(min(offset + cls._SCAN_CHUNK_BYTES, region_end))
                if window.end() <= offset:
                    break
            buffer = window.buffer
            base = window.start
            limit = min(window.start + len(buffer), region_end)
            newline = buffer.find(b"\n", offset - base, limit - base)
            if newline >= 0:
                line_end = newline + 1 + base
            elif limit >= region_end:
                line_end = region_end
            else:
                window.ensure(min(limit + cls._SCAN_CHUNK_BYTES, region_end))
                if min(window.end(), region_end) <= limit:
                    line_end = limit
                else:
                    continue
            if line_end <= offset:
                break
            text = bytes(buffer[offset - base:line_end - base]).decode("utf-8", encoding_errors)
            if text.endswith("\r\n"):
                text = text[:-2] + "\n"
            yield offset, line_end, text
            offset = line_end

    # -----------------------------------------------------------------------
    # 第一级：全局扫描出步骤 / link 的字节区间
    # -----------------------------------------------------------------------

    @classmethod
    def _collect_boundary_events(cls, span, span_start, span_length, events, encoding_errors):
        """在 span 的前 span_length 个字节（由完整行组成）里找出步骤分隔行与 Leave Link 行。

        只传长度而不切片，是为了不给每个 8 MB 大块再多复制一份。
        """
        for marker, kind in ((cls._STEP_SEPARATOR_MARKER, "separator"),
                             (cls._LEAVE_LINK_MARKER, "leave_link")):
            position = 0
            while position < span_length:
                found = span.find(marker, position, span_length)
                if found < 0:
                    break
                line_start = span.rfind(b"\n", 0, found) + 1
                newline = span.find(b"\n", found, span_length)
                line_end = span_length if newline < 0 else newline + 1
                if kind == "leave_link":
                    match = cls._LEAVE_LINK_PATTERN.findall(
                        cls._decode_output_line(bytes(span[line_start:line_end]), encoding_errors))
                    payload = match[0] if match else None
                else:
                    payload = None
                events.append((span_start + line_start, span_start + line_end, kind, payload))
                position = max(line_end, found + 1)

    @classmethod
    def _scan_structure(cls, file_object, file_size, scan_start, carried, encoding_errors):
        """第一级扫描：只找 ``l1.exe`` 与 `` Leave Link``。

        carried 是 refresh() 续扫时带过来的状态字典，全新扫描时传 None。
        返回 (steps, incomplete_tail_start)。
        """
        file_object.seek(scan_start)
        events = []
        buffer = bytearray()
        buffer_start = scan_start
        incomplete_tail_start = None

        while True:
            chunk = file_object.read(cls._SCAN_CHUNK_BYTES)
            if not chunk:
                break
            buffer += chunk
            newline = buffer.rfind(b"\n")
            if newline < 0:
                continue
            cls._collect_boundary_events(buffer, buffer_start, newline + 1, events, encoding_errors)
            del buffer[:newline + 1]
            buffer_start += newline + 1
        if buffer:
            # 文件末尾的残行：readlines() 会把它当作一行，这里同样处理；同时记下它的
            # 起点，refresh() 判断能不能增量续扫要用
            incomplete_tail_start = buffer_start
            cls._collect_boundary_events(buffer, buffer_start, len(buffer), events, encoding_errors)

        events.sort(key=lambda item: item[0])
        # 同一行既含 l1.exe 又匹配 Leave Link 时，按原代码的顺序（先按 l1.exe 切步骤）
        # 只认它是分隔行
        deduplicated = []
        for event in events:
            if deduplicated and deduplicated[-1][0] == event[0]:
                if deduplicated[-1][2] == "separator":
                    continue
                if event[2] == "separator":
                    deduplicated[-1] = event
                    continue
                continue
            deduplicated.append(event)

        if carried:
            steps = list(carried["closed_steps"])
            step_start = carried["open_step_start"]
            regions = list(carried["open_step_regions"])
            leave_time = carried["last_leave_time"]
        else:
            steps = []
            step_start = scan_start
            regions = []
            leave_time = ""
        region_start = regions[-1].end if regions else step_start
        if carried:
            region_start = carried["resume_offset"]

        def close_step(step_end, closed):
            nonlocal regions, region_start
            if region_start < step_end:
                # 收尾链：没有 Leave Link 行，编号待由区域首行推导（规则 3）
                regions.append(cls._Region_Scan(region_start, step_end, None, leave_time, -1))
            if regions:
                steps.append(cls._Step_Scan(step_start, step_end, regions, closed))
            regions = []

        for line_start, line_end, kind, payload in deduplicated:
            if kind == "separator":
                close_step(line_start, True)
                step_start = line_end
                region_start = line_end
                leave_time = ""
            else:
                if payload is not None and payload[0].isnumeric() and region_start < line_start:
                    regions.append(cls._Region_Scan(region_start, line_start, int(payload[0]),
                                                    payload[1], payload[2]))
                    leave_time = payload[1]
                    region_start = line_end
                # 否则这一行留在当前区域里（原代码走 else 分支把它并进 current_link）

        close_step(file_size, False)
        return steps, incomplete_tail_start

    # -----------------------------------------------------------------------
    # 第二级：按规则表限界扫描每个 link 区域
    # -----------------------------------------------------------------------

    @classmethod
    def _collect_coordinate_slab(cls, window, line_start, line_end, line_text,
                                 region_start, region_end, retained):
        """坐标块（规则 6）：命中行前 1 行 + 命中行 + 直到 std_coordinate 失败的后续行。

        原代码对一个命中行会把 marks 里每一个命中的标记都处理一遍（没有 break），
        这里逐字照做。
        """
        for mark, skip in cls._COORDINATE_MARKS.items():
            if mark not in line_text:
                continue
            if line_start == region_start:
                # 原代码 self.lines[count - 1]，count == 0 时取到的是区域最后一行；
                # 用 None 占位，等区域扫完拿到 last_line 再补上
                previous_line = None
            else:
                previous_start = window.line_start_at(
                    line_start - 1, max(region_start, window.start))
                previous_line = window.line_text(previous_start, line_start)
            following = []
            # is_coordinate_line[k] 记下 following[k] 是不是坐标行（下标小于 skip 的行
            # 原代码根本不测，记 None）。收集时本来就要逐行判定，结果顺手留给
            # Gaussian_Output_Link.get_coords 用，省掉一整轮重复的 std_coordinate。
            is_coordinate_line = []
            for _, _, text in cls._iterate_lines(window, line_end, region_end):
                following.append(text)
                index = len(following) - 1
                if index < skip:
                    is_coordinate_line.append(None)
                    continue
                standardized = bool(std_coordinate(text))
                is_coordinate_line.append(standardized)
                # 收集到「下标 >= skip + 1 的第一条非坐标行」为止：不论 Charge 行有没有
                # 让起点后移一行，原代码读到的行都被这个范围覆盖
                if index >= skip + 1 and not standardized:
                    break
            retained.coordinate_slabs.append((mark, skip, previous_line, line_text,
                                              following, is_coordinate_line))

    @classmethod
    def _collect_forward_until(cls, window, line_start, region_end, stop_predicate,
                               include_stop_line=True, skip_hit_line=False):
        """从命中行起向后保留，直到 stop_predicate 为真的那一行为止。"""
        block = []
        for offset, _, text in cls._iterate_lines(window, line_start, region_end):
            if skip_hit_line and offset == line_start:
                continue
            block.append(text)
            if stop_predicate(text):
                if not include_stop_line:
                    block.pop()
                break
        return block

    @classmethod
    def _collect_following_lines(cls, window, line_end, region_end, count):
        """命中行之后的 count 行（不足则短）。"""
        block = []
        for _, _, text in cls._iterate_lines(window, line_end, region_end):
            block.append(text)
            if len(block) >= count:
                break
        return block

    @classmethod
    def _scan_link_region(cls, window, region, is_last_in_step, is_last_of_its_number=True):
        """第二级：按 link 号的标记集限界扫描一个 link 区域。"""
        retained = cls._Retained_Link_Lines()
        region.retained = retained
        region.scanned = True
        region_start, region_end = region.start, region.end

        window.ensure(min(region_start + cls._SCAN_CHUNK_BYTES, region_end))
        first_line_end = window.line_end_at(region_start, region_end)
        retained.first_line = window.line_text(region_start, first_line_end)

        if region.num is None:
            # 规则 3：收尾链的编号取自区域首行的 "(Enter .../lNNN.exe)"
            if "Enter" in retained.first_line:
                found = cls._ENTER_LINK_PATTERN.findall(retained.first_line)
            else:
                found = []
            region.num = int(found[0]) if (found and found[0].isnumeric()) else -1

        if region.num in cls._WHOLE_REGION_LINK_NUMBERS:
            retained.region_lines = [text for _, _, text
                                     in cls._iterate_lines(window, region_start, region_end)]

        markers, needs_line_index = cls._markers_for_link(region.num, is_last_in_step,
                                                          is_last_of_its_number)
        position = region_start
        line_index_offset = region_start
        line_index_value = 0

        while position < region_end:
            window.release(max(region_start, position - cls._SCAN_HISTORY_BYTES))
            window.ensure(min(position + cls._SCAN_CHUNK_BYTES, region_end))
            span_end = min(window.end(), region_end)
            if span_end <= position:
                break
            if span_end < region_end:
                newline = window.rfind_newline(position, span_end)
                if newline < 0:
                    window.ensure(min(span_end + cls._SCAN_CHUNK_BYTES, region_end))
                    if min(window.end(), region_end) <= span_end:
                        break
                    continue
                span_end = newline + 1

            hits = []
            seen = set()
            for marker, purpose in markers:
                search_from = position
                while True:
                    found = window.find(marker, search_from, span_end)
                    if found < 0:
                        break
                    hits.append((found, marker, purpose))
                    search_from = found + len(marker)
            if len(hits) > 1:
                hits.sort(key=lambda item: item[0])

            # 命中行的取行算术在这里全部内联：一份大输出可以有几十万个命中行，每个
            # 少走三四次方法调用是可观的。span_end 一定落在换行符之后，所以命中行
            # 的结束位置必定在本块之内，不需要再读数据。
            buffer = window.buffer
            base = window.start
            lower_bound = max(region_start, base)
            encoding_errors = window.encoding_errors
            for hit_offset, marker, purpose in hits:
                newline = buffer.rfind(b"\n", lower_bound - base, hit_offset - base)
                line_start = lower_bound if newline < 0 else newline + 1 + base
                key = (line_start, marker, purpose)
                if key in seen:
                    continue
                seen.add(key)
                if needs_line_index:
                    line_index_value += buffer.count(b"\n", line_index_offset - base,
                                                     line_start - base)
                    line_index_offset = line_start
                newline = buffer.find(b"\n", line_start - base, span_end - base)
                line_end = region_end if newline < 0 else newline + 1 + base
                line_text = bytes(buffer[line_start - base:line_end - base]).decode(
                    "utf-8", encoding_errors)
                if line_text.endswith("\r\n"):
                    line_text = line_text[:-2] + "\n"
                cls._dispatch_hit(window, retained, purpose, hit_offset, line_start, line_end,
                                  line_text, line_index_value, region_start, region_end)
                # 保留窗口的收集可能把缓冲区扩大 / 前移，重新取一次引用
                buffer = window.buffer
                base = window.start

            if needs_line_index:
                line_index_value += buffer.count(b"\n", line_index_offset - base,
                                                 span_end - base)
                line_index_offset = span_end
            position = span_end

        if any(slab[2] is None for slab in retained.coordinate_slabs):
            lower = max(region_start, window.start, region_end - cls._LAST_LINE_LOOKBACK_BYTES)
            window.ensure(region_end)
            last_start = window.line_start_at(region_end - 1, lower)
            retained.last_line = window.line_text(last_start, region_end)

        return retained

    @classmethod
    def _release_superseded_retained(cls, retained, link_number):
        """同一步骤里出现了更靠后的同号 link 区域，放掉前一个的保留行。"""
        for field in cls._SUPERSEDED_RETAINED_FIELDS[link_number]:
            setattr(retained, field, ())

    @staticmethod
    def _release_link_only_retained(retained):
        """放掉只有 Gaussian_Output_Link 会读的保留行。

        坐标块与 SCF 迭代行是保留行里体积最大的两项（大文件里各有几十万行），
        link 对象一建好它们就没有消费者了。步骤级解析要用的字段留着不动——
        refresh() 重建未完结的步骤时还要用。
        """
        retained.coordinate_slabs = ()
        retained.scf_iteration_hits = ()
        retained.scf_iteration_508_lines = ()
        retained.irc_point_number_blocks = ()
        retained.last_line = ""

    @classmethod
    def _dispatch_hit(cls, window, retained, purpose, hit_offset, line_start, line_end, line_text,
                      line_index, region_start, region_end):
        if purpose == "coordinate":
            cls._collect_coordinate_slab(window, line_start, line_end, line_text,
                                         region_start, region_end, retained)
        elif purpose == "scf_done":
            retained.scf_done_lines.append(line_text)
        elif purpose == "external_energy":
            retained.external_energy_lines.append(line_text)
        elif purpose in ("scf_cycle", "scf_energy"):
            retained.scf_iteration_hits.append((line_index, line_text))
        elif purpose == "scf_iteration_508":
            retained.scf_iteration_508_lines.append(line_text)
        elif purpose == "route":
            # get_routes 的 ``re.findall(r'^\ #', line)``：只认行首的 " #"
            if hit_offset == line_start:
                retained.route_blocks.append(
                    cls._collect_forward_until(window, line_start, region_end,
                                               lambda text: '---' in text))
        elif purpose == "hash_p":
            retained.hash_p_found = True
        elif purpose == "chk":
            retained.chk_blocks.append(cls._collect_chk_block(window, line_start, region_end))
        elif purpose == "basis_from_chk":
            if retained.basis_from_chk_index is None:
                retained.basis_from_chk_index = line_index
        elif purpose == "general_basis":
            if retained.general_basis_index is None:
                retained.general_basis_index = line_index
                retained.general_basis_lines = cls._collect_forward_until(
                    window, line_end, region_end, lambda text: "Ernie" in text)
        elif purpose == "basis_counting":
            following = cls._collect_following_lines(window, line_end, region_end, 1)
            retained.basis_counting_pairs.append((line_text, following[0] if following else ""))
        elif purpose == "pcm":
            retained.pcm_indices.append(line_index)
        elif purpose == "solvent":
            retained.solvent_hits.append((line_index, line_text))
        elif purpose == "freq":
            retained.freq_lines.append(line_text)
        elif purpose == "entropy_header":
            block = [line_text] + cls._collect_following_lines(window, line_end, region_end, 2)
            while len(block) < 3:
                block.append("")
            retained.entropy_blocks.append(block)
        elif purpose == "converged":
            retained.converged_blocks.append(
                [line_text] + cls._collect_following_lines(window, line_end, region_end, 4))
        elif purpose == "scan_count":
            retained.scan_count_found = True
        elif purpose == "optimization_completed":
            retained.optimization_completed_found = True
        elif purpose == "irc_calculating":
            retained.irc_calculating_found = True
        elif purpose == "irc_point_number":
            retained.irc_point_number_blocks.append(
                [line_text] + cls._collect_following_lines(window, line_end, region_end, 2))
        elif purpose == "eigenvalues":
            retained.eigenvalue_lines.append(line_text)
        elif purpose == "excited_state":
            retained.excited_state_lines.append(line_text)
        elif purpose == "summary":
            retained.summary_blocks.append(
                cls._collect_forward_until(window, line_start, region_end,
                                           lambda text: "@" in text))
        elif purpose == "normal_termination":
            retained.normal_termination = True
        elif purpose == "error_termination":
            retained.error_termination = True
        elif purpose == "job_cpu_time":
            retained.job_cpu_time_lines.append(line_text)

    @classmethod
    def _collect_chk_block(cls, window, line_start, region_end):
        """``%chk=`` 命中行 + 续行，直到拼出来的字符串里出现 ``.chk``。"""
        block = []
        accumulated = ""
        for _, _, text in cls._iterate_lines(window, line_start, region_end):
            if not block:
                found = re.findall(r"\%chk\=(.+)", text)
                block.append(text)
                if not found:
                    break
                accumulated = found[0]
            else:
                block.append(text)
                accumulated += text[1:].strip('\n')
            if '.chk' in accumulated:
                break
        return block

    # -----------------------------------------------------------------------
    # 头尾常驻行、文件指纹
    # -----------------------------------------------------------------------

    @staticmethod
    def _encode_line_list(lines, encoding_errors="strict"):
        """把行列表编码成字节，供扫描器复用同一套逻辑。

        行尾没有换行符的元素（例如整份文件用 ``split('\\n')`` 切出来的列表）在编码时
        补上换行符。所有消费点要么用正则、要么 ``strip('\\n')``，补出来的换行符不会
        进入任何存储变量。
        """
        parts = []
        for line in lines:
            if not line.endswith("\n"):
                line = line + "\n"
            parts.append(line.encode("utf-8", encoding_errors))
        return b"".join(parts)

    @classmethod
    def _read_head_lines(cls, path, line_count, encoding_errors):
        """文件开头的 line_count 行原文。"""
        if line_count <= 0:
            return []
        lines = []
        carry = b""
        with open(path, "rb") as file_object:
            while len(lines) < line_count:
                chunk = file_object.read(cls._SCAN_CHUNK_BYTES)
                if not chunk:
                    if carry:
                        lines.append(cls._decode_output_line(carry, encoding_errors))
                    break
                buffer = carry + chunk
                pieces = buffer.split(b"\n")
                carry = pieces.pop()
                for piece in pieces:
                    lines.append(cls._decode_output_line(piece + b"\n", encoding_errors))
                    if len(lines) >= line_count:
                        break
        return lines[:line_count]

    @classmethod
    def _read_tail_lines(cls, path, line_count, encoding_errors, file_size=None):
        """文件末尾的 line_count 行原文。"""
        if line_count <= 0:
            return []
        if file_size is None:
            file_size = os.path.getsize(path)
        window_bytes = 1 << 20
        while True:
            start = max(0, file_size - window_bytes)
            with open(path, "rb") as file_object:
                file_object.seek(start)
                data = file_object.read(file_size - start)
            if start > 0:
                cut = data.find(b"\n")
                data = b"" if cut < 0 else data[cut + 1:]
            lines = cls._split_raw_lines(data, encoding_errors)
            if len(lines) >= line_count or start == 0:
                return lines[-line_count:]
            window_bytes *= 4

    @classmethod
    def _file_fingerprint(cls, path, file_size):
        """(前 64 KB 哈希, 末尾样本起点, 末尾 4 KB 样本哈希)。"""
        sample_start = max(0, file_size - cls._FINGERPRINT_TAIL_BYTES)
        with open(path, "rb") as file_object:
            head = file_object.read(cls._FINGERPRINT_HEAD_BYTES)
            file_object.seek(sample_start)
            sample = file_object.read(cls._FINGERPRINT_TAIL_BYTES)
        return (hashlib.sha256(head).hexdigest(), sample_start,
                hashlib.sha256(sample).hexdigest())


class Gaussian_Output_Step:
    """输出文件里的一个步骤（两条 ``l1.exe`` 分隔行之间的部分）。

    构造参数 ``regions`` 是扫描器给出的 ``Gaussian_Output._Region_Scan`` 列表
    （含各 link 区域的字节区间、link 号与按用途保留下来的行）。本类不再持有
    ``lines``。
    """

    _JOB_CPU_TIME_PATTERN = re.compile(
        r'(\d+) +days +(\d+) +hours +(\d+) +minutes +(\d+.\d) +seconds.')

    def __init__(self, regions, original_filename=""):

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

        # link 对象在扫描阶段就建好了（见 Gaussian_Output._parse）；refresh() 重建未
        # 完结的步骤时直接复用，不需要再解析一遍坐标
        for region in regions:
            if region.link is None:
                region.link = Gaussian_Output_Link(region.num, region.retained,
                                                   region.leave_time, region.cpu_time)
                Gaussian_Output._release_link_only_retained(region.retained)
            self.links.append(region.link)

        try:
            self.last_leave_time = max([x.leave_time for x in self.links])
            self.last_leave_time = Chronyk(datetime.strptime(self.last_leave_time, "%a %b %d %H:%M:%S %Y"))
        except:
            self.last_leave_time = ""
            pass

        # 终止判断只看最后一个 link 区域（原来是 'Normal termination' in
        # "".join(links[-1].lines)）
        self.normal_termination = regions[-1].retained.normal_termination
        self.error_termination = regions[-1].retained.error_termination
        self.summary = self.find_summary(regions)
        self.get_last_coords()
        if self.original_filename:
            for coordinate_object in self.all_coords + [self.last_coord, getattr(self.summary, "coordinate", None)]:
                if coordinate_object is not None:
                    coordinate_object.source_path = str(self.original_filename)
        self.get_routes(regions)

        self.method = ""
        self.basis = ""
        self.mixed_basis_str = ""
        self.mixed_basis_list = []
        # 读不到基函数计数时的占位值。语义是「未知，与任何值比较都不相等（含自身）」，
        # NaN 的 IEEE 语义恰好如此；唯一的消费点是 Gaussian_Output 里
        # last_step.basis_counting == step.basis_counting 这个判断。
        self.basis_counting = math.nan
        self.get_level(regions)

        self.opt_energies = []
        self.get_opt_energies(regions)

        self.converged = [[], [], [], []]
        self.get_converged(regions)

        self.last_scf_iteration = []
        for link in reversed(self.links):
            if link.scf_iteration:
                self.last_scf_iteration = link.scf_iteration
                break

        if 'irc' in self.route_dict:
            self.is_IRC = True
            self.get_irc_coords(regions)

        self.frozen_bonds = []  # a list of 2-tuples represents the frozen bonds
        self.get_freeze(regions)

        self.get_freq_result(regions)

        self.chk_filename = ""
        for region in regions:
            for block in region.retained.chk_blocks:
                re_ret = re.findall(r"\%chk\=(.+)", block[0])
                if not re_ret:
                    continue
                self.chk_filename = re_ret[0]
                for continuation in block[1:]:  # Gaussian一行显示不完chk的文件名
                    if '.chk' in self.chk_filename:
                        break
                    self.chk_filename += continuation[1:].strip('\n')
                break
            if self.chk_filename:
                break

        # Warning, this method only applies to single determinant methods.
        self.is_relaxed_scan = False
        self.converged_relaxed_scan_structures = []
        import collections
        self.converged_relaxed_scan_structures_dict = collections.OrderedDict()  # dict, key is the structure, value is a tuple of step and energy
        self.get_relaxed_scan(regions)

        self.is_solvated = False  # SCF能量里带没带溶剂化
        self.solvent = ""
        self.read_step_solvent(regions)

        self.is_opt = 'opt' in self.route_dict

        # 每步的 CPU 时间（规则 26）
        self.job_cpu_time = None
        self.job_cpu_time_components = None
        self.read_job_cpu_time(regions)

        # 轨道能量与激发态（自 Lib_Obsolete 移植）
        self.alpha_occupied_orbitals = None
        self.alpha_virtual_orbitals = None
        self.beta_occupied_orbitals = None
        self.beta_virtual_orbitals = None
        self.alpha_HOMO = None
        self.alpha_LUMO = None
        self.beta_HOMO = None
        self.beta_LUMO = None
        self.read_orbital_properties(regions)

        self.excitations = None
        self.read_excitation_energies(regions)

    # ------------------------------------------------------------------
    # IOp(9/40) —— TDDFT 激发组分的打印阈值
    # ------------------------------------------------------------------
    @property
    def iop_9_40_value(self) -> int | None:
        """本步骤路由里 ``IOp(9/40=N)`` 的 N；没写这个 IOp 时为 ``None``。"""
        if not isinstance(self.route_dict, Route_Dict):
            return None  # 没解析出路由时 route_dict 是空的普通 dict
        return self.route_dict.iop_9_40_value

    @property
    def iop_9_40_threshold(self) -> float | None:
        """本步骤的 CI 展开系数打印阈值 ``10^-N``；没写这个 IOp 时为 ``None``。"""
        if not isinstance(self.route_dict, Route_Dict):
            return None  # 没解析出路由时 route_dict 是空的普通 dict
        return self.route_dict.iop_9_40_threshold

    @property
    def has_iop_9_40(self) -> bool:
        """本步骤是否写了 ``IOp(9/40=...)``。"""
        return self.iop_9_40_value is not None

    @property
    def is_excited_state(self) -> bool:
        """本步骤是不是激发态任务（``TD`` / ``TDA`` / ``CIS``）。"""
        if not isinstance(self.route_dict, Route_Dict):
            return False  # 没解析出路由时 route_dict 是空的普通 dict
        return self.route_dict.is_excited_state

    def read_step_solvent(self, regions):
        for region in regions:
            if region.num == 301:
                retained = region.retained
                if not retained.pcm_indices:
                    continue
                # 原代码只处理区域里第一条 PCM 行，从它往后找第一条 Solvent 行
                pcm_index = retained.pcm_indices[0]
                for solvent_index, solvent_line in retained.solvent_hits:
                    if solvent_index < pcm_index:
                        continue
                    re_ret = re.findall(r"Solvent\s+:\s*(.+?),", solvent_line)
                    if re_ret:
                        self.is_solvated = True
                        self.solvent = re_ret[0]
                        break

    def get_irc_coords(self, regions):
        for count, region in enumerate(regions):
            if region.retained.irc_calculating_found:
                for coord_link in reversed(self.links[:count]):
                    if coord_link.coords:
                        assert len(coord_link.coords) == 1, 'No. of IRC coords not 1'
                        self.IRC_coords.append(coord_link.coords[0])
                        break

    def get_freq_result(self, regions):
        if "freq" in self.route_dict and 'opt' not in self.route_dict:
            self.has_freq = True

            if 'genchk' in self.route_dict:
                self.is_freq_step_after_opt = True

            for region in reversed(regions):
                if region.num == 716:
                    retained = region.retained

                    # get T, P, rotation symm
                    for line in retained.freq_lines:
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

                    # moment_of_inertia in SI
                    if hasattr(self, 'rotation_constants'):
                        self.moment_of_inertia = [h / (B * 1E9) / 8 / pi ** 2 for B in self.rotation_constants]
                    else:
                        self.moment_of_inertia = [0, 0, 0]

                    # get isotopes
                    # "Atom     1 has atomic number  6 and mass  12.00000"
                    self.isotopes = []
                    for line in retained.freq_lines:
                        re_ret = re.findall(r"Atom\s+\d+\s+has atomic number\s+\d+\s+and mass\s+(\d+\.\d+)", line)
                        if re_ret:
                            self.isotopes.append(float(re_ret[0]))

                    # get frequencies
                    self.harmonic_freqs = []
                    freq_reg = r"Frequencies\s+--\s+(-*\d+\.\d+)\s*(-*\d+\.\d+)*\s*(-*\d+\.\d+)*"
                    for line in retained.freq_lines:
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

                    for line in retained.freq_lines:
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

                    for block in retained.entropy_blocks:
                        if re.findall(entropy_lead_reg, block[0]):
                            match = re.findall(entropy_reg, block[2])
                            if match:
                                self.S = float(match[0][2]) * 4.184
                            break
                    break

            self.imaginaries = [x for x in self.harmonic_freqs if x < 0]
            self.imaginary_count = len(self.imaginaries)

            if self.imaginary_count != 0:
                self.has_imaginary_freq = True

    def find_summary(self, regions):

        l9999_positions = [count for count, region in enumerate(regions) if region.num == 9999]

        if len(l9999_positions) != 1:
            if "Entering Gaussian System" not in self.links[0].first_line:
                return ""
        else:
            position = l9999_positions[0]
            for summary in regions[position].retained.summary_blocks:
                if summary:
                    ret = Gaussian_Summary(summary)
                    link = self.links[position]
                    # refresh() 会复用已经建好的 link 对象重建未完结的步骤，
                    # 这里要保证归档坐标只往它的 coords 里塞一次
                    if not link._summary_coordinate_appended:
                        link.coords.append(ret.coordinate)
                        link._summary_coordinate_appended = True
                    return ret

    def get_last_coords(self):

        self.last_coord = Coordinates()

        self.all_coords = sum([link.coords for link in self.links if link.coords], [])
        if self.all_coords:
            self.last_coord = self.all_coords[-1]

        # last_coord 与 all_coords[-1] 是同一个对象，下面那两句会把电荷 / 自旋多重度
        # 就地写进它。refresh() 重建未完结的步骤时会复用同一批 Coordinates 对象，
        # 而「最后一个坐标」会随着文件增长换成另一个对象——所以每次重建前先把上一
        # 轮盖上去的值还原，否则被盖过的那个对象会一直带着不属于它的电荷。
        for coordinate_object in self.all_coords:
            stamped = getattr(coordinate_object, "_charge_before_last_coord_stamp", None)
            if stamped is None:
                coordinate_object._charge_before_last_coord_stamp = (
                    coordinate_object.charge, coordinate_object.multiplicity)
            else:
                coordinate_object.charge, coordinate_object.multiplicity = stamped

        correct_c_and_m = [coord for coord in self.all_coords if coord.charge != 999]
        if correct_c_and_m:
            self.last_coord.charge = correct_c_and_m[-1].charge
            self.last_coord.multiplicity = correct_c_and_m[-1].multiplicity

    def get_freeze(self, regions):

        if 'opt' in self.route_dict and 'modredundant' in self.route_dict['opt']:
            for region in regions:
                if region.num == 101:
                    region_lines = region.retained.region_lines or []
                    for count, line in enumerate(region_lines):
                        if 'The following ModRedundant input section has been read:' in line:
                            for line2 in region_lines[count + 1:]:
                                re_ret = re.findall(r"B\s+(\d+)\s+(\d+)\s+F", line2)
                                if re_ret:
                                    self.frozen_bonds.append([int(x) - 1 for x in re_ret[0]])
                                else:
                                    break
                            break

    def get_routes(self, regions):

        self.route = ""
        for region in regions:
            if region.num == 1:
                for route_block in region.retained.route_blocks:
                    for route_line in route_block:
                        if '---' in route_line:
                            break
                        if route_line[0] != " ":
                            print("Route Line Process Error!")
                        self.route += route_line[1:].strip('\n')
                break
        self.route_dict = Route_Dict(self.route, remove_genchk=False)

    def get_level(self, regions):
        basis_count = -1
        primitive_count = -1
        cartesian_count = -1
        alpha_count = -1
        beta_count = -1

        if "level" in self.route_dict and ("genecp" in self.route_dict['level'] or 'gen' in self.route_dict['level']):
            self.mixed_basis_list = []
            for region in regions:
                if region.num == 301:
                    retained = region.retained
                    basis_list = []
                    current_basis = []

                    #   394 basis functions,   659 primitive gaussians,   407 cartesian basis functions
                    #   55 alpha electrons       55 beta electrons

                    # 原代码从头逐行扫，谁先出现谁生效
                    from_chk_index = retained.basis_from_chk_index
                    general_index = retained.general_basis_index
                    if from_chk_index is not None and (general_index is None or from_chk_index < general_index):
                        self.mixed_basis_list.append("Check")
                    elif general_index is not None:
                        for basis_line in retained.general_basis_lines:  # 从这一行开始读
                            if "Ernie" in basis_line:
                                break
                            if "****" in basis_line:
                                current_basis.append(basis_line)
                                basis_list.append(current_basis)
                                current_basis = []
                            else:
                                current_basis.append(basis_line)

                    for basis in basis_list:
                        if "****" in basis[-1]:
                            for basis_line in basis:
                                if "Centers:" not in basis_line and "****" not in basis_line:
                                    self.mixed_basis_list.append(basis_line.strip())

                    if -1 in (basis_count, primitive_count, cartesian_count, alpha_count, beta_count):
                        for line, next_line in retained.basis_counting_pairs:
                            re_ret = re.findall(r'''(\d+)\s*basis functions,\s*(\d+)\s*primitive gaussians,\s*(\d+)\s*cartesian basis functions''', line)
                            if re_ret:
                                (basis_count, primitive_count, cartesian_count) = [int(x) for x in re_ret[0]]
                                re_ret = re.findall(r'''(\d+)\s*alpha electrons\s*(\d+)\s*beta electrons''', next_line)
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

    def get_opt_energies(self, regions):
        for region in regions:
            if region.num == 502:
                re_pattern = r"SCF Done:  E\([0-9A-Za-z\-]+\) =\s+(-*\d+\.\d+E*-*\d*)"
                for line in region.retained.scf_done_lines:
                    match = re.findall(re_pattern, line)
                    if len(match) > 0:
                        self.opt_energies.append(''.join(match[0]))
                        break

        if not self.opt_energies:
            # External 读入能量的时候在L402
            for region in regions:
                if region.num == 402:
                    re_pattern = r"Energy\s*=\s+(-*\d+\.\d+)"
                    for line in region.retained.external_energy_lines:
                        match = re.findall(re_pattern, line)
                        if len(match) > 0:
                            self.opt_energies.append(''.join(match[0]))
                            break

    def get_relaxed_scan(self, regions):

        # Warning, this method only applies to single determinant methods.
        last_step = 0
        if 'opt' in self.route_dict and 'modredundant' in self.route_dict['opt']:
            for region in regions:
                if region.num == 103:
                    if region.retained.scan_count_found:
                        self.is_relaxed_scan = True
                        for count, region103 in enumerate(regions):
                            if region103.num == 103:
                                if region103.retained.optimization_completed_found:
                                    relaxed_scan_step = None
                                    relaxed_scan_energy = None
                                    relaxed_scan_structure = None

                                    relaxed_scan_step = count
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

    def get_converged(self, regions):
        for region in regions:
            if region.num == 103:
                for block in region.retained.converged_blocks:
                    for k, line in enumerate(block[1:5]):
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

    def read_job_cpu_time(self, regions):
        """每步的 CPU 时间：``Job cpu time:  0 days  0 hours  3 minutes 21.4 seconds.``"""
        for region in regions:
            for line in region.retained.job_cpu_time_lines:
                match = self._JOB_CPU_TIME_PATTERN.findall(line)
                if match:
                    days, hours, minutes, seconds = match[0]
                    self.job_cpu_time_components = (int(days), int(hours), int(minutes), float(seconds))
                    self.job_cpu_time = (int(days) * 86400 + int(hours) * 3600
                                         + int(minutes) * 60 + float(seconds))

    def read_orbital_properties(self, regions):
        """本步骤最后一个 L601 里的轨道能量（Hartree）。

         Alpha  occ. eigenvalues -- -101.70660-101.69120 -19.33984
         Alpha virt. eigenvalues --    0.11538   0.13213   0.14866
        """
        if not self.normal_termination:
            return None
        _alpha_occupied_orbitals = []
        _alpha_virtual_orbitals = []
        _beta_occupied_orbitals = []
        _beta_virtual_orbitals = []
        for region in reversed(regions):
            if region.num == 601:
                for line in region.retained.eigenvalue_lines:
                    if "eigenvalues" in line:
                        re_ret = [float(x) for x in re.findall(r"-*\d+\.\d+", line)]
                    if "Alpha  occ. eigenvalues" in line:
                        _alpha_occupied_orbitals.extend(re_ret)
                    elif "Alpha virt. eigenvalues" in line:
                        _alpha_virtual_orbitals.extend(re_ret)
                    elif "Beta  occ. eigenvalues" in line:
                        _beta_occupied_orbitals.extend(re_ret)
                    elif "Beta virt. eigenvalues" in line:
                        _beta_virtual_orbitals.extend(re_ret)
                break
        else:
            return None

        self.alpha_virtual_orbitals = _alpha_virtual_orbitals if _alpha_virtual_orbitals else None
        self.alpha_LUMO = _alpha_virtual_orbitals[0] if _alpha_virtual_orbitals else None
        self.alpha_occupied_orbitals = _alpha_occupied_orbitals if _alpha_occupied_orbitals else None
        self.alpha_HOMO = _alpha_occupied_orbitals[-1] if _alpha_occupied_orbitals else None
        self.beta_virtual_orbitals = _beta_virtual_orbitals if _beta_virtual_orbitals else None
        self.beta_LUMO = _beta_virtual_orbitals[0] if _beta_virtual_orbitals else None
        self.beta_occupied_orbitals = _beta_occupied_orbitals if _beta_occupied_orbitals else None
        self.beta_HOMO = _beta_occupied_orbitals[-1] if _beta_occupied_orbitals else None

    def read_excitation_energies(self, regions):
        """本步骤最后一个 L914 里的激发态。

         Excited State   1:      Singlet-?Sym    6.3864 eV  194.14 nm  f=0.5808  <S**2>=0.000
        """
        if not self.normal_termination:
            return None
        for region in reversed(regions):
            if region.num == 914:
                excitation_region = region
                break
        else:
            return None

        _excitation_energies = []
        _excitation_amplitudes = []
        _excitation_multiplicity = []
        for line in excitation_region.retained.excited_state_lines:
            if "Excited State" in line:
                re_ret = re.findall(r"Excited State\s+\d+:.+?\s+(-*\d+\.\d+) eV\s+(-*\d+\.\d+)\s+nm\s+f=(-*\d+\.\d+)\s+<S\*\*2>=(\d+\.\d+)", line)
                if re_ret:
                    excitation_energy, wavelength, amplitude, s_2 = re_ret[0]
                    _excitation_energies.append(float(excitation_energy))
                    _excitation_amplitudes.append(float(amplitude))
                    _excitation_multiplicity.append(float(s_2) * 2 + 1)

        self.excitations = [Excitation(*x) for x in zip(_excitation_energies,
                                                        _excitation_amplitudes,
                                                        _excitation_multiplicity)]


class Gaussian_Output_Link:
    """一个 link 区域的解析结果。本类不再持有 ``lines``。"""

    _IRC_POINT_NUMBER_PATTERN = re.compile(r"Point Number:\s+(\d+)\s+Path Number:\s+(\d+)")

    def __init__(self, link_num, retained, leave_time=0, cpu_time=-1):
        self.num = link_num
        self.first_line = retained.first_line
        self.coords = []
        self._summary_coordinate_appended = False  # find_summary 重建时防止重复塞坐标
        self.get_coords(retained)

        self.leave_time = leave_time
        self.cpu_time = float(cpu_time)
        self.date_class = Date_Class(str(link_num), leave_time)

        self.scf_iteration = []
        self.get_scf_iteration(retained)
        self.get_scf_final_result(retained)

        self.irc_point_number = None
        self.irc_path_number = None
        self.read_irc_point_number(retained)

        # L101 区域整段保留：标题与 ModRedundant 回显要用（其余 link 为 None）
        self.region_lines = retained.region_lines

    def get_scf_final_result(self, retained):
        re_pattern = r"SCF Done:  E\([0-9A-Za-z\-]+\) =\s+(-*\d+\.\d+E*-*\d*)"
        for line in retained.scf_done_lines:
            match = re.findall(re_pattern, line)
            if len(match) > 0:
                self.scf_final_energy = float(''.join(match[0]))
                break

    def get_scf_iteration(self, retained):
        if self.num == 502:
            # 保留下来的是所有含 "Cycle" 与 "E= " 的行，连同它们在区域里的行号；
            # 用行号还原原文件行距之后重放原算法，逐值等价。
            line_by_index = dict(retained.scf_iteration_hits)
            cycle_indices = [index for index, line in retained.scf_iteration_hits
                             if "Cycle" in line]
            for i in reversed(cycle_indices):
                match = re.findall("Cycle +([0-9]+) ", line_by_index[i])
                if match:
                    for j in range(1, 100):
                        neighbour = line_by_index.get(i + j)
                        if neighbour is None:
                            continue
                        if len(re.findall("Cycle +([0-9]+) ", neighbour)) > 0:
                            break

                        findEnergy = re.findall(r"E= (-\d+\.\d+)", neighbour)

                        if len(findEnergy) > 0:
                            energy = float(findEnergy[0])
                            self.scf_iteration.append(energy)
                            break

                    if int(match[0]) == 1:
                        break

        # QC
        if self.num == 508:
            for line in reversed(retained.scf_iteration_508_lines):
                match = re.findall(r"Iteration +(\d+) +EE=", line)
                if match:
                    findEnergy = re.findall(r"Iteration +\d+ +EE= (-\d+\.\d+)", line)
                    if len(findEnergy) > 0:
                        energy = float(findEnergy[0])
                        self.scf_iteration.append(energy)

        self.scf_iteration.reverse()

    def get_coords(self, retained):
        if self.num != 9999:

            # 保留窗口里的 marks 与原代码一致：
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

            for (mark, skip, previous_line, hit_line,
                 following, is_coordinate_line) in retained.coordinate_slabs:
                coords = []
                if previous_line is None:
                    # 原代码 self.lines[count - 1] 在 count == 0 时取到的是区域最后一行
                    previous_line = retained.last_line
                next_line = following[0] if following else ""
                charge_context = next_line + '\n' + previous_line
                start = skip
                pre_run_pattern = "Charge"
                if pre_run_pattern in charge_context:
                    re_result = re.findall(r'Charge = +(-*\d+) Multiplicity = +(\d+)',
                                           charge_context)  # 在某些标题的前面一行有，有的后面一行有
                    if re_result:
                        self.charge, self.multiplicity = re_result[0]  # 如果找到了，就使用新的；如果找不到，沿用上一个
                        start += 1

                for index in range(start, len(following)):
                    if is_coordinate_line[index]:  # 确认这一行中存在坐标（扫描时已判定）
                        coords.append(following[index])
                    else:
                        break
                if coords:
                    self.coords.append(Coordinates(coords, self.charge, self.multiplicity))

    def read_irc_point_number(self, retained):
        """IRC 的点号与路径号（规则 21）。"""
        for block in retained.irc_point_number_blocks:
            re_ret = self._IRC_POINT_NUMBER_PATTERN.findall('\n'.join(block))
            # Point Number:   1  (first point)        Path Number:   1 (forward)
            if re_ret:
                self.irc_point_number = int(re_ret[0][0])
                self.irc_path_number = int(re_ret[0][1])
                break


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

    if isinstance(filename, Gaussian_Output):
        output_object = filename
        filename = output_object.filename
    elif os.path.isfile(filename):
        output_object = Gaussian_Output(filename)
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

    # 既有缺陷修复：原来写的是
    #     solvation_gas = output_object.steps[output_object.solvation_steps[0]]
    # 而 solvation_steps 是「每个结构分组一个 [溶剂化步骤对象, 气相步骤对象] 列表」，
    # 拿它当步骤下标必然抛 TypeError，这一分支从来没有真正跑通过。现在直接用里面存
    # 的步骤对象，并且只在确实识别出配对时才写 __solvation.out；processed 里也改为
    # 存步骤序号（原来存的是步骤对象列表，与后面的 `count in processed` 永远不相等）。
    solvation_pairs = [pair for pair in output_object.solvation_steps if pair]

    if solvation_pairs:
        solvation_sol, solvation_gas = solvation_pairs[0]
        solvation_sol_index = output_object.steps.index(solvation_sol)
        solvation_gas_index = output_object.steps.index(solvation_gas)
        processed += [solvation_sol_index, solvation_gas_index]

        ret = ""

        if solvation_sol_index != 0:
            ret += header

        ret += output_object.step_text(solvation_sol_index) + '\n'
        ret += output_object.step_text(solvation_gas_index) + '\n'

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

        ret += output_object.step_text(count) + '\n'
        processed.append(count)
        if 'opt' in step.route_dict and 'freq' in step.route_dict and count + 1 < len(output_object.steps):
            ret += output_object.step_text(count + 1) + '\n'
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


#: TDDFT 激发段落的标题行；:func:`clean_gaussian_tddft_9_40_small_contributions`
#: 从它出现处开始过滤小组分行。
_GAUSSIAN_TDDFT_SECTION_MARKER = "Excitation energies and oscillator strengths:"


def check_gaussian_tddft_sections_terminated_normally(filename) -> tuple[bool, str]:
    """检查每一个含 TDDFT 激发段落的输出步骤是否都正常结束（Normal termination）。

    :func:`clean_gaussian_tddft_9_40_small_contributions` 只允许处理「对应步骤
    已经正常结束」的输出（2026-08-19 用户裁定）：中途死掉的 TDDFT 输出激发段落
    残缺，为它生成精简副本既浪费空间、又会误导下游把副本当成算完的结果。本函数
    就是那条门槛，也可供调用方（如 :mod:`HPC_Lib.HPC_Gaussian_Postprocess`）在
    批量清理前先行判断、给出友好的跳过信息。

    判定按**输出步骤**进行（以 ``l1.exe`` 行切分，与
    :func:`scan_gaussian_output_step_terminations` 同一套切法）：找出激发段落
    标题行（``Excitation energies and oscillator strengths:``）所在的每一个输出
    步骤，要求它们全部出现 ``Normal termination``。含激发段落的步骤都正常、
    而别的步骤失败（例如 TD 正常、Gaussian 自动追加的 freq 步骤死了）不影响
    判定通过。整个文件里一个激发段落都没有（任务在进入 TDDFT 之前就死了、或
    根本不是 TDDFT 任务）判为不通过——「对应的步骤」不存在，精简无从谈起。
    逐行流式扫描、常数内存，几百 MB 的输出可以安全处理。

    Args:
        filename: Gaussian 输出文件路径（``.out`` / ``.log``，精简副本也可以）。

    Returns:
        ``(通过与否, 原因)``；通过时原因为空字符串。
    """
    output_step_flags: list[dict] = []
    current_step_flags = {"has_section": False, "normal": False}
    inside_l1_run = False

    with open(filename, encoding="utf-8", errors="ignore") as gaussian_output_file:
        for line in gaussian_output_file:
            # 一段连续的 l1.exe 行（初始命令行 + "Entering Link 1 = ..."）只
            # 开启一个输出步骤。
            if "l1.exe" in line:
                if not inside_l1_run:
                    output_step_flags.append(current_step_flags)
                    current_step_flags = {"has_section": False, "normal": False}
                inside_l1_run = True
                continue
            inside_l1_run = False
            if _GAUSSIAN_TDDFT_SECTION_MARKER in line:
                current_step_flags["has_section"] = True
            elif "Normal termination" in line:
                current_step_flags["normal"] = True
    output_step_flags.append(current_step_flags)

    steps_with_section = [flags for flags in output_step_flags if flags["has_section"]]
    if not steps_with_section:
        return False, (f"no '{_GAUSSIAN_TDDFT_SECTION_MARKER}' section found in "
                       f"{filename} — the job likely died before reaching TDDFT")
    abnormal_count = sum(1 for flags in steps_with_section if not flags["normal"])
    if abnormal_count:
        return False, (f"{abnormal_count} of {len(steps_with_section)} output step(s) "
                       f"containing the TDDFT excitation section did not terminate "
                       f"normally in {filename}")
    return True, ""


def clean_gaussian_tddft_9_40_small_contributions(filename, threshold=0.05, output_filename=None):
    """
    Remove small-contribution excitation component lines from a Gaussian TDDFT output.

    TDDFT outputs printed with a lowered CI-coefficient printing threshold
    (e.g. IOp(9/40=5)) list every orbital-pair contribution of every excited
    state (lines like ``  57 ->  60    0.00123`` / ``  57 <-  60   -0.00123``)
    in the "Excitation energies and oscillator strengths:" section, which makes
    the file very long. This copies the file, dropping every contribution line
    whose absolute coefficient is below *threshold*; everything before that
    section and all non-contribution lines are kept verbatim.

    Only outputs whose TDDFT step actually finished may be cleaned: every
    output step containing the excitation section must show ``Normal
    termination`` (checked via
    :func:`check_gaussian_tddft_sections_terminated_normally`), otherwise this
    raises instead of producing a cleaned copy of an incomplete output
    (2026-08-19 user ruling).

    Args:
        filename:        Gaussian TDDFT output filepath.
        threshold:       Minimum absolute CI coefficient to keep (default 0.05).
        output_filename: Target filepath; default replaces the appendix with
                         ``clean.out`` (``X.out`` → ``X.clean.out``).

    Returns:
        The output filepath.

    Raises:
        ValueError: If the file has no TDDFT excitation section, or any output
                    step containing one did not terminate normally.
    """
    check_passed, check_problem = check_gaussian_tddft_sections_terminated_normally(filename)
    if not check_passed:
        raise ValueError(f"refusing to clean TDDFT contributions: {check_problem}")

    if output_filename is None:
        output_filename = filename_class(filename).replace_append_to(f"clean_{threshold}.out")

    started = False
    with open(output_filename, 'w', encoding="utf-8") as clean_output_file:
        with open(filename, encoding="utf-8", errors="ignore") as gaussian_output_file:
            for line in gaussian_output_file:
                if _GAUSSIAN_TDDFT_SECTION_MARKER in line:
                    started = True
                if started:
                    coefficient_match = re.findall(r'\d+[abAB]*\s*\-\>\s*\d+[abAB]*\s*(-*\d+\.\d+)', line)
                    if not coefficient_match:
                        coefficient_match = re.findall(r'\d+[abAB]*\s*\<\-\s*\d+[abAB]*\s*(-*\d+\.\d+)', line)
                    if coefficient_match and abs(float(coefficient_match[0])) < threshold:
                        continue
                clean_output_file.write(line)

    return output_filename


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


def list_related_files(out_file, include_missing=False):
    """列出一个 Gaussian 计算任务的全部相关文件（用于存储空间清理循环）。

    输入是该任务的 .out（或 .log）输出文件路径。相关文件按四条途径收集，
    按可靠性从高到低：

    1. **输出文件头部的 ``Input=`` / ``Output=`` 行**——g16 在输出第 2-3 行
       写出输入/输出文件的绝对路径（输入文件可以是 gjf / inp / com 等任意
       后缀、任意目录），这是定位输入文件的权威来源。
    2. **解析输入文件本体的全部 Link 0 段**——多步任务（``--Link1--`` 分隔）
       每一步的 ``%chk`` / ``%oldchk`` / ``%rwf`` 都只在输入文件里完整声明
       （例如各步 chk 名带 ``_StepN[...]`` 后缀、rwf 在独立的 Gaussian_RWF
       目录），这是 chk / rwf 实际落盘位置最完整的来源。
    3. **输出文件里的 ``%chk=...`` 等回显**（备用，输入文件缺失时才有用）：
       注意回显可能出现在数千行的自定义基组（AtFile）回显之后，且行宽按
       80 列截断——本函数扫描全头部区域，并把疑似被截断的路径丢弃。
    4. **同目录、同文件名主干（stem）的惯例扩展名**：对 out 的 stem 与
       输入文件的 stem 各做一遍——gjf / com / gau / inp（输入）、chk / fchk
       （波函数）、rwf / int / d2e / scr（临时读写文件）、xyz（几何）、
       log（另一种输出后缀）。

    参数:
        out_file:        .out / .log 文件路径（文件本身可以不存在——此时只按
                         惯例扩展名推算候选名）。
        include_missing: True 时把「按惯例应该存在但磁盘上没有」的候选路径
                         也一并返回（远端清理时用来生成待删文件名清单——
                         多余的不存在路径对 ``rm -f`` 无害）。

    返回:
        dict，键为类别、值为 Path 列表（绝对路径，已去重）：
        ``{"output": [...], "input": [...], "chk": [...], "fchk": [...],
        "rwf": [...], "other": [...]}``。
        ``include_missing=False``（默认）时只含磁盘上确实存在的文件。

    示例（远端/本地清理循环）::

        rel = list_related_files("Comp_Files/example_job/opt.out")
        # 成功任务：删 rel["chk"] + rel["rwf"]；失败任务：删除全部类别
    """
    from pathlib import Path

    out_file = Path(out_file).absolute()

    categories = {
        "output": [".out", ".log"],
        "input": [".gjf", ".com", ".gau", ".inp"],
        "chk": [".chk"],
        "fchk": [".fchk"],
        "rwf": [".rwf", ".int", ".d2e", ".scr"],
        "other": [".xyz"],
    }
    result = {cat: [] for cat in categories}

    def _add(cat, path):
        path = Path(path)
        if path in result[cat]:
            return
        if path.exists() or include_missing:
            result[cat].append(path)

    def _add_link0(kind, declared, base_dir):
        """把一条 %chk/%oldchk/%rwf 声明归入对应类别。"""
        declared = Path(declared)
        if not declared.is_absolute():
            declared = base_dir / declared
        cat = "rwf" if kind.lower() == "rwf" else "chk"
        if not declared.suffix:
            declared = declared.with_suffix(".rwf" if cat == "rwf" else ".chk")
        _add(cat, declared)
        if cat == "chk":
            _add("fchk", declared.with_suffix(".fchk"))

    percent_re = re.compile(r"^\s*%(oldchk|chk|rwf)\s*=\s*(.+?)\s*$",
                            re.IGNORECASE)

    # 1) 输出文件头部的 Input= / Output= 行（g16 第 2-3 行；权威输入路径）
    input_file = None
    io_re = re.compile(r"^\s*(Input|Output)=(.+?)\s*$")
    if out_file.is_file():
        try:
            with open(out_file, encoding="utf-8", errors="replace") as f:
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    match = io_re.match(line)
                    if not match:
                        continue
                    kind, declared = match.groups()
                    declared = Path(declared)
                    if not declared.is_absolute():
                        declared = out_file.parent / declared
                    if kind == "Input":
                        input_file = declared
                        _add("input", declared)
                    else:
                        _add("output", declared)
        except OSError:
            pass

    # 2) 解析输入文件全部 Link 0 段的 %chk / %oldchk / %rwf（最完整来源；
    #    多步任务每步声明各自的 chk，rwf 常在别的目录）
    input_parsed = False
    if input_file is None:
        # 没有 Input= 行（如手工重命名过的 .log）：按惯例后缀在 out 同目录找
        for ext in categories["input"]:
            cand = out_file.with_suffix(ext)
            if cand.is_file():
                input_file = cand
                break
    if input_file is not None and input_file.is_file():
        try:
            for line in input_file.read_text(
                    encoding="utf-8", errors="replace").splitlines():
                match = percent_re.match(line)
                if match:
                    _add_link0(match.group(1), match.group(2),
                               input_file.parent)
            input_parsed = True
        except OSError:
            pass

    # 3) 输出文件里的 % 回显（备用）：只在拿不到输入文件本体时才需要。
    #    回显可能排在数千行基组回显之后（扫描窗口放宽到 20000 行），且按
    #    80 列截断——行长达到截断宽度且路径在磁盘上不存在的，视为截断丢弃。
    if out_file.is_file() and not input_parsed:
        try:
            with open(out_file, encoding="utf-8", errors="replace") as f:
                for _ in range(20000):
                    line = f.readline()
                    if not line:
                        break
                    match = percent_re.match(line)
                    if not match:
                        continue
                    kind, declared = match.groups()
                    if len(line.rstrip("\r\n")) >= 79 \
                            and not Path(declared).exists():
                        continue  # 疑似 80 列截断的不完整路径
                    _add_link0(kind, declared, out_file.parent)
        except OSError:
            pass

    # 4) 惯例扩展名：out 的 stem 与输入文件的 stem 各展开一遍
    stems = [out_file.with_suffix("")]
    if input_file is not None:
        in_stem = input_file.with_suffix("")
        if in_stem not in stems:
            stems.append(in_stem)
    for stem_base in stems:
        for cat, exts in categories.items():
            for ext in exts:
                candidate = stem_base.with_suffix(ext)
                if candidate == out_file and cat != "output":
                    continue
                _add(cat, candidate)

    return result


# ---------------------------------------------------------------------------
# chk 清理：formchk 成功之后删掉已经没用的 chk
# ---------------------------------------------------------------------------

#: formchk 转换成功的判据（沿用 clean_chk.py 里的经验值）：fchk 与 chk 的体积
#: 比超过 5%，或者 fchk 本身超过 300 kB。formchk 中途失败留下的 fchk 通常只有
#: 几 kB，两条判据都过不了。
FORMCHK_SUCCESS_SIZE_RATIO = 0.05
FORMCHK_SUCCESS_MINIMUM_SIZE = 300000

#: formchk 产物可能用的扩展名（Linux 下的 g16 写 .fchk，个别版本写 .fch）。
FCHK_EXTENSIONS = (".fchk", ".fch")


def find_fchk_file(chk_file):
    """找出 *chk_file* 对应的、磁盘上确实存在的 fchk 文件。

    Args:
        chk_file: chk 文件路径。

    Returns:
        存在的 fchk 文件路径（``pathlib.Path``）；一个都不存在时返回 ``None``。
    """
    from pathlib import Path

    chk_file = Path(chk_file)
    for extension in FCHK_EXTENSIONS:
        candidate = chk_file.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def formchk_conversion_succeeded(chk_file, fchk_file=None) -> tuple[bool, str]:
    """判断一个 chk 文件是否已经被 formchk 成功转换成了 fchk。

    判据与 ``clean_chk.py`` 相同：fchk 与 chk 的体积比超过
    :data:`FORMCHK_SUCCESS_SIZE_RATIO`，或者 fchk 本身超过
    :data:`FORMCHK_SUCCESS_MINIMUM_SIZE` 字节。转换到一半失败的 fchk 只有几
    kB，两条都满足不了。

    Args:
        chk_file:  chk 文件路径。
        fchk_file: 对应的 fchk 文件路径；为 ``None`` 时用 :func:`find_fchk_file`
                   自动查找。

    Returns:
        ``(是否转换成功, 说明文字)``。说明文字可以直接打印给用户看。
    """
    from pathlib import Path

    chk_file = Path(chk_file)
    if not chk_file.is_file():
        return False, "chk file does not exist"

    if fchk_file is None:
        fchk_file = find_fchk_file(chk_file)
    if fchk_file is None:
        return False, "no fchk file found"

    fchk_file = Path(fchk_file)
    if not fchk_file.is_file():
        return False, f"fchk file does not exist: {fchk_file.name}"

    try:
        chk_size = chk_file.stat().st_size
        fchk_size = fchk_file.stat().st_size
    except OSError as error:
        return False, f"cannot stat chk / fchk: {error}"

    if fchk_size == 0:
        return False, "fchk file is empty"
    if chk_size == 0:
        # chk 是空的，谈不上「转换成功」，也没有删除的价值，交给上层跳过。
        return False, "chk file is empty"

    size_ratio = fchk_size / chk_size
    if size_ratio > FORMCHK_SUCCESS_SIZE_RATIO or fchk_size > FORMCHK_SUCCESS_MINIMUM_SIZE:
        return True, f"fchk/chk size ratio {size_ratio:.3f}, fchk {fchk_size} bytes"
    return False, (f"fchk looks truncated: size ratio {size_ratio:.3f} "
                   f"(<= {FORMCHK_SUCCESS_SIZE_RATIO}) and fchk only {fchk_size} bytes")


def scan_gaussian_output_step_terminations(gaussian_output_file) -> list[dict]:
    """流式扫描 Gaussian 输出文件，按「输入文件里的步骤」给出终止状态。

    输出文件按 ``l1.exe`` 行切分成若干输出步骤（与 :class:`Gaussian_Output`
    的切分方式一致）。注意输出步骤与**输入文件的步骤**并不是一一对应的：
    一个写成 ``opt freq`` 的输入步骤，Gaussian 会自动追加一个 freq 输出步骤，
    它的路由里带 ``GenChk``（``geom=allcheck guess=tcheck`` 这些是用户自己也会
    写的，``GenChk`` 则只有 Gaussian 自动生成的步骤才有）。本函数按这个标志把
    自动追加的输出步骤并回它所属的输入步骤。

    整个扫描逐行进行、常数内存，因此几百 MB 的 TDDFT 输出也可以安全处理——
    不像 :class:`Gaussian_Output` 会把整个文件读进内存。

    Args:
        gaussian_output_file: ``.out`` / ``.log`` 文件路径。

    Returns:
        每个输入步骤一个字典，按出现顺序排列::

            {"normal": bool,               # 该输入步骤的每个输出步骤都正常结束
             "error":  bool,               # 出现过 Error termination
             "output_step_count": int,     # 该输入步骤对应几个输出步骤
             "routes": list[str]}          # 各输出步骤回显的路由
    """
    output_steps: list[dict] = []
    inside_l1_run = False
    collecting_route = False
    current_route = ""

    with open(gaussian_output_file, encoding="utf-8", errors="ignore") as output_file:
        for line in output_file:
            # 一段连续的 l1.exe 行（初始命令行 + "Entering Link 1 = ..."）只开启
            # 一个输出步骤。
            if "l1.exe" in line:
                if not inside_l1_run:
                    output_steps.append({"route": "", "route_done": False,
                                         "normal": False, "error": False})
                    collecting_route = False
                    current_route = ""
                inside_l1_run = True
                continue
            inside_l1_run = False

            if not output_steps:
                continue
            current_output_step = output_steps[-1]

            # 路由回显：以 " #" 打头，到下一条 "-----" 分隔线为止。
            if not current_output_step["route_done"]:
                if collecting_route:
                    if "---" in line:
                        collecting_route = False
                        current_output_step["route"] = current_route.strip()
                        current_output_step["route_done"] = True
                    else:
                        current_route += line[1:].rstrip("\n") if line.startswith(" ") else line.rstrip("\n")
                elif line.startswith(" #"):
                    collecting_route = True
                    current_route = line[1:].rstrip("\n")

            if "Normal termination" in line:
                current_output_step["normal"] = True
            elif "Error termination" in line:
                current_output_step["error"] = True

    # 把 Gaussian 自动追加的输出步骤并回它所属的输入步骤
    input_steps: list[dict] = []
    for output_step in output_steps:
        is_auto_generated = "genchk" in output_step["route"].lower() and input_steps
        if is_auto_generated:
            target = input_steps[-1]
        else:
            target = {"normal": True, "error": False,
                      "output_step_count": 0, "routes": []}
            input_steps.append(target)
        target["output_step_count"] += 1
        target["routes"].append(output_step["route"])
        target["normal"] = target["normal"] and output_step["normal"]
        target["error"] = target["error"] or output_step["error"]

    return input_steps


def _locate_gaussian_input_file(gaussian_output_file):
    """定位一个 Gaussian 输出文件对应的输入文件。

    首选输出文件头部 g16 写的 ``Input=`` 行（权威来源，输入文件可以在别的目录、
    用任意后缀）；没有这一行（例如手工改过名的 ``.log``）时，退回到在输出文件
    同目录下按惯例后缀找同名文件。

    Returns:
        ``pathlib.Path``；找不到时返回 ``None``。
    """
    from pathlib import Path

    gaussian_output_file = Path(gaussian_output_file)
    input_output_pattern = re.compile(r"^\s*(Input|Output)=(.+?)\s*$")
    try:
        with open(gaussian_output_file, encoding="utf-8", errors="replace") as output_file:
            for _ in range(10):
                line = output_file.readline()
                if not line:
                    break
                match = input_output_pattern.match(line)
                if match and match.group(1) == "Input":
                    declared = Path(match.group(2))
                    if not declared.is_absolute():
                        declared = gaussian_output_file.parent / declared
                    if declared.is_file():
                        return declared
                    break
    except OSError:
        pass

    for extension in (".gjf", ".com", ".gau", ".inp"):
        candidate = gaussian_output_file.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def _input_step_chk_and_oldchk_files(gaussian_input_file) -> tuple[list, list]:
    """按步骤顺序取出一个 Gaussian 输入文件里各步骤声明的 ``%chk`` 与 ``%oldchk``。

    首选用 :class:`Gaussian_Input` 解析；它对文件结构（标题段、电荷自旋行）有
    要求，遇到解析不了的文件时退回到最朴素的按 ``--Link1--`` 切分再找 ``%chk=``
    / ``%oldchk=`` 行。

    Returns:
        ``(chk 列表, oldchk 列表)``，两者都是每个输入步骤一项（``pathlib.Path``
        或 ``None``，该步骤没有声明时为 ``None``）。相对路径按输入文件所在目录
        补全，没有扩展名的补上 ``.chk``（Gaussian 自己也是这样补的）。
    """
    from pathlib import Path

    gaussian_input_file = Path(gaussian_input_file)
    try:
        gaussian_input = Gaussian_Input(str(gaussian_input_file))
        declared_chk = [step.chk for step in gaussian_input.steps]
        declared_oldchk = [step.oldchk for step in gaussian_input.steps]
    except Exception:
        try:
            raw_text = gaussian_input_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return [], []
        declared_chk = []
        declared_oldchk = []
        for step_text in re.split(r"(?im)^\s*--link1--\s*$", raw_text):
            chk_match = re.search(r"(?im)^\s*%chk\s*=\s*(.+?)\s*$", step_text)
            oldchk_match = re.search(r"(?im)^\s*%oldchk\s*=\s*(.+?)\s*$", step_text)
            declared_chk.append(chk_match.group(1) if chk_match else "")
            declared_oldchk.append(oldchk_match.group(1) if oldchk_match else "")

    def resolve_declared(declared_list):
        resolved_list = []
        for declared in declared_list:
            declared = (declared or "").strip()
            if not declared:
                resolved_list.append(None)
                continue
            declared_path = Path(declared)
            if not declared_path.is_absolute():
                declared_path = gaussian_input_file.parent / declared_path
            if not declared_path.suffix:
                declared_path = declared_path.with_suffix(".chk")
            resolved_list.append(declared_path)
        return resolved_list

    return resolve_declared(declared_chk), resolve_declared(declared_oldchk)


def clean_chk_files(gaussian_output_file, *, dry_run: bool = False,
                    verbose: bool = True) -> dict:
    """formchk 跑完之后，删掉已经没有保留价值的 chk 文件。

    在 HPC 的 Gaussian 作业脚本末尾（``formchk`` 执行完之后）调用：chk 文件通常
    是几百 MB 到几 GB，而 fchk 里已经有了后续分析需要的全部信息，只有「还要拿来
    重启后续步骤」的那一个 chk 需要留着。

    规则：

    1. 输出文件里每一个步骤都正常结束（Normal termination）时，全部 chk 都可以
       删除——整个任务已经算完，没有需要重启的步骤。
    2. 第 n 步正常结束、第 n+1 步没有正常结束时，**保留第 n 步的 chk**（重启第
       n+1 步要从它出发），只删掉第 1 步到第 n-1 步的 chk。第 n+1 步及其之后各步
       的 chk 一律不动。
    3. 无论走哪条规则，**删除任何一个 chk 之前都必须确认它已经被 formchk 成功
       转换成 fchk**（判据见 :func:`formchk_conversion_succeeded`）。转换失败的
       chk 会被保留下来，以便重新跑一次 formchk。
    4. 同一个 chk 被多个步骤共用（多步任务常常整个任务只用一个 chk）时，只要还有
       一个「要保留的步骤」在用它，就不删。第 n 步之后各步骤的 ``%oldchk`` 指向的
       chk 同样受保护——重启失败步骤时要读它。

    chk 的文件名取自**输入文件**（由输出文件头部的 ``Input=`` 行定位），不是取自
    输出文件里的 ``%chk=`` 回显。回显本身是完整的（长路径按 80 列折行、续写在下
    一行，拼起来能还原），但它不适合本函数：一是 Gaussian 自动追加的那个 freq
    步骤根本不回显 ``%chk``，回显与输入步骤对不上号；二是 ``%oldchk`` 也要读（判断
    哪些 chk 还被未完成的步骤依赖），而输入文件一次就能按步骤取全 ``%chk`` 与
    ``%oldchk``，不需要拼折行。

    Args:
        gaussian_output_file: ``.out`` / ``.log`` 文件路径。
        dry_run:  为 True 时只判断、只打印，不真的删文件。
        verbose:  为 True（默认）时把每一个文件的处理结果打印出来，作业日志里
                  可以直接看到删了什么、为什么留下什么。

    Returns:
        一个字典::

            {"output_file": str,
             "input_file": str | None,
             "input_step_count": int,          # 输入文件声明了几个步骤
             "finished_step_count": int,       # 输出文件里出现了几个输入步骤
             "all_normal": bool,               # 是否每一步都正常结束
             "last_normal_step_index": int,    # 最后一个正常结束的步骤（0 起，-1 表示第一步就没成功）
             "deleted": list[str],             # 实际删掉（dry_run 时是「本该删掉」）的 chk
             "kept": list[tuple[str, str]],    # (保留的 chk, 保留原因)
             "dry_run": bool}
    """
    from pathlib import Path

    def report(message):
        if verbose:
            print(f"[clean_chk_files] {message}")

    gaussian_output_file = Path(gaussian_output_file).absolute()
    result = {
        "output_file": str(gaussian_output_file),
        "input_file": None,
        "input_step_count": 0,
        "finished_step_count": 0,
        "all_normal": False,
        "last_normal_step_index": -1,
        "deleted": [],
        "kept": [],
        "dry_run": dry_run,
    }

    if not gaussian_output_file.is_file():
        report(f"output file not found, nothing to do: {gaussian_output_file}")
        return result

    gaussian_input_file = _locate_gaussian_input_file(gaussian_output_file)
    if gaussian_input_file is None:
        report(f"cannot locate the input file of {gaussian_output_file.name}; "
               f"no chk file will be touched.")
        return result
    result["input_file"] = str(gaussian_input_file)

    step_chk_files, step_oldchk_files = _input_step_chk_and_oldchk_files(gaussian_input_file)
    result["input_step_count"] = len(step_chk_files)
    if not any(chk_file is not None for chk_file in step_chk_files):
        report(f"{gaussian_input_file.name} declares no %chk; nothing to clean.")
        return result

    step_terminations = scan_gaussian_output_step_terminations(gaussian_output_file)
    result["finished_step_count"] = len(step_terminations)
    if len(step_terminations) > len(step_chk_files):
        report(f"WARNING: the output has {len(step_terminations)} job step(s) but "
               f"{gaussian_input_file.name} declares only {len(step_chk_files)}; "
               f"only the first {len(step_chk_files)} are considered.")

    # 从头数出连续正常结束的步骤个数
    normal_prefix_length = 0
    for termination in step_terminations:
        if not termination["normal"]:
            break
        normal_prefix_length += 1

    all_normal = (normal_prefix_length == len(step_terminations)
                  and len(step_terminations) >= len(step_chk_files)
                  and len(step_terminations) > 0)
    result["all_normal"] = all_normal
    result["last_normal_step_index"] = normal_prefix_length - 1

    report(f"{gaussian_output_file.name}: {len(step_terminations)} job step(s) in the "
           f"output, {normal_prefix_length} of them terminated normally in a row "
           f"({len(step_chk_files)} step(s) declared in {gaussian_input_file.name}).")

    if all_normal:
        # 全部算完：所有 chk 都是候选，没有需要保护的步骤。
        deletable_step_indices = range(len(step_chk_files))
        chk_protected_step_indices = range(0)
        oldchk_protected_step_indices = range(0)
        report("every job step terminated normally: all chk files are deletable.")
    else:
        if normal_prefix_length == 0:
            report("the first job step did not terminate normally: no chk file will "
                   "be deleted.")
            return result
        # 保留最后一个正常结束的步骤（第 normal_prefix_length-1 步）及其之后各步
        # 的 chk；重启要从第 normal_prefix_length-1 步的 chk 出发。
        deletable_step_indices = range(normal_prefix_length - 1)
        chk_protected_step_indices = range(normal_prefix_length - 1, len(step_chk_files))
        # 还没算完的步骤（第 normal_prefix_length 步及其之后）读的 %oldchk 也要
        # 保住。注意不包括第 normal_prefix_length-1 步自己的 %oldchk——那一步已经
        # 算完了，它当初读的那个更早的 chk 没有人再需要。
        oldchk_protected_step_indices = range(normal_prefix_length, len(step_chk_files))
        report(f"job step {normal_prefix_length} did not terminate normally: keeping "
               f"the chk of step {normal_prefix_length - 1} for a restart, and "
               f"considering step(s) 0..{normal_prefix_length - 2} for deletion.")

    def resolved(path):
        # normcase：Windows 上统一大小写与斜杠方向，POSIX 上原样返回（那里文件名
        # 是区分大小写的，不能一律转小写）。
        return os.path.normcase(os.path.abspath(str(path)))

    protected_paths = set()
    for path_list, step_indices in ((step_chk_files, chk_protected_step_indices),
                                    (step_oldchk_files, oldchk_protected_step_indices)):
        for step_index in step_indices:
            if step_index < len(path_list) and path_list[step_index] is not None:
                protected_paths.add(resolved(path_list[step_index]))

    candidate_paths = []
    seen_candidate_paths = set()
    for step_index in deletable_step_indices:
        chk_file = step_chk_files[step_index] if step_index < len(step_chk_files) else None
        if chk_file is None:
            continue
        resolved_chk_file = resolved(chk_file)
        if resolved_chk_file in protected_paths:
            reason = "still needed by a job step that has to be restarted"
            if (str(chk_file), reason) not in result["kept"]:
                result["kept"].append((str(chk_file), reason))
                report(f"KEEP   {chk_file}  ({reason})")
            continue
        if resolved_chk_file not in seen_candidate_paths:
            seen_candidate_paths.add(resolved_chk_file)
            candidate_paths.append(chk_file)

    for chk_file in candidate_paths:
        if not Path(chk_file).is_file():
            report(f"SKIP   {chk_file}  (chk file does not exist)")
            continue
        succeeded, explanation = formchk_conversion_succeeded(chk_file)
        if not succeeded:
            result["kept"].append((str(chk_file), explanation))
            report(f"KEEP   {chk_file}  ({explanation})")
            continue
        if dry_run:
            result["deleted"].append(str(chk_file))
            report(f"WOULD DELETE {chk_file}  ({explanation})")
            continue
        try:
            os.remove(chk_file)
        except OSError as error:
            result["kept"].append((str(chk_file), f"deletion failed: {error}"))
            report(f"KEEP   {chk_file}  (deletion failed: {error})")
            continue
        result["deleted"].append(str(chk_file))
        report(f"DELETE {chk_file}  ({explanation})")

    report(f"done: {len(result['deleted'])} chk file(s) "
           f"{'would be deleted' if dry_run else 'deleted'}, "
           f"{len(result['kept'])} kept.")
    return result


# Backward-compatible alias
Gaussian_input = Gaussian_Input
Gaussian_input_step = Gaussian_Input_Step
Route_dict = Route_Dict
Gaussian_summary = Gaussian_Summary
Gaussian_output = Gaussian_Output
Gaussian_output_step = Gaussian_Output_Step
Gaussian_output_link = Gaussian_Output_Link

if __name__ == '__main__':
    pass
