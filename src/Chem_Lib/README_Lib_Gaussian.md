# Lib_Gaussian — LLM Index

File:`src/Chem_Lib/Lib_Gaussian.py`

This document indexes the public classes, methods, and top-level functions
defined in`Lib_Gaussian.py` so an LLM can locate the right entry point
without scanning the whole file (~2200 lines).

---

## Top-level classes

###`Gaussian_Input` — parse / modify / write`.gjf` files

Represents a whole Gaussian input file (one or more steps separated by
`--Link1--`). Constructor takes either a file path or raw text.

**Constructor**
-`Gaussian_Input(path_or_text, *, is_text=False)` — parse from path, or
  from raw text when`is_text=True`.`self.path` is`None` for text input.

**Attributes**
-`steps: list[Gaussian_Input_Step]` — one per`--Link1--` block.
-`step_count: int`
-`annotate_lines: list[str]` — all`!...` lines from anywhere in the file.
-`annotates_dict: dict[str, str]` — parsed from`!__KEY__=VALUE`.
-`run_commands: list[str]` — parsed from`!RUN ...`.
-`chk_files: list[str]`,`rwf_files: list[str]`,`nametag: str | None`.

**`IOp(9/40)` — TDDFT CI-coefficient printing threshold** (properties, always
recomputed from the current route):
-`has_iop_9_40: bool` — any step declares`IOp(9/40=...)`.
-`iop_9_40_threshold: float | None` — the file's printing threshold: the
  **smallest** among the steps (the most verbose step decides how big the
  output gets).`None` when no step declares it.
-`iop_9_40_thresholds: list[float | None]` ·`iop_9_40_values: list[int | None]`
  — per step: the threshold`10^-N`, and the raw`N`.

`IOp(9/40=5)` →`1E-5`,`IOp(9/40=2)` →`0.01`. Per the G16 Rev. C.01 IOps
Reference (L913/L914 entry,`Manual_Gaussian_Full.md` →`##### IOp(9/40)`):
`N` sets`ITHR`,`N=0` sets`ITHR=1`, and`threshold = 10^-ITHR` — so`N=0`
maps to`Route_Dict.IOP_9_40_DEFAULT_THRESHOLD` (`0.1`), the same threshold
Gaussian uses when the IOp is absent. **The same IOp means something else under
L906 (MP2 reference wavefunction: 0=HF, 1=CASSCF, 2=HF)**; these properties
always read it as the L913/L914 printing threshold, which is the only use they
serve. The parsing itself lives on`Route_Dict` (see below); the same four
properties exist on`Gaussian_Output` /`Gaussian_Output_Step`, so an input
and its output are queried identically.

**Modification methods** (all accept`step` =`"ALL"` or`int` index):
-`set_nprocshared(step, value)`
-`set_mem(step, mem_mb)`
-`set_chk(step, chk_path)` ·`set_oldchk(...)` ·`set_rwf(...)`
-`set_keyword_option(step, keyword, options=None)`
  — **value-aware** create-or-replace. The keyword is created if absent.
  Options of the form`name=value` are matched by name: setting`maxstep=10`
  replaces an existing`maxstep=5` instead of accumulating both; plain
  options (`tight`) are appended only if missing. Works for IOp entries
  (`3/76=...`) too.`level` accepts`[method, basis]` or a
`"method/basis"` string.
-`remove_keyword_option(step, keyword, options=None)`
  — **value-aware** option removal. An option given without`=` also removes
  any`name=value` option of the same name (removing`maxstep` drops
`maxstep=5`; removing`5/13` drops the`5/13=...` IOp entry). A keyword
  left empty is dropped, except job-type keywords`opt`/`freq`/`scan`/`irc`
  which stay bare.`options=None` behaves like`clear_keyword`.
-`clear_keyword(step, keyword)`
  — drop the keyword together with **all** of its options (e.g.
`clear_keyword(0, "scrf")` removes`scrf=(solvent=water)` entirely).
`level` is reset to the blank placeholder instead of removed.
-`set_route_keyword(step, keyword, options=None)`
  — **replace** the whole option list verbatim (no name matching). Set
`options=[]` to keep keyword alone.
-`set_route_dict(step, route_mapping: dict)`
  — replace the whole route from a plain dict (`level`/keywords → list of
  options).`level` is preserved from current route if not supplied.
-`set_geometry(step, geometry, *, charge=None, multiplet=None)`
  — replace atomic coordinates.`geometry` accepts a`Coordinates` object,
  a list of raw lines, or a list of`std_coordinate` objects. Automatically
  removes`geom=allcheck` from the route.

**Step selection**
-`extract_steps(indices) → Gaussian_Input`
  — new Gaussian_Input containing only the selected steps. Accepts`int`,
`list[int]`,`slice`, or`"ALL"`.

**Construction from Python data**
-`Gaussian_Input.from_scratch(steps, *, annotates=None, run_commands=None)`
  — build a new file from a list of per-step spec dicts. See the docstring
  for the full dict schema; minimum keys are`route_dict` (or`route_str`)
  plus`charge`/`multiplet`/`geometry` (unless`geom=allcheck`).

**Serialisation**
-`to_string() → str` — rebuild the full`.gjf` text.
-`save(filepath=None)` — write to disk; defaults to`self.path`.

**Helpers**
-`generate_job_name(filepath=None)` — SLURM-safe job name, ≤ 60 chars.

Backward-compat alias:`Gaussian_input = Gaussian_Input`.

---

###`Gaussian_Input_Step` — one step inside a`Gaussian_Input`

Per-step state. You usually interact with this via the`Gaussian_Input`
dispatching methods above, but these methods are also directly callable on
a single step.

**Link-0 attributes**:`proc` (0 = not declared; no`%nprocshared` emitted),
`mem_mb` (0 = not declared),`mem` (GB),`chk`,`oldchk`,
`rwf`,`nosave`,`extra_link0`. Serialisation order is fixed:
`%nprocshared`,`%mem`,`%rwf`,`%nosave`,`%oldchk`,`%chk` —`%nosave`
directly below`%rwf` so it deletes the rwf but not the chk.

**Route / body attributes**:`route_str`,`route_dict` (`Route_Dict`),
`title`,`charge`,`multiplet`,`geom` (list of`std_coordinate`),
`geom_raw` (list of str),`other` (list of paragraphs),`is_allcheck`,
`connectivity`,`coordinate` (`Coordinates` or`None`),
`annotate_lines`,`annotates_dict`,`command_lines`.

**`IOp(9/40)` properties**:`has_iop_9_40: bool`,
`iop_9_40_value: int | None` (the raw`N`),`iop_9_40_threshold: float | None`
(`10^-N`). Recomputed from`route_dict` on every access, so they follow route
edits.

**Modification methods**
-`set_nprocshared(value)` ·`set_mem(mem_mb)` ·`set_chk(...)` ·
`set_oldchk(...)` ·`set_rwf(...)`
-`set_keyword_option(keyword, options=None)` /
`remove_keyword_option(keyword, options=None)` /`clear_keyword(keyword)`
  (value-aware — see`Gaussian_Input` above)
-`set_route_keyword(keyword, options=None)` (verbatim replace)
-`set_route_dict(route_mapping: dict)`
-`set_geometry(geometry, *, charge=None, multiplet=None)`

**Construction**
-`Gaussian_Input_Step.from_spec(spec: dict)` — factory used by
`Gaussian_Input.from_scratch`.

**Serialisation**
-`to_string() → str` — reconstruct the step text.

Backward-compat alias:`Gaussian_input_step = Gaussian_Input_Step`.

---

###`Route_Dict` (subclass of`collections.OrderedDict`)

Parsed route section. Keys are lowercase keyword names, values are lists
of option strings.`level` is always`[method, basis]`.

Methods:`add_item(key, option)`,`remove_item(key, option)`,
`remove_key(key)`,`get_keyword(key)`,`option_exist(key, option)`,
`set_keyword_option(key, options=None)`, `remove_keyword_option(key,
options=None)`, `clear_keyword(key)` (the last three are value-aware:
`name=value` options are matched by name; option order is preserved).

Parsing is case-insensitive throughout (Gaussian input itself is
case-insensitive), including`IOp(...)` in any capitalisation, and keeps the
option order of the file: reading and writing a route back does not reshuffle
`opt=(calcfc,ts)`.

`str(route_dict)` regenerates the textual route.

**Route-level judgements** (the single-route logic behind the properties of
the input/output classes):
-`iop_9_40_value: int | None` (property) — the raw`N` of`IOp(9/40=N)`;
  the last one wins when written twice.
-`iop_9_40_threshold: float | None` (property) —`10^-N`;`N=0` →
`Route_Dict.IOP_9_40_DEFAULT_THRESHOLD` (`0.1`, class constant).
-`has_iop_9_40: bool` (property).
-`is_excited_state: bool` (property) — the route is a`TD` /`TDA` /`CIS`
  job; checks both the standalone keywords (`EXCITED_STATE_KEYWORDS` class
  constant) and the method half of`level` (`CIS(NStates=10)/6-31G*` →`cis`,
  spin prefixes`U`/`R`/`RO` stripped;`EXCITED_STATE_METHODS` class constant).
-`Route_Dict.smallest_iop_9_40_threshold(thresholds)` (staticmethod) — the
  multi-step aggregation rule (smallest = most verbose step wins) used by
`Gaussian_Input.iop_9_40_threshold` /`Gaussian_Output.iop_9_40_threshold`.

Backward-compat alias:`Route_dict = Route_Dict`.

---

###`Keyword`

Lightweight parser for a single route token (e.g.`"opt=(calcfc,ts)"`).
Used internally by`Route_Dict`.

---

## Output parsing (read-only)

###`Gaussian_Output`
```python
Gaussian_Output(output, filename="", *, retained_head_line_count=10000,
                retained_tail_line_count=5000, encoding_errors='strict')
```
Parses a`.out`/`.log` file (pass the path; a list of lines is still accepted
for backward compatibility). Splits into`steps: list[Gaussian_Output_Step]`
by`l1.exe` markers; derives summary, title,`[EXTRACT_GEOM]`/`[FROZEN_BONDS]`
hints, groups steps by identical geometry, computes solvation energies.

Key attributes:`steps`,`title`,`extract_geoms`,
`step_groups_by_structure`,`solvation_energy`,`solvation_level`,
`solvent`,`normal_terminated`,`head_lines`,`tail_lines`.

`IOp(9/40)` properties, identical in meaning to the`Gaussian_Input` ones:
`has_iop_9_40`,`iop_9_40_threshold` (smallest across steps),
`iop_9_40_thresholds`,`iop_9_40_values`.

**Memory**: the file is *never* read whole. It is streamed in 8 MB chunks and
only the lines a parsing function actually reads are decoded and kept, so peak
memory is tens of MB regardless of file size (a 517.8 MB TDDFT output parses in
about 3 seconds with a ~16 MB peak). Two consequences:

- There is no`lines` /`steps_list` /`link.lines` any more. Use`head_lines`
  (first`retained_head_line_count` lines) /`tail_lines` (last
`retained_tail_line_count` lines) for raw text,`step_text(step_index)` for
  one step's original text, or`read_output_tail_lines(path, n)` for an
  arbitrary number of trailing lines.
- Bytes outside the retained regions are never decoded, so`encoding_errors`
  (default`'strict'`, matching the old`open(..., encoding='utf-8')`) only
  ever applies to the lines that are actually parsed.

`refresh() -> bool` re-reads whatever has been appended to the source file and
returns whether anything changed; a rewritten file (shorter, different head, or
a changed tail sample) triggers a full re-parse. Objects built from a list of
lines have no source file and always return`False`. Typical cost of one
`refresh()` on an 800 MB file with one new line is milliseconds — this is what
job monitors should use instead of rebuilding the object every poll.

###`Gaussian_Output_Step`
One`l1.exe` segment. Attributes include`normal_termination`,
`error_termination`,`links`,`summary` (`Gaussian_Summary` or`""`),
`route`,`route_dict`,`method`,`basis`,`mixed_basis_str`,
`harmonic_freqs`,`G`/`H`/`S`,`opt_energies`,`converged`, IRC data,
relaxed-scan data,`is_solvated`/`solvent`,`chk_filename`,
`last_coord`,`all_coords`,`job_cpu_time` (seconds,`float`) /
`job_cpu_time_components` (`(days, hours, minutes, seconds)`),
and the orbital / excited-state results below.

Orbital energies come from the step's **last**`l601` block, in Hartree, and
are`None` unless the step terminated normally:
`alpha_occupied_orbitals`,`alpha_virtual_orbitals`,`beta_occupied_orbitals`,
`beta_virtual_orbitals`,`alpha_HOMO`,`alpha_LUMO`,`beta_HOMO`,`beta_LUMO`.

`excitations` is a list of module-level `Excitation(energy, amplitude,
multiplicity)` dataclasses read from the step's **last** `l914` block —
`energy` in eV,`amplitude` the oscillator strength *f*,`multiplicity`
= 2⟨S²⟩+1. It is`None` when the step did not terminate normally or has no
`l914` block.

`basis_counting` is `(basis, primitive, cartesian, alpha electrons, beta
electrons)` when `l301` printed those counts for a gen/genecp job, and
`math.nan` otherwise — "unknown, compares unequal to everything including
itself", which is exactly what its one consumer (deciding whether a gas-phase
single point belongs to the previous step) needs.

`IOp(9/40)` properties:`has_iop_9_40`,`iop_9_40_value`,
`iop_9_40_threshold`.

###`Gaussian_Output_Link`
One`l<num>.exe` block inside a step. Has`num`,`coords`,
`leave_time`,`cpu_time`,`scf_iteration`,`scf_final_energy`,
`first_line` (the block's first line of raw text — use this instead of the
removed`lines[0]`), and`irc_point_number` /`irc_path_number` (from an
`l123` block's`Point Number: … Path Number: …` line;`None` elsewhere).

###`Gaussian_Summary`
Parses the`1\1\...\@` archive at the end of a step. Extracts
`basic_information`,`route`,`route_dict`,`charge`,`multiplet`,
`coordinate`, and`results` dict (HF, ZeroPoint, Thermal, NImag).

Backward-compat aliases:`Gaussian_output = Gaussian_Output`,
`Gaussian_output_step = Gaussian_Output_Step`,
`Gaussian_output_link = Gaussian_Output_Link`,
`Gaussian_summary = Gaussian_Summary`.

---

## Top-level functions

-`build_Gaussian_input_from_template(method_template, coordinates, output_path=None, *, charge=None, multiplet=None, solvent=None, modredundant=None, remove_atoms=None, include_atoms=None, change_elements=None, nprocshared=None, mem_gb=None, mem_mb=None, title=None, first_step_oldchk=None, delete_rwf_after=True, path_mappings=None, remove_first_geom_allcheck=True, validate_solvent=True)`
  — merge a method/route template (`.gjf` path,`Gaussian_Input`, or raw text)
  with a geometry (`Coordinates`, raw lines, or`std_coordinate` list) into a
  ready-to-run input. Fills`solvent=EDITTHIS` (and only the placeholder — a
  concrete solvent in the template is never replaced; a`solvent` argument with
  nothing to fill is silently ignored) and`B EDITTHIS F` placeholders, drops
`Blank_Method`/`Black_Basis` placeholder halves (`PM6D3/Black_Basis` → bare
`pm6d3`), honours`!__FCHK_TAG__` /`!__EXTRACT_GEOM__` /`!__CHK_READ__`
  annotations (an FCHK_TAG used as an`[EXTRACT_GEOM]` label must not be pure
  numeric or contain a comma), auto-chains`%chk`/`%oldchk` with QM_Creater's
  numbering (first step`_Step0`, later step of 0-based index *i* →
`_Step<i+1>`;`_Step1` never occurs) off a chk base that always derives from
  the output filename — only the first step's`%oldchk` can be set from the
  outside, via`first_step_oldchk` —, writes one shared`%rwf` per job (the
  chk base relocated through`path_mappings` onto the matching remote RWF
  prefix — see below; outside every mapping the rwf sits next to the chks) with
  — when`delete_rwf_after=True` (default) — a`%nosave` line immediately below
`%rwf` and above`%oldchk`/`%chk` (so only the rwf is deleted at job end;
`%nosave` affects only the files declared above it).`%nprocshared`/`%mem`
  are written only when the template or the caller declares them — nothing is
  defaulted (`mem_gb` converts with ×1024). Optional geometry editing
  (`remove_atoms` /`include_atoms` as 1-based numbers or a`"1,5,7-9"`
  selection string;`change_elements` as`{1-based atom number: element}`,
  applied by original numbering before filtering; charge/multiplicity are not
  auto-adjusted) and Windows→Linux path conversion via`path_mappings` — a list
  of`(local_prefix, remote_work_prefix, remote_rwf_prefix)` triples. The
  default`None` loads the **user-level configuration**: the variable
`LOCAL_TO_REMOTE_PATH_MAPPINGS` in
`<My_Program>/My_Lib_Configuration_Private.py` (outside version control),
  loaded via`Python_Lib.My_Lib_File.local_to_remote_path_mappings()`; a
  missing configuration raises immediately instead of silently keeping local
  paths. Prefix matching is
  case-insensitive, both slash directions, and requires a path separator after
  the prefix (`D:\Gaussian_Other` never matches`D:\Gaussian`); earlier entries
  win.`%chk`/`%oldchk`/annotation paths get the remote **work** prefix, the
  shared`%rwf` the remote **RWF** prefix (both keep the relative remainder),
  then all`\` →`/`. The remote prefixes use the`%HOME%` home-path
  placeholder HPC_Lib replaces at submission (legacy files carrying
`/home/gauuser` are replaced there too); pass an empty list to keep Windows
  paths. The helper`map_local_path_to_remote(local_path, mappings=None)`
  (defined in`Python_Lib.My_Lib_File`, same`None` = user configuration)
  returns`(remote_work_path, remote_rwf_path)` or`None`. The written file ends with
  seven blank lines, as QM_Creater. Stamps a template
`!__NAMETAG__=[tag]` into
  the output filename by QM_Creater's rule (explicit`output_path`:
  the last`[...]` segment containing a letter is replaced, else the tag is
  appended; no`output_path`: writes next to the geometry's source file —
`Coordinates.source_path`, stamped by`Gaussian_Input` /`Gaussian_Output` —
  with the stamped source stem. A source .gjf that itself carries a
`!__NAMETAG__` whose tag appears in its filename, i.e. a previous product of
  this function, gets exactly that segment replaced; letter-less segments like
`[000001_000002]` are never touched). Never overwrites an existing file: a
  name clash shifts the output to`<name>_01`,`_02`, ... Actual path →
  returned object's`.path` (`None` if the geometry source is unknown and no
`output_path` was given, in which case nothing is written). GUI-free
  QM_Creater workflow; see its docstring.
  **Charge / spin multiplicity are validated before anything is written**, and
  the validation cannot be switched off: every step that carries an explicit
  geometry (a`geom=allcheck` step has none) is checked with
`Coordinates.charge_and_multiplicity_problem()`, and a`ValueError` is raised
  when the pair is an unresolved`999` /`99` placeholder, when the multiplicity
  is not a positive integer, or when the multiplicity contradicts the parity of
  the electron count. The check runs **after**`remove_atoms` /`include_atoms` /
`change_elements`, so deleting an atom without passing the matching`multiplet`
  is rejected here rather than by Gaussian on the cluster.
-`open_with_gview(filename)` — launch GaussView on a`.gjf` path or
`Coordinates` object (Windows-only; silently skips if GView absent).
-`split_gaussian_output_file_steps(filename_or_output)` — split a
  multi-step`.out` into one`.out` per step (keeping`opt+freq` paired
  and gas/sol SMD pair merged). Returns`True` if split,`False` if
  single-step.
-`split_gaussian_file(filename, file_content=None, only_get_file_names=False, required_step=None)`
  — split a`.log` by`Normal termination` boundaries, writing per-step
`.log` files. Returns a list of output filenames (or a single filename
  if`required_step` was a single int).
-`check_gaussian_tddft_sections_terminated_normally(filename) → (bool, str)`
  — check that every output step containing the TDDFT excitation section
  ("Excitation energies and oscillator strengths:") shows`Normal termination`.
  Returns`(ok, reason)`;`reason` is empty when`ok`. Fails when no excitation
  section exists at all (the job died before reaching TDDFT). Streaming,
  constant memory. This is the gate for the cleaning function below.
-`clean_gaussian_tddft_9_40_small_contributions(filename, threshold=0.05, output_filename=None)`
  — copy a TDDFT`.out` printed with a lowered CI-coefficient printing
  threshold (e.g.`IOp(9/40=5)`), dropping excitation contribution lines
  (`57 -> 60  0.00123` /`57 <- 60 -0.00123`) whose absolute coefficient is
  below`threshold`, from the "Excitation energies and oscillator strengths:"
  section onward. Default output filepath:`X.out` →`X.clean_<threshold>.out`.
  Returns the output filepath. **Refuses to run (raises`ValueError`) unless
  every output step containing the excitation section terminated normally**
  (checked via`check_gaussian_tddft_sections_terminated_normally`; 2026-08-19
  user ruling) — a cleaned copy of an incomplete output would waste space and
  mislead downstream consumers. Streams line by line, so the input size does not
  matter. Cleaning is composable: cleaning an already-cleaned file at a coarser
  threshold gives exactly the same result as cleaning the original at that
  threshold, for far less I/O (what`HPC_Gaussian_Postprocess` exploits).
  (The former module helpers`iop_9_40_value_from_route` /
`iop_9_40_threshold_from_route` /`is_excited_state_route` are gone: the
  route-level`IOp(9/40)` and excited-state logic now lives on`Route_Dict`
  as properties — see the`Route_Dict` section.)
-`clean_chk_files(gaussian_output_file, *, dry_run=False, verbose=True) → dict`
  — delete the`.chk` files of a finished Gaussian job that are no longer worth
  keeping. Called at the end of the HPC job script, after`formchk`. Rules:
  every step terminated normally → all chk are deletable; step *n* normal but
  step *n+1* not → keep step *n*'s chk (that is the restart point) and delete
  steps`0..n-1`, leaving step *n+1* onward untouched; step 0 not normal →
  delete nothing. **A chk is only ever deleted after confirming its fchk exists
  and the conversion succeeded.** A chk still referenced by a step that has to
  be re-run — its`%chk`, or the`%oldchk` of any not-yet-finished step (an
`!__CHK_READ__` offset can point several steps back) — is protected.
  chk names come from the **input** file (located via the`Input=` line in the
  output header), not from the output's`%chk=` echo. The echo is complete — a
  long path wraps at 80 columns and continues on the next line — but it is the
  wrong source here: the freq step Gaussian appends to`opt freq` echoes no
`%chk` at all, so echoes do not line up with input steps, and`%oldchk` is
  needed too. Returns a dict with`deleted`,`kept` (path + reason),
`all_normal`,`last_normal_step_index`, ...
-`formchk_conversion_succeeded(chk_file, fchk_file=None) → (bool, str)`
  — did`formchk` produce a usable fchk? Criterion carried over from
`clean_chk.py`: fchk/chk size ratio >`FORMCHK_SUCCESS_SIZE_RATIO` (0.05), or
  fchk larger than`FORMCHK_SUCCESS_MINIMUM_SIZE` (300 kB). A half-written fchk
  is only a few kB and passes neither.`find_fchk_file(chk_file)` returns the
  existing`.fchk`/`.fch` next to a chk, or`None`.
-`scan_gaussian_output_step_terminations(gaussian_output_file) → list[dict]`
  — **constant-memory** streaming scan giving one entry per *input* step:
`{"normal", "error", "output_step_count", "routes"}`. An`opt freq` input step
  produces two output steps in the file (Gaussian appends the freq step, whose
  echoed route carries`GenChk`); this folds those back into the input step they
  belong to. Cheaper than`Gaussian_Output` when only termination status is
  needed, though`Gaussian_Output` itself is no longer memory-hungry.
-`read_output_tail_lines(filename, line_count, encoding_errors='strict')`
  — the last`line_count` lines of an output file, read backwards from the end
  without loading the file. Use when`Gaussian_Output.tail_lines` (which keeps
`retained_tail_line_count` lines) is not enough.
-`split_gaussian_output_file_steps(filename)` — write one`__STEP_n.out` per
  step (and a`__solvation.out` for a recognised solvated/gas pair) by copying
  each step's byte range straight out of the source file. Accepts a path or an
  existing`Gaussian_Output`.
-`print_link_List(data=None, running=False, modify_time=...)` — format
  the link-by-link timing report used by the monitor UI.

---

## Common usage patterns

```python
from Chem_Lib.Lib_Gaussian import Gaussian_Input, Gaussian_Output

# Read → tweak resources → save.
gaussian_input = Gaussian_Input("job.gjf")
gaussian_input.set_nprocshared("ALL", 32)
gaussian_input.set_mem("ALL", 64000)
gaussian_input.save("job_modified.gjf")

# Extract only steps 0 and 2 into a new file.
sub = gaussian_input.extract_steps([0, 2])
sub.save("job_steps_0_2.gjf")

# Replace the route of step 1 entirely from a dict.
gaussian_input.set_route_dict(1, {
    "level": ["b3lyp", "6-31g(d)"],
    "opt":   ["calcfc", "ts"],
    "freq":  [],
})

# Replace geometry of step 0.
gaussian_input.set_geometry(0, ["H 0.0 0.0 0.0", "H 0.0 0.0 0.74"],
                charge=0, multiplet=1)

# Build a brand-new input from scratch.
new = Gaussian_Input.from_scratch([
    {
        "nprocshared": 16, "mem_mb": 32000,
        "chk": "new_job.chk",
        "route_dict": {"level": ["b3lyp", "6-31g(d)"], "opt": []},
        "title": "New job",
        "charge": 0, "multiplet": 1,
        "geometry": ["H 0.0 0.0 0.0", "H 0.0 0.0 0.74"],
    }
], annotates={"NAMETAG": "MyNewJob"})
new.save("new_job.gjf")

# Read an output. Even a 500 MB TDDFT log costs only tens of MB of memory.
gaussian_output = Gaussian_Output("job.out")
print(gaussian_output.normal_terminated, len(gaussian_output.steps))
print(gaussian_output.steps[-1].last_coord)          # geometry
print(gaussian_output.steps[-1].alpha_HOMO)          # orbital energies, Hartree
print(gaussian_output.steps[-1].excitations)         # TDDFT excited states

# Watch a running job: build once, then only read what has been appended.
while True:
    if gaussian_output.refresh():
        print(gaussian_output.steps[-1].links[-1].first_line)
    time.sleep(30)

# Raw text without loading the file.
print("".join(gaussian_output.tail_lines[-20:]))
```
