# -*- coding: utf-8 -*-
"""
My_Lib_HPC - HPC Job Submission Utility
========================================

Purpose:
    This module provides utilities for generating and submitting SLURM job scripts
    on HPC clusters. It auto-detects the current HPC environment by scanning for a
    configuration file, then uses that configuration to build and submit jobs.

Configuration:
    Each HPC environment requires a configuration file placed in the My_Program
    root directory (4 levels above this file):

        {My_Program}/My_Lib_HPC_Configuration_Secret_{HPC_Name}.py

    Exactly ONE such file must exist. The file defines cluster-specific settings
    such as paths, SLURM account, partition/QoS presets, node resources, etc.
    See the example file  My_Lib_HPC_Configuration_Secret_Negishi.py  for all
    available settings and their documentation.

Directory Layout (example):
    E:/My_Program/                                              → My_Program root (found by searching upward for 'My_Program')
    ├── My_Lib_HPC_Configuration_Secret_Negishi.py              → config file
    └── Python_Lib/
        └── src/
            └── Python_Lib/
                └── My_Lib_HPC.py                               → this file

Usage (command line):
    python  .../My_Lib_HPC.py  submit  <python_file>  [--qos PRIORITY]  [--cores N]  [--mem SIZE]  [--email]  [script_args...]

    Commands:
        submit              Submit a file to the HPC cluster
        stop                Cancel one or more jobs
        stop all            Cancel ALL of your currently queued/running jobs

    Submit options:
        <python_file>       Absolute or relative path to the Python script to run
        --qos PRIORITY      Optional. "high" (normal QoS, up to 2 weeks) or
                            "normal" (standby QoS, up to 4 hours, default)
                            Synonyms: --priority, -qos, -priority
        --cores N           Optional. Number of CPU cores to request.
                            If exceeds CORES_PER_NODE, requires confirmation.
                            When specified alone, memory is auto-calculated proportionally.
                            Synonyms: --cpu, -cores, -cpu
        --mem SIZE          Optional. Memory to request (e.g., 20GB, 100000MB, 1TB).
                            Default unit is GB. If exceeds MEMORY_PER_NODE_MB, requires confirmation.
                            Synonyms: --memory, -mem, -memory
        --email             Optional. Send email notification when the job ends
                            (regardless of exit status). Without this flag, email is only
                            sent on failure. Requires MAIL_USER in config.
        [script_args...]    Additional arguments forwarded to the submitted script
        
        Note: All parameters are case-insensitive (--CORES, --Cores, --cores all work).
              Single dash multi-char options (-cores) work the same as double dash (--cores).
              Short options (e.g., -c) are NOT recognized and will be forwarded to the script.

    Interactive mode:
        python  .../My_Lib_HPC.py  submit
        (prompts for file paths, priority, cores, and memory)

    Examples:
        # Basic submission
        python /scratch/.../My_Lib_HPC.py  submit  my_script.py
        
        # High priority
        python /scratch/.../My_Lib_HPC.py  submit  my_script.py  --qos high
        
        # Custom cores (memory auto-calculated)
        python /scratch/.../My_Lib_HPC.py  submit  my_script.py  --cores 16
        
        # Using synonyms and single dash (case-insensitive)
        python /scratch/.../My_Lib_HPC.py  submit  my_script.py  -CPU 8  -MEMORY 50GB
        
        # Custom memory
        python /scratch/.../My_Lib_HPC.py  submit  my_script.py  --mem 50GB
        
        # Custom cores and memory
        python /scratch/.../My_Lib_HPC.py  submit  my_script.py  --cores 8  --mem 32GB
        
        # With script arguments (note: -b is forwarded to script, not intercepted)
        python /scratch/.../My_Lib_HPC.py  submit  my_script.py  --qos high  --cores 16  -b 64  --epochs 100
        
        # With custom script arguments and HPC resources mixed
        python /scratch/.../My_Lib_HPC.py  submit  my_script.py  --parm_for_my_script1 abc  --parm_for_my_script2 cde  --CPU 8
        
        # Stop jobs
        python /scratch/.../My_Lib_HPC.py  stop  12345 12346 12350-12360

        # Stop all YOUR currently queued/running jobs
        python /scratch/.../My_Lib_HPC.py  stop  all
        python /scratch/.../My_Lib_HPC.py  stop  all  qos=high
        python /scratch/.../My_Lib_HPC.py  stop  all  --qos  normal

        # Tail a log file (prints last 200 lines, then follows; waits if file absent)
        python /scratch/.../My_Lib_HPC.py  tail  job_output.log
        python /scratch/.../My_Lib_HPC.py  tail  job_output.log  -n 50
        # Tail by SLURM job ID (auto-resolves StdOut/StdErr path)
        python /scratch/.../My_Lib_HPC.py  tail  34450001

        # Compress files/folders into a .tar.gz archive
        python /scratch/.../My_Lib_HPC.py  compress  /path/to/folder
        python /scratch/.../My_Lib_HPC.py  compress  file1.py  file2.py
        python /scratch/.../My_Lib_HPC.py  compress                         # interactive mode

        # Show node availability
        python /scratch/.../My_Lib_HPC.py  avail
        python /scratch/.../My_Lib_HPC.py  avail  --idle

        # Show SLURM + scheduler queue (your jobs by default)
        python /scratch/.../My_Lib_HPC.py  queue
        python /scratch/.../My_Lib_HPC.py  queue  --all
        python /scratch/.../My_Lib_HPC.py  queue  --user  li5876

        # Show recently completed/failed jobs
        python /scratch/.../My_Lib_HPC.py  history
        python /scratch/.../My_Lib_HPC.py  history  100
        python /scratch/.../My_Lib_HPC.py  history  1d
        python /scratch/.../My_Lib_HPC.py  history  1000  48h

        # Show sub-commands (queue, avail, history, job details)
        python /scratch/.../My_Lib_HPC.py  show  queue
        python /scratch/.../My_Lib_HPC.py  show  avail  --idle
        python /scratch/.../My_Lib_HPC.py  show  history  1d
        python /scratch/.../My_Lib_HPC.py  show  34450001       # detailed info for a SLURM job
        python /scratch/.../My_Lib_HPC.py  show  20260310-111136_01  # detailed info for a scheduler job

        # Add a job to the scheduler queue (deferred submission)
        python /scratch/.../My_Lib_HPC.py  schedule  my_script.py
        python /scratch/.../My_Lib_HPC.py  schedule  my_script.py  --qos high  --priority 100
        python /scratch/.../My_Lib_HPC.py  schedule  my_script.py  --vip  --cores 16
        python /scratch/.../My_Lib_HPC.py  schedule  my_script.py  --after 001,002
        python /scratch/.../My_Lib_HPC.py  schedule  my_script.py  --after-any 003  --priority 50
        python /scratch/.../My_Lib_HPC.py  schedule  cancel  005
        python /scratch/.../My_Lib_HPC.py  schedule  cancel  001  002  003

        # Manage the scheduler daemon
        python /scratch/.../My_Lib_HPC.py  handler                # start daemon
        python /scratch/.../My_Lib_HPC.py  handler  start         # same as above
        python /scratch/.../My_Lib_HPC.py  handler  stop          # cancel running daemon
        python /scratch/.../My_Lib_HPC.py  handler  restart       # stop + start
        python /scratch/.../My_Lib_HPC.py  handler  status        # show daemon status

Usage (as a library):
    from My_Lib_HPC import submit_python_file
    
    # Basic usage
    submit_python_file("/path/to/my_script.py", priority="high")
    
    # With custom resources
    submit_python_file("/path/to/my_script.py", cores=16, memory_mb=32768)
    
    # With script arguments
    submit_python_file("/path/to/my_script.py", 
                      priority="high",
                      cores=8,
                      script_args=["--batch_size", "64", "--epochs", "100"])

Supported File Types (extensible):
    .py        → submit_python_file()    Runs script with configured Python interpreter.
                                         
    .gjf/.com  → submit_Gaussian_file()  Runs Gaussian 16.

Command-line summary:
    submit      Submit a file to the HPC cluster
    stop        Cancel one or more jobs
    tail        Follow a log file (like tail -f)
    compress    Pack files/folders into a .tar.gz archive
    avail       Show node availability (sinfo)  [alias for 'show avail']
    queue       Show SLURM + scheduler queue; defaults to your own jobs  [alias for 'show queue']
    schedule    Add a job to the scheduler queue (deferred submission)
    schedule cancel <ID>  Cancel a previously scheduled job
    schedule [start|stop|restart|status]  Alias for 'handler ...'
    handler     Manage the scheduler daemon (start / stop / restart / status)
    scheduler   Alias for 'handler'
    show        Display commands: 'show queue', 'show avail', 'show history', 'show <slurm_id|schedule_id>'
    history     Show recently completed/failed/cancelled jobs  [alias for 'show history']

Scheduler system:
    The scheduler is a long-running daemon that manages job submissions on your
    behalf.  Use ``schedule`` to add jobs with optional ``--priority`` or
    ``--vip`` flags, and the scheduler will submit them to SLURM when resources
    allow, respecting priority ordering and per-QoS resource limits.

    Jobs get timestamp IDs like ``20260310-111136`` or ``20260310-111136_01``.
    You can set up job dependencies with ``--after ID[,ID,...]`` (run only after
    listed jobs succeed) or ``--after-any ID[,ID,...]`` (run after listed jobs
    finish even if some failed).

    Quick-start:
        python My_Lib_HPC.py schedule my_script.py           # queue a job (gets a timestamp ID)
        python My_Lib_HPC.py schedule my_script.py --vip     # VIP priority
        python My_Lib_HPC.py schedule post.py --after 20260310-111136,20260310-111136_01
        python My_Lib_HPC.py schedule cancel 20260310-111136_01
        python My_Lib_HPC.py schedule cancel 20260310-111136 20260310-111136_01
        python My_Lib_HPC.py schedule restart                 # alias for handler restart
        python My_Lib_HPC.py handler                          # start daemon
        python My_Lib_HPC.py handler restart                  # restart daemon
        python My_Lib_HPC.py scheduler status                 # alias for handler status
        python My_Lib_HPC.py show queue                        # view status
        python My_Lib_HPC.py handler stop                     # stop daemon

Module structure:
    This module (My_Lib_HPC.py) contains the CLI entry point, configuration
    loading, and job submission functions.  SLURM query utilities are in
    My_Lib_HPC_Slurm.py.  The scheduler system is in My_Lib_HPC_Scheduler.py.
    Both are re-exported from this module for backward compatibility.
                                 
"""

__author__ = 'LiYuanhe'

import sys
import os
import re
import glob
import pathlib
import subprocess
import shutil
import time
import unicodedata
import random
import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Optional, Sequence

_MODULE_BOOT_T0 = time.perf_counter()


def _should_print_config_banner(argv: list[str] | None = None) -> bool:
    """Return whether the module-level HPC configuration banner should print."""
    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 3:
        return True

    command = argv[1].lower()
    subcommand = argv[2].lower()

    if command == "schedule" and subcommand == "_debounced_submit_check":
        return False
    if command in ("handler", "scheduler") and subcommand == "_run":
        return False
    return True

# ---------------------------------------------------------------------------
# Resolve paths and import configuration
# ---------------------------------------------------------------------------
from Python_Lib.My_Lib_Stock import get_input_with_while_cycle, parse_range_selection
from Python_Lib.My_Lib_File import *
from datetime import datetime

# Static-analysis fallback:
# Import Example config so IDE/Pylance can resolve global names even though
# the real config is loaded dynamically from My_Program root at runtime.
from HPC_Lib.HPC_Configuration_Example import (
    HPC_NAME,
    USER_NAME,
    HOME_PATH,
    PYTHON_PATH,
    MY_PROGRAM_PATH,
    JOB_SCRIPT_DIR,
    SLURM_ACCOUNT,
    SLURM_PRESETS,
    DEFAULT_PRESET,
    CONCURRENT_SCHEDULED_MISSION_COUNT,
    CONCURRENT_RUNNING_MISSION_COUNT,
    CONGESTED_QUEUE,
    MEMORY_FRACTION,
    CORE_FRACTION,
    NODES,
    MAIL_USER,
    SUBMIT_COMMAND,
    CANCEL_COMMAND,
    )

# Import the HPC configuration module based on filename pattern matching
# Search upward from this file's directory until we find a folder named "My_Program"
def _find_my_program_root() -> str:
    """Walk up from __file__ until we reach a directory named 'My_Program'."""
    current = pathlib.Path(__file__).resolve().parent
    while True:
        if current.name == "My_Program":
            return str(current)
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                "Could not find a parent directory named 'My_Program' "
                f"starting from: {pathlib.Path(__file__).resolve()}"
            )
        current = parent

My_Program_path = _find_my_program_root()
_CONFIG_PATTERN = "My_Lib_HPC_Configuration_Secret_*.py"
_config_files = glob.glob(os.path.join(My_Program_path, _CONFIG_PATTERN))
_config_files = [f for f in _config_files if f.endswith(".py") and "__pycache__" not in f]

# Reject if the only config found is the Example template
_config_files = [
    f for f in _config_files
    if not os.path.basename(f).startswith("My_Lib_HPC_Configuration_Secret_Example")
]

if len(_config_files) == 0:
    raise FileNotFoundError(
        f"No HPC configuration file matching '{_CONFIG_PATTERN}' found in:\n"
        f"  {My_Program_path}\n"
        f"(The _Example template does NOT count as a valid configuration.)\n"
        f"Please copy the Example file to My_Lib_HPC_Configuration_Secret_<YourHPC>.py\n"
        f"and fill in the values for your cluster."
    )
if len(_config_files) > 1:
    raise RuntimeError(
        f"Multiple HPC configuration files found in {My_Program_path}:\n"
        + "\n".join(f"  - {os.path.basename(f)}" for f in _config_files)
        + "\nExactly one configuration file is expected per machine."
    )

_config_file = _config_files[0]

# Extract HPC_Name from filename
_config_basename = os.path.basename(_config_file)
_match = re.match(r"My_Lib_HPC_Configuration_Secret_(.+)\.py$", _config_basename)
if not _match:
    raise RuntimeError(f"Could not parse HPC name from config filename: {_config_basename}")

_HPC_NAME_FROM_FILE = _match.group(1)

# Import the configuration module dynamically
_spec = importlib.util.spec_from_file_location(
    f"My_Lib_HPC_Configuration_Secret_{_HPC_NAME_FROM_FILE}", _config_file
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load configuration module spec from: {_config_file}")
_config_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_module)

# Pull all public names into this module's namespace
for _name in dir(_config_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_config_module, _name)

if _should_print_config_banner():
    print(f"[My_Lib_HPC] Loaded configuration for HPC: {HPC_NAME}")


def _get_gaussian_input_class():
    """Import Gaussian_Input lazily so non-Gaussian commands start faster."""
    from Chem_Lib.Lib_Gaussian import Gaussian_Input as _Gaussian_Input
    return _Gaussian_Input


def _get_filetype_tools():
    """Import file_type/Filetype lazily so schedule/queue avoid heavy imports."""
    from Chem_Lib.Lib_Filetype import file_type as _file_type, Filetype as _Filetype
    return _file_type, _Filetype


def _lazy_file_type(path: str):
    """Thin wrapper used where the scheduler needs file_type as a callback."""
    _file_type, _ = _get_filetype_tools()
    return _file_type(path)


# ===========================================================================
# Helper utilities
# ===========================================================================

def _get_terminal_width(fallback: int = 220) -> int:
    """Return the current terminal column width, defaulting to *fallback* when not a tty."""
    return shutil.get_terminal_size(fallback=(fallback, 40)).columns


def _str_display_width(text: str) -> int:
    """Return the terminal display width of *text*.

    Full-width (W) and fullwidth (F) Unicode characters (e.g. CJK) occupy
    2 terminal columns; all other characters occupy 1.
    """
    w = 0
    for ch in text:
        cat = unicodedata.east_asian_width(ch)
        w += 2 if cat in ('W', 'F') else 1
    return w


def _truncate_to_display_width(text: str, max_width: int) -> str:
    """Truncate *text* so its terminal display width is ≤ *max_width*.

    Appends '…' (1 column) if truncation is necessary.  Never returns a
    string that is wider than *max_width* columns.
    """
    if _str_display_width(text) <= max_width:
        return text
    budget = max_width - 1  # reserve 1 column for '…'
    w = 0
    for i, ch in enumerate(text):
        cw = 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
        if w + cw > budget:
            return text[:i] + '…'
        w += cw
    return text[:max_width - 1] + '…'


def _prefix_for_display_width(text: str, max_width: int) -> str:
    """Return the longest prefix of *text* whose display width is ≤ *max_width*."""
    w = 0
    for i, ch in enumerate(text):
        cw = 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
        if w + cw > max_width:
            return text[:i]
        w += cw
    return text


def _suffix_for_display_width(text: str, max_width: int) -> str:
    """Return the longest suffix of *text* whose display width is ≤ *max_width*."""
    w = 0
    for i in range(len(text) - 1, -1, -1):
        cw = 2 if unicodedata.east_asian_width(text[i]) in ('W', 'F') else 1
        if w + cw > max_width:
            return text[i + 1:]
        w += cw
    return text


def _truncate_task_display(text: str, max_width: int) -> str:
    """Smart TASK-column truncation for 'script.py [first_arg ...]' strings.

    Keeps the full script name (up to the first space) and, when the first
    argument would overflow, shows its head and tail separated by ' ... '.

    Example (55 cols available for the argument)::

        A_DeepSeek_OCR.py Hennig「2020 - Edition C ... ira」Chelonian Eggs.pdf

    Falls back to plain end-truncation when the pattern does not apply.
    """
    if _str_display_width(text) <= max_width:
        return text

    space_idx = text.find(' ')
    if space_idx == -1:
        return _truncate_to_display_width(text, max_width)

    script_part = text[:space_idx]
    args_part   = text[space_idx + 1:]
    script_w    = _str_display_width(script_part)
    avail       = max_width - script_w - 1   # 1 for the separating space

    SEP   = ' ... '
    sep_w = _str_display_width(SEP)          # 5

    if avail < sep_w + 2:                    # not enough room for meaningful head+tail
        return _truncate_to_display_width(text, max_width)

    head_budget = (avail - sep_w) // 2
    tail_budget = avail - sep_w - head_budget

    head = _prefix_for_display_width(args_part, head_budget)
    tail = _suffix_for_display_width(args_part, tail_budget)
    return script_part + ' ' + head + SEP + tail


def _build_task_display_name(filepath: str, script_args: list[str] | None = None) -> str:
    """Return the queue/task display name used by both Scheduler and SLURM.

    Format: ``script_basename [first_arg_display]``.
    If the first argument looks like a path, only its basename is shown so the
    queue focuses on the distinguishing payload instead of long parent paths.
    """
    task_parts = [os.path.basename(filepath)]
    if script_args:
        first_arg = str(script_args[0]).strip()
        if first_arg:
            if os.sep in first_arg or '/' in first_arg or '\\' in first_arg:
                first_arg = os.path.basename(first_arg)
            task_parts.append(first_arg)
    return " ".join(task_parts)


def _ensure_job_script_dir():
    """Create the job script directory if it does not exist."""
    os.makedirs(JOB_SCRIPT_DIR, exist_ok=True)


def _generate_script_filename(source_file: str, suffix: str = ".sh") -> str:
    """
    Generate a unique job script filename in JOB_SCRIPT_DIR.

    Returns the absolute path to the generated script file.
    """
    _ensure_job_script_dir()
    script_id = datetime.now().strftime("%y%m%d_%H%M%S_%f")[:17]  # yymmdd_hhmmss_ms (3 digits)
    base = os.path.basename(source_file)
    script_name = f"auto_generated_script_{base}_{script_id}{suffix}"
    return os.path.join(JOB_SCRIPT_DIR, script_name)


def _resolve_preset(priority: str | None = None) -> dict:
    """
    Resolve a priority string ("high" / "normal") to a SLURM preset dict.

    Falls back to DEFAULT_PRESET if priority is None.
    """
    if priority is None:
        priority = DEFAULT_PRESET

    priority = priority.lower().strip()
    if priority not in SLURM_PRESETS:
        available = ", ".join(SLURM_PRESETS.keys())
        raise ValueError(
            f"Unknown priority '{priority}'. Available presets: {available}"
        )
    return SLURM_PRESETS[priority]


def _compute_resources(preset: dict) -> tuple[int, int]:
    """
    Compute the number of cores and memory (MB) to request based on the preset.

    Uses preset['cores_per_node'] and preset['memory_per_node_mb'] together with
    the global CORE_FRACTION and MEMORY_FRACTION settings.

    Returns:
        (cores, memory_mb)
    """
    cores_per_node = int(preset['cores_per_node'])
    cores = max(1, int(cores_per_node * CORE_FRACTION))
    memory_mb = _calculate_auto_memory_mb(preset, cores)
    return cores, memory_mb


def _calculate_auto_memory_mb(preset: dict, cores: int) -> int:
    """Calculate auto memory strictly from the configured node-memory fraction formula."""
    cores_per_node = int(preset['cores_per_node'])
    memory_per_node_mb = int(preset['memory_per_node_mb'])
    memory_mb = int(memory_per_node_mb * MEMORY_FRACTION * cores / cores_per_node)
    if memory_mb <= 0:
        raise ValueError(
            "Automatic memory calculation produced a non-positive value: "
            f"cores={cores}, cores_per_node={cores_per_node}, "
            f"memory_per_node_mb={memory_per_node_mb}, MEMORY_FRACTION={MEMORY_FRACTION}"
        )
    # print(f"[My_Lib_HPC] Memory calculation: {memory_per_node_mb} MB (node) × {MEMORY_FRACTION} (MEMORY_FRACTION) × {cores}/{cores_per_node} (cores ratio) = {memory_mb} MB")
    return memory_mb


def _resolve_resources(preset: dict, cores: int | None, memory_mb: int | None) -> tuple[int, int]:
    """
    Resolve and validate the requested cores and memory against the given preset.

    Fills in defaults when either value is missing, warns and asks for
    confirmation when the requested value exceeds the preset node limits.

    Args:
        preset:    SLURM preset dict (must contain 'cores_per_node' and 'memory_per_node_mb').
        cores:     Requested CPU cores, or None to use default.
        memory_mb: Requested memory in MB, or None to calculate automatically.

    Returns:
        (cores, memory_mb) — both guaranteed to be positive ints.
    """
    cores_per_node = int(preset['cores_per_node'])
    memory_per_node_mb = int(preset['memory_per_node_mb'])

    if cores is None and memory_mb is None:
        return _compute_resources(preset)

    if cores is not None and memory_mb is None:
        if cores > cores_per_node:
            print(f"[My_Lib_HPC] WARNING: Requested cores ({cores}) exceeds preset cores_per_node ({cores_per_node})")
            if input("Continue with submission? (y/n): ").lower() != 'y':
                print("[My_Lib_HPC] Submission aborted.")
                sys.exit(0)
        memory_mb = _calculate_auto_memory_mb(preset, cores)
        return cores, memory_mb

    if cores is None and memory_mb is not None:
        cores = max(1, int(cores_per_node * CORE_FRACTION))
        if memory_mb > memory_per_node_mb:
            print(f"[My_Lib_HPC] WARNING: Requested memory ({memory_mb} MB) exceeds preset memory_per_node_mb ({memory_per_node_mb} MB)")
            if input("Continue with submission? (y/n): ").lower() != 'y':
                print("[My_Lib_HPC] Submission aborted.")
                sys.exit(0)
        return cores, memory_mb

    # Type narrowing for static analyzers
    if cores is None or memory_mb is None:
        raise RuntimeError("Internal error: cores/memory resolution failed.")

    # Both specified
    if cores > cores_per_node:
        print(f"[My_Lib_HPC] WARNING: Requested cores ({cores}) exceeds preset cores_per_node ({cores_per_node})")
        if input("Continue with submission? (y/n): ").lower() != 'y':
            print("[My_Lib_HPC] Submission aborted.")
            sys.exit(0)
    if memory_mb > memory_per_node_mb:
        print(f"[My_Lib_HPC] WARNING: Requested memory ({memory_mb} MB) exceeds preset memory_per_node_mb ({memory_per_node_mb} MB)")
        if input("Continue with submission? (y/n): ").lower() != 'y':
            print("[My_Lib_HPC] Submission aborted.")
            sys.exit(0)
    return cores, memory_mb


def _parse_memory_string(mem_str: str) -> int:
    """
    Parse a memory string with optional unit suffix to MB.

    Supported formats:
        - "20" or "20GB" → 20480 MB (default unit is GB)
        - "100000MB" → 100000 MB
        - "1TB" → 1048576 MB

    Args:
        mem_str: Memory string (e.g., "20GB", "100000MB", "1TB", or "20")

    Returns:
        Memory in MB

    Raises:
        ValueError: If the format is invalid
    """
    mem_str = mem_str.strip().upper()
    
    # Extract number and unit
    import re
    match = re.match(r'^([0-9.]+)\s*(GB|MB|TB)?$', mem_str)
    if not match:
        raise ValueError(f"Invalid memory format: '{mem_str}'. Expected format: number + optional unit (GB/MB/TB)")
    
    value = float(match.group(1))
    unit = match.group(2) or 'GB'  # Default to GB
    
    # Convert to MB
    if unit == 'MB':
        return int(value)
    elif unit == 'GB':
        return int(value * 1024)
    elif unit == 'TB':
        return int(value * 1024 * 1024)
    else:
        raise ValueError(f"Unsupported memory unit: {unit}")


# ===========================================================================
# Job submission functions
# ===========================================================================

def submit_python_file(python_file: str, priority: str | None = None, script_args: list[str] | None = None, cores: int | None = None, memory_mb: int | None = None, ntasks_per_node: int = 1, email: bool = False) -> str:
    """
    Generate a SLURM job script that runs a Python file and submit it.

    The generated script will:
        1. Print the active Python interpreter path (``which python``).
        2. Run the specified Python file using the configured PYTHON_PATH.

    Args:
        python_file:      Absolute path to the .py file to execute on the cluster.
        priority:         "high" (normal QoS, 2 weeks) or "normal" (standby QoS,
                          4 hours).  Defaults to DEFAULT_PRESET from configuration.
                          Supplied via ``--qos`` on the command line.
        script_args:      Extra arguments to forward to the Python script (e.g.
                          ``['--num_experts', '8', '--hidden_dim', '64']``).
        cores:            Number of CPU cores to request. If None, uses default
                          (preset['cores_per_node'] * CORE_FRACTION). If specified
                          and exceeds preset's cores_per_node, requires confirmation.
        memory_mb:        Memory in MB to request. If None, calculated automatically
                          based on cores. If specified and exceeds preset's
                          memory_per_node_mb, requires user confirmation.
        ntasks_per_node:  Number of MPI tasks per node (--ntasks-per-node).
                          Defaults to 1 for single-task/threaded jobs.
        email:            If True, send email notification when the job ends
                          (regardless of exit status). If False (default), email is
                          only sent on failure. Requires MAIL_USER in config.

    Returns:
        The path to the generated job script.
    """
    python_file = os.path.abspath(python_file)
    if not os.path.isfile(python_file):
        raise FileNotFoundError(f"Python file not found: {python_file}")

    preset = _resolve_preset(priority)
    cores, memory_mb = _resolve_resources(preset, cores, memory_mb)

    job_name = _build_task_display_name(python_file, script_args)
    script_path = _generate_script_filename(python_file)
    output_file = os.path.splitext(script_path)[0] + ".out"

    mail_type = "END" if email else "FAIL"
    header = _build_sbatch_header(
        job_name,
        output_file,
        preset,
        cores,
        memory_mb,
        _get_hpc_config(),
        mail_type=mail_type,
        ntasks_per_node=ntasks_per_node,
    )

    # Get the directory of the python file to use as working directory
    python_file_dir = os.path.dirname(python_file)
    
    # Search for .venv upwards
    venv_root = None
    current_search_dir = python_file_dir
    stop_search_dir = os.path.abspath(My_Program_path)
    
    while True:
        venv_path = os.path.join(current_search_dir, ".venv")
        if os.path.isdir(venv_path):
            if os.path.isfile(os.path.join(venv_path, "bin", "python")) or \
               os.path.isfile(os.path.join(venv_path, "Scripts", "python.exe")):
                venv_root = current_search_dir
                break
        
        if current_search_dir == stop_search_dir:
            break
            
        parent_dir = os.path.dirname(current_search_dir)
        if parent_dir == current_search_dir:
            break
        current_search_dir = parent_dir

    import shlex
    extra_args_str = " ".join(shlex.quote(arg) for arg in script_args) if script_args else ""
    
    if venv_root:
        working_dir = venv_root
        run_cmd = f"uv run {python_file}"
        if extra_args_str:
            run_cmd += f" {extra_args_str}"
        env_diag = [
            "# --- Environment diagnostics ---",
            "echo '=== Python interpreter (uv) ==='",
            "which uv",
            f"echo 'Found .venv at: {venv_root}'",
        ]
    else:
        working_dir = python_file_dir
        run_cmd = f"{PYTHON_PATH} {python_file}"
        if extra_args_str:
            run_cmd += f" {extra_args_str}"
        env_diag = [
            "# --- Environment diagnostics ---",
            "echo '=== Python interpreter ==='",
            "which python",
            f"echo 'Configured interpreter: {PYTHON_PATH}'",
        ]

    body_lines = [
        "",
    ] + env_diag + [
        "",
        "# --- Change to working directory ---",
        f"cd {working_dir}",
        f"echo 'Working directory: {working_dir}'",
        "",
        "# --- Run the Python script ---",
        run_cmd,
        "",
    ]

    script_content = header + "\n" + "\n".join(body_lines)

    with open(script_path, "w", newline="\n") as f:
        f.write(script_content)

    print(f"[My_Lib_HPC] Preset: {preset['description']}")
    print(f"[My_Lib_HPC] Resources: {cores} cores, {memory_mb} MB memory")
    print(f"[My_Lib_HPC] Output file: {output_file}")
    print(f"\n>>> {SUBMIT_COMMAND} {script_path}\n")

    # input("Confirm submission (press Enter to continue, Ctrl+C to cancel)...")

    result = subprocess.run(
        [SUBMIT_COMMAND, script_path],
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)

    if result.returncode != 0:
        print(f"[My_Lib_HPC] WARNING: {SUBMIT_COMMAND} returned exit code {result.returncode}", file=sys.stderr)

    print("\n------------------------------------------------------------------------\n")
    return script_path


def submit_Gaussian_file(gaussian_file: str, priority: str | None = None, cores: int | None = None, memory_mb: int | None = None, email: bool = False) -> str:
    """
    Generate a SLURM job script that runs a Gaussian 16 input file and submit it.

    Uses :class:`Gaussian_Input` from ``Chem_Lib.Lib_Gaussian`` to parse the
    input file, modify ``%nprocshared``/``%mem``, and extract metadata (chk/rwf
    paths, run commands, etc.).

    The generated script will:
        1. Create necessary directories for checkpoint and rwf files.
        2. Execute any ``!RUN`` commands embedded in the input file.
        3. Run ``g16`` with the input file, writing output to ``<stem>.out``.
        4. Run ``formchk`` on every checkpoint file listed in the input.

    Before submission the function **modifies the input file in-place** to set
    ``%nprocshared`` and ``%mem`` according to the requested (or default)
    resources.  The Gaussian ``%mem`` is set to 80 % of the SLURM memory
    allocation so that Gaussian does not exceed the job's memory limit.

    Args:
        gaussian_file: Absolute path to the ``.gjf`` / ``.com`` file.
        priority:      "high" (normal QoS, 2 weeks) or "normal" (standby QoS,
                       4 hours).  Defaults to DEFAULT_PRESET.
        cores:         Number of CPU cores.  When ``None`` the default
                       (preset['cores_per_node'] 脳 CORE_FRACTION) is used.  If
                       ``memory_mb`` is also ``None``, memory is calculated
                       proportionally.
        memory_mb:     SLURM memory in MB.  When ``None`` it is derived from
                       *cores* (or the default fraction).

    Returns:
        The path to the generated job script.
    """

    # --- validate input file -----------------------------------------------
    gaussian_file = os.path.abspath(gaussian_file)
    if not os.path.isfile(gaussian_file):
        raise FileNotFoundError(f"Gaussian input file not found: {gaussian_file}")

    if " " in gaussian_file:
        raise ValueError(
            f"Gaussian file path must not contain spaces: {gaussian_file}"
        )

    # --- resolve resources --------------------------------------------------
    preset = _resolve_preset(priority)
    cores, memory_mb = _resolve_resources(preset, cores, memory_mb)

    # --- parse input file with Gaussian_Input ------------------------------
    Gaussian_Input = _get_gaussian_input_class()
    Gaussian_input_object = Gaussian_Input(gaussian_file)

    # --- modify %nprocshared and %mem in every step ------------------------
    # Gaussian gets 80 % of the SLURM allocation so it doesn't OOM the job.
    gaussian_mem_mb = max(1024, int(memory_mb * 0.8))
    Gaussian_input_object.set_nprocshared("ALL", cores)
    Gaussian_input_object.set_mem("ALL", gaussian_mem_mb)

    # Write modified file back
    Gaussian_input_object.save()

    # --- extract info from parsed object -----------------------------------
    chk_files = Gaussian_input_object.chk_files
    rwf_files = Gaussian_input_object.rwf_files
    run_commands = Gaussian_input_object.run_commands
    charge = Gaussian_input_object.steps[0].charge if Gaussian_input_object.steps else None
    multiplicity = Gaussian_input_object.steps[0].multiplet if Gaussian_input_object.steps else None

    # --- job name -----------------------------------------------------------
    job_name = os.path.basename(gaussian_file)

    # --- paths --------------------------------------------------------------
    gaussian_file_dir = os.path.dirname(gaussian_file)
    gaussian_output_file = os.path.splitext(gaussian_file)[0] + ".out"

    script_path = _generate_script_filename(gaussian_file)
    slurm_log = os.path.splitext(script_path)[0] + ".slurm.log"

    # --- determine g16 / formchk executable path ----------------------------
    g16_exe_dir = globals().get("GAUSSIAN_EXE_DIR", "")
    if g16_exe_dir:
        g16_cmd = os.path.join(g16_exe_dir, "g16")
        formchk_cmd = os.path.join(g16_exe_dir, "formchk")
    else:
        g16_cmd = "g16"
        formchk_cmd = "formchk"

    # --- build script -------------------------------------------------------
    mail_type = "END" if email else "FAIL"
    header = _build_sbatch_header(
        job_name,
        slurm_log,
        preset,
        cores,
        memory_mb,
        _get_hpc_config(),
        mail_type=mail_type,
    )

    body_lines: list[str] = [""]

    # Change to input file directory
    body_lines.append(f"cd {gaussian_file_dir}")
    body_lines.append("")

    # Create directories for chk / rwf files if they don't exist
    dirs_to_create: set[str] = set()
    for chk in chk_files:
        d = os.path.dirname(chk)
        if d:
            dirs_to_create.add(d)
    for rwf in rwf_files:
        d = os.path.dirname(rwf)
        if d:
            dirs_to_create.add(d)
    for d in sorted(dirs_to_create):
        body_lines.append(f"mkdir -p {d}")
    if dirs_to_create:
        body_lines.append("")

    # Embedded !RUN commands
    for cmd in run_commands:
        body_lines.append(cmd)
    if run_commands:
        body_lines.append("")

    # Gaussian execution
    body_lines.append(f"{g16_cmd} {gaussian_file} {gaussian_output_file}")
    body_lines.append("")

    # formchk on every checkpoint file
    for chk in chk_files:
        body_lines.append(f"{formchk_cmd} {chk}")
    if chk_files:
        body_lines.append("")

    script_content = header + "\n" + "\n".join(body_lines)

    with open(script_path, "w", newline="\n") as f:
        f.write(script_content)

    # --- print summary ------------------------------------------------------
    print(f"[My_Lib_HPC] Gaussian submission for: {gaussian_file}")
    print(f"[My_Lib_HPC] Preset: {preset['description']}")
    print(f"[My_Lib_HPC] Resources: {cores} cores, {memory_mb} MB SLURM memory "
          f"({gaussian_mem_mb} MB for Gaussian)")
    print(f"[My_Lib_HPC] Job name: {job_name}")
    if charge is not None:
        print(f"[My_Lib_HPC] Charge: {charge}   Multiplicity: {multiplicity}")
    for i, chk in enumerate(chk_files):
        print(f"[My_Lib_HPC] Chk file {i}: {chk}")
    print(f"[My_Lib_HPC] Gaussian output: {gaussian_output_file}")
    print(f"[My_Lib_HPC] SLURM log: {slurm_log}")
    print(f"\n>>> {SUBMIT_COMMAND} {script_path}\n")

    result = subprocess.run(
        [SUBMIT_COMMAND, script_path],
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)

    if result.returncode != 0:
        print(f"[My_Lib_HPC] WARNING: {SUBMIT_COMMAND} returned exit code {result.returncode}", file=sys.stderr)

    print("\n------------------------------------------------------------------------\n")
    return script_path


# ===========================================================================
# Dispatcher — choose handler based on file extension
# ===========================================================================


def submit_file(filepath: str, priority: str | None = None, script_args: list[str] | None = None, cores: int | None = None, memory_mb: int | None = None, email: bool = False) -> str:
    """
    Dispatch a file to the appropriate submission handler based on its extension.

    Args:
        filepath:    Path to the input file.
        priority:    "high" or "normal" (see submit_python_file for details).
        script_args: Extra arguments forwarded to the submitted script.
        cores:       Number of CPU cores to request (see submit_python_file for details).
        memory_mb:   Memory in MB to request (see submit_python_file for details).
        email:       If True, send email notification when the job ends.

    Returns:
        The path to the generated job script.

    Raises:
        ValueError / SystemExit: If the file extension is not supported.
    """
    filepath = os.path.abspath(filepath)
    ext = filename_last_append(filepath).lower()

    if ext == "py":
        return submit_python_file(filepath, priority, script_args, cores, memory_mb, email=email)
    file_type, Filetype = _get_filetype_tools()
    ftype = file_type(filepath)
    if ftype == Filetype.gaussian_input:
        return submit_Gaussian_file(filepath, priority, cores, memory_mb, email=email)
    else:
        print(f"Unsupported file extension '{ext}'.")
        sys.exit(1)


# ===========================================================================
# Re-exports from split modules
# ===========================================================================
# SLURM query utilities (moved to HPC_Slurm.py)
from HPC_Lib.HPC_Slurm import (
    _build_sbatch_header,
    SlurmJobInfo, get_job_info,
    get_job_batch_script,
    NodeInfo, get_node_availability,
    QueueEntry, get_queue,
    estimate_job_runnability,
    HistoryEntry, get_recent_jobs,
)

# Scheduler system (moved to HPC_Scheduler.py)
from HPC_Lib.HPC_Scheduler import (
    ScheduledJob,
    _get_scheduler_dir,
    _get_scheduler_queue_entries,
    _scheduler_log,
    _read_scheduler_state,
    _write_scheduler_state,
    _next_schedule_id,
    _save_scheduled_job,
    _load_scheduled_job,
    _load_all_scheduled_jobs,
    _delete_scheduled_job_file,
    _is_scheduler_running,
    _submit_scheduler_slurm_job,
    _cancel_slurm_job,
    _parse_time_limit_seconds,
    cancel_scheduled_job,
    cancel_scheduled_job_and_collect_slurm_job,
    cancel_scheduled_jobs_and_collect_slurm_jobs,
    scheduler_main_loop,
    _are_dependencies_met,
    _can_submit_more,
    _count_submitted_and_running,
    _read_heartbeat_age,
    _read_scheduler_launch_age,
    _write_scheduler_submit_check_request,
    _read_scheduler_submit_check_request,
    _is_scheduler_submit_check_token_current,
    _SCHEDULER_HEARTBEAT_STALE,
    _SCHEDULER_SUBMIT_CHECK_DEBOUNCE_DELAY,
)


def _get_hpc_config() -> dict:
    """
    Build and return a dict of all HPC configuration values.

    This is used to pass configuration to the Scheduler module (which
    lives in a separate file and cannot access this module's globals
    directly at import time).
    """
    keys = [
        "HPC_NAME", "USER_NAME", "HOME_PATH", "PYTHON_PATH",
        "MY_PROGRAM_PATH", "JOB_SCRIPT_DIR", "SLURM_ACCOUNT",
        "SLURM_PRESETS", "DEFAULT_PRESET",
        "CONCURRENT_SCHEDULED_MISSION_COUNT", "CONCURRENT_RUNNING_MISSION_COUNT",
        "CONGESTED_QUEUE",
        "MEMORY_FRACTION", "CORE_FRACTION", "NODES",
        "MAIL_USER", "SUBMIT_COMMAND", "CANCEL_COMMAND",
    ]
    cfg: dict = {}
    for k in keys:
        if k in globals():
            cfg[k] = globals()[k]
    return cfg


# ===========================================================================
# CLI entry point
# ===========================================================================

def handle_submit_command(args: list[str]):
    """
    Handle the 'submit' command.

    Args:
        args: Command-line arguments after 'submit' command.
              Expected: [file_to_submit] [--qos high|normal] [--cores N] [--mem SIZE] [--email] [script_args...]
              
              Parameters are case-insensitive and accept single/double dash (but not short -c style).
              Synonyms: --qos/--priority, --cores/--cpu, --mem/--memory

    Usage:
        python My_Lib_HPC.py submit <file_to_submit> [--qos high|normal] [--cores N] [--mem SIZE] [--email] [script_args...]
        python My_Lib_HPC.py submit <file_to_submit> [-CORES 16] [-MEM 50GB]  (case-insensitive, single dash works)
        python My_Lib_HPC.py submit my_script.py --parm_for_my_script1 abc --parm_for_my_script2 cde --CPU 8
        python My_Lib_HPC.py submit my_script.py --email
        python My_Lib_HPC.py submit  (interactive mode - enter multiple files)
    """
    if len(args) < 1:
        # Interactive mode: ask for multiple files
        print("[My_Lib_HPC] No input file provided. Entering interactive mode.")
        print("[My_Lib_HPC] Enter file paths to submit (one per line, empty line to finish):")
        print("")
        
        input_lines = get_input_with_while_cycle(
            break_condition=lambda x: not x.strip(),
            input_prompt="File path: ",
            strip_quote=True
        )
        
        if not input_lines:
            print("[My_Lib_HPC] No files provided. Exiting.")
            sys.exit(0)
        
        # Ask for priority once for all files
        priority_input = input("Priority [high/normal, default=normal]: ").strip().lower()
        priority = priority_input if priority_input in ['high', 'normal'] else None

        _preview_preset = _resolve_preset(priority)
        _default_cores, _ = _compute_resources(_preview_preset)

        # Ask for cores
        cores_input = input(f"Number of cores [default={_default_cores}]: ").strip()
        cores = int(cores_input) if cores_input else None
        
        # Ask for memory
        memory_mb = None
        memory_input = input("Memory [e.g., 20GB, 100000MB, or empty for auto]: ").strip()
        if memory_input:
            try:
                memory_mb = _parse_memory_string(memory_input)
            except ValueError as e:
                print(f"[My_Lib_HPC] ERROR: {e}")
                sys.exit(1)
        
        # Ask for email notification
        email_input = input("Send email on job end? [y/n, default=n]: ").strip().lower()
        email = email_input in ('y', 'yes')
        
        # Submit all files
        print(f"\n[My_Lib_HPC] Submitting {len(input_lines)} file(s)...")
        for filepath in input_lines:
            filepath = filepath.strip()
            if filepath:
                filepath = os.path.abspath(filepath)
                print(f"\n{'='*60}")
                print(f"[My_Lib_HPC] Processing: {filepath}")
                print('='*60)
                try:
                    submit_file(filepath, priority, cores=cores, memory_mb=memory_mb, email=email)
                except Exception as e:
                    print(f"[My_Lib_HPC] ERROR submitting {filepath}: {e}", file=sys.stderr)
                    continue
        
        print(f"\n[My_Lib_HPC] All submissions complete.")
    else:
        # Command-line mode: single file
        # Parse --qos <value>, --cores <value>, and --mem <value> from args;
        # all other flags/values are forwarded to the submitted script.
        filepath = os.path.abspath(args[0])
        remaining = args[1:]

        priority = None
        cores = None
        memory_mb = None
        email = False
        script_args = []
        i = 0
        while i < len(remaining):
            arg = remaining[i]
            
            # Normalize argument: case-insensitive, support single/double dash
            # but exclude short options (single dash + single char)
            arg_lower = arg.lower()
            arg_name = None
            
            if arg_lower.startswith('--') and len(arg_lower) > 2:
                # Double dash: --cores, --CORES, etc.
                arg_name = arg_lower[2:]
            elif arg_lower.startswith('-') and len(arg_lower) > 2:
                # Single dash with multi-char: -cores, -CORES, etc.
                arg_name = arg_lower[1:]
            # else: short option like -c, or not an option at all
            
            # Match recognized parameters
            if arg_name in ['qos', 'priority'] and i + 1 < len(remaining):
                priority = remaining[i + 1]
                i += 2
            elif arg_name in ['cores', 'cpu'] and i + 1 < len(remaining):
                try:
                    cores = int(remaining[i + 1])
                except ValueError:
                    print(f"[My_Lib_HPC] ERROR: --cores must be an integer, got '{remaining[i + 1]}'")
                    sys.exit(1)
                i += 2
            elif arg_name in ['mem', 'memory'] and i + 1 < len(remaining):
                try:
                    memory_mb = _parse_memory_string(remaining[i + 1])
                except ValueError as e:
                    print(f"[My_Lib_HPC] ERROR: {e}")
                    sys.exit(1)
                i += 2
            elif arg_name == 'email':
                email = True
                i += 1
            else:
                # Not a recognized parameter, forward to script
                script_args.append(remaining[i])
                i += 1

        submit_file(filepath, priority, script_args if script_args else None, cores, memory_mb, email=email)


def _batch_cancel_jobs(job_ids: Sequence[int | str], cancel_command: str, batch_size: int = 20):
    """
    Cancel SLURM jobs in batches using *cancel_command* (e.g. ``scancel``).

    Passes up to *batch_size* job IDs per invocation so that the final list of
    jobs can be cancelled with as few shell calls as possible instead of one
    call per job.  Higher numeric job IDs are cancelled first.

    Args:
        job_ids:        List of SLURM job IDs to cancel.
        cancel_command: The executable to run (e.g. ``"scancel"``).
        batch_size:     Maximum number of IDs per single invocation.
    """
    ordered_job_ids = sorted(
        (str(job_id) for job_id in job_ids),
        key=lambda job_id: int(job_id) if str(job_id).isdigit() else str(job_id),
        reverse=True,
    )

    for start in range(0, len(ordered_job_ids), batch_size):
        chunk = ordered_job_ids[start:start + batch_size]
        cmd = [cancel_command] + [str(j) for j in chunk]
        print(f"\n>>> {' '.join(cmd)}\n")
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"[My_Lib_HPC] Error running cancel command: {e}")


def handle_stop_command(args: list[str]):
    """
    Handle the 'stop' command to cancel jobs.

    Accepts SLURM job IDs (integers/ranges) and scheduler IDs (e.g.
    ``20260306-143025``).  If a numeric ID is not found in the SLURM queue
    but matches a scheduler ID, the scheduler job is cancelled instead.

        When the resolved list contains **both** SLURM queue jobs and scheduler
        jobs the user is prompted to choose:
      1 – cancel all (scheduler first, then SLURM queue)
      2 – cancel scheduler jobs only
      3 – cancel SLURM queue jobs only

        When the resolved list contains **both** pending jobs and non-pending
        jobs the user is also prompted to choose:
            1 – cancel all
            2 – cancel pending jobs only
            3 – cancel non-pending jobs only

    Args:
        args: Command-line arguments after 'stop' command.
              Expected: [job_id ...] where job IDs can be individual numbers,
              ranges, scheduler IDs, or 'all'.
              Examples: "12345", "12346 12347", "12350-12360",
                        "20260306-143025", "all"
              If empty, enters interactive mode.
    
    Usage:
        python My_Lib_HPC.py stop 12345 12346 12350-12360
        python My_Lib_HPC.py stop 20260306-143025       (cancel a scheduler job)
        python My_Lib_HPC.py stop all                    (cancel all SLURM + pending scheduler jobs)
        python My_Lib_HPC.py stop all qos=high           (cancel only high-QoS jobs)
        python My_Lib_HPC.py stop all --qos normal       (same as above)
        python My_Lib_HPC.py stop       (interactive mode)
    """
    # Check for CANCEL_COMMAND global variable (from config)
    if not globals().get("CANCEL_COMMAND"):
        print(f"[My_Lib_HPC] Error: CANCEL_COMMAND is not defined in the configuration file.")
        sys.exit(1)
        
    global CANCEL_COMMAND # Just to be safe for linter, though defined dynamically

    cancel_cmd = globals()["CANCEL_COMMAND"]
    job_ids_to_cancel: list[int] = []
    scheduler_ids_to_cancel: list[str] = []
    list_already_printed = False
    stop_all_qos_filter: str | None = None
    scheduler_daemon_job_name = "HPC_Scheduler"
    cfg_cache: dict | None = None
    scheduler_jobs_cache: list[ScheduledJob] | None = None

    def _classify_stop_args(tokens: list[str]):
        """Separate tokens into SLURM integer IDs and scheduler string IDs."""
        slurm_ids: list[int] = []
        sched_ids: list[str] = []
        # Tokens that are purely numeric (or numeric ranges like 100-200)
        numeric_tokens: list[str] = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            # Check if this looks like a numeric token or range
            if re.match(r'^[\d\s,\-]+$', tok) and not re.match(r'^\d{8}-\d{6}', tok):
                numeric_tokens.append(tok)
            else:
                # Looks like a scheduler ID (contains non-range hyphens, letters, etc.)
                sched_ids.append(tok)
        if numeric_tokens:
            combined = " ".join(numeric_tokens)
            try:
                parsed = parse_range_selection(combined, decrease_by_1=False)
                if parsed:
                    slurm_ids.extend(parsed)
            except Exception:
                pass
        return slurm_ids, sched_ids

    def _parse_stop_all_options(tokens: list[str]) -> str | None:
        """Parse optional filters for ``stop all``."""
        qos_filter: str | None = None
        i = 0
        while i < len(tokens):
            token = tokens[i].strip()
            lowered = token.lower()
            if lowered in ("--qos", "-qos", "--priority", "-priority"):
                if i + 1 >= len(tokens):
                    print(f"[My_Lib_HPC] Error: {token} requires a QoS value.")
                    sys.exit(1)
                qos_filter = tokens[i + 1].strip()
                i += 2
                continue
            if lowered.startswith(("--qos=", "-qos=", "--priority=", "-priority=", "qos=", "priority=")):
                qos_filter = token.split("=", 1)[1].strip()
                i += 1
                continue
            print(f"[My_Lib_HPC] Error: Unsupported option for 'stop all': {token}")
            print("[My_Lib_HPC] Supported forms: 'qos=<name>' or '--qos <name>'.")
            sys.exit(1)

        if qos_filter is not None and not qos_filter:
            print("[My_Lib_HPC] Error: QoS filter cannot be empty.")
            sys.exit(1)
        return qos_filter.lower() if qos_filter else None

    def _get_cfg_cached() -> dict:
        nonlocal cfg_cache
        if cfg_cache is None:
            cfg_cache = _get_hpc_config()
        return cfg_cache

    def _get_scheduler_jobs_cached() -> list[ScheduledJob]:
        nonlocal scheduler_jobs_cache
        if scheduler_jobs_cache is None:
            scheduler_jobs_cache = _load_all_scheduled_jobs(_get_cfg_cached())
        return scheduler_jobs_cache

    def _is_pending_slurm_state(state: str) -> bool:
        return (state or "").strip().upper() == "PD"

    def _is_pending_scheduler_status(status: str) -> bool:
        return (status or "").strip().lower() == "pending"

    # ------------------------------------------------------------------
    # 'stop all' — cancel every queued/running job for this user
    # ------------------------------------------------------------------
    if args and args[0].lower() == "all":
        stop_all_qos_filter = _parse_stop_all_options(args[1:])
        qos_desc = f" with qos='{stop_all_qos_filter}'" if stop_all_qos_filter else ""
        print(f"[My_Lib_HPC] Querying all non-scheduler jobs for user '{USER_NAME}'{qos_desc}...")
        queue_entries = get_queue(user=USER_NAME)
        queue_entries = [e for e in queue_entries if e.job_name != scheduler_daemon_job_name]
        if stop_all_qos_filter:
            queue_entries = [
                e for e in queue_entries
                if (e.qos or "").strip().lower() == stop_all_qos_filter
            ]
        if not queue_entries:
            print(f"[My_Lib_HPC] No matching SLURM jobs found in the queue for user '{USER_NAME}'.")
        else:
            job_ids_to_cancel = sorted({int(e.job_id) for e in queue_entries if e.job_id.isdigit()}, reverse=True)
            print(f"[My_Lib_HPC] Found {len(job_ids_to_cancel)} SLURM job(s) in the queue:")
            for e in queue_entries:
                state_str = f"[{e.state}]"
                print(f"  {e.job_id:<12} {state_str:<6}  {e.job_name}")

        # Also cancel all pending scheduler jobs
        cfg = _get_cfg_cached()
        all_sched = _get_scheduler_jobs_cached()
        pending_sched = [j for j in all_sched if j.status == "pending"]
        if stop_all_qos_filter:
            default_qos = (_resolve_preset(None) or {}).get("qos", "")
            pending_sched = [
                j for j in pending_sched
                if ((j.qos or default_qos) or "").strip().lower() == stop_all_qos_filter
            ]
        if pending_sched:
            scheduler_ids_to_cancel = [j.schedule_id for j in pending_sched]
            print(f"[My_Lib_HPC] Found {len(pending_sched)} pending scheduler job(s):")
            for j in pending_sched:
                print(f"  {j.schedule_id:<20}  {os.path.basename(j.filepath)}")

        list_already_printed = True

        if not job_ids_to_cancel and not scheduler_ids_to_cancel:
            print("[My_Lib_HPC] Nothing to cancel.")
            return

    elif args:
        # Command-line mode: args are job IDs (e.g. "stop 1 2 3-5")
        # or scheduler IDs (e.g. "stop 20260306-143025")
        slurm_ids, sched_ids = _classify_stop_args(args)
        job_ids_to_cancel.extend(slurm_ids)
        scheduler_ids_to_cancel.extend(sched_ids)

    else:
        # Interactive mode
        print("[My_Lib_HPC] Enter job IDs to cancel (e.g. 12345, 12346, 12350-12360).")
        print("[My_Lib_HPC] Scheduler IDs (e.g. 20260306-143025) are also accepted.")
        print("[My_Lib_HPC] Enter 'all' to cancel all YOUR jobs currently in the queue.")
        print("[My_Lib_HPC] Empty line to finish.")
        
        print("Job IDs to stop (ends with empty line):")
        input_lines = get_input_with_while_cycle(
            break_condition=lambda x: not x.strip(),
            strip_quote=True
        )

        for line in input_lines:
            line = line.strip()
            if not line:
                continue
            if line.lower() == "all":
                handle_stop_command(["all"])
                return
            slurm_ids, sched_ids = _classify_stop_args([line])
            job_ids_to_cancel.extend(slurm_ids)
            scheduler_ids_to_cancel.extend(sched_ids)

    # --- Check SLURM IDs that are actually scheduler IDs in disguise ---
    # For integer IDs not found in the SLURM queue, check the scheduler.
    _queue_entry_by_id: dict = {}
    if job_ids_to_cancel:
        queue_entries = get_queue(user=globals().get("USER_NAME"))
        queued_ids = {int(e.job_id) for e in queue_entries if e.job_id.isdigit()}
        _queue_entry_by_id = {int(e.job_id): e for e in queue_entries if e.job_id.isdigit()}
        slurm_found: list[int] = []
        slurm_not_found: list[int] = []
        for jid in job_ids_to_cancel:
            if jid in queued_ids:
                slurm_found.append(jid)
            else:
                slurm_not_found.append(jid)

        # For IDs not in SLURM, check if they match a scheduler ID
        if slurm_not_found:
            all_sched = _get_scheduler_jobs_cached()
            sched_id_set = {j.schedule_id for j in all_sched if j.status in ("pending", "submitted", "running")}
            for jid in slurm_not_found:
                sid = str(jid)
                if sid in sched_id_set:
                    scheduler_ids_to_cancel.append(sid)
                    print(f"[My_Lib_HPC] ID {jid} not in SLURM queue → found in scheduler, will cancel scheduler job.")
                else:
                    # Keep it as a SLURM ID anyway (user may know what they're doing)
                    slurm_found.append(jid)

        job_ids_to_cancel = sorted(set(slurm_found), reverse=True)

    # Deduplicate scheduler IDs
    scheduler_ids_to_cancel = list(dict.fromkeys(scheduler_ids_to_cancel))

    if not job_ids_to_cancel and not scheduler_ids_to_cancel:
        print("[My_Lib_HPC] No valid job IDs provided to stop.")
        return

    if not list_already_printed:
        total = len(job_ids_to_cancel) + len(scheduler_ids_to_cancel)
        print(f"[My_Lib_HPC] Planning to cancel {total} job(s)...")
        print("------------------------------------------------------------------------")
        if job_ids_to_cancel:
            print("SLURM jobs to cancel:")
            for j in job_ids_to_cancel:
                entry = _queue_entry_by_id.get(j)
                if entry:
                    state_str = f"[{entry.state}]" if entry.state else ""
                    print(f"  {j:<12} {state_str:<6}  {entry.job_name}")
                else:
                    print(f"  - {j}")
        if scheduler_ids_to_cancel:
            print("Scheduler jobs to cancel:")
            for s in scheduler_ids_to_cancel:
                print(f"  - {s}")

    print("------------------------------------------------------------------------")

    special_prompt_used = False

    # If both SLURM queue jobs and scheduler jobs are present, ask the user what to cancel
    if job_ids_to_cancel and scheduler_ids_to_cancel:
        print("[My_Lib_HPC] The list contains BOTH SLURM queue job(s) and scheduler job(s).")
        print("  1  Cancel BOTH (all)")
        print("  2  Cancel only scheduler jobs")
        print("  3  Cancel only SLURM queue jobs")
        choice = input("Your choice (1/2/3, or any other key to abort): ").strip()
        if choice == '2':
            job_ids_to_cancel = []
            print(f"[My_Lib_HPC] Will cancel {len(scheduler_ids_to_cancel)} scheduler job(s) only.")
        elif choice == '3':
            scheduler_ids_to_cancel = []
            print(f"[My_Lib_HPC] Will cancel {len(job_ids_to_cancel)} SLURM queue job(s) only.")
        elif choice == '1':
            total = len(job_ids_to_cancel) + len(scheduler_ids_to_cancel)
            print(f"[My_Lib_HPC] Will cancel all {total} job(s) (scheduler first, then SLURM).")
        else:
            print("[My_Lib_HPC] Cancellation aborted.")
            return
        special_prompt_used = True

    slurm_state_by_id: dict[int, str] = {}
    if job_ids_to_cancel:
        queue_entries = get_queue(user=globals().get("USER_NAME"))
        slurm_state_by_id = {
            int(entry.job_id): (entry.state or "").strip().upper()
            for entry in queue_entries
            if entry.job_id.isdigit()
        }
    scheduler_status_by_id = {
        job.schedule_id: (job.status or "").strip().lower()
        for job in _get_scheduler_jobs_cached()
    }

    pending_slurm_ids = [jid for jid in job_ids_to_cancel if _is_pending_slurm_state(slurm_state_by_id.get(jid, ""))]
    nonpending_slurm_ids = [jid for jid in job_ids_to_cancel if jid not in pending_slurm_ids]
    pending_scheduler_ids = [
        sid for sid in scheduler_ids_to_cancel
        if _is_pending_scheduler_status(scheduler_status_by_id.get(sid, ""))
    ]
    nonpending_scheduler_ids = [sid for sid in scheduler_ids_to_cancel if sid not in pending_scheduler_ids]

    pending_total = len(pending_slurm_ids) + len(pending_scheduler_ids)
    nonpending_total = len(nonpending_slurm_ids) + len(nonpending_scheduler_ids)
    if pending_total and nonpending_total:
        print("[My_Lib_HPC] The list contains BOTH pending job(s) and non-pending job(s).")
        print("  1  Cancel BOTH (all)")
        print("  2  Cancel only pending jobs")
        print("  3  Cancel only non-pending jobs")
        choice = input("Your choice (1/2/3, or any other key to abort): ").strip()
        if choice == '2':
            job_ids_to_cancel = pending_slurm_ids
            scheduler_ids_to_cancel = pending_scheduler_ids
            print(f"[My_Lib_HPC] Will cancel {pending_total} pending job(s) only.")
        elif choice == '3':
            job_ids_to_cancel = nonpending_slurm_ids
            scheduler_ids_to_cancel = nonpending_scheduler_ids
            print(f"[My_Lib_HPC] Will cancel {nonpending_total} non-pending job(s) only.")
        elif choice == '1':
            total = pending_total + nonpending_total
            print(f"[My_Lib_HPC] Will cancel all {total} job(s) across both state groups.")
        else:
            print("[My_Lib_HPC] Cancellation aborted.")
            return
        special_prompt_used = True

    if not special_prompt_used:
        total = len(job_ids_to_cancel) + len(scheduler_ids_to_cancel)
        while True:
            confirm = input(f"Are you sure you want to cancel {total} job(s)? (y/n): ").strip().lower()
            if confirm == 'y':
                break
            elif confirm == 'n':
                print("[My_Lib_HPC] Cancellation aborted.")
                return
            else:
                print("[My_Lib_HPC] Please enter 'y' or 'n'.")

    # Cancel scheduler jobs first, then SLURM queue jobs
    slurm_ids_from_scheduler: list[str] = []
    if scheduler_ids_to_cancel:
        cfg = _get_cfg_cached()
        slurm_ids_from_scheduler = cancel_scheduled_jobs_and_collect_slurm_jobs(
            scheduler_ids_to_cancel,
            cfg,
        )

    slurm_job_ids_to_cancel = [str(jid) for jid in job_ids_to_cancel]
    slurm_job_ids_to_cancel.extend(slurm_ids_from_scheduler)
    slurm_job_ids_to_cancel = list(dict.fromkeys(slurm_job_ids_to_cancel))

    if slurm_job_ids_to_cancel:
        _batch_cancel_jobs(slurm_job_ids_to_cancel, cancel_cmd)


def handle_tail_command(args: list):
    """
    Handle the 'tail' command to monitor a file.

    Behaves like ``tail -f`` on Linux:
      - Prints the last N lines of the file (default 200).
      - Then follows the file, printing new content as it is appended.
      - If the file does not yet exist, waits silently until it appears.
      - Gracefully handles file truncation / rotation.
      - Runs until interrupted with Ctrl+C.

    Args:
        args: Command-line arguments after 'tail' command.
              Expected: <file_or_job_id> [-n N]

    Usage:
        python My_Lib_HPC.py tail <file>
        python My_Lib_HPC.py tail <job_id>
        python My_Lib_HPC.py tail <file_or_job_id> -n 100
        python My_Lib_HPC.py tail -n 100 <file_or_job_id>
    """
    import time

    DEFAULT_LINES = 200
    POLL_INTERVAL = 0.5  # seconds between size-check polls

    n_lines = DEFAULT_LINES
    filepath = None

    # -----------------------------------------------------------------------
    # Parse arguments: -n / --lines N  and a positional file path
    # -----------------------------------------------------------------------
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.lstrip("-").lower() in ("n", "lines"):
            i += 1
            if i >= len(args):
                print("[My_Lib_HPC] Error: -n requires a numeric argument.")
                sys.exit(1)
            try:
                n_lines = int(args[i])
            except ValueError:
                print(f"[My_Lib_HPC] Error: -n value must be an integer, got '{args[i]}'.")
                sys.exit(1)
        elif filepath is None and not arg.startswith("-"):
            filepath = arg
        i += 1

    if filepath is None:
        print("[My_Lib_HPC] Error: No file path specified for 'tail' command.")
        print("Usage: python My_Lib_HPC.py tail <file_or_job_id> [-n N]")
        sys.exit(1)

    # If the target looks like a SLURM job ID and no such path exists, resolve
    # the job's StdOut/StdErr path automatically.
    target = filepath.strip()
    if target.isdigit() and not os.path.exists(target):
        job_info = get_job_info(target)
        if not job_info.job_id:
            print(f"[My_Lib_HPC] Error: Could not resolve job ID '{target}'.")
            print("Usage: python My_Lib_HPC.py tail <file_or_job_id> [-n N]")
            sys.exit(1)

        resolved = (job_info.stdout or "").strip() or (job_info.stderr or "").strip()
        if not resolved:
            print(f"[My_Lib_HPC] Error: Job {target} has no StdOut/StdErr path available.")
            print("(This commonly happens for historical jobs only available via sacct.)")
            sys.exit(1)

        if "%j" in resolved:
            resolved = resolved.replace("%j", target)
        if "%A" in resolved:
            resolved = resolved.replace("%A", target)
        if "%x" in resolved and job_info.job_name:
            resolved = resolved.replace("%x", job_info.job_name)

        filepath = resolved
        print(f"[My_Lib_HPC] Resolved job {target} -> {filepath}")

    filepath = os.path.abspath(filepath)

    # -----------------------------------------------------------------------
    # Wait for the file to appear if it doesn't exist yet
    # -----------------------------------------------------------------------
    if not os.path.exists(filepath):
        print(f"[My_Lib_HPC] File not found: {filepath}")
        print(f"[My_Lib_HPC] Waiting for file to appear... (Ctrl+C to stop)")
        try:
            while not os.path.exists(filepath):
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print()
            print("[My_Lib_HPC] Interrupted while waiting for file.")
            return
        print(f"[My_Lib_HPC] File appeared: {filepath}")

    # -----------------------------------------------------------------------
    # Print the last n_lines of the current file contents
    # -----------------------------------------------------------------------
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        lines = raw.splitlines(keepends=True)
        tail_lines = lines[-n_lines:] if len(lines) >= n_lines else lines
        sys.stdout.buffer.write(b"".join(tail_lines))
        sys.stdout.buffer.flush()

        current_pos = len(raw)
    except PermissionError:
        print(f"[My_Lib_HPC] Error: Permission denied reading file: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"[My_Lib_HPC] Error reading file: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Follow the file (like tail -f)
    # -----------------------------------------------------------------------
    try:
        while True:
            time.sleep(POLL_INTERVAL)

            # Handle file disappearing (e.g. rotation)
            try:
                new_size = os.path.getsize(filepath)
            except FileNotFoundError:
                print(f"\n[My_Lib_HPC] File disappeared: {filepath}")
                print(f"[My_Lib_HPC] Waiting for file to reappear... (Ctrl+C to stop)")
                while not os.path.exists(filepath):
                    time.sleep(POLL_INTERVAL)
                print(f"[My_Lib_HPC] File reappeared: {filepath}")
                current_pos = 0  # re-read from beginning
                new_size = os.path.getsize(filepath)

            # Handle truncation
            if new_size < current_pos:
                print("\n[My_Lib_HPC] File was truncated; reading from beginning.")
                current_pos = 0

            # Read and print newly appended bytes
            if new_size > current_pos:
                with open(filepath, "rb") as f:
                    f.seek(current_pos)
                    new_bytes = f.read()
                sys.stdout.buffer.write(new_bytes)
                sys.stdout.buffer.flush()
                current_pos += len(new_bytes)

    except KeyboardInterrupt:
        print()
        print("[My_Lib_HPC] Tail stopped.")


def _get_tar_gz_output_path(paths: list[str]) -> str:
    """
    Determine the output .tar.gz path for a compress operation.

    Rules:
      - Single item  → <parent>/<item_basename>.tar.gz
      - Multiple items → <common_parent>/<common_parent_basename>.tar.gz

    If the computed path already exists, appends ``_01``, ``_02``, … before
    ``.tar.gz`` (e.g. ``20251209_LYH_01.tar.gz``) — equivalent to what
    ``get_unused_filename`` does but correctly handles the double extension.

    Args:
        paths: List of absolute, normalised filesystem paths.

    Returns:
        An absolute path string that does not currently exist on disk.
    """
    # Determine base name stem and parent directory
    if len(paths) == 1:
        base_stem = os.path.basename(paths[0].rstrip("/").rstrip("\\"))
        parent_dir = os.path.dirname(os.path.normpath(paths[0]))
    else:
        common = os.path.commonpath(paths)
        # commonpath returns the deepest common directory
        parent_dir = common
        base_stem = os.path.basename(common)

    # Build candidate paths, avoiding existing files
    candidate = os.path.join(parent_dir, base_stem + ".tar.gz")
    if not os.path.exists(candidate):
        return candidate

    number = 1
    while True:
        candidate = os.path.join(parent_dir, f"{base_stem}_{number:02d}.tar.gz")
        if not os.path.exists(candidate):
            return candidate
        number += 1


def _do_compress(output_path: str, items: list[str], delete_after: bool):
    """
    Create a .tar.gz archive at *output_path* containing all *items*, verify it,
    and optionally delete the originals.

    Args:
        output_path:  Absolute path for the output archive.
        items:        List of absolute paths to add (files or directories).
        delete_after: Whether to delete originals after successful verification.
    """
    import tarfile
    import shutil

    print(f"[My_Lib_HPC] Creating archive: {output_path}")
    try:
        with tarfile.open(output_path, "w:gz") as tar:
            for p in items:
                arcname = os.path.basename(p)
                tar.add(p, arcname=arcname)
                print(f"  Added: {p}")
    except Exception as e:
        print(f"[My_Lib_HPC] Error creating archive: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        return

    print(f"[My_Lib_HPC] Archive created: {output_path}")

    # Verify archive integrity
    verified = False
    try:
        with tarfile.open(output_path, "r:gz") as tar:
            members = tar.getnames()
        if members:
            print(f"[My_Lib_HPC] Archive verified: {len(members)} member(s).")
            verified = True
        else:
            print("[My_Lib_HPC] Warning: Archive is empty; originals will NOT be deleted.")
    except Exception as e:
        print(f"[My_Lib_HPC] Error verifying archive: {e}")

    # Delete originals (only if verified)
    if delete_after and verified:
        print("[My_Lib_HPC] Deleting original files/folders...")
        import shutil as _shutil
        for p in items:
            try:
                if os.path.isdir(p):
                    _shutil.rmtree(p)
                    print(f"  Deleted directory: {p}")
                else:
                    os.remove(p)
                    print(f"  Deleted file     : {p}")
            except Exception as e:
                print(f"  Error deleting '{p}': {e}")
        print("[My_Lib_HPC] Done.")
    elif delete_after and not verified:
        print("[My_Lib_HPC] Archive verification failed — original files NOT deleted.")


def handle_compress_command(args: list):
    """
    Handle the 'compress' command: pack files/folders into .tar.gz archive(s).

    When multiple items are provided the user picks one of two modes:

      individual — each item → its own ``<parent>/<item_name>.tar.gz``
      combined   — all items → one ``<common_parent>/<common_parent_name>.tar.gz``

    Single items always go to ``<parent>/<item_name>.tar.gz`` directly.

    The user is asked *before* compression starts whether to delete the
    originals afterwards.  Originals are only removed if the archive is
    successfully created and verified.

    Args:
        args: Paths provided on the command line.  When empty, interactive
              mode is used.

    Usage::

        python My_Lib_HPC.py compress /path/to/folder
        python My_Lib_HPC.py compress file1.py file2.py
        python My_Lib_HPC.py compress          # interactive mode
    """
    # ------------------------------------------------------------------
    # Collect paths
    # ------------------------------------------------------------------
    if args:
        raw_paths = [a.strip().strip('"') for a in args if a.strip()]
    else:
        print("[My_Lib_HPC] Enter file/folder paths to compress (empty line to finish):")
        input_lines = get_input_with_while_cycle(
            break_condition=lambda x: not x.strip(),
            input_prompt="  Path: ",
            strip_quote=True,
        )
        raw_paths = [l.strip() for l in input_lines if l.strip()]

    if not raw_paths:
        print("[My_Lib_HPC] No paths provided.")
        return

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    valid_paths = []
    for p in raw_paths:
        p_abs = os.path.normpath(os.path.abspath(p))
        if not os.path.exists(p_abs):
            print(f"[My_Lib_HPC] Warning: Path does not exist, skipping: {p_abs}")
        else:
            valid_paths.append(p_abs)

    if not valid_paths:
        print("[My_Lib_HPC] No valid paths to compress.")
        return

    print("[My_Lib_HPC] Items to compress:")
    for p in valid_paths:
        print(f"  {p}")
    print("")

    # ------------------------------------------------------------------
    # For multiple items: ask individual vs combined mode
    # ------------------------------------------------------------------
    individual_mode = False
    planned: list[tuple[str, str]] = []
    combined_output = ""
    if len(valid_paths) > 1:
        print("[My_Lib_HPC] Compression mode:")
        print("  1 - Individual: each item → its own <item_name>.tar.gz")
        print("  2 - Combined  : all items → one <common_parent_name>.tar.gz")
        mode_input = input("Choose mode (1/2): ").strip()
        if mode_input == "1":
            individual_mode = True
        elif mode_input == "2":
            individual_mode = False
        else:
            print("[My_Lib_HPC] Invalid choice, aborted.")
            return
        print("")

    # ------------------------------------------------------------------
    # Show planned output paths
    # ------------------------------------------------------------------
    if individual_mode:
        planned = [(p, _get_tar_gz_output_path([p])) for p in valid_paths]
        for src, dst in planned:
            print(f"  {src}")
            print(f"    →{dst}")
    else:
        combined_output = _get_tar_gz_output_path(valid_paths)
        print(f"[My_Lib_HPC] Output archive: {combined_output}")
    print("")

    # ------------------------------------------------------------------
    # Ask about deletion BEFORE compression starts
    # ------------------------------------------------------------------
    delete_after = input("Delete original files/folders after successful compression? (y/n): ").strip().lower() == 'y'

    # ------------------------------------------------------------------
    # Final confirmation
    # ------------------------------------------------------------------
    if individual_mode:
        confirm = input(f"Create {len(valid_paths)} individual archive(s)? (y/n): ").strip().lower()
    else:
        confirm = input(f"Compress {len(valid_paths)} item(s) into '{os.path.basename(combined_output)}'? (y/n): ").strip().lower()
    if confirm != 'y':
        print("[My_Lib_HPC] Aborted.")
        return

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------
    if individual_mode:
        for src, dst in planned:
            print("")
            _do_compress(dst, [src], delete_after)
    else:
        _do_compress(combined_output, valid_paths, delete_after)


def handle_avail_command(args: list[str]):
    """
    Handle the 'avail' command: display current node availability.

    Runs ``sinfo`` and prints a formatted table of node states and free resources.

    Usage:
        python My_Lib_HPC.py avail
        python My_Lib_HPC.py avail --idle      # show only fully-idle nodes
        python My_Lib_HPC.py avail --mixed     # show only mixed (partially used) nodes
    """
    filter_state: str | None = None
    for arg in args:
        arg_lower = arg.lstrip("-").lower()
        if arg_lower in ("idle", "mixed", "allocated", "drain", "draining"):
            filter_state = arg_lower

    nodes = get_node_availability()
    if not nodes:
        print("[My_Lib_HPC] No node information available.")
        return

    if filter_state:
        nodes = [n for n in nodes if filter_state in n.state.lower()]

    # Print table
    header = f"{'NODE':<16} {'CPUS(A/I/T)':<14} {'MEM(GB)':<10} {'FREE_MEM(GB)':<14} {'STATE':<20}"
    print(header)
    print("-" * len(header))
    for n in nodes:
        cpus_str = f"{n.cpus_alloc}/{n.cpus_idle}/{n.total_cpus}"
        mem_gb = n.memory_mb // 1024
        free_gb = n.free_memory_mb // 1024
        print(f"{n.node_name:<16} {cpus_str:<14} {mem_gb:<10} {free_gb:<14} {n.state:<20}")

    # Summary
    total_nodes = len(nodes)
    idle_nodes = sum(1 for n in nodes if "idle" in n.state.lower())
    mixed_nodes = sum(1 for n in nodes if "mixed" in n.state.lower())
    total_idle_cpus = sum(n.cpus_idle for n in nodes)
    total_free_mem_gb = sum(n.free_memory_mb for n in nodes) // 1024
    print("")
    print(f"Total nodes shown: {total_nodes}  (idle: {idle_nodes}, mixed: {mixed_nodes})")
    print(f"Total idle CPUs: {total_idle_cpus}   Total free memory: {total_free_mem_gb} GB")


def handle_queue_command(args: list[str]):
    """
    Handle the 'queue' command: display both the SLURM job queue and the
    scheduler's internal queue.

    Defaults to showing only the current user's jobs (``--me``).
    Use ``--all`` to show all users' jobs.

    This command is equivalent to ``show queue`` and supports the same
    column-filter flags.

    Usage::

        python My_Lib_HPC.py queue              # show only YOUR jobs (default)
        python My_Lib_HPC.py queue --all        # show all users' jobs
        python My_Lib_HPC.py queue --user NAME  # show jobs for a specific user
        python My_Lib_HPC.py queue --ST RUNN    # show only running jobs
        python My_Lib_HPC.py queue -task D2A    # show jobs whose task contains D2A
    """
    handle_show_queue_command(args)


def _generate_scheduler_submit_check_token() -> str:
    """Return a unique token for one deferred scheduler daemon check."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    return f"{timestamp}-{os.getpid()}-{random.randint(1000, 9999)}"


def _ensure_scheduler_daemon_for_scheduled_jobs(cfg: dict):
    """Ensure that a scheduler daemon exists for queued scheduler jobs.

    This function contains the expensive part of ``schedule``: it may read the
    heartbeat/launch files, query ``squeue`` for ``HPC_Scheduler``, cancel a
    stuck scheduler, or submit a fresh scheduler SLURM job.

    ``handle_schedule_command`` no longer calls this synchronously.  Instead it
    writes the job JSON immediately, then a detached helper calls this function
    after a short non-blocking delay.  If a newer ``schedule`` command arrives
    before the delay expires, the older helper exits before touching SLURM.
    """
    # Auto-start or restart the scheduler if needed.
    # Strategy: check heartbeat file first (cheap, no squeue).
    #   - Heartbeat fresh  → scheduler is alive, nothing to do.
    #   - Heartbeat stale / missing → check launch timestamp first
    #     to avoid restart storms when many schedule commands run
    #     concurrently.  Only query squeue if no recent launch:
    #       * RUNNING in squeue but stale heartbeat → stuck, restart it.
    #       * PENDING in squeue → wait for it to start.
    #       * Not in squeue → submit a new scheduler.
    check_started_at = time.perf_counter()

    heartbeat_started_at = time.perf_counter()
    heartbeat_age = _read_heartbeat_age(cfg)
    heartbeat_elapsed = time.perf_counter() - heartbeat_started_at
    if heartbeat_age is not None and heartbeat_age < _SCHEDULER_HEARTBEAT_STALE:
        # Scheduler heartbeat is fresh — it's alive, nothing to do.
        state = _read_scheduler_state(cfg)
        sid = state.get("slurm_job_id", "?") if state else "?"
        print(f"\n[Scheduler] Scheduler daemon is alive (heartbeat {heartbeat_age:.0f}s ago, SLURM {sid}).")
        total_elapsed = time.perf_counter() - check_started_at
        print(f"[Scheduler] Deferred daemon check finished in {total_elapsed:.3f}s (heartbeat read {heartbeat_elapsed:.3f}s, no squeue needed).")
        return

    # Before querying squeue, check if a scheduler was launched
    # very recently by another 'schedule' process (anti-storm guard).
    launch_started_at = time.perf_counter()
    launch_age = _read_scheduler_launch_age(cfg)
    launch_elapsed = time.perf_counter() - launch_started_at
    if launch_age is not None and launch_age < 120:
        print(
            f"\n[Scheduler] Scheduler was recently launched ({launch_age:.0f}s ago), "
            f"skipping restart check."
        )
        total_elapsed = time.perf_counter() - check_started_at
        print(f"[Scheduler] Deferred daemon check finished in {total_elapsed:.3f}s (heartbeat {heartbeat_elapsed:.3f}s, launch-age {launch_elapsed:.3f}s, no squeue needed).")
        return

    # Heartbeat stale or missing — query squeue to decide what to do.
    if heartbeat_age is not None:
        print(f"\n[Scheduler] Heartbeat is stale ({heartbeat_age:.0f}s old), checking squeue...")
    else:
        print(f"\n[Scheduler] No recent scheduler heartbeat found, checking squeue...")
    squeue_started_at = time.perf_counter()
    sched_entries = _get_scheduler_queue_entries(cfg)
    squeue_elapsed = time.perf_counter() - squeue_started_at
    running_entries = [e for e in sched_entries if e.state == "R"]
    pending_entries = [e for e in sched_entries if e.state == "PD"]
    action_elapsed = 0.0
    action_desc = "observe"

    if running_entries and heartbeat_age is not None and heartbeat_age >= _SCHEDULER_HEARTBEAT_STALE:
        # Scheduler SLURM job is running but heartbeat is stale → stuck.
        stuck_ids = ", ".join(e.job_id for e in running_entries)
        print(f"[Scheduler] Scheduler {stuck_ids} appears stuck (heartbeat {heartbeat_age:.0f}s old). Restarting...")
        action_desc = "restart-stuck"
        action_started_at = time.perf_counter()
        for e in running_entries:
            _cancel_slurm_job(e.job_id, cfg)
        sid = _submit_scheduler_slurm_job(
            cfg,
            _ensure_job_script_dir=_ensure_job_script_dir,
            _resolve_preset=_resolve_preset,
            _build_sbatch_header=_build_sbatch_header,
            _calculate_auto_memory_mb=_calculate_auto_memory_mb,
        )
        if sid:
            print(f"[Scheduler] New scheduler submitted as SLURM job {sid}")
        else:
            print(f"[Scheduler] WARNING: Failed to start scheduler. Run 'handler restart' manually.")
        action_elapsed = time.perf_counter() - action_started_at
    elif pending_entries:
        ids_desc = ", ".join(e.job_id for e in pending_entries)
        action_desc = "pending-in-queue"
        print(f"[Scheduler] Scheduler is pending in queue ({ids_desc}), waiting for it to start.")
    elif sched_entries:
        # Has entries (running) but heartbeat is None (first time / no file yet)
        ids_desc = ", ".join(e.job_id for e in sched_entries)
        action_desc = "already-active"
        print(f"[Scheduler] Scheduler daemon is active ({ids_desc}).")
    else:
        # Not in squeue at all — submit new scheduler.
        action_desc = "submit-new"
        print(f"[Scheduler] Starting scheduler daemon...")
        action_started_at = time.perf_counter()
        sid = _submit_scheduler_slurm_job(
            cfg,
            _ensure_job_script_dir=_ensure_job_script_dir,
            _resolve_preset=_resolve_preset,
            _build_sbatch_header=_build_sbatch_header,
            _calculate_auto_memory_mb=_calculate_auto_memory_mb,
        )
        if sid:
            print(f"[Scheduler] Scheduler submitted as SLURM job {sid}")
        else:
            print(f"[Scheduler] WARNING: Failed to start scheduler. Run 'handler' manually.")
        action_elapsed = time.perf_counter() - action_started_at

    total_elapsed = time.perf_counter() - check_started_at
    print(
        f"[Scheduler] Deferred daemon check finished in {total_elapsed:.3f}s "
        f"(heartbeat {heartbeat_elapsed:.3f}s, launch-age {launch_elapsed:.3f}s, "
        f"squeue {squeue_elapsed:.3f}s, action {action_elapsed:.3f}s, result={action_desc})."
    )


def _arm_debounced_scheduler_submit_check(
    cfg: dict,
    delay_seconds: float = _SCHEDULER_SUBMIT_CHECK_DEBOUNCE_DELAY,
) -> tuple[str, bool]:
    """Schedule a detached non-blocking helper to check the scheduler later.

    The helper process sleeps for ``delay_seconds`` without blocking the user's
    terminal, then compares its token to the latest token stored under
    ``HOME_PATH``.  Only the newest helper is allowed to run the expensive
    squeue/sbatch logic.
    """
    token = _generate_scheduler_submit_check_token()
    _write_scheduler_submit_check_request(token, delay_seconds, cfg)

    python_path = cfg.get("PYTHON_PATH") or sys.executable or "python"
    helper_cmd = [
        python_path,
        os.path.abspath(__file__),
        "schedule",
        "_debounced_submit_check",
        "--token",
        token,
        "--delay",
        f"{float(delay_seconds):.3f}",
    ]

    popen_kwargs = {
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_kwargs["stdout"] = subprocess.DEVNULL
        popen_kwargs["stderr"] = subprocess.DEVNULL
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        popen_kwargs["start_new_session"] = True

    subprocess.Popen(helper_cmd, **popen_kwargs)
    return token, True


def _run_debounced_scheduler_submit_check(
    token: str,
    delay_seconds: float,
    cfg: dict,
) -> bool:
    """Sleep, verify token freshness, then run the real scheduler check.

    Example:
        1. ``schedule file1`` writes token A and starts helper A.
        2. ``schedule file2`` arrives within 2 seconds, overwrites the token
           file with token B, and starts helper B.
        3. Helper A wakes up, sees token B is newer, and exits without reading
           squeue.
        4. Helper B wakes up later, still owns the newest token, so it performs
           the real scheduler daemon check and submits/restarts the scheduler if
           required.
    """
    delay_seconds = max(0.0, float(delay_seconds))
    if delay_seconds > 0:
        time.sleep(delay_seconds)

    if not _is_scheduler_submit_check_token_current(token, cfg):
        return False

    print(f"[Scheduler] Running deferred scheduler daemon check now ({delay_seconds:.1f}s debounce elapsed).")
    _ensure_scheduler_daemon_for_scheduled_jobs(cfg)
    return True


def _handle_debounced_scheduler_submit_check_command(args: list[str]):
    """Internal command used by detached helpers started from ``schedule``.

    This command is not meant for manual use.  It implements the non-blocking
    debounce logic requested for bulk pastes such as::

        hpc schedule 1
        hpc schedule 2
        hpc schedule 3

    In that example, three helpers are started, but only the final helper whose
    token is still current after the 2-second delay is allowed to touch SLURM.
    The earlier two helpers exit quietly before querying ``squeue``.
    """
    token = ""
    delay_seconds = _SCHEDULER_SUBMIT_CHECK_DEBOUNCE_DELAY

    i = 0
    while i < len(args):
        arg = args[i].lower()
        if arg == "--token" and i + 1 < len(args):
            token = args[i + 1]
            i += 2
        elif arg == "--delay" and i + 1 < len(args):
            delay_seconds = float(args[i + 1])
            i += 2
        else:
            i += 1

    if not token:
        print("[Scheduler] Usage: python My_Lib_HPC.py schedule _debounced_submit_check --token TOKEN [--delay SEC]")
        sys.exit(1)

    cfg = _get_hpc_config()
    _run_debounced_scheduler_submit_check(token, delay_seconds, cfg)


def handle_schedule_command(args: list[str]):
    """
    Handle the 'schedule' command: add a job to the scheduler's queue,
    cancel a previously scheduled job, or forward handler-style sub-commands.

    Sub-commands::

        python My_Lib_HPC.py schedule <file> [options]        # schedule a new job
        python My_Lib_HPC.py schedule cancel <id> [<id>...]   # cancel scheduled job(s)
        python My_Lib_HPC.py schedule restart                 # alias for handler restart
        python My_Lib_HPC.py schedule status                  # alias for handler status

    Like ``submit``, but the job is NOT submitted immediately.  Instead, a JSON
    config is written to ``{HOME_PATH}/HPC_Scheduler/`` and the scheduler
    daemon will submit it when resources allow, respecting priority ordering.

    Extra parameters compared to ``submit``:

    - ``--priority N`` / ``-priority N``:  Integer priority (default 0).
      Higher values are submitted first.  Case-insensitive.
    - ``--vip`` / ``-vip``:  Infinite priority — submitted before all non-VIP
      jobs.  Equivalent to ``--priority 9999999999``.
    - ``--after ID[,ID,...]``:  Dependency — run only after ALL listed jobs
      have completed successfully.  IDs are short schedule IDs (e.g. ``001``).
    - ``--after-any ID[,ID,...]``:  Like ``--after`` but runs even if some
      dependencies failed (only waits for them to finish).

    All other parameters (``--qos``, ``--cores``, ``--mem``, plus arbitrary
    script arguments) are parsed identically to ``submit``.

     Deferred daemon check:

     ``schedule`` now writes the job JSON immediately and defers the expensive
     "is the scheduler daemon alive / should a new HPC_Scheduler SLURM job be
     submitted" check by 2 seconds in a detached helper process.

     The delay is non-blocking for the terminal.  If another ``schedule``
     command arrives before those 2 seconds expire, the older pending helper is
     superseded and exits before it queries ``squeue``.  Only the newest helper
     is allowed to touch SLURM.

     Example::

          hpc schedule 1
          hpc schedule 2
          hpc schedule 3

     Effective order:

        1. ``schedule 1`` writes its JSON immediately and arms delayed helper A.
        2. ``schedule 2`` writes its JSON immediately and arms delayed helper B.
            Helper A becomes obsolete before its 2-second timer expires.
        3. ``schedule 3`` writes its JSON immediately and arms delayed helper C.
            Helper B becomes obsolete before its timer expires.
        4. After 2 seconds, only helper C still owns the newest token, so only C
            performs the real scheduler-daemon check and possible submission.

    Usage::

        python My_Lib_HPC.py schedule my_script.py
        python My_Lib_HPC.py schedule my_script.py --qos high --priority 100
        python My_Lib_HPC.py schedule my_script.py --vip --cores 16
        python My_Lib_HPC.py schedule my_script.py --after 001,002
        python My_Lib_HPC.py schedule my_script.py --after-any 003 --priority 50
        python My_Lib_HPC.py schedule cancel 005
        python My_Lib_HPC.py schedule restart

    Example output::

        [Scheduler] Job scheduled:
          ID:       005
          File:     /scratch/.../my_script.py
          Priority: 100
          QoS:      high
          Cores:    16 (auto)
          Memory:   auto
          After:    001, 002 (all must succeed)
          Config:   /home/li5876/HPC_Scheduler/005.json

        [Scheduler] Scheduler daemon is already running (SLURM job 34456789).

    Notes:

    - ``--priority`` is the scheduler's own priority, independent of SLURM's
      QoS-based priority.  It controls the ORDER in which the scheduler submits
      jobs.  SLURM's own scheduling priority (fair-share, QoS weight, etc.)
      takes over once the job is submitted.
    - For congested queues, priority only affects submission order.  Once a job
      is in SLURM's queue, it cannot be reordered by the scheduler.
    - For non-congested queues, the scheduler may cancel a lower-priority
      pending SLURM job to submit a higher-priority one first.
    - Schedule IDs are short sequential integers (``001``, ``002``, …) — easy
      to read and type.
    """
    command_started_at = time.perf_counter()
    cfg = _get_hpc_config()

    if args and args[0].lower() == "_debounced_submit_check":
        _handle_debounced_scheduler_submit_check_command(args[1:])
        return

    # --- Sub-command: cancel ---
    if args and args[0].lower() == "cancel":
        if len(args) < 2:
            print("[Scheduler] Usage: python My_Lib_HPC.py schedule cancel <id> [<id> ...]")
            sys.exit(1)
        cancel_ids = args[1:]
        for cid in cancel_ids:
            cancel_scheduled_job(cid.strip(), cfg)
        return

    if not args:
        print("[My_Lib_HPC] Error: 'schedule' command requires a file path.")
        print("")
        print("Usage: python My_Lib_HPC.py schedule <file> [options]")
        print("       python My_Lib_HPC.py schedule cancel <id> [<id> ...]")
        print("")
        print("Options:")
        print("  --qos high|normal      QoS preset (default: normal)")
        print("  --cores N              CPU cores to request")
        print("  --mem SIZE             Memory (e.g., 20GB, 100000MB)")
        print("  --priority N           Scheduler priority (higher = submit first, default: 0)")
        print("  --vip                  Infinite priority (submitted before all non-VIP jobs)")
        print("  --email                Send email notification when job ends")
        print("  --after ID[,ID,...]    Run only after listed jobs succeed")
        print("  --after-any ID[,ID,...] Run after listed jobs finish (even if failed)")
        print("  [script_args...]       Extra arguments forwarded to the script")
        sys.exit(1)

    filepath = os.path.abspath(args[0])
    remaining = args[1:]

    qos: str | None = None
    cores: int | None = None
    memory_mb: int | None = None
    is_vip = False
    email = False
    priority_level = 0
    depends_on: list[str] = []
    depend_mode: str = "success"
    script_args: list[str] = []

    i = 0
    while i < len(remaining):
        arg = remaining[i]
        arg_lower = arg.lower()
        arg_name = None

        if arg_lower.startswith("--") and len(arg_lower) > 2:
            arg_name = arg_lower[2:]
        elif arg_lower.startswith("-") and len(arg_lower) > 2:
            arg_name = arg_lower[1:]

        if arg_name in ("qos", "priority_qos") and i + 1 < len(remaining):
            qos = remaining[i + 1]
            i += 2
        elif arg_name in ("cores", "cpu") and i + 1 < len(remaining):
            try:
                cores = int(remaining[i + 1])
            except ValueError:
                print(f"[My_Lib_HPC] ERROR: --cores must be an integer, got '{remaining[i + 1]}'")
                sys.exit(1)
            i += 2
        elif arg_name in ("mem", "memory") and i + 1 < len(remaining):
            try:
                memory_mb = _parse_memory_string(remaining[i + 1])
            except ValueError as e:
                print(f"[My_Lib_HPC] ERROR: {e}")
                sys.exit(1)
            i += 2
        elif arg_name == "priority" and i + 1 < len(remaining):
            try:
                priority_level = int(remaining[i + 1])
            except ValueError:
                print(f"[My_Lib_HPC] ERROR: --priority must be an integer, got '{remaining[i + 1]}'")
                sys.exit(1)
            i += 2
        elif arg_name == "vip":
            is_vip = True
            i += 1
        elif arg_name == "email":
            email = True
            i += 1
        elif arg_name == "after" and i + 1 < len(remaining):
            depends_on = [x.strip() for x in remaining[i + 1].split(",") if x.strip()]
            depend_mode = "success"
            i += 2
        elif arg_name in ("after-any", "after_any") and i + 1 < len(remaining):
            depends_on = [x.strip() for x in remaining[i + 1].split(",") if x.strip()]
            depend_mode = "any"
            i += 2
        else:
            script_args.append(remaining[i])
            i += 1

    args_parsed_at = time.perf_counter()

    # Validate file exists
    if not os.path.isfile(filepath):
        print(f"[My_Lib_HPC] ERROR: File not found: {filepath}")
        sys.exit(1)

    # Create the scheduled job
    schedule_id = _next_schedule_id(cfg)
    job = ScheduledJob(
        schedule_id=schedule_id,
        filepath=filepath,
        priority_level=priority_level,
        is_vip=is_vip,
        qos=qos,
        cores=cores,
        memory_mb=memory_mb,
        script_args=script_args,
        status="pending",
        created_at=datetime.now().isoformat(),
        depends_on=depends_on,
        depend_mode=depend_mode,
        email=email,
    )
    save_started_at = time.perf_counter()
    _save_scheduled_job(job, cfg)
    save_elapsed = time.perf_counter() - save_started_at

    # Print confirmation
    prio_str = "VIP (infinite)" if is_vip else str(priority_level)
    print(f"[Scheduler] Job scheduled:")
    print(f"  ID:       {schedule_id}")
    print(f"  File:     {filepath}")
    print(f"  Priority: {prio_str}")
    print(f"  QoS:      {qos or 'default'}")
    print(f"  Cores:    {cores or 'auto'}")
    print(f"  Memory:   {f'{memory_mb} MB' if memory_mb else 'auto'}")
    if depends_on:
        mode_desc = "all must succeed" if depend_mode == "success" else "any finish"
        print(f"  After:    {', '.join(depends_on)} ({mode_desc})")
    if email:
        print(f"  Email:    yes (notify on job end)")
    print(f"  Config:   {job.config_file}")

    try:
        arm_started_at = time.perf_counter()
        _arm_debounced_scheduler_submit_check(cfg)
        arm_finished_at = time.perf_counter()
        # arm_elapsed = arm_finished_at - arm_started_at
        # command_finished_at = time.perf_counter()

        # startup_elapsed = max(0.0, command_started_at - _MODULE_BOOT_T0)
        # parse_elapsed = max(0.0, args_parsed_at - command_started_at)
        # presave_elapsed = max(0.0, save_started_at - args_parsed_at)
        # finalize_elapsed = max(0.0, command_finished_at - arm_finished_at)
        # wall_elapsed = max(
        #     0.0,
        #     startup_elapsed + parse_elapsed + presave_elapsed + save_elapsed + arm_elapsed + finalize_elapsed,
        # )
        # print(
        #     f"[Scheduler] Deferred daemon check armed (+{float(_SCHEDULER_SUBMIT_CHECK_DEBOUNCE_DELAY):.1f}s, non-blocking, newer schedule calls supersede older pending checks)."
        # )
        # print(
        #     f"[Scheduler] schedule timing: total {wall_elapsed:.3f}s "
        #     f"[startup/import {startup_elapsed:.3f}s, arg-parse {parse_elapsed:.3f}s, pre-save {presave_elapsed:.3f}s, job-json {save_elapsed:.3f}s, helper-arm {arm_elapsed:.3f}s, finalize {finalize_elapsed:.3f}s]"
        # )
    except Exception as e:
        print(f"[Scheduler] WARNING: Failed to start detached delayed helper ({e}). Falling back to an immediate scheduler check.")
        _ensure_scheduler_daemon_for_scheduled_jobs(cfg)


def handle_handler_command(args: list[str]):
    """
    Handle the 'handler' command: manage the scheduler daemon.

    Sub-commands::

        python My_Lib_HPC.py handler               # Start the scheduler (if not running)
        python My_Lib_HPC.py handler start          # Same as above
        python My_Lib_HPC.py handler stop           # Cancel the running scheduler
        python My_Lib_HPC.py handler restart        # Stop + start the scheduler
        python My_Lib_HPC.py handler status         # Show scheduler status
        python My_Lib_HPC.py handler _run [opts]    # Internal: run the scheduler loop
                                                    #   (called by the SLURM job script)

    When invoked without arguments (or with ``start``), the handler:
      1. Creates ``{HOME_PATH}/HPC_Scheduler/`` if it does not exist.
      2. Checks if a scheduler is already running.
      3. If not, submits a scheduler SLURM job (high QoS, 1 CPU, max time).

    The ``restart`` sub-command stops any running scheduler and immediately
    starts a new one.

    The scheduler job runs ``python My_Lib_HPC.py handler _run`` which enters
    the main scheduling loop.  The loop:

            - Monitors ``HPC_Scheduler/`` for pending job configs.
            - Submits jobs to SLURM based on priority and resource constraints.
            - Self-renews before its time limit expires (< 2 hours remaining).
            - Terminates after 24h with no active (pending/submitted/running) jobs.

    Self-renewal mechanism:
      When the scheduler's remaining wall time drops below 2 hours, it submits
      a new scheduler job with ``--replace_job_id <current_id>``.  When the
      replacement starts running, it cancels the old scheduler and takes over.
      This ensures continuous scheduling across SLURM time limits.
    """
    cfg = _get_hpc_config()

    subcmd = args[0].lower() if args else "start"

    if subcmd in ("start",):
        # Start the scheduler
        scheduler_dir = _get_scheduler_dir(cfg)
        print(f"[Scheduler] Scheduler directory: {scheduler_dir}")

        if _is_scheduler_running(cfg):
            entries = _get_scheduler_queue_entries(cfg)
            running = [e for e in entries if e.state == "R"]
            pending = [e for e in entries if e.state == "PD"]
            ids_desc = ", ".join(
                f"{e.job_id}({'running' if e.state == 'R' else 'pending'})"
                for e in entries
            )
            print(f"[Scheduler] Scheduler is already active: {ids_desc}")
            print(f"[Scheduler] Use 'handler stop' to cancel, or 'handler status' to check.")
            return

        sid = _submit_scheduler_slurm_job(
            cfg,
            _ensure_job_script_dir=_ensure_job_script_dir,
            _resolve_preset=_resolve_preset,
            _build_sbatch_header=_build_sbatch_header,
            _calculate_auto_memory_mb=_calculate_auto_memory_mb,
        )
        if sid:
            print(f"[Scheduler] Scheduler submitted as SLURM job {sid}")
        else:
            print(f"[Scheduler] ERROR: Failed to submit scheduler job.")
            sys.exit(1)

    elif subcmd == "stop":
        # Cancel ALL scheduler jobs in the queue
        entries = _get_scheduler_queue_entries(cfg)
        if not entries:
            print("[Scheduler] No scheduler jobs found in the queue.")
            return
        for e in entries:
            print(f"[Scheduler] Cancelling scheduler job {e.job_id} (state={e.state})...")
            _cancel_slurm_job(e.job_id, cfg)
        print(f"[Scheduler] Done.")

    elif subcmd == "restart":
        # Stop all existing schedulers in the queue
        entries = _get_scheduler_queue_entries(cfg)
        if entries:
            for e in entries:
                print(f"[Scheduler] Stopping scheduler {e.job_id} (state={e.state})...")
                _cancel_slurm_job(e.job_id, cfg)
            import time
            time.sleep(2)  # brief wait for cancellation to register
        # Start a new one
        new_sid = _submit_scheduler_slurm_job(
            cfg,
            _ensure_job_script_dir=_ensure_job_script_dir,
            _resolve_preset=_resolve_preset,
            _build_sbatch_header=_build_sbatch_header,
            _calculate_auto_memory_mb=_calculate_auto_memory_mb,
        )
        if new_sid:
            print(f"[Scheduler] Scheduler restarted as SLURM job {new_sid}")
        else:
            print(f"[Scheduler] ERROR: Failed to submit new scheduler job.")
            sys.exit(1)

    elif subcmd == "status":
        state = _read_scheduler_state(cfg)
        if not state:
            print("[Scheduler] No scheduler state found.")
            return

        sid = state.get("slurm_job_id", "")
        started = state.get("started_at", "?")
        renewal = state.get("renewal_submitted", False)
        renewal_id = state.get("renewal_job_id", "")

        is_running = _is_scheduler_running(cfg)
        status_str = "RUNNING" if is_running else "NOT RUNNING"

        print(f"[Scheduler] Status: {status_str}")
        print(f"  SLURM job ID:     {sid}")
        print(f"  Started at:       {started}")
        print(f"  Renewal submitted: {renewal}")
        if renewal_id:
            print(f"  Renewal job ID:   {renewal_id}")

        # Show scheduled jobs summary
        jobs = _load_all_scheduled_jobs(cfg)
        if jobs:
            pending = sum(1 for j in jobs if j.status == "pending")
            submitted = sum(1 for j in jobs if j.status == "submitted")
            running = sum(1 for j in jobs if j.status == "running")
            completed = sum(1 for j in jobs if j.status == "completed")
            failed = sum(1 for j in jobs if j.status == "failed")
            print(f"\n  Scheduled jobs: {len(jobs)} total")
            print(f"    Pending:   {pending}")
            print(f"    Submitted: {submitted}")
            print(f"    Running:   {running}")
            if completed:
                print(f"    Completed: {completed}")
            if failed:
                print(f"    Failed:    {failed}")

    elif subcmd == "_run":
        # Internal: actually run the scheduler loop
        job_id = ""
        time_limit_seconds = 0
        replace_job_id_val = ""

        i = 1
        while i < len(args):
            arg = args[i].lstrip("-").lower()
            if arg == "job_id" and i + 1 < len(args):
                job_id = args[i + 1]
                i += 2
            elif arg == "time_limit" and i + 1 < len(args):
                try:
                    time_limit_seconds = int(args[i + 1])
                except ValueError:
                    time_limit_seconds = _parse_time_limit_seconds(args[i + 1])
                i += 2
            elif arg == "replace_job_id" and i + 1 < len(args):
                replace_job_id_val = args[i + 1]
                i += 2
            else:
                i += 1

        scheduler_main_loop(
            job_id, time_limit_seconds, replace_job_id_val,
            cfg,
            _ensure_job_script_dir=_ensure_job_script_dir,
            _resolve_preset=_resolve_preset,
            _build_sbatch_header=_build_sbatch_header,
            _calculate_auto_memory_mb=_calculate_auto_memory_mb,
            _compute_resources=_compute_resources,
            _submit_python_file=submit_python_file,
            _submit_gaussian_file=submit_Gaussian_file,
            _file_type=_lazy_file_type,
        )

    else:
        print(f"[My_Lib_HPC] Error: Unknown handler sub-command '{args[0]}'.")
        print("")
        print("Usage:")
        print("  python My_Lib_HPC.py handler          Start the scheduler")
        print("  python My_Lib_HPC.py handler start     Same as above")
        print("  python My_Lib_HPC.py handler stop      Cancel the running scheduler")
        print("  python My_Lib_HPC.py handler restart   Stop + start the scheduler")
        print("  python My_Lib_HPC.py handler status    Show scheduler status")
        sys.exit(1)


def handle_show_queue_command(args: list[str]):
    """
    Handle the 'show queue' command: display both SLURM queue and scheduler queue.

    Shows:
      1. The current SLURM queue (default: current user's jobs only).
      2. The scheduler's internal queue (pending/submitted/running jobs from
         ``HPC_Scheduler/``).
      3. Scheduler daemon status.

    Usage::

        python My_Lib_HPC.py show queue
        python My_Lib_HPC.py show queue --all
        python My_Lib_HPC.py show queue --user NAME
        python My_Lib_HPC.py show queue --ST RUNN
        python My_Lib_HPC.py show queue -task D2A

    Args:
        args: Optional flags:
            ``--all``            Show jobs for all users
            ``--user NAME``      Show jobs for a specific user
            (default)            Show only the current user's jobs
            ``--COLNAME VALUE``  Filter table rows: only show rows where the
                                 column named COLNAME contains VALUE as a
                                 case-insensitive substring.  Any column header
                                 can be used (e.g. ST, TASK, QOS, MEM, JOBID).
                                 Both single-dash and double-dash are accepted.

    Example output::

        =====================================================================
        Scheduler Queue
        =====================================================================
        ID                  TASK                          PRIO  ST       QOS     CORES  DEPENDENCIES      REASON
        -----------------------------------------------------------------------
        20260306-143025     my_script.py                 VIP   ⏳ PEND  high    16                       queue
        20260306-143030     postprocess.py result.json   50    ⏳ PEND  normal  auto   20260306-143025   queue

        Scheduler: 2 pending job(s)

        =====================================================================
        SLURM Queue (li5876)
        =====================================================================
        JOBID    TASK                QOS     ST     TIME     CPUS  MEM
        -----------------------------------------------------------------------
        3445001  my_train.py         normal  ▶ RUNN 1:23:45  128   256GB   a042
        3445002  my_train.py         normal  ◔ PEND 0:00     128   256GB   (Priority)

        Total: 2 job(s)   Pending: 1   Running: 1   Completing: 0
        Scheduler: RUNNING (34456789) - 5 pending job(s)
    """
    cfg = _get_hpc_config()

    # --- Parse arguments ---
    current_user: str | None = cfg.get("USER_NAME") or None
    user: str | None = current_user
    filters: dict[str, str] = {}  # column name (upper) → substring to match (case-insensitive)
    i = 0
    while i < len(args):
        arg = args[i].lstrip("-").lower()
        if arg == "all":
            user = None
        elif arg == "me":
            user = current_user
            if not user:
                print("[My_Lib_HPC] WARNING: USER_NAME not defined in config; showing all jobs.")
        elif arg == "user" and i + 1 < len(args):
            user = args[i + 1]
            i += 1
        elif i + 1 < len(args) and not args[i + 1].startswith("-"):
            # Generic column filter: --COLNAME value  or  -COLNAME value
            filters[arg.upper()] = args[i + 1]
            i += 1
        i += 1

    def _state_symbol(code: str) -> str:
        # 使用严格的单宽计算字符(Single-width Unicode Symbols)，避免 Emoji 双宽及 \ufe0f 变体导致的表格错位
        mapping = {
            "PD": ("◔", "PEND"),
            "CF": ("⚙", "CONF"),
            "R":  ("▶", "RUNN"),
            "CG": ("↻", "CMPT"),
            "CD": ("✔", "CMPD"),
            "F":  ("✖", "FAIL"),
            "TO": ("⧖", "TOUT"),
            "CA": ("⊘", "CANC"),
            "S":  ("⏸", "SUSP"),
        }
        res = mapping.get((code or "").upper())
        if res:
            symbol, abbr = res
            return f"{symbol} {abbr}"
        return (code or "").upper()

    def _state_order(code: str) -> tuple[int, str]:
        rank = {"PD": 0, "R": 1, "CG": 2}
        c = (code or "").upper()
        return rank.get(c, 99), c

    def _memory_to_gb_str(mem_text: str) -> str:
        s = (mem_text or "").strip().upper()
        m = re.match(r"^([0-9]+(?:\.[0-9]+)?)([KMGTP]?)(?:B)?(?:\+)?$", s)
        if not m:
            return s or "-"
        value = float(m.group(1))
        unit = m.group(2)
        scale_mb = {
            "": 1.0,
            "K": 1.0 / 1024.0,
            "M": 1.0,
            "G": 1024.0,
            "T": 1024.0 * 1024.0,
            "P": 1024.0 * 1024.0 * 1024.0,
        }
        mb = value * scale_mb.get(unit, 1.0)
        gb = mb / 1024.0
        return f"{gb:>3.0f} GB"

    def _print_table(
        title: str,
        columns: list[str],
        rows: list[list[str]],
        trailing_col: list[str] | None = None,
        left_align_cols: set[int] | None = None,
        repeat_threshold: int = 18,
    ):
        """Print a formatted table with dynamic column widths, capped to terminal width.

        Column 1 (TASK / FILE) is truncated with '…' when the table would
        otherwise exceed the current terminal width.

        Args:
            left_align_cols: set of column indices that should be left-aligned.
                             All other columns are center-aligned.
        """
        trailing_col = trailing_col or [""] * len(rows)
        left_align_cols = left_align_cols or set()

        # Compute natural column widths (in terminal display columns)
        widths: list[int] = []
        for col_idx, col_name in enumerate(columns):
            max_len = _str_display_width(col_name)
            for row in rows:
                if col_idx < len(row):
                    max_len = max(max_len, _str_display_width(str(row[col_idx])))
            widths.append(max_len)

        # Compute trailing column width (for separator lines)
        trailing_w = 0
        for t in trailing_col:
            if t:
                trailing_w = max(trailing_w, _str_display_width(t))

        # --- Cap to terminal width by shrinking column 1 (TASK / FILE) ---
        if len(widths) > 1:
            term_w = _get_terminal_width()
            sep_w = 4 * (len(widths) - 1)
            trail_total = (4 + trailing_w) if trailing_w else 0
            total_w = sum(widths) + sep_w + trail_total
            if total_w > term_w:
                other_w = sum(w for i, w in enumerate(widths) if i != 1) + sep_w + trail_total
                avail = term_w - other_w
                min_col1 = max(_str_display_width(columns[1]), 10)
                widths[1] = max(avail, min_col1)

        def _fmt_cell(text: str, width: int, left: bool) -> str:
            """Render *text* into a field of exactly *width* terminal columns."""
            if left:
                text = _truncate_task_display(text, width)
            else:
                text = _truncate_to_display_width(text, width)
            dw = _str_display_width(text)
            pad = width - dw
            if left:
                return text + ' ' * pad
            # centre
            lpad = pad // 2
            return ' ' * lpad + text + ' ' * (pad - lpad)

        def _format_header() -> str:
            parts = []
            for i, (col, w) in enumerate(zip(columns, widths)):
                parts.append(_fmt_cell(col, w, i in left_align_cols))
            return "    ".join(parts)

        def _format_row(row_vals: list[str], trailing: str) -> str:
            parts = []
            for i, (v, w) in enumerate(zip(row_vals, widths)):
                parts.append(_fmt_cell(str(v), w, i in left_align_cols))
            left = "    ".join(parts)
            if trailing:
                return f"{left}    {trailing}"
            return left

        header_line = _format_header()
        rendered_rows = [_format_row(r, t) for r, t in zip(rows, trailing_col)]
        # Total width includes the trailing column area
        main_width = _str_display_width(header_line)
        full_width = main_width + (4 + trailing_w if trailing_w else 0)
        content_width = max(full_width, _str_display_width(title), 65)
        border = "=" * content_width
        separator = "-" * content_width

        print(border)
        print(title)
        print(border)

        print(header_line)
        print(separator)
        for line in rendered_rows:
            print(line)

        if len(rows) >= repeat_threshold:
            print(separator)
            print(header_line)
        print(border)

    # Part 0: Recent job history — shown at the top so it scrolls off while queue stays visible
    _print_history_entries(cfg, count=200, max_hours=48, title_suffix=" (recent)")
    print("")

    current_user_queue = get_queue(user=current_user)

    # Part 1: Scheduler queue (pending only — submitted/running already appear in SLURM)
    jobs = _load_all_scheduled_jobs(cfg)
    pending_sched_jobs = [j for j in jobs if j.status == "pending"]

    if not pending_sched_jobs:
        print("=" * 65)
        print("Scheduler Queue")
        print("=" * 65)
        print("  (empty)")
        print("=" * 65)
        print("\n\n\n")
    else:
        sched_rows: list[list[str]] = []
        all_jobs = jobs  # full list for dependency checks
        submitted_count = len(current_user_queue)
        running_count = sum(1 for e in current_user_queue if (e.state or "").upper() in ("R", "CG"))
        max_scheduled = int(cfg.get("CONCURRENT_SCHEDULED_MISSION_COUNT", 10**9))
        max_running   = int(cfg.get("CONCURRENT_RUNNING_MISSION_COUNT",   10**9))
        concurrent_limit_hit = (submitted_count >= max_scheduled or running_count >= max_running)

        for j in pending_sched_jobs:
            prio_str = "VIP" if j.is_vip else str(j.priority_level)
            cores_str = str(j.cores) if j.cores else "auto"
            deps_str = ",".join(j.depends_on) if j.depends_on else ""
            task_text = _build_task_display_name(j.filepath, j.script_args)
            qos_str = j.qos if j.qos else _resolve_preset(None)['qos']

            # Reason why not yet submitted
            if not _are_dependencies_met(j, all_jobs):
                # Build a compact description: show dep IDs still pending/running
                dep_status = {d.schedule_id: d.status for d in all_jobs}
                blocking = [
                    dep_id for dep_id in j.depends_on
                    if dep_status.get(dep_id, "completed") not in ("completed", "failed")
                ]
                reason_str = "DEP: " + ",".join(blocking) if blocking else "DEP: failed"
            elif concurrent_limit_hit:
                reason_str = f"MAX ({submitted_count}/{max_scheduled} slots)"
            else:
                reason_str = "queue"

            sched_rows.append([
                j.schedule_id,
                task_text,
                prio_str,
                "⏳ PEND",
                qos_str,
                cores_str,
                deps_str,
                reason_str,
            ])

        # Apply column filters to scheduler rows
        if filters:
            _sched_col = ["ID", "TASK", "PRIO", "ST", "QOS", "CORES", "DEPENDENCIES", "REASON"]
            sched_rows = [
                row for row in sched_rows
                if all(
                    fv.lower() in str(row[_sched_col.index(fk)]).lower()
                    for fk, fv in filters.items()
                    if fk in _sched_col
                )
            ]

        # ID (0) and FILE (1) columns are left-aligned; all others center-aligned
        _print_table(
            title="Scheduler Queue",
            columns=["ID", "TASK", "PRIO", "ST", "QOS", "CORES", "DEPENDENCIES", "REASON"],
            rows=sched_rows,
            left_align_cols={0, 1},
        )

        print(f"Scheduler: {len(sched_rows)} of {len(pending_sched_jobs)} pending job(s)" if filters else f"Scheduler: {len(pending_sched_jobs)} pending job(s)")
        print("\n\n\n")

    # Part 2: SLURM queue
    print("")
    entries = current_user_queue if user == current_user else get_queue(user=user)
    entries.sort(key=lambda e: (0 if (e.state or "").upper() == "PD" else 1, -int(e.job_id) if e.job_id.isdigit() else 0))

    if not entries:
        print("=" * 65)
        print(f"SLURM Queue ({user or 'all users'})")
        print("=" * 65)
        print("  (empty)")
        print("=" * 65)
        print("\n\n\n")
    else:
        slurm_rows: list[list[str]] = []
        slurm_tail: list[str] = []
        for e in entries:
            c_text = str(e.num_cpus)
            if e.num_nodes and e.num_nodes != 1:
                c_text = f"{e.num_cpus}×{e.num_nodes}"

            slurm_rows.append([
                e.job_id,
                e.job_name or "-",
                e.qos or "-",
                _state_symbol(e.state),
                e.elapsed,
                c_text,
                _memory_to_gb_str(e.min_memory),
            ])
            slurm_tail.append(e.reason_or_nodelist)

        # Apply column filters to SLURM rows
        if filters:
            _slurm_col = ["JOBID", "TASK", "QOS", "ST", "TIME", "CPUS", "MEM"]
            _slurm_rows_f: list[list[str]] = []
            _slurm_tail_f: list[str] = []
            for _row, _tail in zip(slurm_rows, slurm_tail):
                if all(
                    fv.lower() in str(_row[_slurm_col.index(fk)]).lower()
                    for fk, fv in filters.items()
                    if fk in _slurm_col
                ):
                    _slurm_rows_f.append(_row)
                    _slurm_tail_f.append(_tail)
            slurm_rows, slurm_tail = _slurm_rows_f, _slurm_tail_f

        # TASK column (index 1) is left-aligned; all others center-aligned
        _print_table(
            title=f"SLURM Queue ({user or 'all users'})",
            columns=["JOBID", "TASK", "QOS", "ST", "TIME", "CPUS", "MEM"],
            rows=slurm_rows,
            trailing_col=slurm_tail,
            left_align_cols={1},
        )

        running = sum(1 for e in entries if e.state == "R")
        pending = sum(1 for e in entries if e.state == "PD")
        completing = sum(1 for e in entries if e.state == "CG")
        total_shown = len(slurm_rows)
        total_all = len(entries)
        filter_note = f" (filtered: {total_shown} of {total_all})" if filters and total_shown != total_all else ""
        print(f"Total: {total_all} job(s)   Pending: {pending}   Running: {running}   Completing: {completing}{filter_note}")
        print("\n")

    # Scheduler daemon status
    print("[My_Lib_HPC] Checking scheduler daemon status (querying squeue)...")
    scheduler_running = _is_scheduler_running(cfg)
    all_scheduler_jobs_for_status = _load_all_scheduled_jobs(cfg)
    n_sched_pending = sum(1 for j in all_scheduler_jobs_for_status if j.status == "pending")
    heartbeat_age = _read_heartbeat_age(cfg)
    if scheduler_running:
        state = _read_scheduler_state(cfg)
        sid = state.get("slurm_job_id", "?") if state else "?"
        pending_suffix = f" - {n_sched_pending} pending job(s)"
        hb_str = f", heartbeat {heartbeat_age:.0f}s ago" if heartbeat_age is not None else ""
        print(f"Scheduler: RUNNING ({sid}){pending_suffix}{hb_str}")
        if heartbeat_age is not None and heartbeat_age >= _SCHEDULER_HEARTBEAT_STALE:
            print(f"  ⚠ WARNING: Heartbeat is stale ({heartbeat_age:.0f}s old). Scheduler may be stuck!")
            print(f"  ⚠ Run 'hpc handler restart' to restart the scheduler.")
    else:
        print("Scheduler: NOT RUNNING")
        # Warn if there are pending jobs but scheduler is not running
        if n_sched_pending:
            print(f"  ⚠ WARNING: {n_sched_pending} pending job(s) but scheduler is NOT RUNNING!")
            print(f"  ⚠ Run 'hpc handler' or 'hpc handler restart' to start the scheduler.")

    # Show when the scheduler last submitted a job
    # if all_scheduler_jobs_for_status:
    #     submitted_times = [
    #         j.submitted_at for j in all_scheduler_jobs_for_status
    #         if j.submitted_at
    #     ]
    #     if submitted_times:
    #         last_submitted = max(submitted_times)
    #         print(f"  Last job submitted to SLURM by scheduler: {last_submitted}")

    # Clean up stale completed/failed/cancelled config files left behind by crashed scheduler
    for j in all_scheduler_jobs_for_status:
        if j.status in ("completed", "failed", "cancelled"):
            _delete_scheduled_job_file(j)



def _parse_history_time_arg(arg: str) -> float | None:
    """
    Parse a duration argument like '48h', '1d', '15m' into hours.

    Returns None if the argument is not a time-duration format.
    """
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(h|d|m)$', arg.lower())
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2)
    if unit == 'h':
        return value
    elif unit == 'd':
        return value * 24
    elif unit == 'm':
        return value / 60
    return None


def _elapsed_to_seconds(elapsed: str) -> float:
    """
    Parse an elapsed-time string from sacct (DD-HH:MM:SS, HH:MM:SS, MM:SS)
    into total seconds.
    """
    try:
        parts = elapsed.strip()
        days = 0
        if '-' in parts:
            day_part, parts = parts.split('-', 1)
            days = int(day_part)
        segments = parts.split(':')
        if len(segments) == 3:
            h, m, s = int(segments[0]), int(segments[1]), int(segments[2])
        elif len(segments) == 2:
            h, m, s = 0, int(segments[0]), int(segments[1])
        else:
            return 0
        return days * 86400 + h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return 0


def _format_duration_short(seconds: float) -> str:
    """
    Format a duration in seconds into a human-friendly short string.

    Examples: '2.3 h', '15 min', '■■ E ■■' (< 20s implies error).
    """
    if seconds < 20:
        return ' ■■ E ■■'
    if seconds < 360:  # < 6 min
        return f'{seconds / 60:>5.1f} min'
    hours = seconds / 3600
    if hours < 100:
        return f'{hours:>6.1f} h'
    days = hours / 24
    return f'{days:>5.1f} d'


def _format_ago(seconds: float) -> str:
    """
    Format a 'time ago' value from seconds into a human-friendly string.

    Examples: '5 min ago', '2.3 h ago', '1.2 d ago'.
    """
    if seconds < 60:
        return f'{int(seconds)} s ago'
    if seconds < 3600:
        return f'{seconds / 60:.0f} min ago'
    if seconds < 86400:
        return f'{seconds / 3600:.1f} h ago'
    return f'{seconds / 86400:.1f} d ago'


def _print_history_entries(
    cfg: dict,
    count: int = 200,
    max_hours: float = 48,
    title_suffix: str = "",
):
    """
    Fetch and print recent job history in a formatted table.

    Args:
        cfg:          HPC configuration dict.
        count:        Max number of entries to show.
        max_hours:    How far back to query (hours).
        title_suffix: Appended to the table title.
    """
    user = cfg.get("USER_NAME") or None

    # Progressive sacct queries: try shorter time windows first (fast),
    # only expand if we haven't collected enough entries yet.
    _PROGRESSIVE_HOURS = [1, 4, 12, 24, 48]
    entries: list[HistoryEntry] = []
    for trial_hours in _PROGRESSIVE_HOURS:
        if trial_hours > max_hours:
            break
        trial_str = f"now-{int(trial_hours)}hours"
        entries = get_recent_jobs(user=user, since=trial_str, count=count)
        if len(entries) >= count:
            break
    else:
        # Final attempt with the full max_hours if none of the progressive steps covered it
        if not entries or len(entries) < count:
            since_str = f"now-{int(max_hours)}hours" if max_hours < 24 * 365 else f"now-{int(max_hours / 24)}days"
            entries = get_recent_jobs(user=user, since=since_str, count=count)

    if not entries:
        print("=" * 65)
        print(f"Job History{title_suffix}")
        print("=" * 65)
        print("  (no completed jobs found)")
        print("=" * 65)
        print("\n\n\n")
        return

    # Reverse so that the newest jobs appear at the bottom
    entries = list(reversed(entries))

    now_ts = datetime.now().timestamp()

    def _state_symbol(state_str: str) -> str:
        s = state_str.upper()
        if "COMPLETED" in s:
            return "✔ CMPD"
        if "FAIL" in s:
            return "✖ FAIL"
        if "TIMEOUT" in s:
            return "⧖ TOUT"
        if "CANCEL" in s:
            return "⊘ CANC"
        if "OUT_OF_MEMORY" in s or "OOM" in s:
            return "✖ OOM"
        return s[:6]

    def _memory_to_gb_str(mem_text: str) -> str:
        s = (mem_text or "").strip().upper()
        m = re.match(r"^([0-9]+(?:\.[0-9]+)?)([KMGTP]?)(?:B)?(?:\+)?$", s)
        if not m:
            return s or "-"
        value = float(m.group(1))
        unit = m.group(2)
        scale_mb = {
            "": 1.0, "K": 1.0 / 1024.0, "M": 1.0,
            "G": 1024.0, "T": 1024.0 * 1024.0, "P": 1024.0 * 1024.0 * 1024.0,
        }
        mb = value * scale_mb.get(unit, 1.0)
        gb = mb / 1024.0
        return f"{gb:>3.0f} GB"

    rows: list[list[str]] = []
    trailing: list[str] = []

    for e in entries:
        # Parse end_time to compute "ago"
        ago_str = ""
        try:
            from datetime import datetime as _dt
            end_dt = _dt.fromisoformat(e.end_time.replace("T", " ").split(".")[0])
            ago_seconds = now_ts - end_dt.timestamp()
            ago_str = _format_ago(ago_seconds)
        except Exception:
            ago_str = ""

        # Duration
        elapsed_sec = _elapsed_to_seconds(e.elapsed)
        duration_str = _format_duration_short(elapsed_sec)

        # Cancelled < 20s → show ●● C ●●
        if elapsed_sec < 20 and "CANCEL" in e.state.upper():
            duration_str = " ●● C ●●"

        rows.append([
            e.job_id,
            duration_str,
            ago_str,
            e.job_name or "-",
            e.qos or "-",
            _state_symbol(e.state),
            str(e.num_cpus),
            _memory_to_gb_str(e.req_mem),
        ])
        trailing.append(e.node_list)

    # Column order: JOBID, DURATION, AGO, TASK, QOS, ST, CPUS, MEM
    columns = ["JOBID", "DURATION", "AGO", "TASK", "QOS", "ST", "CPUS", "MEM"]
    _TASK_COL = 3
    left_align_cols = {_TASK_COL}  # TASK column

    # Compute natural column widths (in terminal display columns)
    widths: list[int] = []
    for col_idx, col_name in enumerate(columns):
        max_len = _str_display_width(col_name)
        for row in rows:
            if col_idx < len(row):
                max_len = max(max_len, _str_display_width(str(row[col_idx])))
        widths.append(max_len)

    trailing_w = max((_str_display_width(t) for t in trailing if t), default=0)

    # --- Cap to terminal width by shrinking TASK column ---
    if len(widths) > 1:
        term_w = _get_terminal_width()
        sep_w = 4 * (len(widths) - 1)
        trail_total = (4 + trailing_w) if trailing_w else 0
        total_w = sum(widths) + sep_w + trail_total
        if total_w > term_w:
            other_w = sum(w for i, w in enumerate(widths) if i != _TASK_COL) + sep_w + trail_total
            avail = term_w - other_w
            min_col1 = max(_str_display_width(columns[_TASK_COL]), 10)
            widths[_TASK_COL] = max(avail, min_col1)

    def _fmt_cell(text: str, width: int, left: bool) -> str:
        """Render *text* into a field of exactly *width* terminal columns."""
        text = _truncate_to_display_width(text, width)
        dw = _str_display_width(text)
        pad = width - dw
        if left:
            return text + ' ' * pad
        lpad = pad // 2
        return ' ' * lpad + text + ' ' * (pad - lpad)

    def _format_header() -> str:
        parts = []
        for i, (col, w) in enumerate(zip(columns, widths)):
            parts.append(_fmt_cell(col, w, i in left_align_cols))
        return "    ".join(parts)

    def _format_row(row_vals: list[str], trail: str) -> str:
        parts = []
        for i, (v, w) in enumerate(zip(row_vals, widths)):
            if i == _TASK_COL:
                # Smart truncation: keep script name, show head + ' ... ' + tail
                text = _truncate_task_display(str(v), w)
                dw   = _str_display_width(text)
                parts.append(text + ' ' * (w - dw))
            else:
                parts.append(_fmt_cell(str(v), w, i in left_align_cols))
        left_part = "    ".join(parts)
        if trail:
            return f"{left_part}    {trail}"
        return left_part

    header_line = _format_header()
    main_width = _str_display_width(header_line)
    full_width = main_width + (4 + trailing_w if trailing_w else 0)
    content_width = max(full_width, 65)
    border = "=" * content_width
    separator = "-" * content_width

    print(border)
    print(f"Job History{title_suffix}")
    print(border)
    print(header_line)
    print(separator)
    for row, t in zip(rows, trailing):
        print(_format_row(row, t))

    if len(rows) >= 18:
        print(separator)
        print(header_line)
    print(border)
    print(f"Showing {len(rows)} completed job(s) within last {max_hours:.0f} hours")
    print("\n\n\n")


def handle_show_history_command(args: list[str]):
    """
    Handle 'show history': display recently completed/failed/cancelled jobs.

    Usage::

        python My_Lib_HPC.py show history              # last 200 jobs, within 48h
        python My_Lib_HPC.py show history 100           # last 100 jobs, within 48h
        python My_Lib_HPC.py show history 48h           # last 200 jobs, within 48h
        python My_Lib_HPC.py show history 1d            # last 200 jobs, within 24h
        python My_Lib_HPC.py show history 15m           # last 200 jobs, within 15 minutes
        python My_Lib_HPC.py show history 1000 48h      # last 1000 jobs, within 48h

    Args:
        args: Optional arguments: [count] [time_window]
    """
    cfg = _get_hpc_config()
    count = 200
    max_hours = 48.0

    for arg in args:
        # Try as integer count
        if re.match(r'^\d+$', arg):
            count = int(arg)
            continue
        # Try as duration
        parsed_hours = _parse_history_time_arg(arg)
        if parsed_hours is not None:
            max_hours = parsed_hours
            continue
        print(f"[My_Lib_HPC] WARNING: Unrecognized argument '{arg}' (expected number or time like 48h/1d/15m)")

    _print_history_entries(cfg, count=count, max_hours=max_hours)


def handle_show_job_command(args: list[str]):
    """
    Handle 'show <job_id>': display detailed information about a SLURM job.

    Fetches job details via ``scontrol show job`` (running/pending jobs) or
    ``sacct`` (completed/historical jobs) and prints them in a readable format.

    Usage::

        python My_Lib_HPC.py show <job_id>

    Example output::

        =====================================================================
        Job 34450001
        =====================================================================
        Name       : my_train.py
        State      : RUNNING
        QOS        : normal
        Partition  : normal
        Account    : myaccount
        ─────────────────────────────────────────────────────────────────────
        Nodes      : 1   CPUs: 128   Memory: 256000M
        Node list  : a042
        ─────────────────────────────────────────────────────────────────────
        Submitted  : 2026-03-04T10:00:00
        Started    : 2026-03-04T10:01:23
        Elapsed    : 1:23:45   Time limit: 14-00:00:00
        ─────────────────────────────────────────────────────────────────────
        Command    : /scratch/.../my_train.sh
        Work dir   : /scratch/.../my_project/
        Stdout     : /scratch/.../my_train.slurm.log
    """
    if not args:
        print("[My_Lib_HPC] Usage: python My_Lib_HPC.py show <job_id>")
        return

    job_id = args[0]
    info = get_job_info(job_id)

    if not info.job_id:
        print(f"[My_Lib_HPC] No information found for job {job_id}.")
        return

    border = "=" * 65
    divider = "-" * 65
    label_w = 10

    def _row(label: str, value: str):
        if value:
            print(f"  {label:<{label_w}}: {value}")

    batch_script = get_job_batch_script(job_id)
    if batch_script:
        print(border)
        print(f"Job {info.job_id}  (sacct --batch-script)")
        print(border)
        print(batch_script)
        print("")

    print(border)
    print(f"Job {info.job_id}  ({info.source})")
    print(border)

    _row("Name", info.job_name)
    _row("State", info.state + (f"  ({info.reason})" if info.reason else ""))
    _row("QOS", info.qos)
    _row("Partition", info.partition)
    _row("Account", info.account)
    _row("User", info.user_id)
    _row("Priority", info.priority)

    print(divider)
    nodes_line = f"{info.num_nodes}" if info.num_nodes else ""
    cpus_line = f"{info.num_cpus}" if info.num_cpus else ""
    resources = "  ".join(x for x in [
        f"Nodes: {nodes_line}" if nodes_line else "",
        f"CPUs: {cpus_line}" if cpus_line else "",
        f"Memory: {info.req_mem}" if info.req_mem else "",
    ] if x)
    if resources:
        print(f"  {resources}")
    _row("Node list", info.node_list)
    if info.max_rss or info.max_vm_size:
        _row("Peak RSS", info.max_rss)
        _row("Peak VM", info.max_vm_size)

    print(divider)
    _row("Submitted", info.submit_time)
    _row("Started", info.start_time)
    _row("End time", info.end_time)
    elapsed_line = info.elapsed
    if info.time_limit:
        elapsed_line = f"{info.elapsed}   Time limit: {info.time_limit}"
    _row("Elapsed", elapsed_line)
    _row("Exit code", info.exit_code if info.exit_code and info.exit_code != "0:0" else "")

    if info.command or info.work_dir or info.stdout:
        print(divider)
        _row("Command", info.command)
        _row("Work dir", info.work_dir)
        _row("Stdout", info.stdout)
        if info.stderr and info.stderr != info.stdout:
            _row("Stderr", info.stderr)


def handle_show_scheduled_job_command(args: list[str]):
    """
    Handle 'show <schedule_id>': display detailed information about a scheduler job.

    Finds the job with the given schedule ID in the scheduler queue (both
    pending and submitted/running jobs) and prints its configuration and
    current status.

    Usage::

        python My_Lib_HPC.py show <schedule_id>

    Example output::

        =====================================================================
        Scheduler Job 20260310-111136_01
        =====================================================================
        File       : /scratch/.../my_script.py
        Status     : submitted
        QOS        : high
        Priority   : VIP
        Cores      : 16
        Memory     : 32768 MB
        SLURM job  : 34748625
        ─────────────────────────────────────────────────────────────────────
        SH script  : /scratch/.../auto_generated_script_my_script.py_yymmdd.sh
        Output     : /scratch/.../auto_generated_script_my_script.py_yymmdd.out
        ─────────────────────────────────────────────────────────────────────
        Created    : 2026-03-10T11:11:36
        Submitted  : 2026-03-10T11:12:05
        Dependencies: 20260310-111100 (success)
    """
    if not args:
        print("[My_Lib_HPC] Usage: python My_Lib_HPC.py show <schedule_id>")
        return

    schedule_id = args[0]
    cfg = _get_hpc_config()
    all_jobs = _load_all_scheduled_jobs(cfg)

    job: ScheduledJob | None = next(
        (j for j in all_jobs if j.schedule_id == schedule_id), None
    )

    if job is None:
        print(f"[My_Lib_HPC] No scheduler job found with ID '{schedule_id}'.")
        return

    border = "=" * 65
    divider = "-" * 65
    label_w = 10

    def _row(label: str, value: str):
        if value:
            print(f"  {label:<{label_w}}: {value}")

    _status_symbols = {
        "pending":   "⏳ PEND",
        "submitted": "◔ SUBM",
        "running":   "▶ RUNN",
        "completed": "✔ DONE",
        "failed":    "✖ FAIL",
        "cancelled": "⊘ CANC",
    }
    status_str = _status_symbols.get(job.status, job.status.upper())

    print(border)
    print(f"Scheduler Job {job.schedule_id}")
    print(border)

    _row("File", job.filepath)
    _row("Status", status_str)
    _row("QOS", job.qos or "auto")
    _row("Priority", "VIP" if job.is_vip else (str(job.priority_level) if job.priority_level else "0"))
    _row("Cores", str(job.cores) if job.cores else "auto")
    _row("Memory", f"{job.memory_mb} MB" if job.memory_mb else "auto")
    _row("SLURM job", job.slurm_job_id or "")
    _row("Email", "yes" if job.email else "")

    if job.sh_script_path:
        print(divider)
        _row("SH script", job.sh_script_path)
        ext = os.path.splitext(job.filepath)[1].lower().lstrip(".")
        out_suffix = ".slurm.log" if ext in ("gjf", "com") else ".out"
        _row("Output", os.path.splitext(job.sh_script_path)[0] + out_suffix)

    if job.created_at or job.submitted_at or job.completed_at:
        print(divider)
        _row("Created", job.created_at)
        _row("Submitted", job.submitted_at)
        _row("Completed", job.completed_at)

    if job.depends_on:
        print(divider)
        dep_str = ", ".join(job.depends_on) + f"  (mode: {job.depend_mode})"
        _row("Depends on", dep_str)

    if job.error_message:
        print(divider)
        _row("Error", job.error_message)

    if job.script_args:
        print(divider)
        _row("Script args", " ".join(job.script_args))

    if job.config_file:
        print(divider)
        _row("Config file", job.config_file)

    # If the job has a SLURM ID, also show scontrol/sacct info
    if job.slurm_job_id:
        print("")
        handle_show_job_command([job.slurm_job_id])


def _show_help():
    """Print help for the 'show' sub-commands."""
    print("""\
Usage: python My_Lib_HPC.py show <sub-command> [options]

Sub-commands:
  show queue [--all | --user NAME] [--COLNAME VALUE ...]
      Show both the SLURM queue and the scheduler's internal queue.
      Default: shows only your own jobs.
      Arguments:
          [--all]            Show jobs for all users
          [--user NAME]      Show jobs for a specific user
          [--COLNAME VALUE]  Filter rows: only show entries where the column
                             named COLNAME contains VALUE (case-insensitive).
                             Any column header works (ST, TASK, QOS, MEM, JOBID, ...).
                             Both -COLNAME and --COLNAME are accepted.

      Examples:
          python My_Lib_HPC.py show queue
          python My_Lib_HPC.py show queue --all
          python My_Lib_HPC.py show queue --user li5876
          python My_Lib_HPC.py show queue --ST RUNN
          python My_Lib_HPC.py show queue -task D2A
          python My_Lib_HPC.py show --ST RUNN   (shorthand, no 'queue' needed)

  show avail [--idle|--mixed|--allocated|--drain]
      Show current node availability (sinfo).
      Arguments:
          [--idle]       Show only idle nodes
          [--mixed]      Show only mixed (partially used) nodes
          [--allocated]  Show only fully allocated nodes
          [--drain]      Show only draining/drained nodes

      Examples:
          python My_Lib_HPC.py show avail
          python My_Lib_HPC.py show avail --idle

  show history [count] [time_window]
      Show recently completed/failed/cancelled jobs.
      Arguments:
          [count]        Number of jobs to show (default: 200)
          [time_window]  Time window (e.g., 48h, 1d, 15m; default: 48h)
                         Can combine: show history 1000 48h

      Examples:
          python My_Lib_HPC.py show history
          python My_Lib_HPC.py show history 100
          python My_Lib_HPC.py show history 1d
          python My_Lib_HPC.py show history 1000 48h

  show <job_id>
      Show detailed information about a specific SLURM job.

      Examples:
          python My_Lib_HPC.py show 34450001

  show <schedule_id>
      Show detailed information about a scheduler job.
      Schedule IDs have the form  yyyymmdd-hhmmss  or  yyyymmdd-hhmmss_NN.
      If the scheduler job has already been submitted, also shows the
      underlying SLURM job details.

      Examples:
          python My_Lib_HPC.py show 20260310-111136
          python My_Lib_HPC.py show 20260310-111136_01\
""")


def handle_show_command(args: list[str]):
    """
    Handle the 'show' command dispatcher.

    Dispatches to:
      - ``show queue``          → handle_show_queue_command
      - ``show avail``          → handle_avail_command
      - ``show history``        → handle_show_history_command
      - ``show <job_id>``       → handle_show_job_command   (numeric SLURM ID)
      - ``show <schedule_id>``  → handle_show_scheduled_job_command
      - ``show --COLNAME VAL``  → handle_show_queue_command (filter shorthand)
      - ``show`` (no args)      → print show-related help

    Usage::

        python My_Lib_HPC.py show queue [--all | --user NAME] [--COLNAME VALUE ...]
        python My_Lib_HPC.py show avail [--idle|--mixed]
        python My_Lib_HPC.py show <job_id>
        python My_Lib_HPC.py show <schedule_id>
        python My_Lib_HPC.py show --ST RUNN    (filter shorthand)
        python My_Lib_HPC.py show -task D2A    (filter shorthand)
        python My_Lib_HPC.py show
    """
    if not args:
        _show_help()
        return

    sub = args[0].lower()

    if sub == "queue":
        handle_show_queue_command(args[1:])
    elif sub == "avail":
        handle_avail_command(args[1:])
    elif sub == "history":
        handle_show_history_command(args[1:])
    elif re.match(r"^\d+$", args[0]):
        handle_show_job_command(args)
    elif re.match(r"^\d{8}-\d{6}(_\d+)?$", args[0]):
        handle_show_scheduled_job_command(args)
    elif args[0].startswith("-"):
        # Filter flags without explicit 'queue' sub-command: e.g. "hpc show --ST RUNN"
        handle_show_queue_command(args)
    else:
        print(f"[My_Lib_HPC] Unknown 'show' sub-command: '{args[0]}'")
        print("")
        _show_help()


def print_usage():
    """
    Print usage information for the CLI.
    """
    print("""\
Usage: python My_Lib_HPC.py <command> [arguments...]

Available commands:

  submit <file> [--qos high|normal] [--cores N] [--mem SIZE] [--email] [script_args...]
      Submit a file to the HPC cluster.
      Arguments:
          <file>              Path to the file to submit (.py, .gjf, .com)
          [--qos high|normal] Optional QoS/priority (default: normal)
                              Synonyms: --priority, -qos, -priority
          [--cores N]         Optional number of CPU cores to request
                              Synonyms: --cpu, -cores, -cpu
          [--mem SIZE]        Optional memory to request (e.g., 20GB, 100000MB, default unit: GB)
                              Synonyms: --memory, -mem, -memory
          [--email]           Send email notification when the job ends
                              (regardless of exit status). Requires MAIL_USER in config.
          [script_args...]    Extra arguments forwarded to the submitted script

      Note: Parameters are case-insensitive and support single/double dash.
            Short options (e.g., -c) are forwarded to the script.

    stop [job_id... | all [qos=<name> | --qos <name>]]
      Cancel one or more jobs (SLURM or scheduler).
      Arguments:
          [job_id...]         SLURM job IDs (supports ranges, e.g. 1 2 3-5 100-200)
                              or scheduler IDs (e.g. 20260306-143025).
                              If a numeric ID is not in the SLURM queue but matches
                              a scheduler job, the scheduler job is cancelled instead.
                              If omitted, interactive mode is used.
                    all                 Cancel ALL SLURM + pending scheduler jobs.
                                                            The SLURM daemon job 'HPC_Scheduler' is excluded.
                    [qos=<name>]        Optional filter for 'stop all'.
                    [--qos <name>]      Same as above.

      Examples:
          python My_Lib_HPC.py submit my_script.py
          python My_Lib_HPC.py submit my_script.py --qos high
          python My_Lib_HPC.py submit my_script.py --cores 16
          python My_Lib_HPC.py submit my_script.py -CPU 8  (synonym, case-insensitive)
          python My_Lib_HPC.py submit my_script.py --mem 20GB
          python My_Lib_HPC.py submit my_script.py --email
          python My_Lib_HPC.py submit my_script.py --cores 8 --mem 50GB
          python My_Lib_HPC.py submit my_script.py -CORES 8 -b 64  (-b forwarded to script)
          python My_Lib_HPC.py submit my_script.py --parm_for_my_script1 abc --parm_for_my_script2 cde --CPU 8
          python My_Lib_HPC.py submit    (interactive mode)
          python My_Lib_HPC.py stop 12345 12346
          python My_Lib_HPC.py stop 12350-12360
          python My_Lib_HPC.py stop 20260306-143025             (cancel a scheduler job)
          python My_Lib_HPC.py stop all
          python My_Lib_HPC.py stop all qos=high
          python My_Lib_HPC.py stop all --qos normal
          python My_Lib_HPC.py stop      (interactive mode)

    tail <file_or_job_id> [-n N]
      Monitor a file, printing new content as it is appended (like tail -f).
      Arguments:
                    <file_or_job_id>
                                        Path to the file to monitor, or a SLURM job ID
                                        (auto-resolves job StdOut/StdErr path)
          [-n N]    Number of lines to show initially (default: 200)
      Notes:
          If the file does not exist yet, waits until it appears.
          Press Ctrl+C to stop.

      Examples:
          python My_Lib_HPC.py tail job_output.log
          python My_Lib_HPC.py tail 34450001
          python My_Lib_HPC.py tail job_output.log -n 50

  compress [path ...]
      Pack files/folders into a .tar.gz archive.
      Arguments:
          [path ...]  Paths to files or folders to compress.
                      If omitted, interactive mode is used.
      Output naming:
          Single item  → <parent>/<item_name>.tar.gz
          Multiple items → <common_parent>/<common_parent_name>.tar.gz
          If the target name is taken, _01 / _02 … is appended.
      Notes:
          Asks whether to delete originals before compression starts.
          Originals are only deleted after successful verification.

      Examples:
          python My_Lib_HPC.py compress /scratch/.../Art_Thesis_Agents/
          python My_Lib_HPC.py compress file1.py file2.py
          python My_Lib_HPC.py compress    (interactive mode)

  queue [--all | --user NAME] [--COLNAME VALUE ...]
      Show both the SLURM queue and the scheduler's internal queue.
      Default: shows only your own jobs.
      Arguments:
          [--all]            Show jobs for all users
          [--user NAME]      Show jobs for a specific user
          [--COLNAME VALUE]  Filter rows by column value (case-insensitive substring).
                             Any column header works: ST, TASK, QOS, MEM, JOBID, ...
                             Both -COLNAME and --COLNAME are accepted.
      (Equivalent to 'show queue'.)

  avail [--idle|--mixed|--allocated|--drain]
      Show current node availability (sinfo).
      (Alias for 'show avail'; the 'show avail' form is preferred.)
      Arguments:
          [--idle]       Show only idle nodes
          [--mixed]      Show only mixed (partially used) nodes
          [--allocated]  Show only fully allocated nodes
          [--drain]      Show only draining/drained nodes

  show <sub-command> [options]
      Collection of display commands. Run 'show' with no arguments for help.

    show queue [--all | --user NAME] [--COLNAME VALUE ...]
        Show both the SLURM queue and the scheduler's internal queue.
        Default: shows only your own jobs.
        Arguments:
            [--all]            Show jobs for all users
            [--user NAME]      Show jobs for a specific user
            [--COLNAME VALUE]  Filter rows by column value (case-insensitive substring).
                               Any column header works: ST, TASK, QOS, MEM, JOBID, ...
                               Both -COLNAME and --COLNAME are accepted.

        Examples:
            python My_Lib_HPC.py show queue
            python My_Lib_HPC.py show queue --all
            python My_Lib_HPC.py show queue --user li5876
            python My_Lib_HPC.py show queue --ST RUNN
            python My_Lib_HPC.py show queue -task D2A
            python My_Lib_HPC.py show --ST RUNN         (shorthand, no 'queue' needed)

    show avail [--idle|--mixed|--allocated|--drain]
        Show current node availability (sinfo).
        Arguments:
            [--idle]       Show only idle nodes
            [--mixed]      Show only mixed (partially used) nodes
            [--allocated]  Show only fully allocated nodes
            [--drain]      Show only draining/drained nodes

        Examples:
            python My_Lib_HPC.py show avail
            python My_Lib_HPC.py show avail --idle

    show <job_id>
        Show detailed information about a specific SLURM job using scontrol/sacct.

        Examples:
            python My_Lib_HPC.py show 34450001

    show history [count] [time_window]
        Show recently completed/failed/cancelled jobs.
        Arguments:
            [count]        Number of jobs to show (default: 200)
            [time_window]  Time window (e.g., 48h, 1d, 15m; default: 48h)

        Examples:
            python My_Lib_HPC.py show history
            python My_Lib_HPC.py show history 100
            python My_Lib_HPC.py show history 1d
            python My_Lib_HPC.py show history 1000 48h

  history [count] [time_window]
      Show recently completed/failed/cancelled jobs.
      (Shortcut for 'show history'.)

      Examples:
          python My_Lib_HPC.py history
          python My_Lib_HPC.py history 100

  schedule <file> [options]
      Add a job to the scheduler queue (deferred submission).
      The scheduler daemon will submit it when resources allow,
      respecting priority ordering.  Jobs get short sequential IDs
      (001, 002, …) for easy reference.
      Arguments:
          <file>                   Path to the file to schedule (.py, .gjf, .com)
          [--priority N]           Scheduler priority (higher = submit first, default: 0)
          [--vip]                  Infinite priority — submitted before all non-VIP jobs
          [--qos high|normal]      Optional QoS/priority (default: normal)
          [--cores N]              Optional number of CPU cores
          [--mem SIZE]             Optional memory (e.g., 20GB, 100000MB)
          [--email]                Send email notification when the job ends
          [--after ID[,ID,...]]    Run only after listed jobs succeed
          [--after-any ID[,ID,...]] Run after listed jobs finish (even if some failed)
          [script_args...]         Extra arguments forwarded to the script

      Examples:
          python My_Lib_HPC.py schedule my_script.py
          python My_Lib_HPC.py schedule my_script.py --qos high --priority 100
          python My_Lib_HPC.py schedule my_script.py --vip --cores 16
          python My_Lib_HPC.py schedule my_script.py --after 001,002
          python My_Lib_HPC.py schedule my_script.py --after-any 003 --priority 50
          python My_Lib_HPC.py schedule my_script.py --email

  schedule cancel <id> [<id> ...]
      Cancel one or more previously scheduled jobs by ID.
      If the job is pending, it is removed.  If submitted/running,
      the SLURM job is also cancelled.

      Examples:
          python My_Lib_HPC.py schedule cancel 20260306-143025
          python My_Lib_HPC.py schedule cancel 20260306-143025 20260306-143030

  handler [start|stop|restart|status]
      Manage the scheduler daemon.
      Sub-commands:
          (no argument)   Start the scheduler daemon (high QoS, 1 CPU, max time)
          start           Same as above
          stop            Cancel the running scheduler
          restart         Stop the current scheduler and start a new one
          status          Show scheduler status and job summary

      The scheduler daemon monitors HPC_Scheduler/ for pending jobs and
      submits them to SLURM based on priority and resource limits.
      It self-renews before its time limit expires.

      Examples:
          python My_Lib_HPC.py handler
          python My_Lib_HPC.py handler status
          python My_Lib_HPC.py handler stop
          python My_Lib_HPC.py handler restart

For more information, see module docstring or configuration file.

Shortcuts:
  Most 'show' sub-commands can be used directly as top-level commands:
    python My_Lib_HPC.py history   →  python My_Lib_HPC.py show history
    python My_Lib_HPC.py queue     →  python My_Lib_HPC.py show queue
    python My_Lib_HPC.py avail     →  python My_Lib_HPC.py show avail\
""")


def main():
    """
    Command-line entry point with subcommand dispatch.

    Usage:
        python My_Lib_HPC.py <command> [arguments...]

    Commands:
        submit <file> [--qos high|normal] [--cores N] [--mem SIZE] [--email] [script_args...]
            Submit a file (.py / .gjf / .com) to the HPC cluster.
            Without <file>, enters interactive mode (prompts for paths, priority,
            cores, memory and email notification).

        stop [job_id... | all [qos=<name> | --qos <name>]]
            Cancel one or more SLURM or scheduler jobs.
            Accepts individual IDs, ranges (100-200) and scheduler IDs.
            Without arguments, enters interactive mode.
            'all' cancels every queued/running SLURM job + pending scheduler jobs.
            The SLURM daemon job 'HPC_Scheduler' is excluded.
            'stop all' supports 'qos=<name>' and '--qos <name>'.

        tail <file_or_job_id> [-n N]
            Follow a log file (like ``tail -f``).
            Accepts a file path OR a SLURM job ID (auto-resolves StdOut path).
            Waits if the file does not yet exist.  Default: last 200 lines.

        compress [path ...]
            Pack files/folders into a .tar.gz archive.
            Prompts for individual-vs-combined mode and optional deletion of originals.
            Without paths, enters interactive mode.

        avail [--idle|--mixed|--allocated|--drain]
            Show current node availability (sinfo).  Alias for 'show avail'.

        queue [--all | --user NAME]
            Show SLURM queue and scheduler internal queue.
            Defaults to the current user's jobs.  Alias for 'show queue'.

        history [count] [time_window]
            Show recently completed/failed/cancelled jobs (sacct).
            Examples: history 100 / history 1d / history 1000 48h.
            Alias for 'show history'.

        show <sub-command> [options]
            Collection of display commands:
                show queue  [--all | --user NAME]
                show avail  [--idle | --mixed | --allocated | --drain]
                show history [count] [time_window]
                show <slurm_id>    — detailed info via scontrol / sacct
                show <schedule_id> — scheduler job info (e.g. 20260310-111136_01)

        schedule <file> [options]
            Add a job to the scheduler queue (deferred SLURM submission).
            Options: --priority N, --vip, --qos, --cores, --mem, --email,
                     --after ID[,ID,...], --after-any ID[,ID,...].

        schedule cancel <id> [<id> ...]
            Cancel previously scheduled jobs by short ID (e.g. 001, 002).

        schedule [start|stop|restart|status]
            Alias for the handler daemon commands.

        handler [start|stop|restart|status]
            Manage the scheduler daemon.
            start    — submit a scheduler SLURM job (if not already running)
            stop     — cancel the running scheduler
            restart  — stop + start
            status   — show daemon status, heartbeat, and job summary

        scheduler [start|stop|restart|status]
            Alias for handler.

    For detailed help, run without arguments or see the module docstring.
    """
    if len(sys.argv) < 2:
        print("[My_Lib_HPC] Error: No command specified.")
        print("")
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    args = sys.argv[2:]  # Arguments after the command
    handler_alias_subcommands = {"start", "stop", "restart", "status", "_run"}

    # Command dispatch dictionary - add new commands here
    commands = {
        "submit": handle_submit_command,
        "stop": handle_stop_command,
        "tail": handle_tail_command,
        "compress": handle_compress_command,
        "avail": handle_avail_command,        # alias: 'show avail' is preferred
        "queue": handle_queue_command,
        "schedule": handle_schedule_command,
        "handler": handle_handler_command,
        "scheduler": handle_handler_command,
        "show": handle_show_command,
        "history": lambda a: handle_show_history_command(a),  # shortcut for 'show history'
    }

    # Sub-commands under 'show' that can also be used as top-level shortcuts.
    # e.g. "hpc history" → "hpc show history"
    show_shortcuts = {"history", "queue", "avail"}  # queue & avail already exist as direct commands

    if command == "schedule" and (not args or args[0].lower() in handler_alias_subcommands):
        handle_handler_command(args)
    elif command in commands:
        commands[command](args)
    elif command in show_shortcuts:
        # If it somehow wasn't in commands but is a show shortcut, dispatch via show
        handle_show_command([command] + args)
    else:
        print(f"[My_Lib_HPC] Error: Unknown command '{command}'.")
        print("")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
