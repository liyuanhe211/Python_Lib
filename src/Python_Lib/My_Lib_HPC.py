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
    E:/My_Program/                                              ← My_Program root
    ├── My_Lib_HPC_Configuration_Secret_Negishi.py              ← config file
    └── Python_Lib/
        └── src/
            └── Python_Lib/
                └── My_Lib_HPC.py                               ← this file

Usage (command line):
    python  .../My_Lib_HPC.py  submit  <python_file>  [--qos PRIORITY]  [--cores N]  [--mem SIZE]  [script_args...]

    Commands:
        submit              Submit a file to the HPC cluster
        stop                Cancel one or more jobs

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

        # Tail a log file (prints last 200 lines, then follows; waits if file absent)
        python /scratch/.../My_Lib_HPC.py  tail  job_output.log
        python /scratch/.../My_Lib_HPC.py  tail  job_output.log  -n 50

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
    .py   → submit_python_file()    Generates a job that runs the script with the
                                     configured Python interpreter.
    (Other file types such as .gjf can be added in the future.)
"""

__author__ = 'LiYuanhe'

import sys
import os
import re
import glob
import pathlib
import subprocess
import random
import importlib
import importlib.util

# ---------------------------------------------------------------------------
# Resolve paths and import configuration
# ---------------------------------------------------------------------------
Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)

from My_Lib_Stock import get_input_with_while_cycle, parse_range_selection
from My_Lib_File import *
from datetime import datetime

# My_Program root is 4 levels up from this file:
#   this file   → Python_Lib/  → src/  → Python_Lib/  → My_Program/
My_Program_path = str(pathlib.Path(__file__).parent.parent.parent.parent.resolve())

# ---------------------------------------------------------------------------
# Auto-discover the HPC configuration file
# ---------------------------------------------------------------------------
_CONFIG_PATTERN = "My_Lib_HPC_Configuration_Secret_*.py"
_config_files = glob.glob(os.path.join(My_Program_path, _CONFIG_PATTERN))

# Filter out __pycache__ or .pyc artefacts just in case
_config_files = [f for f in _config_files if f.endswith(".py") and "__pycache__" not in f]

if len(_config_files) == 0:
    raise FileNotFoundError(
        f"No HPC configuration file matching '{_CONFIG_PATTERN}' found in:\n"
        f"  {My_Program_path}\n"
        f"Please create one (see My_Lib_HPC_Configuration_Secret_Negishi.py as an example)."
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
_config_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_module)

# Pull all public names into this module's namespace
for _name in dir(_config_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_config_module, _name)

print(f"[My_Lib_HPC] Loaded configuration for HPC: {HPC_NAME}")


# ===========================================================================
# Helper utilities
# ===========================================================================

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


def _compute_resources() -> tuple[int, int]:
    """
    Compute the number of cores and memory (MB) to request based on config.

    Returns:
        (cores, memory_mb)
    """
    cores = max(1, int(CORES_PER_NODE * CORE_FRACTION))
    memory_mb = max(1024, int(MEMORY_PER_NODE_MB * MEMORY_FRACTION))
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


def _build_sbatch_header(
    job_name: str,
    output_file: str,
    preset: dict,
    cores: int,
    memory_mb: int,
) -> str:
    """
    Build the #SBATCH comment block for a job script.

    Args:
        job_name:    Job name (--job-name).
        output_file: Path pattern for stdout/stderr (--output).
        preset:      SLURM preset dict with partition, qos, time_limit.
        cores:       Number of CPU cores to request (-c).
        memory_mb:   Memory in MB to request (--mem).

    Returns:
        A multi-line string with the #!/bin/bash shebang and all #SBATCH lines.
    """
    lines = [
        "#!/bin/bash",
        f"#SBATCH --output={output_file}",
        f"#SBATCH --partition={preset['partition']}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={SLURM_ACCOUNT}",
        f"#SBATCH --qos={preset['qos']}",
        f"#SBATCH --get-user-env",
        f"#SBATCH --nodes={NODES}",
        f"#SBATCH --ntasks-per-node={NTASKS_PER_NODE}",
        f"#SBATCH --time={preset['time_limit']}",
        f"#SBATCH --cpus-per-task={cores}",
        f"#SBATCH --mem={memory_mb}",
    ]

    if MAIL_USER:
        lines.append(f"#SBATCH --mail-type={MAIL_TYPE}")
        lines.append(f"#SBATCH --mail-user={MAIL_USER}")

    return "\n".join(lines)


# ===========================================================================
# Job submission functions
# ===========================================================================

def submit_python_file(python_file: str, priority: str | None = None, script_args: list[str] | None = None, cores: int | None = None, memory_mb: int | None = None) -> str:
    """
    Generate a SLURM job script that runs a Python file and submit it.

    The generated script will:
        1. Print the active Python interpreter path (``which python``).
        2. Run the specified Python file using the configured PYTHON_PATH.

    Args:
        python_file:  Absolute path to the .py file to execute on the cluster.
        priority:     "high" (normal QoS, 2 weeks) or "normal" (standby QoS,
                      4 hours).  Defaults to DEFAULT_PRESET from configuration.
                      Supplied via ``--qos`` on the command line.
        script_args:  Extra arguments to forward to the Python script (e.g.
                      ``['--num_experts', '8', '--hidden_dim', '64']``).
        cores:        Number of CPU cores to request. If None, uses default
                      calculation (CORES_PER_NODE * CORE_FRACTION). If specified
                      and exceeds CORES_PER_NODE, requires user confirmation.
        memory_mb:    Memory in MB to request. If None, calculated automatically
                      based on cores. If specified and exceeds MEMORY_PER_NODE_MB,
                      requires user confirmation.

    Returns:
        The path to the generated job script.
    """
    python_file = os.path.abspath(python_file)
    if not os.path.isfile(python_file):
        raise FileNotFoundError(f"Python file not found: {python_file}")

    preset = _resolve_preset(priority)
    
    # Handle cores and memory allocation
    if cores is None and memory_mb is None:
        # Both unspecified: use defaults
        cores, memory_mb = _compute_resources()
    elif cores is not None and memory_mb is None:
        # Cores specified, memory not: validate cores and calculate memory proportionally
        if cores > CORES_PER_NODE:
            print(f"[My_Lib_HPC] WARNING: Requested cores ({cores}) exceeds CORES_PER_NODE ({CORES_PER_NODE})")
            confirm = input("Continue with submission? (y/n): ")
            if confirm.lower() != 'y':
                print("[My_Lib_HPC] Submission aborted.")
                sys.exit(0)
        # Calculate memory proportionally based on cores
        memory_mb = max(1024, int(MEMORY_PER_NODE_MB * (cores / CORES_PER_NODE)))
    elif cores is None and memory_mb is not None:
        # Memory specified, cores not: validate memory and use default cores
        cores = max(1, int(CORES_PER_NODE * CORE_FRACTION))
        if memory_mb > MEMORY_PER_NODE_MB:
            print(f"[My_Lib_HPC] WARNING: Requested memory ({memory_mb} MB) exceeds MEMORY_PER_NODE_MB ({MEMORY_PER_NODE_MB} MB)")
            confirm = input("Continue with submission? (y/n): ")
            if confirm.lower() != 'y':
                print("[My_Lib_HPC] Submission aborted.")
                sys.exit(0)
    else:
        # Both specified: validate both
        if cores > CORES_PER_NODE:
            print(f"[My_Lib_HPC] WARNING: Requested cores ({cores}) exceeds CORES_PER_NODE ({CORES_PER_NODE})")
            confirm = input("Continue with submission? (y/n): ")
            if confirm.lower() != 'y':
                print("[My_Lib_HPC] Submission aborted.")
                sys.exit(0)
        if memory_mb > MEMORY_PER_NODE_MB:
            print(f"[My_Lib_HPC] WARNING: Requested memory ({memory_mb} MB) exceeds MEMORY_PER_NODE_MB ({MEMORY_PER_NODE_MB} MB)")
            confirm = input("Continue with submission? (y/n): ")
            if confirm.lower() != 'y':
                print("[My_Lib_HPC] Submission aborted.")
                sys.exit(0)

    job_name = os.path.basename(python_file)
    script_path = _generate_script_filename(python_file)
    output_file = os.path.splitext(script_path)[0] + ".out"

    header = _build_sbatch_header(job_name, output_file, preset, cores, memory_mb)

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
    print(f"[My_Lib_HPC] Submitting with: {SUBMIT_COMMAND} {script_path}")
    
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


# ===========================================================================
# Dispatcher — choose handler based on file extension
# ===========================================================================


def submit_file(filepath: str, priority: str | None = None, script_args: list[str] | None = None, cores: int | None = None, memory_mb: int | None = None) -> str:
    """
    Dispatch a file to the appropriate submission handler based on its extension.

    Args:
        filepath:    Path to the input file.
        priority:    "high" or "normal" (see submit_python_file for details).
        script_args: Extra arguments forwarded to the submitted script.
        cores:       Number of CPU cores to request (see submit_python_file for details).
        memory_mb:   Memory in MB to request (see submit_python_file for details).

    Returns:
        The path to the generated job script.

    Raises:
        ValueError: If the file extension is not supported.
    """
    filepath = os.path.abspath(filepath)
    ext = filename_last_append(filepath).lower()

    if ext == "py":
        return submit_python_file(filepath, priority, script_args, cores, memory_mb)
    else:
        print(f"Unsupported file extension '{ext}'.")
        sys.exit(1)


# ===========================================================================
# CLI entry point
# ===========================================================================

def handle_submit_command(args: list[str]):
    """
    Handle the 'submit' command.

    Args:
        args: Command-line arguments after 'submit' command.
              Expected: [file_to_submit] [--qos high|normal] [--cores N] [--mem SIZE] [script_args...]
              
              Parameters are case-insensitive and accept single/double dash (but not short -c style).
              Synonyms: --qos/--priority, --cores/--cpu, --mem/--memory

    Usage:
        python My_Lib_HPC.py submit <file_to_submit> [--qos high|normal] [--cores N] [--mem SIZE] [script_args...]
        python My_Lib_HPC.py submit <file_to_submit> [-CORES 16] [-MEM 50GB]  (case-insensitive, single dash works)
        python My_Lib_HPC.py submit my_script.py --parm_for_my_script1 abc --parm_for_my_script2 cde --CPU 8
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
        
        # Ask for cores
        cores_input = input(f"Number of cores [default={int(CORES_PER_NODE * CORE_FRACTION)}]: ").strip()
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
                    submit_file(filepath, priority, cores=cores, memory_mb=memory_mb)
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
            else:
                # Not a recognized parameter, forward to script
                script_args.append(remaining[i])
                i += 1

        submit_file(filepath, priority, script_args if script_args else None, cores, memory_mb)


def handle_stop_command(args: list[str]):
    """
    Handle the 'stop' command to cancel jobs.

    Args:
        args: Command-line arguments after 'stop' command.
              Expected: [job_id ...] where job IDs can be individual numbers or ranges.
              Examples: "12345", "12346 12347", "12350-12360"
              If empty, enters interactive mode.
    
    Usage:
        python My_Lib_HPC.py stop 12345 12346 12350-12360
        python My_Lib_HPC.py stop  (interactive mode)
    """
    # Check for CANCEL_COMMAND global variable (from config)
    if not globals().get("CANCEL_COMMAND"):
        print(f"[My_Lib_HPC] Error: CANCEL_COMMAND is not defined in the configuration file.")
        sys.exit(1)
        
    global CANCEL_COMMAND # Just to be safe for linter, though defined dynamically

    job_ids_to_cancel = []

    if args:
        # Command-line mode: args are job IDs (e.g. "stop 1 2 3-5")
        # Concatenate arguments into a single string for parsing.
        input_str = " ".join(args)
        try:
            parsed_ids = parse_range_selection(input_str, decrease_by_1=False)
            if parsed_ids:
                job_ids_to_cancel.extend(parsed_ids)
        except Exception as e:
            print(f"[My_Lib_HPC] Warning: Could not parse '{input_str}': {e}")
            
    else:
        # Interactive mode
        print("[My_Lib_HPC] Enter job IDs to cancel (e.g. 12345, 12346, 12350-12360).")
        print("[My_Lib_HPC] Empty line to finish.")
        
        # We need to manually call get_input_with_while_cycle here as in submit
        print("Job IDs to stop (ends with empty line):")
        input_lines = get_input_with_while_cycle(
            break_condition=lambda x: not x.strip(),
            strip_quote=True
        )
        
        for line in input_lines:
            line = line.strip()
            if line:
                try:
                    parsed_ids = parse_range_selection(line, decrease_by_1=False)
                    if parsed_ids:
                        job_ids_to_cancel.extend(parsed_ids)
                except Exception as e:
                    print(f"[My_Lib_HPC] Warning: Could not parse '{line}': {e}")

    # Remove duplicates and sort
    job_ids_to_cancel = sorted(list(set(job_ids_to_cancel)))

    if not job_ids_to_cancel:
        print("[My_Lib_HPC] No valid job IDs provided to stop.")
        return

    print(f"[My_Lib_HPC] Planning to cancel {len(job_ids_to_cancel)} job(s)...")
    print("------------------------------------------------------------------------")
    print("Jobs to cancel:")
    for i in job_ids_to_cancel:
        print(f"  - {i}")

    confirm = input("Are you sure you want to cancel these jobs? (y/n): ")
    if confirm.lower() != 'y':
        print("[My_Lib_HPC] Cancellation aborted.")
        return

    for job_id in job_ids_to_cancel:
        cmd = [globals()["CANCEL_COMMAND"], str(job_id)]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"[My_Lib_HPC] Error running command for job {job_id}: {e}")


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
              Expected: <file> [-n N]

    Usage:
        python My_Lib_HPC.py tail <file>
        python My_Lib_HPC.py tail <file> -n 100
        python My_Lib_HPC.py tail -n 100 <file>
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
        print("Usage: python My_Lib_HPC.py tail <file> [-n N]")
        sys.exit(1)

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


def print_usage():
    """
    Print usage information for the CLI.
    """
    print("Usage: python My_Lib_HPC.py <command> [arguments...]")
    print("")
    print("Available commands:")
    print("")
    print("  submit <file> [--qos high|normal] [--cores N] [--mem SIZE] [script_args...]")
    print("      Submit a file to the HPC cluster.")
    print("      Arguments:")
    print("          <file>              Path to the file to submit (.py, etc.)")
    print("          [--qos high|normal] Optional QoS/priority (default: normal)")
    print("                              Synonyms: --priority, -qos, -priority")
    print("          [--cores N]         Optional number of CPU cores to request")
    print("                              Synonyms: --cpu, -cores, -cpu")
    print("          [--mem SIZE]        Optional memory to request (e.g., 20GB, 100000MB, default unit: GB)")
    print("                              Synonyms: --memory, -mem, -memory")
    print("          [script_args...]    Extra arguments forwarded to the submitted script")
    print("")
    print("      Note: Parameters are case-insensitive and support single/double dash.")
    print("            Short options (e.g., -c) are forwarded to the script.")
    print("")
    print("  stop [job_id...]")
    print("      Cancel one or more jobs.")
    print("      Arguments:")
    print("          [job_id...]         Start/End job IDs (e.g. 1 2 3-5).")
    print("                              If omitted, interactive mode is used.")
    print("")
    print("      Examples:")
    print("          python My_Lib_HPC.py submit my_script.py")
    print("          python My_Lib_HPC.py submit my_script.py --qos high")
    print("          python My_Lib_HPC.py submit my_script.py --cores 16")
    print("          python My_Lib_HPC.py submit my_script.py -CPU 8  (synonym, case-insensitive)")
    print("          python My_Lib_HPC.py submit my_script.py --mem 20GB")
    print("          python My_Lib_HPC.py submit my_script.py --cores 8 --mem 50GB")
    print("          python My_Lib_HPC.py submit my_script.py -CORES 8 -b 64  (-b forwarded to script)")
    print("          python My_Lib_HPC.py submit my_script.py --parm_for_my_script1 abc --parm_for_my_script2 cde --CPU 8")
    print("          python My_Lib_HPC.py submit    (interactive mode)")
    print("          python My_Lib_HPC.py stop 12345 12346")
    print("          python My_Lib_HPC.py stop      (interactive mode)")
    print("")
    print("  tail <file> [-n N]")
    print("      Monitor a file, printing new content as it is appended (like tail -f).")
    print("      Arguments:")
    print("          <file>    Path to the file to monitor")
    print("          [-n N]    Number of lines to show initially (default: 200)")
    print("      Notes:")
    print("          If the file does not exist yet, waits until it appears.")
    print("          Press Ctrl+C to stop.")
    print("")
    print("      Examples:")
    print("          python My_Lib_HPC.py tail job_output.log")
    print("          python My_Lib_HPC.py tail job_output.log -n 50")
    print("")
    print("For more information, see module docstring or configuration file.")


def main():
    """
    Command-line entry point with subcommand dispatch.

    Usage:
        python My_Lib_HPC.py <command> [arguments...]

    Commands:
        submit [file] [--qos PRIORITY] [--cores N] [--mem SIZE] [script_args...]
            Submit files to the HPC cluster.
            Without [file], enters interactive mode.
            
        stop [job_id...]
            Cancel jobs by ID.
            Without [job_id...], enters interactive mode.
    
    For detailed help, run without arguments or see module docstring.
    """
    if len(sys.argv) < 2:
        print("[My_Lib_HPC] Error: No command specified.")
        print("")
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    args = sys.argv[2:]  # Arguments after the command
    
    # Command dispatch dictionary - add new commands here
    commands = {
        "submit": handle_submit_command,
        "stop": handle_stop_command,
        "tail": handle_tail_command,
    }
    
    if command in commands:
        commands[command](args)
    else:
        print(f"[My_Lib_HPC] Error: Unknown command '{command}'.")
        print("")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
