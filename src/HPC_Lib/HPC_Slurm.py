# -*- coding: utf-8 -*-
"""
My_Lib_HPC_Slurm — SLURM Query Utilities
==========================================

This module contains dataclasses and functions for querying SLURM cluster state:

- :class:`SlurmJobInfo` / :func:`get_job_info` — detailed info about a single job
- :class:`NodeInfo` / :func:`get_node_availability` — node-level resource info
- :class:`QueueEntry` / :func:`get_queue` — current job queue
- :func:`estimate_job_runnability` — heuristic assessment of whether a job can start

All functions invoke SLURM command-line utilities (``scontrol``, ``sacct``,
``sinfo``, ``squeue``) via :mod:`subprocess`.

This module is imported by ``My_Lib_HPC.py`` — it should not be run directly.
"""

__author__ = 'LiYuanhe'

import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass


# ===========================================================================
# Job information query
# ===========================================================================

@dataclass
class SlurmJobInfo:
    """
    Information about a SLURM job, collected from ``scontrol show job`` (for
    running/pending jobs) or ``sacct`` (for completed/historical jobs).

    Fields marked *scontrol-only* are populated only when the job is still in
    the scheduler's queue; fields marked *sacct-only* are populated only after
    the job has finished and its accounting record is available.  All other
    fields are populated from whichever source was used.
    """

    # ── identity ─────────────────────────────────────────────────────────────
    job_id: str = ""
    job_name: str = ""
    user_id: str = ""          # e.g. "li5876(2093117)" (scontrol) or "li5876" (sacct)
    source: str = ""           # "scontrol" or "sacct"

    # ── status ───────────────────────────────────────────────────────────────
    state: str = ""            # RUNNING / PENDING / COMPLETED / FAILED / …
    reason: str = ""           # None / Priority / Resources / … (mostly scontrol)
    exit_code: str = ""        # e.g. "0:0"

    # ── scheduling ───────────────────────────────────────────────────────────
    partition: str = ""
    account: str = ""
    qos: str = ""
    priority: str = ""

    # ── timing ───────────────────────────────────────────────────────────────
    submit_time: str = ""      # ISO-8601
    start_time: str = ""       # ISO-8601
    end_time: str = ""         # ISO-8601
    elapsed: str = ""          # RunTime (scontrol) / Elapsed (sacct)  DD-HH:MM:SS
    time_limit: str = ""       # e.g. "14-00:00:00"

    # ── resources ────────────────────────────────────────────────────────────
    num_nodes: int = 0
    num_cpus: int = 0          # AllocCPUS
    num_tasks: int = 0
    req_mem: str = ""          # requested memory string (e.g. "16448M" or "1917M")
    max_rss: str = ""          # sacct-only  peak resident set size (e.g. "1308K")
    max_vm_size: str = ""      # sacct-only  peak virtual memory    (e.g. "733708K")

    # ── nodes ────────────────────────────────────────────────────────────────
    node_list: str = ""        # scontrol → NodeList;  sacct → MaxRSSNode of batch step

    # ── file paths ───────────────────────────────────────────────────────────
    command: str = ""          # path to the submitted shell script (scontrol-only)
    work_dir: str = ""         # working directory
    stdout: str = ""           # StdOut file (scontrol-only)
    stderr: str = ""           # StdErr file (scontrol-only)


def _parse_scontrol_kv(output: str) -> dict[str, str]:
    """
    Parse ``scontrol show job`` output into a flat key→value dictionary.

    The format is a sequence of ``KEY=VALUE`` tokens separated by whitespace.
    Values end at the next ``WORD=`` boundary so they may not contain spaces
    (which matches the actual scontrol output format).
    Empty/null marker values (``(null)``, ``N/A``, ``None``) are stored as
    empty strings.
    """
    kv: dict[str, str] = {}
    for m in re.finditer(r'(\w+)=(\S+)', output):
        key = m.group(1)
        value = m.group(2)
        if value in ('(null)', 'N/A', 'None'):
            value = ""
        kv[key] = value
    return kv


def _parse_sacct_rows(output: str) -> list[dict[str, str]]:
    """
    Parse ``sacct --parsable2`` output into a list of header→value dict rows.

    All pipe-delimited fields are included verbatim.
    """
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    headers = lines[0].split("|")
    rows: list[dict[str, str]] = []

    for line in lines[1:]:
        values = line.split("|")
        if len(values) < 2:
            continue
        if len(values) < len(headers):
            values += [""] * (len(headers) - len(values))
        elif len(values) > len(headers):
            values = values[:len(headers)]
        rows.append(dict(zip(headers, values)))

    return rows


def get_job_info(job_id: str | int) -> SlurmJobInfo:
    """
    Retrieve information about a SLURM job by its numeric job ID.

    The function first tries ``scontrol show job <job_id>`` (suitable for
    running or recently queued jobs).  If scontrol reports that the job ID is
    invalid (i.e. the job has already left the queue), it falls back to
    ``sacct`` with an explicit ``--format`` field list to retrieve rich
    accounting data for completed/historical jobs.

    Args:
        job_id: Numeric SLURM job ID (``int`` or ``str``).

    Returns:
        A :class:`SlurmJobInfo` dataclass populated with whatever information
        could be retrieved.  Fields that are not available from the chosen
        source are left as empty strings / zeros.

    Example::

        info = get_job_info(34445128)
        print(info.state, info.elapsed, info.node_list)
    """
    job_id = str(job_id)
    info = SlurmJobInfo(job_id=job_id)

    # 1. Try scontrol
    _scontrol_cmd = ["scontrol", "show", "job", job_id]
    print(f"\n>>> {' '.join(_scontrol_cmd)}\n")
    scontrol_result = subprocess.run(
        _scontrol_cmd,
        capture_output=True,
        text=True,
    )

    scontrol_ok = (
        scontrol_result.returncode == 0
        and "Invalid job id" not in scontrol_result.stdout
        and "Invalid job id" not in scontrol_result.stderr
        and scontrol_result.stdout.strip().startswith("JobId=")
    )

    if scontrol_ok:
        kv = _parse_scontrol_kv(scontrol_result.stdout)

        info.source = "scontrol"
        info.job_name   = kv.get("JobName", "")
        info.user_id    = kv.get("UserId", "")
        info.state      = kv.get("JobState", "")
        info.reason     = kv.get("Reason", "")
        info.exit_code  = kv.get("ExitCode", "")
        info.partition  = kv.get("Partition", "")
        info.account    = kv.get("Account", "")
        info.qos        = kv.get("QOS", "")
        info.priority   = kv.get("Priority", "")
        info.submit_time = kv.get("SubmitTime", "")
        info.start_time  = kv.get("StartTime", "")
        info.end_time    = kv.get("EndTime", "")
        info.elapsed     = kv.get("RunTime", "")
        info.time_limit  = kv.get("TimeLimit", "")
        info.node_list   = kv.get("NodeList", "")
        info.command     = kv.get("Command", "")
        info.work_dir    = kv.get("WorkDir", "")
        info.stdout      = kv.get("StdOut", "")
        info.stderr      = kv.get("StdErr", "")

        try:
            info.num_nodes = int(kv.get("NumNodes", 0))
        except (ValueError, TypeError):
            info.num_nodes = 0
        try:
            info.num_cpus = int(kv.get("NumCPUs", 0))
        except (ValueError, TypeError):
            info.num_cpus = 0
        try:
            info.num_tasks = int(kv.get("NumTasks", 0))
        except (ValueError, TypeError):
            info.num_tasks = 0

        req_tres = kv.get("ReqTRES", "")
        mem_match = re.search(r'mem=([^,]+)', req_tres)
        info.req_mem = mem_match.group(1) if mem_match else ""

        return info

    # 2. Fall back to sacct (explicit rich field list)
    sacct_fields = [
        "JobIDRaw",
        "JobID",
        "JobName",
        "User",
        "Partition",
        "State",
        "Reason",
        "ExitCode",
        "Elapsed",
        "Timelimit",
        "Submit",
        "Start",
        "End",
        "AllocCPUS",
        "NTasks",
        "NNodes",
        "ReqMem",
        "NodeList",
        "Account",
        "QOS",
        "Priority",
        "WorkDir",
        "StdOut",
        "StdErr",
        "MaxRSS",
        "MaxRSSNode",
        "MaxVMSize",
    ]
    _sacct_cmd = [
        "sacct",
        "-j", job_id,
        "--format", ",".join(sacct_fields),
        "-P",
    ]
    print(f"\n>>> {' '.join(_sacct_cmd)}\n")
    sacct_result = subprocess.run(
        _sacct_cmd,
        capture_output=True,
        text=True,
    )

    if sacct_result.returncode != 0 or not sacct_result.stdout.strip():
        print(f"[My_Lib_HPC] ERROR: Could not retrieve info for job {job_id}."
              f" scontrol: {scontrol_result.stderr.strip()}"
              f" sacct: {sacct_result.stderr.strip()}")
        return info

    rows = _parse_sacct_rows(sacct_result.stdout)
    if not rows:
        print(f"[My_Lib_HPC] ERROR: Could not parse sacct output for job {job_id}.")
        return info

    def _row_jobid(row: dict[str, str]) -> str:
        return (row.get("JobIDRaw") or row.get("JobID") or "").strip()

    def _base_id(row: dict[str, str]) -> str:
        jid = _row_jobid(row)
        return jid.split(".", 1)[0]

    def _select_main_row(all_rows: list[dict[str, str]]) -> dict[str, str]:
        for r in all_rows:
            if _row_jobid(r) == job_id:
                return r
        for r in all_rows:
            jid = _row_jobid(r)
            if _base_id(r) == job_id and "." not in jid:
                return r
        for r in all_rows:
            if _base_id(r) == job_id:
                return r
        return all_rows[0]

    def _select_batch_row(all_rows: list[dict[str, str]]) -> dict[str, str]:
        for r in all_rows:
            jid = _row_jobid(r)
            if _base_id(r) == job_id and jid.endswith(".batch"):
                return r
        return {}

    main_row = _select_main_row(rows)
    batch_row = _select_batch_row(rows)

    info.source     = "sacct"
    info.job_name   = main_row.get("JobName", "")
    info.partition  = main_row.get("Partition", "")
    info.state      = main_row.get("State", "")
    info.reason     = main_row.get("Reason", "")
    info.exit_code  = main_row.get("ExitCode", "")
    info.elapsed    = main_row.get("Elapsed", "")
    info.req_mem    = main_row.get("ReqMem", "")
    info.submit_time = main_row.get("Submit", "")
    info.start_time  = main_row.get("Start", "")
    info.end_time    = main_row.get("End", "")
    info.account     = main_row.get("Account", "")
    info.qos         = main_row.get("QOS", "")
    info.priority    = main_row.get("Priority", "")
    info.time_limit  = main_row.get("Timelimit", "")
    info.work_dir    = main_row.get("WorkDir", "")
    info.user_id     = main_row.get("User", "")
    info.node_list   = main_row.get("NodeList", "")
    info.stdout      = main_row.get("StdOut", "")
    info.stderr      = main_row.get("StdErr", "")

    try:
        info.num_cpus = int(main_row.get("AllocCPUS", 0) or 0)
    except (ValueError, TypeError):
        info.num_cpus = 0
    try:
        info.num_tasks = int(main_row.get("NTasks", 0) or 0)
    except (ValueError, TypeError):
        info.num_tasks = 0
    try:
        info.num_nodes = int(main_row.get("NNodes", 0) or 0)
    except (ValueError, TypeError):
        info.num_nodes = 0

    if batch_row:
        info.max_rss = batch_row.get("MaxRSS", "") or info.max_rss
        info.max_vm_size = batch_row.get("MaxVMSize", "") or info.max_vm_size
        if not info.node_list:
            info.node_list = batch_row.get("MaxRSSNode", "")

    if not info.max_rss or not info.max_vm_size:
        for r in rows:
            if _base_id(r) != job_id:
                continue
            if not info.max_rss and r.get("MaxRSS", ""):
                info.max_rss = r.get("MaxRSS", "")
            if not info.max_vm_size and r.get("MaxVMSize", ""):
                info.max_vm_size = r.get("MaxVMSize", "")

    return info


def get_job_batch_script(job_id: str | int) -> str:
    """
    Retrieve the submitted batch script text for a job via ``sacct --batch-script``.

    Returns an empty string when the script is unavailable (e.g. accounting does
    not store scripts, job has no batch script, or permission is restricted).
    """
    job_id = str(job_id)
    _batch_script_cmd = ["sacct", "-j", job_id, "--batch-script"]
    print(f"\n>>> {' '.join(_batch_script_cmd)}\n")
    result = subprocess.run(
        _batch_script_cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return ""

    text = (result.stdout or "").strip()
    if not text:
        return ""

    if text.upper() == "NONE":
        return ""

    lines = text.splitlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("#!"):
            return "\n".join(lines[i:]).strip()

    filtered: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            if filtered and filtered[-1] != "":
                filtered.append("")
            continue
        if s.upper() == "NONE":
            continue
        if re.match(r"^[-=]{3,}$", s):
            continue
        if re.search(r"\bjobid\b", s, re.IGNORECASE):
            continue
        filtered.append(line.rstrip())

    return "\n".join(filtered).strip()


# ===========================================================================
# Cluster availability query
# ===========================================================================

@dataclass
class NodeInfo:
    """
    Information about a single cluster node, collected from
    ``sinfo -N -o "%15N %10c %20C %12m %10e %10X %10Y %10Z %25G %10f %10w %20T"``.
    """
    node_name: str = ""          # NODELIST
    total_cpus: int = 0          # CPUS (total on node)
    cpus_alloc: int = 0          # A (allocated) from CPUS(A/I/O/T)
    cpus_idle: int = 0           # I (idle/free) from CPUS(A/I/O/T)
    cpus_other: int = 0          # O (other/unavailable) from CPUS(A/I/O/T)
    memory_mb: int = 0           # MEMORY (total, MB)
    free_memory_mb: int = 0      # FREE_MEM (MB)
    sockets: int = 0             # SOCKETS
    cores_per_socket: int = 0    # CORES (per socket)
    threads_per_core: int = 0    # THREADS
    gres: str = ""               # GRES (generic resources, e.g. "hp_cpu:128")
    avail_features: str = ""     # AVAIL_FEAT
    weight: int = 0              # WEIGHT
    state: str = ""              # STATE (e.g. "idle", "mixed", "allocated", "drain")


def get_node_availability() -> list[NodeInfo]:
    """
    Query SLURM node availability using ``sinfo``.

    Runs::

        sinfo -N -o "%15N %10c %20C %12m %10e %10X %10Y %10Z %25G %10f %10w %20T"

    Returns:
        A list of :class:`NodeInfo` dataclass instances, one per node.
        Returns an empty list if ``sinfo`` fails or produces no output.
    """
    _sinfo_cmd = [
        "sinfo", "-N",
        "-o", "%15N %10c %20C %12m %10e %10X %10Y %10Z %25G %10f %10w %20T",
    ]
    print(f"\n>>> {' '.join(_sinfo_cmd)}\n")
    result = subprocess.run(
        _sinfo_cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[My_Lib_HPC] ERROR: sinfo failed: {result.stderr.strip()}")
        return []

    lines = result.stdout.splitlines()
    if not lines:
        return []

    nodes: list[NodeInfo] = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=11)
        if len(parts) < 12:
            continue

        node = NodeInfo()
        node.node_name = parts[0]

        try:
            node.total_cpus = int(parts[1])
        except ValueError:
            pass

        cpu_aiot = parts[2]
        try:
            a, i, o, t = cpu_aiot.split("/")
            node.cpus_alloc = int(a)
            node.cpus_idle = int(i)
            node.cpus_other = int(o)
        except Exception:
            pass

        try:
            node.memory_mb = int(parts[3])
        except ValueError:
            pass

        try:
            node.free_memory_mb = int(parts[4])
        except ValueError:
            pass

        try:
            node.sockets = int(parts[5])
        except ValueError:
            pass

        try:
            node.cores_per_socket = int(parts[6])
        except ValueError:
            pass

        try:
            node.threads_per_core = int(parts[7])
        except ValueError:
            pass

        node.gres = parts[8]
        node.avail_features = parts[9]

        try:
            node.weight = int(parts[10])
        except ValueError:
            pass

        node.state = parts[11]

        nodes.append(node)

    return nodes


# ===========================================================================
# Queue query
# ===========================================================================

@dataclass
class QueueEntry:
    """
    Information about a single job in the SLURM queue, collected from
    ``squeue -o "%i|%u|%t|%M|%D|%C|%m|%q|%j|%R"``.
    """
    job_id: str = ""             # JOBID
    user: str = ""               # USER
    state: str = ""              # ST  (e.g. "R"=running, "PD"=pending, "CG"=completing)
    elapsed: str = ""            # TIME  (wall-clock elapsed, e.g. "0:05")
    num_nodes: int = 0           # NODES
    num_cpus: int = 0            # CPUS
    min_memory: str = ""         # MIN_MEMORY  (e.g. "2000M")
    qos: str = ""                # QOS
    job_name: str = ""           # NAME  (job name, e.g. "my_script.py")
    reason_or_nodelist: str = "" # NODELIST(REASON)  (node name when running, reason when pending)


def get_queue(user: str | None = None) -> list[QueueEntry]:
    """
    Query the current SLURM job queue using ``squeue``.

    Runs::

        squeue -o "%i|%u|%t|%M|%D|%C|%m|%q|%j|%R"

    Args:
        user: If given, filter results to this username only.
              Defaults to ``None`` (all users).

    Returns:
        A list of :class:`QueueEntry` dataclass instances.
        Returns an empty list if ``squeue`` fails or produces no jobs.
    """
    cmd = ["squeue", "-o", "%i|%u|%t|%M|%D|%C|%m|%q|%j|%R"]
    if user:
        cmd += ["-u", user]

    _sep = "-" * shutil.get_terminal_size(fallback=(220, 40)).columns
    print(f"\n{_sep}\n\n>>> {' '.join(cmd)}\n\n{_sep}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[My_Lib_HPC] ERROR: squeue failed: {result.stderr.strip()}")
        return []

    lines = result.stdout.splitlines()
    if len(lines) < 2:
        return []

    entries: list[QueueEntry] = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split("|")
        if len(parts) < 10:
            continue

        entry = QueueEntry()
        entry.job_id = parts[0]
        entry.user = parts[1]
        entry.state = parts[2]
        entry.elapsed = parts[3]

        try:
            entry.num_nodes = int(parts[4])
        except ValueError:
            pass

        try:
            entry.num_cpus = int(parts[5])
        except ValueError:
            pass

        entry.min_memory = parts[6]
        entry.qos = parts[7]
        entry.job_name = parts[8]
        entry.reason_or_nodelist = parts[9]

        entries.append(entry)

    return entries


def _build_sbatch_header(
    job_name: str,
    output_file: str,
    preset: dict,
    cores: int,
    memory_mb: int,
    hpc_config: dict,
    mail_type: str = "FAIL",
    ntasks_per_node: int = 1,
) -> str:
    """
    Build the #SBATCH comment block for a job script.

    Args:
        job_name:        Job name (--job-name).
        output_file:     Path pattern for stdout/stderr (--output).
        preset:          SLURM preset dict with partition, qos, time_limit.
        cores:           Number of CPU cores to request (--cpus-per-task).
        memory_mb:       Memory in MB to request (--mem).
        hpc_config:      HPC configuration dict with account, node, mail, and home-path values.
        mail_type:       When to send email (e.g. "ALL", "FAIL", "END").
                         This is per-task-type; not read from platform config.
        ntasks_per_node: Number of MPI tasks per node (--ntasks-per-node).
                         Defaults to 1 (single-task / threaded jobs).

    Returns:
        A multi-line string with the #!/bin/bash shebang, all #SBATCH lines,
        and a sourced .bashrc line immediately after the SBATCH block when
        HOME_PATH is available in the HPC configuration.
    """
    effective_mail_type = mail_type
    slurm_account = hpc_config.get("SLURM_ACCOUNT", "")
    nodes = hpc_config.get("NODES", 1)
    mail_user = hpc_config.get("MAIL_USER", "")
    home_path = str(hpc_config.get("HOME_PATH", "") or "").strip()

    lines = [
        "#!/bin/bash",
        f"#SBATCH --output={output_file}",
        f"#SBATCH --partition={preset['partition']}",
        f"#SBATCH --job-name='{job_name}'",
        f"#SBATCH --account={slurm_account}",
        f"#SBATCH --qos={preset['qos']}",
        "#SBATCH --get-user-env",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks-per-node={ntasks_per_node}",
        f"#SBATCH --time={preset['time_limit']}",
        f"#SBATCH --cpus-per-task={cores}",
        f"#SBATCH --mem={memory_mb}",
    ]

    if mail_user:
        lines.append(f"#SBATCH --mail-type={effective_mail_type}")
        lines.append(f"#SBATCH --mail-user={mail_user}")

    # if home_path:
    #     bashrc_path = home_path.rstrip("/\\") + "/.bashrc"
    #     lines.append(f"source {shlex.quote(bashrc_path)}")

    return "\n".join(lines)


# ===========================================================================
# Runnability estimation
# ===========================================================================

def estimate_job_runnability(
    cores: int,
    memory_mb: int,
    qos: str | None = None,
    *,
    _resolve_preset=None,
) -> dict:
    """
    Heuristic assessment of whether a job *could* start on the cluster right
    now, given the requested cores and memory.

    .. rubric:: How SLURM decides which job runs next

    SLURM uses a **multi-factor priority** formula to rank pending jobs.  The
    factors include (weights are cluster-specific):

    1. **Fair-share** — accounts / users who have consumed fewer resources
       recently get higher priority.
    2. **QoS weight** — each QoS level can carry a different priority bonus.
       ``normal`` QoS typically has a higher weight than ``standby``.
    3. **Job age** — how long the job has been pending.
    4. **Partition weight** — partition-level priority.
    5. **TRES factors** — trackable resource (CPU, memory, GPU) adjustments.

    Once SLURM computes a priority for every pending job, it tries to start the
    highest-priority job first by checking if enough resources are available.

    **Backfill scheduling**:  Even if the top-priority job cannot start (e.g.
    waiting for a large block of cores), SLURM's backfill scheduler will look
    further down the queue for *smaller* jobs that can fit without delaying the
    top-priority job's expected start time.

    **Resource reservations**:  SLURM may create a reservation for a
    top-priority job, preventing lower-priority jobs from using those resources.

    **Job preemption**:  If the QoS has ``PreemptMode=cancel`` or ``requeue``
    enabled, a higher-priority job can preempt (cancel/requeue) a lower-priority
    running job.  ``standby`` QoS jobs are typically preemptable.

    **Practical implication**:  Even if there are 8 idle cores on a node, an
    8-core job won't necessarily start immediately — a higher-priority job may
    have a reservation on those cores, or SLURM's scheduling cycle may not have
    run yet.

    Args:
        cores:      Number of CPU cores the job would request.
        memory_mb:  Memory in MB the job would request.
        qos:        Optional QoS preset name.
        _resolve_preset: (internal) preset resolver function injected by
                    My_Lib_HPC to avoid circular imports.

    Returns:
        A dict with keys:

        - ``can_fit`` (bool): Whether at least one node has enough idle
          resources right now.
        - ``fitting_nodes`` (list[str]): Names of nodes that could host the job.
        - ``pending_ahead`` (int): Number of pending jobs in the queue.
        - ``assessment`` (str): Human-readable summary.

    Example::

        {
            'can_fit': True,
            'fitting_nodes': ['a008', 'a009'],
            'pending_ahead': 12,
            'assessment': 'Resources available on 2 node(s), but 12 jobs '
                          'are pending ahead — start time depends on SLURM '
                          'priority and backfill.'
        }
    """
    nodes = get_node_availability()
    queue = get_queue()

    fitting_nodes: list[str] = []
    for n in nodes:
        if n.cpus_idle >= cores and n.free_memory_mb >= memory_mb:
            state_lower = n.state.lower()
            if "drain" in state_lower or "down" in state_lower:
                continue
            fitting_nodes.append(n.node_name)

    can_fit = len(fitting_nodes) > 0

    pending_ahead = sum(1 for e in queue if e.state == "PD")

    if can_fit and pending_ahead == 0:
        assessment = (
            f"Resources available on {len(fitting_nodes)} node(s) and no "
            f"pending jobs — job will very likely start immediately."
        )
    elif can_fit:
        assessment = (
            f"Resources available on {len(fitting_nodes)} node(s), but "
            f"{pending_ahead} job(s) are pending — start time depends on "
            f"SLURM priority, fair-share, and backfill scheduling."
        )
    else:
        assessment = (
            f"No single node currently has {cores} idle cores and "
            f"{memory_mb} MB free memory.  The job will be queued until "
            f"resources become available."
        )

    return {
        "can_fit": can_fit,
        "fitting_nodes": fitting_nodes,
        "pending_ahead": pending_ahead,
        "assessment": assessment,
    }


# ===========================================================================
# Job history query
# ===========================================================================

@dataclass
class HistoryEntry:
    """
    A completed/failed/cancelled SLURM job from ``sacct``.
    """
    job_id: str = ""
    job_name: str = ""
    state: str = ""           # COMPLETED / FAILED / CANCELLED / TIMEOUT / ...
    elapsed: str = ""         # e.g. "01:23:45"
    start_time: str = ""      # ISO-8601
    end_time: str = ""        # ISO-8601
    num_cpus: int = 0
    req_mem: str = ""         # e.g. "76000M"
    qos: str = ""
    node_list: str = ""
    exit_code: str = ""       # e.g. "0:0"


def get_recent_jobs(
    user: str | None = None,
    since: str = "now-48hours",
    count: int = 200,
) -> list[HistoryEntry]:
    """
    Query completed/finished jobs from ``sacct``.

    Args:
        user:  Filter to this user.  If None, queries all users.
        since: ``--starttime`` value for sacct (e.g. ``"now-48hours"``).
        count: Maximum number of entries to return.

    Returns:
        List of :class:`HistoryEntry`, sorted by end_time descending
        (most recent first), limited to *count* entries.
    """
    # NOTE: When --state is specified with --starttime but NOT --endtime,
    # sacct defaults endtime to the same value as starttime (zero-width window).
    # We must always pass --endtime now explicitly to get the full time range.
    cmd = [
        "sacct",
        "--format", "JobIDRaw,JobName,State,Elapsed,Start,End,AllocCPUS,ReqMem,QOS,NodeList,ExitCode",
        "-P",
        "--starttime", since,
        "--endtime", "now",
        "--state", "CD,F,TO,CA,NF,PR,OOM,DL,RQ",
    ]
    if user:
        cmd += ["-u", user]

    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[My_Lib_HPC] ERROR: sacct failed: {result.stderr.strip()}")
        return []

    rows = _parse_sacct_rows(result.stdout)
    if not rows:
        return []

    entries: list[HistoryEntry] = []
    seen_ids: set[str] = set()
    for row in rows:
        jid_raw = (row.get("JobIDRaw") or "").strip()
        # Skip sub-steps like "12345.batch", "12345.extern"
        if "." in jid_raw:
            continue
        if jid_raw in seen_ids:
            continue
        seen_ids.add(jid_raw)

        e = HistoryEntry()
        e.job_id = jid_raw
        e.job_name = row.get("JobName", "")
        e.state = row.get("State", "")
        e.elapsed = row.get("Elapsed", "")
        e.start_time = row.get("Start", "")
        e.end_time = row.get("End", "")
        e.qos = row.get("QOS", "")
        e.req_mem = row.get("ReqMem", "")
        e.node_list = row.get("NodeList", "")
        e.exit_code = row.get("ExitCode", "")
        try:
            e.num_cpus = int(row.get("AllocCPUS", 0) or 0)
        except (ValueError, TypeError):
            pass

        entries.append(e)

    # Sort by end_time descending (most recent first)
    def _end_sort_key(entry: HistoryEntry) -> str:
        return entry.end_time or "0"

    entries.sort(key=_end_sort_key, reverse=True)
    return entries[:count]
