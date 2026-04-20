# -*- coding: utf-8 -*-
"""
My_Lib_HPC_Scheduler — Scheduler System
=========================================

A daemon-style job scheduler that manages deferred SLURM job submissions with
priority ordering, dependency chains, and resource-aware scheduling.

Architecture::

    HPC_Scheduler/                           ← inside HOME_PATH
    ├── .scheduler_state.json                ← daemon's SLURM job ID, status
    ├── scheduler.log                        ← timestamped log (max 20 MB)
    ├── 20260306-143025-123.json             ← pending job configs (timestamp IDs)
    ├── 20260306-143030-456.json
    ├── Submitted/                           ← submitted/running job configs
    │   ├── 20260306-142500-789.json
    │   └── ...
    └── ...

Workflow:

1. ``python My_Lib_HPC.py schedule <file> [opts]`` writes a JSON job config
   and auto-starts the scheduler daemon if not running.

2. ``python My_Lib_HPC.py handler [start|stop|status|restart]`` manages the
   daemon, which runs as a SLURM job (high QoS, 1 CPU, max time).

3. The daemon loop:
   - Scans ``HPC_Scheduler/`` and ``Submitted/`` for jobs.
   - Updates submitted/running job statuses via squeue/sacct.
   - Respects concurrency and core-budget limits per QoS.
   - Submits highest-priority jobs first (even in congested mode).
   - Checks and honours dependency chains.
   - Moves job config files to ``Submitted/`` upon SLURM submission.
   - Deletes job config files only when jobs complete successfully.
   - Self-renews before its SLURM time limit expires.
    - Auto-terminates after 24h with no active jobs remaining.

Schedule IDs:
    Jobs use timestamp-based IDs in the format
    ``yyyymmdd-hhmmss-{ms}`` (e.g. ``20260306-143025-123``).  If two jobs
    are created in the same millisecond a dedup suffix ``_1``, ``_2``, …
    is appended.  IDs are unique and never reused.

Dependencies:
    ``--after ID[,ID,...]``  — wait for listed schedule IDs to complete.
    ``--after-any ID[,ID,...]`` — same, but run even if dependencies fail.
    Dependencies are stored in the JSON config and checked each loop cycle.

This module is imported by ``My_Lib_HPC.py`` — it should not be run directly.
"""

__author__ = 'LiYuanhe'

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from HPC_Lib.HPC_Slurm import QueueEntry, get_queue, get_job_info

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEDULER_POLL_INTERVAL_NORMAL = 30   # seconds — quiet mode
_SCHEDULER_POLL_INTERVAL_FAST = 5      # seconds — after a new schedule event
_SCHEDULER_FAST_POLL_DURATION = 180    # seconds — stay fast for 3 min after last new job
_SCHEDULER_RENEWAL_THRESHOLD = 2 * 3600  # renew when < 2 hours remain
_SCHEDULER_JOB_NAME = "HPC_Scheduler"
_SCHEDULER_LOG_MAX_BYTES = 20 * 1024 * 1024  # 20 MB
_SCHEDULER_HEARTBEAT_INTERVAL = 10     # seconds — write heartbeat this often
_SCHEDULER_HEARTBEAT_STALE = 60        # seconds — heartbeat older than this ⇒ stuck
_SCHEDULER_STATUS_RECONCILE_BATCH = 50  # max departed jobs to resolve via sacct per loop
_SCHEDULER_IDLE_EXIT_THRESHOLD = 24 * 3600  # exit after 24h of continuous inactivity
_SCHEDULER_SUBMIT_CHECK_DEBOUNCE_DELAY = 2.0  # seconds — non-blocking delay before touching squeue/sbatch


# ===========================================================================
# ScheduledJob dataclass
# ===========================================================================

@dataclass
class ScheduledJob:
    """
    A job managed by the scheduler, stored as a JSON file in HPC_Scheduler/.

    Status transitions::

        pending  →  submitted  →  running  →  completed
                                            →  failed
                 →  cancelled

    Fields:
        schedule_id:    Short sequential integer ID (e.g. ``"001"``).
        filepath:       Absolute path to the file to submit.
        priority_level: Integer priority (higher = submit first). Default 0.
        is_vip:         Whether this is a VIP (infinite priority) job.
        qos:            SLURM QoS preset name (e.g. "high", "normal"), or None
                        to use DEFAULT_PRESET.
        cores:          Requested CPU cores, or None for auto.
        memory_mb:      Requested memory in MB, or None for auto.
        script_args:    Extra arguments forwarded to the submitted script.
        status:         Current status string.
        slurm_job_id:   SLURM job ID once submitted (empty string if not yet).
        created_at:     ISO-8601 timestamp of when the job was scheduled.
        submitted_at:   ISO-8601 timestamp of SLURM submission.
        completed_at:   ISO-8601 timestamp of completion/failure.
        error_message:  Error description if the job failed.
        config_file:    Absolute path to the JSON config file on disk.
        depends_on:     List of schedule IDs that must complete first.
        depend_mode:    ``"success"`` (default) — run only if ALL deps succeeded.
                        ``"any"`` — run even if some deps failed.
        email:          If True, send email notification when the job ends.
    """
    schedule_id: str = ""
    filepath: str = ""
    priority_level: int = 0
    is_vip: bool = False
    qos: str | None = None
    cores: int | None = None
    memory_mb: int | None = None
    script_args: list[str] = field(default_factory=list)
    status: str = "pending"       # pending | submitted | running | completed | failed | cancelled
    slurm_job_id: str = ""
    created_at: str = ""
    submitted_at: str = ""
    completed_at: str = ""
    error_message: str = ""
    config_file: str = ""
    depends_on: list[str] = field(default_factory=list)
    depend_mode: str = "success"  # "success" or "any"
    email: bool = False
    sh_script_path: str = ""  # path to the generated .sh script (set after submission)

    @property
    def effective_priority(self) -> int:
        """Return priority for sorting (VIP → sys.maxsize)."""
        return sys.maxsize if self.is_vip else self.priority_level

    def to_dict(self) -> dict:
        return {
            "schedule_id": self.schedule_id,
            "filepath": self.filepath,
            "priority_level": self.priority_level,
            "is_vip": self.is_vip,
            "qos": self.qos,
            "cores": self.cores,
            "memory_mb": self.memory_mb,
            "script_args": self.script_args,
            "status": self.status,
            "slurm_job_id": self.slurm_job_id,
            "created_at": self.created_at,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "depends_on": self.depends_on,
            "depend_mode": self.depend_mode,
            "email": self.email,
            "sh_script_path": self.sh_script_path,
        }

    @classmethod
    def from_dict(cls, d: dict, config_file: str = "") -> "ScheduledJob":
        return cls(
            schedule_id=d.get("schedule_id", ""),
            filepath=d.get("filepath", ""),
            priority_level=d.get("priority_level", 0),
            is_vip=d.get("is_vip", False),
            qos=d.get("qos"),
            cores=d.get("cores"),
            memory_mb=d.get("memory_mb"),
            script_args=d.get("script_args", []),
            status=d.get("status", "pending"),
            slurm_job_id=d.get("slurm_job_id", ""),
            created_at=d.get("created_at", ""),
            submitted_at=d.get("submitted_at", ""),
            completed_at=d.get("completed_at", ""),
            error_message=d.get("error_message", ""),
            config_file=config_file,
            depends_on=d.get("depends_on", []),
            depend_mode=d.get("depend_mode", "success"),
            email=d.get("email", False),
            sh_script_path=d.get("sh_script_path", ""),
        )


# ===========================================================================
# Scheduler directory & file I/O
# ===========================================================================

def _get_scheduler_dir(hpc_config: dict) -> str:
    """
    Return the absolute path to the HPC_Scheduler directory, creating it
    if necessary.  Located at ``{HOME_PATH}/HPC_Scheduler/``.
    """
    home = hpc_config.get("HOME_PATH", os.path.expanduser("~"))
    d = os.path.join(home, "HPC_Scheduler")
    os.makedirs(d, exist_ok=True)
    return d


def _scheduler_state_file(hpc_config: dict) -> str:
    """Path to ``.scheduler_state.json`` inside the scheduler directory."""
    home = hpc_config.get("HOME_PATH", os.path.expanduser("~"))
    return os.path.join(home, ".scheduler_state.json")


def _scheduler_heartbeat_file(hpc_config: dict) -> str:
    """Path to ``.scheduler_heartbeat`` inside the HOME directory (not HPC_Scheduler/)."""
    home = hpc_config.get("HOME_PATH", os.path.expanduser("~"))
    return os.path.join(home, ".scheduler_heartbeat")


def _write_heartbeat(hpc_config: dict):
    """Write the current UTC timestamp to the heartbeat file."""
    path = _scheduler_heartbeat_file(hpc_config)
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(datetime.now().isoformat())
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
    except OSError:
        pass


def _read_heartbeat_age(hpc_config: dict) -> float | None:
    """Return the age (in seconds) of the last heartbeat, or None if no file."""
    path = _scheduler_heartbeat_file(hpc_config)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            ts_str = f.read().strip()
        ts = datetime.fromisoformat(ts_str)
        return (datetime.now() - ts).total_seconds()
    except (ValueError, OSError):
        return None


def _delete_heartbeat(hpc_config: dict):
    """Remove the heartbeat file so future schedule commands restart immediately."""
    path = _scheduler_heartbeat_file(hpc_config)
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _scheduler_launch_file(hpc_config: dict) -> str:
    """Path to ``.scheduler_last_launch`` inside the scheduler directory."""
    home = hpc_config.get("HOME_PATH", os.path.expanduser("~"))
    return os.path.join(home, ".scheduler_last_launch")


def _scheduler_submit_check_file(hpc_config: dict) -> str:
    """Path to the debounced scheduler-submit request file inside HOME_PATH.

    ``schedule`` commands update this file instead of touching SLURM
    immediately.  Detached helper processes sleep for a short delay, then only
    the helper whose token still matches this file is allowed to query squeue
    or submit/restart the scheduler daemon.
    """
    home = hpc_config.get("HOME_PATH", os.path.expanduser("~"))
    return os.path.join(home, ".scheduler_submit_check.json")


def _write_scheduler_submit_check_request(
    token: str,
    delay_seconds: float,
    hpc_config: dict,
):
    """Persist the newest deferred scheduler-check request atomically.

    The most recent ``schedule`` invocation overwrites any previous request.
    Older detached helpers will notice the token mismatch after their sleep and
    exit without querying SLURM.
    """
    path = _scheduler_submit_check_file(hpc_config)
    payload = {
        "token": token,
        "delay_seconds": float(delay_seconds),
        "requested_at": datetime.now().isoformat(),
        "run_after_epoch": time.time() + max(0.0, float(delay_seconds)),
    }
    target_dir = os.path.dirname(path)
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _read_scheduler_submit_check_request(hpc_config: dict) -> dict | None:
    """Read the newest deferred scheduler-check request, if any."""
    path = _scheduler_submit_check_file(hpc_config)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _is_scheduler_submit_check_token_current(token: str, hpc_config: dict) -> bool:
    """Return True only if *token* is still the newest deferred request token."""
    request = _read_scheduler_submit_check_request(hpc_config)
    if not request:
        return False
    return request.get("token") == token


def _write_scheduler_launch_time(hpc_config: dict):
    """Record that a scheduler job was just submitted.

    Used by ``handle_schedule_command`` to avoid a restart storm: many
    concurrent ``schedule`` CLI invocations can detect a stale heartbeat
    and each try to cancel + restart the scheduler.  By writing a launch
    timestamp, subsequent invocations within 120 s will skip the restart.
    """
    path = _scheduler_launch_file(hpc_config)
    try:
        with open(path, "w") as f:
            f.write(datetime.now().isoformat())
    except OSError:
        pass


def _read_scheduler_launch_age(hpc_config: dict) -> float | None:
    """Return seconds since the last scheduler launch, or None if unknown."""
    path = _scheduler_launch_file(hpc_config)
    if not os.path.isfile(path):
        return None
    try:
        # Use file mtime — more robust than parsing the content
        return time.time() - os.path.getmtime(path)
    except OSError:
        return None


def _scheduler_log_file(hpc_config: dict) -> str:
    """Path to ``scheduler.log`` inside the HOME directory (not HPC_Scheduler/)."""
    home = hpc_config.get("HOME_PATH", os.path.expanduser("~"))
    return os.path.join(home, "scheduler.log")


def _rotate_log_if_needed(hpc_config: dict):
    """
    If ``scheduler.log`` exceeds 20 MB, rotate it:
    rename current → ``scheduler.log.1`` (overwriting any existing backup),
    then start a fresh log.
    """
    log_path = _scheduler_log_file(hpc_config)
    try:
        if os.path.isfile(log_path) and os.path.getsize(log_path) > _SCHEDULER_LOG_MAX_BYTES:
            backup = log_path + ".1"
            if os.path.isfile(backup):
                os.remove(backup)
            os.rename(log_path, backup)
    except OSError:
        pass


def _scheduler_log(msg: str, hpc_config: dict):
    """Append a timestamped line to the scheduler log and also print it."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        _rotate_log_if_needed(hpc_config)
        with open(_scheduler_log_file(hpc_config), "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _read_scheduler_state(hpc_config: dict) -> dict | None:
    """
    Read the scheduler state file.

    Returns:
        A dict with keys ``slurm_job_id``, ``started_at``,
        ``renewal_submitted``, ``renewal_job_id``; or ``None`` if the file
        does not exist or cannot be parsed.
    """
    path = _scheduler_state_file(hpc_config)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_scheduler_state(state: dict, hpc_config: dict):
    """Write the scheduler state dict to ``.scheduler_state.json``."""
    path = _scheduler_state_file(hpc_config)
    target_dir = os.path.dirname(path)
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # atomic on POSIX
    except BaseException:
        os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Timestamp-based schedule ID generation
# ---------------------------------------------------------------------------

def _next_schedule_id(hpc_config: dict) -> str:
    """
    Generate a timestamp-based schedule ID.

    Format: ``yyyymmdd-hhmmss`` (e.g. ``20260306-143025``).
    If the ID already exists (same second), a dedup suffix ``_01``,
    ``_02``, … is appended (e.g. ``20260306-143025_01``).

    The config file is atomically claimed via ``O_CREAT | O_EXCL`` to
    guarantee uniqueness even when many processes run in the same second.
    """
    now = datetime.now()
    base_id = now.strftime("%Y%m%d-%H%M%S")

    sdir = _get_scheduler_dir(hpc_config)
    os.makedirs(sdir, exist_ok=True)

    candidates = [base_id] + [f"{base_id}_{s:02d}" for s in range(1, 100)]
    for candidate in candidates:
        path = os.path.join(sdir, f"{candidate}.json")
        try:
            # O_CREAT | O_EXCL: atomic create, fails if file already exists
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return candidate
        except FileExistsError:
            continue

    # Extremely unlikely fallback: add PID + random to guarantee uniqueness
    import random
    fallback_id = f"{base_id}_{os.getpid()}_{random.randint(0, 9999):04d}"
    path = os.path.join(sdir, f"{fallback_id}.json")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    return fallback_id


# ---------------------------------------------------------------------------
# Submitted directory helpers
# ---------------------------------------------------------------------------

def _get_submitted_dir(hpc_config: dict) -> str:
    """
    Return the absolute path to the ``Submitted/`` subdirectory inside the
    scheduler directory, creating it if necessary.
    """
    d = os.path.join(_get_scheduler_dir(hpc_config), "Submitted")
    os.makedirs(d, exist_ok=True)
    return d


def _move_job_to_submitted(job: ScheduledJob, hpc_config: dict):
    """
    Move a job's JSON config file from the scheduler root directory into
    the ``Submitted/`` subdirectory.  Updates ``job.config_file`` in place.
    """
    if not job.config_file or not os.path.isfile(job.config_file):
        return
    submitted_dir = _get_submitted_dir(hpc_config)
    dest = os.path.join(submitted_dir, os.path.basename(job.config_file))
    try:
        os.replace(job.config_file, dest)
        job.config_file = dest
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Job file I/O
# ---------------------------------------------------------------------------

def _save_scheduled_job(job: ScheduledJob, hpc_config: dict):
    """
    Write (or update) the scheduled-job JSON config file.

    If the file was already pre-claimed by ``_next_schedule_id``, it is
    overwritten in place.  Otherwise a new file is created atomically.
    """
    if not job.config_file:
        job.config_file = os.path.join(
            _get_scheduler_dir(hpc_config), f"{job.schedule_id}.json"
        )
    # Ensure the target directory exists (guards against race with cleanup)
    target_dir = os.path.dirname(job.config_file)
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(job.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, job.config_file)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_scheduled_job(filepath: str) -> ScheduledJob | None:
    """Load a single ScheduledJob from a JSON file. Returns None on error."""
    try:
        with open(filepath) as f:
            d = json.load(f)
        return ScheduledJob.from_dict(d, config_file=filepath)
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def _load_all_scheduled_jobs(
    hpc_config: dict,
    _keep_heartbeat_alive: bool = False,
) -> list[ScheduledJob]:
    """
    Load every ``*.json`` file in the scheduler directory **and** its
    ``Submitted/`` subdirectory (excluding dotfiles).

    Args:
        hpc_config:             Configuration dict.
        _keep_heartbeat_alive:  If True, write heartbeat every 200 files
                                to prevent stale-heartbeat detection when
                                the directory contains many thousands of
                                job configs.  Only the scheduler daemon
                                should set this to True.

    Returns:
        List of :class:`ScheduledJob` instances, sorted by effective priority
        (highest first), then by creation time (oldest first).
    """
    sdir = _get_scheduler_dir(hpc_config)
    submitted_dir = _get_submitted_dir(hpc_config)
    jobs: list[ScheduledJob] = []
    _files_read = 0
    for d in (sdir, submitted_dir):
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.startswith(".") or not fname.endswith(".json"):
                continue
            full = os.path.join(d, fname)
            job = _load_scheduled_job(full)
            if job:
                jobs.append(job)
            _files_read += 1
            # Keep heartbeat fresh during very large directory scans
            if _keep_heartbeat_alive and _files_read % 200 == 0:
                _write_heartbeat(hpc_config)
    # Sort: highest priority first, ties broken by earliest creation
    jobs.sort(key=lambda j: (-j.effective_priority, j.created_at))
    return jobs


def _delete_scheduled_job_file(job: ScheduledJob):
    """Remove the JSON config file for a completed job."""
    if job.config_file and os.path.isfile(job.config_file):
        try:
            os.remove(job.config_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Dependency checking
# ---------------------------------------------------------------------------

def _are_dependencies_met(
    job: ScheduledJob,
    all_jobs: list[ScheduledJob],
) -> bool:
    """
    Check whether all dependencies of *job* have been satisfied.

    - ``depend_mode == "success"``: all deps must have status ``"completed"``.
    - ``depend_mode == "any"``:     all deps must have status ``"completed"``
      or ``"failed"`` (i.e. finished, regardless of outcome).

    If a dependency ID is not found in *all_jobs* (file was deleted because
    it already ran and completed), it is treated as satisfied.

    Returns True if the job is ready to be submitted.
    """
    if not job.depends_on:
        return True

    deps_by_id: dict[str, ScheduledJob] = {j.schedule_id: j for j in all_jobs}

    for dep_id in job.depends_on:
        dep = deps_by_id.get(dep_id)
        if dep is None:
            # Dependency file gone → already completed/cleaned up → satisfied
            continue

        if job.depend_mode == "any":
            if dep.status not in ("completed", "failed"):
                return False
        else:
            # "success" mode
            if dep.status == "failed":
                # Dependency failed → this job cannot run
                return False
            if dep.status != "completed":
                return False

    return True


def _mark_dependency_blocked(
    job: ScheduledJob,
    all_jobs: list[ScheduledJob],
    hpc_config: dict,
):
    """
    If a job has ``depend_mode == "success"`` and a dependency has failed,
    mark the job as failed too.
    """
    if not job.depends_on or job.depend_mode != "success":
        return
    deps_by_id = {j.schedule_id: j for j in all_jobs}
    for dep_id in job.depends_on:
        dep = deps_by_id.get(dep_id)
        if dep and dep.status == "failed":
            job.status = "failed"
            job.error_message = f"Dependency {dep_id} failed"
            job.completed_at = datetime.now().isoformat()
            _save_scheduled_job(job, hpc_config)
            _scheduler_log(
                f"  Job {job.schedule_id} BLOCKED: dependency {dep_id} failed",
                hpc_config,
            )
            return


# ---------------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------------

# In-memory squeue result cache — avoids duplicate squeue calls within the
# same process (e.g. main-loop fetch + _check_and_self_renew in the same tick).
_SQUEUE_RESULT_CACHE: dict = {"time": 0.0, "result": [], "user": ""}
_SQUEUE_RESULT_CACHE_TTL = 10  # seconds


def _get_user_slurm_jobs(hpc_config: dict) -> list[QueueEntry]:
    """Get current user's SLURM jobs via squeue.

    Results are cached in-memory for 10 seconds so that multiple callers
    within the same loop tick (or rapid CLI calls in the same process)
    don't each fire a separate ``squeue`` invocation.
    """
    global _SQUEUE_RESULT_CACHE
    user = hpc_config.get("USER_NAME", "")
    now = time.time()
    if (
        now - _SQUEUE_RESULT_CACHE["time"] < _SQUEUE_RESULT_CACHE_TTL
        and _SQUEUE_RESULT_CACHE["user"] == user
    ):
        return _SQUEUE_RESULT_CACHE["result"]

    _squeue_desc = f"squeue -o %i|%u|%t|%M|%D|%C|%m|%q|%j|%R" + (f" -u {user}" if user else "")
    _scheduler_log(f"Running: {_squeue_desc}", hpc_config)
    if not user:
        result = get_queue()
    else:
        result = get_queue(user=user)
    _SQUEUE_RESULT_CACHE = {"time": now, "result": result, "user": user}
    return result


def _update_submitted_job_statuses(
    jobs: list[ScheduledJob],
    hpc_config: dict,
    queue: list["QueueEntry"] | None = None,
    max_terminal_checks: int | None = None,
) -> dict[str, int]:
    """
    For every job with status ``submitted`` or ``running``, check its current
    SLURM state and update the status + JSON file accordingly.

    Uses ``squeue`` for fast batch lookup, then falls back to ``get_job_info``
    for jobs that have left the queue.

    Jobs that complete successfully have their config file **deleted**.
    Failed jobs' config files are kept in ``Submitted/`` for inspection.

    Args:
        queue: Optional pre-fetched squeue result.  When provided the squeue
               call is skipped (avoids re-querying within the same loop tick).
        max_terminal_checks: Maximum number of jobs that have already left
               ``squeue`` to reconcile via ``get_job_info`` in this loop.

    Returns:
        Dict with keys ``resolved_missing`` and ``deferred_missing``.
    """
    if queue is None:
        queue = _get_user_slurm_jobs(hpc_config)
    queue_by_id: dict[str, QueueEntry] = {e.job_id: e for e in queue}
    stale_jobs: list[ScheduledJob] = []
    resolved_missing = 0

    for job in jobs:
        if job.status not in ("submitted", "running"):
            continue
        if not job.slurm_job_id:
            job.status = "failed"
            job.completed_at = datetime.now().isoformat()
            job.error_message = "Missing SLURM job ID after submission"
            _save_scheduled_job(job, hpc_config)
            _scheduler_log(
                f"  Job {job.schedule_id} marked FAILED: missing SLURM job ID",
                hpc_config,
            )
            continue

        qe = queue_by_id.get(job.slurm_job_id)
        if qe:
            # Still in queue
            if qe.state == "R" and job.status != "running":
                job.status = "running"
                _scheduler_log(
                    f"  Job {job.schedule_id} ({os.path.basename(job.filepath)}) "
                    f"is now RUNNING (SLURM {job.slurm_job_id})",
                    hpc_config,
                )
                _save_scheduled_job(job, hpc_config)
            elif qe.state == "PD" and job.status == "running":
                # Edge case: went back to pending (requeue?)
                job.status = "submitted"
                _save_scheduled_job(job, hpc_config)
        else:
            stale_jobs.append(job)

    deferred_missing = 0
    for idx, job in enumerate(stale_jobs):
        if max_terminal_checks is not None and resolved_missing >= max_terminal_checks:
            deferred_missing = len(stale_jobs) - idx
            break

        # Not in squeue → finished or failed; check sacct
        _scheduler_log(f"Running: scontrol show job {job.slurm_job_id}  (falls back to sacct if gone)", hpc_config)
        info = get_job_info(job.slurm_job_id)
        # Write heartbeat after each scontrol call so we don't appear stuck
        # when processing many finished jobs sequentially.
        _write_heartbeat(hpc_config)
        resolved_missing += 1
        state_upper = info.state.upper()
        now_iso = datetime.now().isoformat()
        if "COMPLETED" in state_upper:
            job.status = "completed"
            job.completed_at = now_iso
            _scheduler_log(
                f"  Job {job.schedule_id} ({os.path.basename(job.filepath)}) "
                f"COMPLETED (SLURM {job.slurm_job_id})",
                hpc_config,
            )
            # Delete config file on success
            _delete_scheduled_job_file(job)
        elif "FAIL" in state_upper or "TIMEOUT" in state_upper or "CANCEL" in state_upper:
            job.status = "failed"
            job.completed_at = now_iso
            job.error_message = f"SLURM state: {info.state}"
            _scheduler_log(
                f"  Job {job.schedule_id} ({os.path.basename(job.filepath)}) "
                f"FAILED/CANCELLED: {info.state} (SLURM {job.slurm_job_id})",
                hpc_config,
            )
            # Keep config file for failed jobs (for inspection)
            _save_scheduled_job(job, hpc_config)
        else:
            job.status = "completed"
            job.completed_at = now_iso
            _scheduler_log(
                f"  Job {job.schedule_id} unknown final state: {info.state}",
                hpc_config,
            )
            _delete_scheduled_job_file(job)

    return {
        "resolved_missing": resolved_missing,
        "deferred_missing": deferred_missing,
    }


def _effective_scheduler_job_state(
    job: ScheduledJob,
    queue_by_id: dict[str, QueueEntry] | None = None,
) -> str | None:
    """Return the effective active scheduler state for accounting purposes.

    When *queue_by_id* is provided, jobs that have already left ``squeue`` are
    treated as inactive so they do not block fresh submissions while their
    terminal state is being reconciled lazily via ``sacct``.
    """
    if job.status not in ("submitted", "running"):
        return None
    if not job.slurm_job_id:
        return None
    if queue_by_id is None:
        return job.status

    qe = queue_by_id.get(job.slurm_job_id)
    if qe is None:
        return None
    if qe.state in ("R", "CG"):
        return "running"
    if qe.state in ("PD", "CF"):
        return "submitted"
    return "submitted"


# ---------------------------------------------------------------------------
# Resource accounting
# ---------------------------------------------------------------------------

def _count_cores_in_use_by_preset(
    queue: list[QueueEntry],
    preset_name: str,
    hpc_config: dict,
) -> int:
    """
    Count total CPU cores currently used (running + pending) in SLURM by the
    current user.  This is an approximation: counts ALL user's cores.
    """
    total = 0
    for e in queue:
        total += e.num_cpus
    return total


def _count_submitted_and_running(
    jobs: list[ScheduledJob],
    queue: list[QueueEntry] | None = None,
) -> tuple[int, int]:
    """Count jobs that are submitted-to-SLURM (pending in SLURM) and running."""
    queue_by_id = {e.job_id: e for e in queue} if queue is not None else None
    submitted = 0
    running = 0
    for j in jobs:
        effective_state = _effective_scheduler_job_state(j, queue_by_id)
        if effective_state == "submitted":
            submitted += 1
        elif effective_state == "running":
            submitted += 1
            running += 1
    return submitted, running


def _count_total_queued_and_running(
    queue: list[QueueEntry],
) -> tuple[int, int]:
    """Count all queued jobs and all currently running/completing jobs."""
    total = len(queue)
    running = sum(1 for e in queue if (e.state or "").upper() in ("R", "CG"))
    return total, running


def _can_submit_more(
    jobs: list[ScheduledJob],
    hpc_config: dict,
    queue: list[QueueEntry] | None = None,
) -> bool:
    """Check global constraints: CONCURRENT_SCHEDULED/RUNNING_MISSION_COUNT."""
    if queue is not None:
        submitted, running = _count_total_queued_and_running(queue)
    else:
        submitted, running = _count_submitted_and_running(jobs, queue=queue)

    max_scheduled = int(hpc_config.get("CONCURRENT_SCHEDULED_MISSION_COUNT", 1e10))
    max_running = int(hpc_config.get("CONCURRENT_RUNNING_MISSION_COUNT", 1e10))

    if submitted >= max_scheduled:
        return False
    if running >= max_running:
        return False
    return True


def _cores_budget_remaining(
    jobs: list[ScheduledJob],
    preset_name: str,
    hpc_config: dict,
    _compute_resources=None,
    queue: list[QueueEntry] | None = None,
) -> float:
    """How many more cores can be submitted for a given preset?"""
    presets = hpc_config.get("SLURM_PRESETS", {})
    preset = presets.get(preset_name, {})
    limit = float(preset.get("total_cores_available", 1e10))
    queue_by_id = {e.job_id: e for e in queue} if queue is not None else None

    used = 0
    for j in jobs:
        if _effective_scheduler_job_state(j, queue_by_id) is not None:
            job_preset = (j.qos or hpc_config.get("DEFAULT_PRESET", "normal"))
            if job_preset == preset_name:
                if j.cores is not None:
                    used += j.cores
                elif _compute_resources:
                    c, _ = _compute_resources(preset)
                    used += c

    return limit - used


# ---------------------------------------------------------------------------
# Submission decision
# ---------------------------------------------------------------------------

def _decide_next_submissions(
    jobs: list[ScheduledJob],
    queue: list[QueueEntry],
    hpc_config: dict,
    _compute_resources=None,
) -> tuple[list[ScheduledJob], list[str]]:
    """
    Decide which pending scheduled jobs should be submitted next.

    **Both congested and non-congested modes submit in priority order.**

    For congested queues: submits up to the concurrency limit, in priority
    order.  Does NOT cancel existing SLURM jobs.

    For non-congested queues: keeps at most 1 pending in SLURM.  May cancel
    a lower-priority pending SLURM job to make way for a higher-priority one.

    Returns:
        (to_submit, to_cancel)
    """
    congested = bool(hpc_config.get("CONGESTED_QUEUE", True))
    presets = hpc_config.get("SLURM_PRESETS", {})
    max_scheduled = int(hpc_config.get("CONCURRENT_SCHEDULED_MISSION_COUNT", 1e10))
    max_running = int(hpc_config.get("CONCURRENT_RUNNING_MISSION_COUNT", 1e10))

    def _get_job_requested_cores(job: ScheduledJob) -> int:
        """Return the effective core request for a scheduled job."""
        preset_name = job.qos or hpc_config.get("DEFAULT_PRESET", "normal")
        preset = presets.get(preset_name, {})
        if job.cores is not None:
            return job.cores
        if _compute_resources:
            cores, _ = _compute_resources(preset)
            return cores
        return int(preset.get("cores_per_node", 1))

    # Filter to pending jobs whose dependencies are met
    pending_jobs = [
        j for j in jobs
        if j.status == "pending" and _are_dependencies_met(j, jobs)
    ]
    # Already sorted by priority (highest first) from _load_all_scheduled_jobs
    if not pending_jobs:
        return [], []

    to_submit: list[ScheduledJob] = []
    to_cancel: list[str] = []

    # Snapshot current usage once, then include jobs planned during this same
    # decision pass.  Without this, a single loop tick can overshoot limits by
    # many jobs because newly selected jobs do not yet have SLURM IDs.
    planned_submitted, planned_running = _count_total_queued_and_running(queue)
    planned_cores_by_preset: dict[str, int] = {}
    queue_by_id = {e.job_id: e for e in queue}
    for j in jobs:
        effective_state = _effective_scheduler_job_state(
            j,
            queue_by_id,
        )
        if effective_state is None:
            continue
        preset_name = j.qos or hpc_config.get("DEFAULT_PRESET", "normal")
        planned_cores_by_preset[preset_name] = (
            planned_cores_by_preset.get(preset_name, 0) + _get_job_requested_cores(j)
        )

    if congested:
        # Congested: submit highest-priority jobs first, up to limits.
        for job in pending_jobs:
            if planned_submitted >= max_scheduled or planned_running >= max_running:
                break
            preset_name = job.qos or hpc_config.get("DEFAULT_PRESET", "normal")
            preset = presets.get(preset_name, {})
            job_cores = _get_job_requested_cores(job)
            limit = float(preset.get("total_cores_available", 1e10))
            remaining = limit - planned_cores_by_preset.get(preset_name, 0)
            if job_cores <= remaining:
                to_submit.append(job)
                planned_submitted += 1
                planned_cores_by_preset[preset_name] = (
                    planned_cores_by_preset.get(preset_name, 0) + job_cores
                )
    else:
        # Non-congested: keep at most 1 pending in SLURM.
        slurm_pending_ids: list[str] = []
        slurm_pending_priorities: dict[str, int] = {}

        for j in jobs:
            if j.status == "submitted" and j.slurm_job_id:
                qe = next((e for e in queue if e.job_id == j.slurm_job_id), None)
                if qe and qe.state == "PD":
                    slurm_pending_ids.append(j.slurm_job_id)
                    slurm_pending_priorities[j.slurm_job_id] = j.effective_priority

        if pending_jobs and planned_submitted < max_scheduled and planned_running < max_running:
            top_pending = pending_jobs[0]
            preset_name = top_pending.qos or hpc_config.get("DEFAULT_PRESET", "normal")
            preset = presets.get(preset_name, {})
            top_cores = _get_job_requested_cores(top_pending)
            limit = float(preset.get("total_cores_available", 1e10))
            remaining = limit - planned_cores_by_preset.get(preset_name, 0)

            if top_cores <= remaining:
                if slurm_pending_ids:
                    for sid in slurm_pending_ids:
                        if slurm_pending_priorities.get(sid, 0) < top_pending.effective_priority:
                            to_cancel.append(sid)
                    if to_cancel or not slurm_pending_ids:
                        to_submit.append(top_pending)
                else:
                    to_submit.append(top_pending)

    return to_submit, to_cancel


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def _submit_one_scheduled_job(
    job: ScheduledJob,
    hpc_config: dict,
    _resolve_preset=None,
    _resolve_resources=None,
    _submit_python_file=None,
    _submit_gaussian_file=None,
    _file_type=None,
) -> bool:
    """
    Submit a single scheduled job to SLURM.

    Updates the job's status and slurm_job_id on success.
    Returns True if submission succeeded.
    """
    try:
        # Keep the daemon heartbeat fresh while a large submission burst is
        # in progress.  Otherwise the outer loop heartbeat may not be written
        # for many minutes if thousands of jobs are submitted sequentially.
        _write_heartbeat(hpc_config)

        _scheduler_log(
            f"  Submitting: {os.path.basename(job.filepath)} "
            f"(id={job.schedule_id}, priority={job.effective_priority}, "
            f"qos={job.qos or 'default'}, cores={job.cores or 'auto'}, "
            f"mem={job.memory_mb or 'auto'}MB)",
            hpc_config,
        )
        _scheduler_log(f"    Path: {job.filepath}", hpc_config)
        if job.script_args:
            _scheduler_log(f"    Args: {' '.join(job.script_args)}", hpc_config)

        buf_out = io.StringIO()
        buf_err = io.StringIO()

        ext = os.path.splitext(job.filepath)[1].lower().lstrip(".")

        ret_val = None
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            if ext == "py" and _submit_python_file:
                ret_val = _submit_python_file(
                    job.filepath, job.qos, job.script_args or None,
                    job.cores, job.memory_mb, email=job.email,
                )
            elif ext in ("gjf", "com") and _submit_gaussian_file:
                ret_val = _submit_gaussian_file(
                    job.filepath, job.qos, job.cores, job.memory_mb, email=job.email,
                )
            elif _file_type and _submit_gaussian_file:
                # Try file_type detection
                from Chem_Lib.Lib_Filetype import Filetype
                ftype = _file_type(job.filepath)
                if ftype == Filetype.gaussian_input:
                    ret_val = _submit_gaussian_file(
                        job.filepath, job.qos, job.cores, job.memory_mb, email=job.email,
                    )
                else:
                    raise ValueError(f"Unsupported file type: {job.filepath}")
            else:
                raise ValueError(f"No submit function available for: {job.filepath}")
        if ret_val:
            job.sh_script_path = ret_val

        output = buf_out.getvalue()
        err_output = buf_err.getvalue()

        match = re.search(r"Submitted batch job\s+(\d+)", output)
        if match:
            job.slurm_job_id = match.group(1)
        else:
            _scheduler_log(
                f"    WARNING: Could not parse SLURM job ID from output",
                hpc_config,
            )
            _scheduler_log(f"    stdout: {output.strip()}", hpc_config)
            if err_output.strip():
                _scheduler_log(f"    stderr: {err_output.strip()}", hpc_config)

        job.status = "submitted"
        job.submitted_at = datetime.now().isoformat()
        _save_scheduled_job(job, hpc_config)

        # Move config file to Submitted/ subdirectory
        _move_job_to_submitted(job, hpc_config)

        _write_heartbeat(hpc_config)
        _scheduler_log(f"    → SLURM job ID: {job.slurm_job_id}", hpc_config)
        return True

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        job.completed_at = datetime.now().isoformat()
        _save_scheduled_job(job, hpc_config)
        _write_heartbeat(hpc_config)
        _scheduler_log(f"    SUBMISSION FAILED: {e}", hpc_config)
        return False


# ---------------------------------------------------------------------------
# SLURM job cancel / resubmit
# ---------------------------------------------------------------------------

def _cancel_slurm_job(slurm_job_id: str, hpc_config: dict):
    """Cancel a SLURM job by ID using the configured CANCEL_COMMAND."""
    cancel_cmd = hpc_config.get("CANCEL_COMMAND", "scancel")
    try:
        _cancel_cmd = [cancel_cmd, str(slurm_job_id)]
        _scheduler_log(f"Running: {' '.join(_cancel_cmd)}", hpc_config)
        print(f"\n>>> {' '.join(_cancel_cmd)}\n")
        subprocess.run(
            _cancel_cmd, check=False,
            capture_output=True, text=True,
        )
    except Exception as e:
        _scheduler_log(f"    WARNING: Failed to cancel {slurm_job_id}: {e}", hpc_config)


def _resubmit_cancelled_jobs(
    jobs: list[ScheduledJob],
    cancelled_ids: list[str],
    hpc_config: dict,
):
    """
    Re-mark scheduler jobs whose SLURM submission was cancelled back to
    'pending' so they get re-submitted in a future cycle.
    """
    for j in jobs:
        if j.slurm_job_id in cancelled_ids and j.status in ("submitted",):
            old_id = j.slurm_job_id
            j.status = "pending"
            j.slurm_job_id = ""
            j.submitted_at = ""
            _save_scheduled_job(j, hpc_config)
            _scheduler_log(
                f"  Re-queued (pending): {os.path.basename(j.filepath)} "
                f"(was SLURM {old_id})",
                hpc_config,
            )


# ---------------------------------------------------------------------------
# Self-renewal
# ---------------------------------------------------------------------------

def _parse_time_limit_seconds(time_str: str) -> int:
    """
    Parse a SLURM time-limit string to total seconds.

    Supported formats::

        MM:SS
        HH:MM:SS
        D-HH:MM:SS
        D-HH:MM
        (minutes)

    Returns 0 on parse failure.
    """
    try:
        if "-" in time_str:
            days_part, rest = time_str.split("-", 1)
            days = int(days_part)
            parts = rest.split(":")
        else:
            days = 0
            parts = time_str.split(":")

        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, s = int(parts[0]), int(parts[1]), 0
        elif len(parts) == 1:
            return int(parts[0]) * 60
        else:
            return 0

        return days * 86400 + h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return 0


def _submit_scheduler_slurm_job(
    hpc_config: dict,
    replace_job_id: str = "",
    _ensure_job_script_dir=None,
    _resolve_preset=None,
    _build_sbatch_header=None,
    _calculate_auto_memory_mb=None,
) -> str:
    """
    Submit the scheduler itself as a SLURM job (high QoS, 1 CPU, max time).

    Returns the SLURM job ID of the submitted scheduler job, or empty string
    on failure.
    """
    if _calculate_auto_memory_mb is None:
        from HPC_Lib.HPC import _calculate_auto_memory_mb

    if _ensure_job_script_dir:
        _ensure_job_script_dir()

    if _resolve_preset:
        preset = _resolve_preset("high")
    else:
        preset = hpc_config.get("SLURM_PRESETS", {}).get("high", {})

    time_limit = preset.get("time_limit", "14-00:00:00")
    time_limit_seconds = _parse_time_limit_seconds(time_limit)
    scheduler_cores = 1
    scheduler_memory_mb = _calculate_auto_memory_mb(preset, scheduler_cores)

    scheduler_dir = _get_scheduler_dir(hpc_config)
    script_id = datetime.now().strftime("%y%m%d_%H%M%S")
    job_script_dir = hpc_config.get("JOB_SCRIPT_DIR", scheduler_dir)
    script_path = os.path.join(
        job_script_dir,
        f"auto_generated_script_scheduler_{script_id}.sh",
    )
    output_file = os.path.splitext(script_path)[0] + ".out"

    if _build_sbatch_header:
        header = _build_sbatch_header(
            job_name=_SCHEDULER_JOB_NAME,
            output_file=output_file,
            preset=preset,
            cores=scheduler_cores,
            memory_mb=scheduler_memory_mb,
            hpc_config=hpc_config,
            mail_type="FAIL",
            ntasks_per_node=1,
        )
    else:
        # Minimal fallback header
        header = "#!/bin/bash\n#SBATCH --job-name=HPC_Scheduler\n"

    python_path = hpc_config.get("PYTHON_PATH", "python")
    # Find the path to HPC.py (this module's sibling)
    my_lib_hpc_path = os.path.join(os.path.dirname(__file__), "HPC.py")

    replace_arg = f" --replace_job_id {replace_job_id}" if replace_job_id else ""

    body = f"""
# --- Scheduler job ---
echo "Scheduler starting on $(hostname) at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

{python_path} {my_lib_hpc_path} handler _run \\
    --job_id $SLURM_JOB_ID \\
    --time_limit {time_limit_seconds}{replace_arg}

echo "Scheduler finished at $(date)"
"""

    script_content = header + "\n" + body
    with open(script_path, "w", newline="\n") as f:
        f.write(script_content)

    submit_cmd = hpc_config.get("SUBMIT_COMMAND", "sbatch")
    _submit_cmd = [submit_cmd, script_path]
    print(f"\n>>> {' '.join(_submit_cmd)}\n")
    result = subprocess.run(
        _submit_cmd,
        capture_output=True, text=True,
    )

    slurm_job_id = ""
    if result.stdout.strip():
        m = re.search(r"Submitted batch job\s+(\d+)", result.stdout)
        if m:
            slurm_job_id = m.group(1)
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)

    if slurm_job_id:
        _scheduler_log(f"Scheduler SLURM job submitted: {slurm_job_id}", hpc_config)
        _scheduler_log(f"  Script: {script_path}", hpc_config)
        _scheduler_log(f"  Output: {output_file}", hpc_config)
        _scheduler_log(f"  Command: {python_path} {my_lib_hpc_path} handler _run ...", hpc_config)
        # Invalidate the entries cache so subsequent calls see the new scheduler
        global _SCHED_QUEUE_ENTRIES_CACHE
        _SCHED_QUEUE_ENTRIES_CACHE["time"] = 0.0
        # Write launch timestamp so concurrent schedule commands know a
        # scheduler was recently submitted and don't trigger a restart storm.
        _write_scheduler_launch_time(hpc_config)
    else:
        _scheduler_log(f"WARNING: Failed to submit scheduler job", hpc_config)

    return slurm_job_id


_SCHED_QUEUE_ENTRIES_CACHE: dict = {"time": 0.0, "result": [], "user": ""}
_SCHED_QUEUE_ENTRIES_TTL = 30  # seconds — reuse squeue result for rapid schedule calls


def _get_scheduler_queue_entries(hpc_config: dict) -> list[QueueEntry]:
    """Return all HPC_Scheduler jobs currently running or pending in the queue.

    Matches by job name (``_SCHEDULER_JOB_NAME``), which is more robust than
    relying on the state file alone.

    Results are cached for ``_SCHED_QUEUE_ENTRIES_TTL`` seconds so that
    rapid successive calls (e.g. running ``schedule`` for many jobs) do not
    each fire a separate squeue invocation.
    """
    global _SCHED_QUEUE_ENTRIES_CACHE
    user = hpc_config.get("USER_NAME", "")
    now = time.time()
    if (
        now - _SCHED_QUEUE_ENTRIES_CACHE["time"] < _SCHED_QUEUE_ENTRIES_TTL
        and _SCHED_QUEUE_ENTRIES_CACHE["user"] == user
    ):
        return _SCHED_QUEUE_ENTRIES_CACHE["result"]

    queue = _get_user_slurm_jobs(hpc_config)
    result = [
        e for e in queue
        if e.job_name == _SCHEDULER_JOB_NAME and e.state in ("R", "PD")
    ]
    _SCHED_QUEUE_ENTRIES_CACHE = {"time": now, "result": result, "user": user}
    return result


def _is_scheduler_running(hpc_config: dict) -> bool:
    """Check whether any HPC_Scheduler job is running or pending in the queue.

    Checks the SLURM queue by job name rather than relying solely on the
    state file, so stale state files or manually submitted schedulers are
    handled correctly.
    """
    return len(_get_scheduler_queue_entries(hpc_config)) > 0


def _check_and_self_renew(
    start_time: datetime,
    time_limit_seconds: int,
    current_job_id: str,
    hpc_config: dict,
    _ensure_job_script_dir=None,
    _resolve_preset=None,
    _build_sbatch_header=None,
    _calculate_auto_memory_mb=None,
) -> bool:
    """
    Check whether the scheduler should submit a renewal job.

    If remaining wall time < 2 hours and renewal has not yet been submitted,
    submits a new scheduler job with ``--replace_job_id`` pointing to
    *current_job_id*.

    Returns True if renewal was submitted (or already submitted).
    """
    if time_limit_seconds <= 0:
        return False

    elapsed = (datetime.now() - start_time).total_seconds()
    remaining = time_limit_seconds - elapsed

    if remaining > _SCHEDULER_RENEWAL_THRESHOLD:
        return False

    state = _read_scheduler_state(hpc_config)
    if state and state.get("renewal_submitted"):
        return True

    # Check if there's already a pending scheduler in the queue (besides us).
    # If so, a renewal is already queued — no need to submit another.
    pending_schedulers = [
        e for e in _get_scheduler_queue_entries(hpc_config)
        if e.job_id != current_job_id and e.state == "PD"
    ]
    if pending_schedulers:
        _scheduler_log(
            f"Self-renewal: pending scheduler already exists "
            f"({pending_schedulers[0].job_id}), skipping.",
            hpc_config,
        )
        return True

    _scheduler_log(
        f"Self-renewal: {remaining:.0f}s remaining, submitting replacement.",
        hpc_config,
    )
    new_id = _submit_scheduler_slurm_job(
        hpc_config,
        replace_job_id=current_job_id,
        _ensure_job_script_dir=_ensure_job_script_dir,
        _resolve_preset=_resolve_preset,
        _build_sbatch_header=_build_sbatch_header,
        _calculate_auto_memory_mb=_calculate_auto_memory_mb,
    )

    if new_id and state:
        state["renewal_submitted"] = True
        state["renewal_job_id"] = new_id
        _write_scheduler_state(state, hpc_config)

    return bool(new_id)


# ---------------------------------------------------------------------------
# Cancel a scheduled job
# ---------------------------------------------------------------------------

def cancel_scheduled_job(
    schedule_id: str,
    hpc_config: dict,
) -> bool:
    """
    Cancel a scheduled job by its schedule ID.

    - If the job is ``pending``, it is marked ``cancelled`` and deleted.
    - If the job is ``submitted`` (pending in SLURM), the SLURM job is
      cancelled and the config file is deleted.
    - If the job is ``running``, the SLURM job is cancelled.
    - If the job is already completed/failed/cancelled, nothing happens.

    Returns True if something was cancelled.
    """
    jobs = _load_all_scheduled_jobs(hpc_config)
    target = None
    for j in jobs:
        if j.schedule_id == schedule_id:
            target = j
            break

    if not target:
        print(f"[Scheduler] No job with schedule ID '{schedule_id}' found.")
        return False

    if target.status in ("completed", "failed", "cancelled"):
        print(f"[Scheduler] Job {schedule_id} is already {target.status}.")
        return False

    if target.status in ("submitted", "running") and target.slurm_job_id:
        print(f"[Scheduler] Cancelling SLURM job {target.slurm_job_id}...")
        _cancel_slurm_job(target.slurm_job_id, hpc_config)

    target.status = "cancelled"
    target.completed_at = datetime.now().isoformat()
    _delete_scheduled_job_file(target)
    print(f"[Scheduler] Job {schedule_id} cancelled.")
    _scheduler_log(f"Job {schedule_id} cancelled by user.", hpc_config)
    return True


def cancel_scheduled_job_and_collect_slurm_job(
    schedule_id: str,
    hpc_config: dict,
) -> tuple[bool, str]:
    """
    Cancel a scheduler-managed job record and return its linked SLURM job ID.

    This updates/removes the scheduler record immediately, but does not invoke
    ``scancel`` itself. The caller can batch multiple returned SLURM job IDs
    into a single cancel command.

    Returns:
        ``(cancelled, slurm_job_id)`` where ``slurm_job_id`` is empty when the
        scheduler job was only pending in the scheduler queue.
    """
    jobs = _load_all_scheduled_jobs(hpc_config)
    target = None
    for j in jobs:
        if j.schedule_id == schedule_id:
            target = j
            break

    if not target:
        print(f"[Scheduler] No job with schedule ID '{schedule_id}' found.")
        return False, ""

    if target.status in ("completed", "failed", "cancelled"):
        print(f"[Scheduler] Job {schedule_id} is already {target.status}.")
        return False, ""

    slurm_job_id = ""
    if target.status in ("submitted", "running") and target.slurm_job_id:
        slurm_job_id = str(target.slurm_job_id)
        print(f"[Scheduler] Job {schedule_id} detached from SLURM job {slurm_job_id}.")

    target.status = "cancelled"
    target.completed_at = datetime.now().isoformat()
    _delete_scheduled_job_file(target)
    print(f"[Scheduler] Job {schedule_id} cancelled.")
    _scheduler_log(f"Job {schedule_id} cancelled by user.", hpc_config)
    return True, slurm_job_id


def cancel_scheduled_jobs_and_collect_slurm_jobs(
    schedule_ids: list[str],
    hpc_config: dict,
    progress_every: int = 200,
) -> list[str]:
    """
    Cancel many scheduler-managed jobs in one pass and collect linked SLURM IDs.

    This avoids repeatedly scanning the scheduler directory once per schedule ID,
    which becomes prohibitively slow for large cancellations.

    Args:
        schedule_ids: Scheduler IDs to cancel.
        hpc_config: HPC configuration dict.
        progress_every: Print a progress line every N processed jobs when the
            request is large.

    Returns:
        Linked SLURM job IDs that should be cancelled in a later batched
        ``scancel`` call.
    """
    ordered_ids = list(dict.fromkeys(schedule_ids))
    if not ordered_ids:
        return []

    jobs = _load_all_scheduled_jobs(hpc_config)
    jobs_by_schedule_id = {job.schedule_id: job for job in jobs}

    total = len(ordered_ids)
    slurm_job_ids: list[str] = []
    cancelled_count = 0
    missing_count = 0
    skipped_count = 0

    print(f"[Scheduler] Cancelling {total} scheduler job(s)...")

    for index, schedule_id in enumerate(ordered_ids, start=1):
        target = jobs_by_schedule_id.get(schedule_id)

        if not target:
            print(f"[Scheduler] No job with schedule ID '{schedule_id}' found.")
            missing_count += 1
        elif target.status in ("completed", "failed", "cancelled"):
            print(f"[Scheduler] Job {schedule_id} is already {target.status}.")
            skipped_count += 1
        else:
            if target.status in ("submitted", "running") and target.slurm_job_id:
                slurm_job_ids.append(str(target.slurm_job_id))

            target.status = "cancelled"
            target.completed_at = datetime.now().isoformat()
            _delete_scheduled_job_file(target)
            cancelled_count += 1

        if total <= 20:
            if target and target.status == "cancelled":
                print(f"[Scheduler] Job {schedule_id} cancelled.")
        elif index == 1 or index % progress_every == 0 or index == total:
            print(
                f"[Scheduler] Progress: {index}/{total} processed, "
                f"{cancelled_count} cancelled, {len(slurm_job_ids)} linked SLURM job(s)."
            )

    _scheduler_log(
        f"Bulk user cancellation: {cancelled_count} scheduler job(s) cancelled, "
        f"{len(slurm_job_ids)} linked SLURM job(s), {missing_count} missing, "
        f"{skipped_count} already finished.",
        hpc_config,
    )
    return list(dict.fromkeys(slurm_job_ids))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def scheduler_main_loop(
    current_job_id: str,
    time_limit_seconds: int,
    replace_job_id: str,
    hpc_config: dict,
    _ensure_job_script_dir=None,
    _resolve_preset=None,
    _build_sbatch_header=None,
    _calculate_auto_memory_mb=None,
    _compute_resources=None,
    _submit_python_file=None,
    _submit_gaussian_file=None,
    _file_type=None,
):
    """
    Main scheduler loop.  Runs inside a SLURM job.

    This function:
      1. Handles replacement of a previous scheduler.
      2. Writes its own state to ``.scheduler_state.json``.
      3. Enters a polling loop that:
         a. Loads all scheduled jobs.
         b. Updates statuses of submitted jobs (deletes config on RUNNING).
         c. Checks dependency chains and marks blocked jobs.
         d. Decides which pending jobs to submit next (in priority order).
         e. Submits them.
         f. Checks self-renewal.
         g. Auto-terminates after 24h of continuous inactivity.
      4. Adaptive polling: 5s when new jobs appear, 30s otherwise.

    Args:
        current_job_id:     This scheduler's own SLURM job ID.
        time_limit_seconds: Wall-time limit of this SLURM job.
        replace_job_id:     SLURM job ID of the previous scheduler to cancel.
        hpc_config:         Configuration dict (from My_Lib_HPC globals).
        _ensure_job_script_dir, _resolve_preset, _build_sbatch_header,
        _compute_resources, _submit_python_file, _submit_gaussian_file,
        _file_type: Callback functions from My_Lib_HPC to avoid circular imports.
    """
    start_time = datetime.now()
    _scheduler_log("=" * 60, hpc_config)
    _scheduler_log(f"Scheduler starting  (SLURM job {current_job_id})", hpc_config)
    _scheduler_log(
        f"Time limit: {time_limit_seconds}s  "
        f"Replace: {replace_job_id or '(none)'}",
        hpc_config,
    )
    _scheduler_log("=" * 60, hpc_config)

    # Step 1: Cancel ALL other HPC_Scheduler jobs in the queue (except self).
    # This covers the explicit replace_job_id case as well as any stale or
    # duplicate scheduler jobs that may exist.
    other_schedulers = [
        e for e in _get_scheduler_queue_entries(hpc_config)
        if e.job_id != current_job_id
    ]
    if other_schedulers:
        for e in other_schedulers:
            _scheduler_log(
                f"Cancelling other scheduler: {e.job_id} (state={e.state})",
                hpc_config,
            )
            _cancel_slurm_job(e.job_id, hpc_config)
        time.sleep(2)
    elif replace_job_id:
        # Fallback: replace_job_id was specified but may have already left the queue
        _scheduler_log(f"Cancelling previous scheduler: {replace_job_id}", hpc_config)
        _cancel_slurm_job(replace_job_id, hpc_config)
        time.sleep(2)

    # Step 2: Write state + first heartbeat
    state = {
        "slurm_job_id": current_job_id,
        "started_at": start_time.isoformat(),
        "renewal_submitted": False,
        "renewal_job_id": "",
    }
    _write_scheduler_state(state, hpc_config)
    _write_heartbeat(hpc_config)

    # Step 3: Main loop with adaptive polling
    consecutive_errors = 0
    last_known_job_count = 0
    last_new_job_time = 0.0  # epoch time of last detected new job
    last_heartbeat_time = time.time()  # monotonic clock for heartbeat writes
    idle_since: float | None = None
    last_idle_log_hours = -1

    while True:
        try:
            # 3a. Load all scheduled jobs (with heartbeat to stay alive
            #     during very large directory scans)
            jobs = _load_all_scheduled_jobs(
                hpc_config, _keep_heartbeat_alive=True,
            )

            # Detect new jobs for fast-poll mode
            current_job_count = len(jobs)
            if current_job_count > last_known_job_count and last_known_job_count > 0:
                last_new_job_time = time.time()
                _scheduler_log(
                    f"  New job(s) detected ({current_job_count - last_known_job_count} new), "
                    f"switching to fast poll ({_SCHEDULER_POLL_INTERVAL_FAST}s)",
                    hpc_config,
                )
            last_known_job_count = current_job_count

            # 3b. Fetch SLURM queue once — reuse for status update + submission decision
            current_slurm_queue = _get_user_slurm_jobs(hpc_config)

            # Update statuses (modifies job objects in-place, saves
            # individual files; deletes completed configs).  No need to
            # reload the entire directory — the in-memory list is
            # authoritative until next tick.
            status_refresh = _update_submitted_job_statuses(
                jobs,
                hpc_config,
                queue=current_slurm_queue,
                max_terminal_checks=_SCHEDULER_STATUS_RECONCILE_BATCH,
            )

            if status_refresh["deferred_missing"] > 0:
                _scheduler_log(
                    f"  Deferred terminal-state reconciliation for "
                    f"{status_refresh['deferred_missing']} departed job(s)",
                    hpc_config,
                )

            # Filter out completed/cancelled jobs whose config files were
            # already deleted, so they don't clutter subsequent logic.
            jobs = [j for j in jobs if j.status not in ("completed", "cancelled")]

            # 3c. Check dependencies — mark blocked jobs as failed
            #     (_mark_dependency_blocked modifies objects in-place)
            for j in jobs:
                if j.status == "pending" and j.depends_on:
                    _mark_dependency_blocked(j, jobs, hpc_config)

            # 3d. Summarise active jobs and track continuous idle time.
            active_statuses = ("pending", "submitted", "running")
            active_jobs = [j for j in jobs if j.status in active_statuses]

            pending_count = sum(1 for j in active_jobs if j.status == "pending")
            submitted_count = sum(1 for j in active_jobs if j.status == "submitted")
            running_count = sum(1 for j in active_jobs if j.status == "running")
            if active_jobs:
                if idle_since is not None:
                    idle_seconds = time.time() - idle_since
                    _scheduler_log(
                        f"Active jobs detected again after {idle_seconds / 3600:.1f}h idle; clearing idle shutdown timer.",
                        hpc_config,
                    )
                    idle_since = None
                    last_idle_log_hours = -1
                _scheduler_log(
                    f"Status: {pending_count} pending, {submitted_count} submitted, "
                    f"{running_count} running",
                    hpc_config,
                )
            else:
                now_ts = time.time()
                if idle_since is None:
                    idle_since = now_ts
                    last_idle_log_hours = 0
                    _scheduler_log(
                        "No active jobs. Idle shutdown timer started; scheduler will exit after 24h of continuous inactivity.",
                        hpc_config,
                    )
                else:
                    idle_for = now_ts - idle_since
                    if idle_for >= _SCHEDULER_IDLE_EXIT_THRESHOLD:
                        _scheduler_log("No active jobs for 24h. Scheduler exiting.", hpc_config)
                        break

                    idle_hours = int(idle_for // 3600)
                    if idle_hours > last_idle_log_hours:
                        remaining_hours = (
                            _SCHEDULER_IDLE_EXIT_THRESHOLD - idle_for
                        ) / 3600
                        _scheduler_log(
                            f"No active jobs. Idle for {idle_for / 3600:.1f}h; exiting after {remaining_hours:.1f}h more if still idle.",
                            hpc_config,
                        )
                        last_idle_log_hours = idle_hours

            # 3e. Reuse already-fetched queue for submission decision (no extra squeue call)
            to_submit, to_cancel = _decide_next_submissions(
                jobs, current_slurm_queue, hpc_config, _compute_resources
            )

            # Log why nothing is submitted (helps diagnose stalls)
            if not to_submit and pending_count > 0:
                submitted_r, running_r = _count_total_queued_and_running(current_slurm_queue)
                max_sched = hpc_config.get("CONCURRENT_SCHEDULED_MISSION_COUNT", "∞")
                max_run   = hpc_config.get("CONCURRENT_RUNNING_MISSION_COUNT",   "∞")
                # Check core budgets for each preset
                presets = hpc_config.get("SLURM_PRESETS", {})
                budget_parts = []
                for pname, preset in presets.items():
                    limit = preset.get("total_cores_available", 1e10)
                    if limit < 1e9:
                        remaining = _cores_budget_remaining(
                            jobs, pname, hpc_config, _compute_resources,
                            queue=current_slurm_queue,
                        )
                        budget_parts.append(f"{pname}: {remaining:.0f}/{limit:.0f} cores free")
                budget_str = "; ".join(budget_parts) if budget_parts else "no core limits"
                _scheduler_log(
                    f"  No submissions: {pending_count} pending held back "
                    f"(submitted+running={submitted_r}, "
                    f"max_scheduled={max_sched}, max_running={max_run}, "
                    f"core budget: {budget_str})",
                    hpc_config,
                )

            # Cancel lower-priority pending jobs (non-congested mode)
            if to_cancel:
                for sid in to_cancel:
                    _scheduler_log(f"  Cancelling lower-priority pending job: {sid}", hpc_config)
                    _cancel_slurm_job(sid, hpc_config)
                _resubmit_cancelled_jobs(jobs, to_cancel, hpc_config)

            # Submit
            for job in to_submit:
                _submit_one_scheduled_job(
                    job, hpc_config,
                    _resolve_preset=_resolve_preset,
                    _resolve_resources=None,
                    _submit_python_file=_submit_python_file,
                    _submit_gaussian_file=_submit_gaussian_file,
                    _file_type=_file_type,
                )
                # Update last_new_job_time to keep fast polling after submission
                last_new_job_time = time.time()

            # 3f. Self-renewal check
            _check_and_self_renew(
                start_time, time_limit_seconds, current_job_id,
                hpc_config,
                _ensure_job_script_dir=_ensure_job_script_dir,
                _resolve_preset=_resolve_preset,
                _build_sbatch_header=_build_sbatch_header,
                _calculate_auto_memory_mb=_calculate_auto_memory_mb,
            )

            # 3g. Clean up completed configs (keep failed for inspection)
            for j in jobs:
                if j.status == "completed":
                    _delete_scheduled_job_file(j)

            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            _scheduler_log(f"ERROR in scheduler loop: {e}", hpc_config)
            import traceback
            _scheduler_log(traceback.format_exc(), hpc_config)
            if consecutive_errors >= 10:
                _scheduler_log("Too many consecutive errors. Scheduler aborting.", hpc_config)
                break

        # Write heartbeat at the end of each loop iteration
        _write_heartbeat(hpc_config)
        last_heartbeat_time = time.time()

        # Adaptive polling interval — sleep in small increments so we can
        # keep writing heartbeats even during quiet periods.
        time_since_new_job = time.time() - last_new_job_time if last_new_job_time > 0 else float("inf")
        if time_since_new_job < _SCHEDULER_FAST_POLL_DURATION:
            poll_interval = _SCHEDULER_POLL_INTERVAL_FAST
        else:
            poll_interval = _SCHEDULER_POLL_INTERVAL_NORMAL

        # Sleep in chunks of heartbeat interval so heartbeat stays fresh
        sleep_remaining = poll_interval
        while sleep_remaining > 0:
            chunk = min(sleep_remaining, _SCHEDULER_HEARTBEAT_INTERVAL)
            time.sleep(chunk)
            sleep_remaining -= chunk
            if sleep_remaining > 0:
                _write_heartbeat(hpc_config)

    _delete_heartbeat(hpc_config)
    _scheduler_log("Scheduler loop ended.", hpc_config)
