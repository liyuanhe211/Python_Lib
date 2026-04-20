# -*- coding: utf-8 -*-
"""
My_Lib_HPC Configuration — EXAMPLE / TEMPLATE
===============================================

This file serves TWO purposes:

  1. **Template**: Copy this file to your ``My_Program/`` root directory and
     rename it to ``My_Lib_HPC_Configuration_Secret_{YourHPC}.py``, then
     fill in the values for your cluster.

  2. **IDE autocomplete**: Because the *real* configuration file lives outside
     the Python_Lib source tree (in My_Program root, which is 4 levels up),
     IDEs cannot resolve the variable names injected via ``globals()``.
     This file provides type-annotated stubs so that editors like VS Code /
     PyCharm can offer autocompletion and type checking.

.. warning::

   **This file is NOT a valid configuration by itself.**  If ``My_Lib_HPC.py``
   detects that the only configuration file found is this example file, it will
   raise an error.  You MUST have at least one properly-named configuration
   file (not ``_Example``) in the ``My_Program/`` directory.

To use:
   1. Copy this file to  ``E:/My_Program/``  (or the equivalent on your HPC).
   2. Rename to ``My_Lib_HPC_Configuration_Secret_Negishi.py``
      (replace *Negishi* with your HPC's name).
   3. Fill in every field below.
   4. Keep the file out of version control (add to ``.gitignore``).

See also:
   ``My_Lib_HPC.py`` — the main module that imports this configuration.
"""

__author__ = 'LiYuanhe'

# ========================== Cluster Identity ==========================
HPC_NAME: str = "Example"
"""Short name of the HPC cluster (e.g. ``"Negishi"``, ``"Anvil"``)."""

USER_NAME: str = ""
"""Your UNIX username on this cluster (e.g. ``"li5876"``)."""

# ========================== Paths ==========================
HOME_PATH: str = ""
"""Absolute path to your home directory (e.g. ``"/home/li5876"``)."""

PYTHON_PATH: str = ""
"""
Absolute path to the Python interpreter on this HPC.
Example: ``"/home/li5876/anaconda3/bin/python"``
"""

MY_PROGRAM_PATH: str = ""
"""
Absolute path to the ``My_Program/`` directory on this HPC.
Example: ``"/scratch/negishi/li5876/My_Program"``
"""

JOB_SCRIPT_DIR: str = ""
"""
Directory where auto-generated SLURM job scripts (``.sh``) and their
output files (``.out``) will be stored.
Example: ``"/home/li5876/Scripts"``
"""

# ========================== SLURM Account ==========================
SLURM_ACCOUNT: str = ""
"""
SLURM account name to charge jobs to (``#SBATCH --account``).
Example: ``"meji"``
"""

# ========================== SLURM Partition & QoS Presets ==========================
SLURM_PRESETS: dict[str, dict] = {
    "high": {
        "partition": "cpu",
        "qos": "normal",
        "time_limit": "14-00:00:00",   # 2 weeks
        "description": "High priority (normal QoS), up to 2 weeks",
        "cores_per_node": 128,
        "memory_per_node_mb": 263168,  # 257 GB ≈ 263168 MB
        "total_cores_available": 128,
    },
    "normal": {
        "partition": "cpu",
        "qos": "standby",
        "time_limit": "04:00:00",       # 4 hours
        "description": "Low priority (standby QoS), up to 4 hours",
        "cores_per_node": 128,
        "memory_per_node_mb": 263168,
        "total_cores_available": 1E10,
    },
}
"""
Each key is a QoS preset name (used via ``--qos``).  Values are dicts with:

- ``partition``          — SLURM partition name.
- ``qos``                — SLURM QoS string.
- ``time_limit``         — Wall-time limit (``D-HH:MM:SS`` or ``HH:MM:SS``).
- ``description``        — Human-readable description.
- ``cores_per_node``     — Maximum CPU cores available per node in this QoS.
- ``memory_per_node_mb`` — Maximum memory (MB) per node in this QoS.
- ``total_cores_available`` — Scheduler limit: max total CPU cores across
  all simultaneously submitted jobs in this QoS.  Set to ``1E10`` for no limit.
"""

DEFAULT_PRESET: str = "normal"
"""Preset used when ``--qos`` is not specified."""

# ========================== Scheduler Settings ==========================
CONCURRENT_SCHEDULED_MISSION_COUNT: float = 1E10
"""
Maximum number of scheduler-managed jobs that may be submitted to SLURM
at the same time (pending + running).  Set ``1E10`` for no limit.
"""

CONCURRENT_RUNNING_MISSION_COUNT: float = 1E10
"""
Maximum number of scheduler-managed jobs that may be *running* at the same
time.  Set ``1E10`` for no limit.
"""

CONGESTED_QUEUE: bool = True
"""
Whether the SLURM queue is typically congested (many long-pending jobs).

- ``True``  — The scheduler submits up to the concurrency limits.  Once a job
  is in SLURM's queue, it is never cancelled for priority reasons.  Priority
  only controls the order in which *new* jobs are submitted.
- ``False`` — The scheduler keeps at most one pending job in SLURM.  If a
  higher-priority job appears, the pending one is cancelled, the high-priority
  job is submitted, and the cancelled job is re-queued.
"""

# ========================== Job Defaults ==========================
MEMORY_FRACTION: float = 0.95
"""Fraction of ``memory_per_node_mb`` to request per job (default 0.95)."""

CORE_FRACTION: float = 1.0
"""Fraction of ``cores_per_node`` to request per job (default 1.0)."""

NODES: int = 1
"""Number of nodes to request per job (single-node jobs)."""

# ========================== Notifications ==========================
MAIL_USER: str = ""
"""
Email address for SLURM job notifications.
Leave empty to disable email notifications.
Example: ``"user+HPC@gmail.com"``
"""

# ========================== Submit / Cancel Commands ==========================
SUBMIT_COMMAND: str = "sbatch"
"""Command used to submit jobs (default ``"sbatch"``)."""

CANCEL_COMMAND: str = "scancel"
"""Command used to cancel jobs (default ``"scancel"``)."""

# ========================== Optional: Gaussian ==========================
# Uncomment if you submit Gaussian jobs:
# GAUSSIAN_EXE_DIR: str = "/opt/gaussian/g16/bsd"
