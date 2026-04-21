"""
Sampling Among Numbered Files
==============================

This script selectively keeps a subset of files whose names contain a numbered
capture group (e.g. epoch number, step number) and moves the rest to the
recycle bin.

Configuration (constants at the top of the script):
----------------------------------------------------
SCAN_FOLDERS : list[str]
    List of absolute folder paths to scan.
    Example:
        [
            r"E:\\My_Program\\Machine_Learning\\Triton\\Offset_Recovery\\20260117_LYH"
            r"\\Checkpoints\\LYH_B0_Offset_Recovery_Model_Training.py_20260125_104835",
        ]

FILENAME_REGEX : str
    A regex applied to each *filename* (not full path).  It must contain
    exactly one capture group that matches a non-negative integer.
    Example:  ``Epoch_(\\d+)_Sample_0\\.png``

SCAN_DESCENDANTS : bool
    * True  – recursively scan all descendant folders of each entry in
              SCAN_FOLDERS.  Matching files are grouped by their immediate
              parent directory; each group is sampled independently.
    * False – scan only the folder itself (no recursion).

Interactive prompts (per folder or shared across all folders):
-------------------------------------------------------------
1. **Number of files to keep**
   - Integer >= 1        → absolute count of files to keep.
   - Float in (0, 1)     → ratio of total matched files to keep.
     The kept count is computed as int(total * ratio).
   - Values <= 0 or non-integer values > 1 raise an exception.

2. **Keep strategy** (one of three choices):
   - ``largest``  – keep the *n* files with the largest matched number.
   - ``smallest`` – keep the *n* files with the smallest matched number.
   - ``even``     – keep *n* files evenly distributed by *list index*
     (files sorted by matched number ascending).  The first and last files
     are always kept.
     Example: matched numbers sorted = [0,1,2,3,4,5,100,200,300,400,500,600,700],
     keep 4 → indices 0, 4, 8, 12 → numbers [0, 4, 300, 700].

3. **Apply same settings to all folders?**
   - Yes → the keep-count and strategy entered once are reused for every folder.
   - No  → the user is prompted separately for each folder.

Deletion behaviour:
-------------------
Files that are NOT selected for keeping are moved to the **recycle bin**
(via ``send2trash``), not permanently deleted.

Sampling is strictly per-folder: the decision to keep or delete a file depends
only on the matched files within that same folder.
"""

import os
import re
from pathlib import Path
from send2trash import send2trash

# ============================================================================
# User configuration
# ============================================================================

# True:  recursively scan all descendant folders of each entry in SCAN_FOLDERS;
#        matched files are grouped and sampled independently per immediate parent folder.
# False: scan only the folders listed in SCAN_FOLDERS themselves (no recursion).
SCAN_DESCENDANTS = True

# List of folder paths to scan for matching files.
# One path per line. Raw strings (r"...") are recommended to avoid backslash issues.
# When SCAN_DESCENDANTS is False, only these exact folders are scanned.
# When SCAN_DESCENDANTS is True, all descendant subfolders are scanned, with
# matched files grouped independently by their immediate parent directory.
SCAN_FOLDERS = r'''E:\My_Program\Machine_Learning\Song2025\20251209_LYH\0_Traning_Record_A11'''
SCAN_FOLDERS = [line.strip().strip('"') for line in SCAN_FOLDERS.strip().splitlines() if line.strip()]

# Regex pattern applied to each *filename* (not the full path).
# Must contain exactly one capture group that matches a non-negative integer
# (e.g. an epoch number, step index). The integer determ
# ines sort order and
# the "largest"/"smallest" keep strategies.
# IMPORTANT: use \\. to match a literal dot; a bare . matches any character.
FILENAME_REGEX = r"Linear_Fit_\[E(\d+)\]\.json"

# ============================================================================
# Core logic
# ============================================================================


def collect_matched_files(folder: str, pattern: re.Pattern) -> dict[str, list[tuple[Path, int]]]:
    """
    Scan *folder* for files whose names match *pattern*.

    Returns a dict mapping each scanned directory (as string) to a list of
    (file_path, matched_integer) tuples, sorted ascending by matched_integer.
    """
    results: dict[str, list[tuple[Path, int]]] = {}
    folder_path = Path(folder)

    if SCAN_DESCENDANTS:
        file_iter = folder_path.rglob("*")
    else:
        file_iter = folder_path.iterdir()

    for f in file_iter:
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m is None:
            continue
        raw = m.group(1)
        # Verify the captured group is a non-negative integer
        if not re.fullmatch(r"\d+", raw):
            raise ValueError(
                f"Capture group matched non-integer value '{raw}' in file: {f}\n"
                "The regex capture group must match a non-negative integer (0 or positive)."
            )
        number = int(raw)
        parent = str(f.parent)
        results.setdefault(parent, []).append((f, number))

    # Sort each folder's list by matched number ascending
    for key in results:
        results[key].sort(key=lambda x: x[1])

    return results


def ask_keep_count(total: int) -> int:
    """Interactively ask how many files to keep and return an absolute count."""
    while True:
        raw = input(
            f"  Total matched files: {total}.\n"
            "  How many to keep?\n"
            "    - Integer >= 1  : exact number of files to keep  (e.g. 10)\n"
            "    - Float (0, 1)  : fraction of files to keep, result truncated with int()  (e.g. 0.2 keeps 20%)\n"
            "  > "
        ).strip()
        try:
            value = float(raw)
        except ValueError:
            print("  Invalid input. Enter a number.")
            continue

        if value <= 0:
            raise ValueError(f"Keep count/ratio must be > 0, got {value}")

        if value >= 1:
            if value != int(value):
                raise ValueError(
                    f"Values >= 1 must be integers (exact file count), got {value}"
                )
            n = int(value)
        else:
            # ratio in (0, 1)
            n = int(total * value)

        if n < 1:
            n = 1
        if n > total:
            n = total
        return n


def ask_strategy() -> str:
    """Interactively ask which keep strategy to use."""
    while True:
        raw = input("  Strategy – 'largest', 'smallest', or 'even': ").strip().lower()
        if raw in ("largest", "smallest", "even"):
            return raw
        print("  Invalid choice. Enter 'largest', 'smallest', or 'even'.")


def select_files_to_keep(
    files: list[tuple[Path, int]], n: int, strategy: str
) -> list[tuple[Path, int]]:
    """
    Select *n* files to keep from *files* (already sorted ascending by number).

    Returns the list of kept (path, number) tuples.
    """
    total = len(files)
    if n >= total:
        return list(files)

    if strategy == "largest":
        return files[total - n:]
    elif strategy == "smallest":
        return files[:n]
    elif strategy == "even":
        if n == 1:
            return [files[-1]]
        # Pick n indices evenly spaced, always including first (0) and last (total-1)
        indices = set()
        for i in range(n):
            frac = i / (n - 1)
            idx = round(frac * (total - 1))
            indices.add(idx)
        # If rounding caused collisions, greedily fill from nearest ideal positions
        if len(indices) < n:
            ideal = [i / (n - 1) * (total - 1) for i in range(n)]
            all_idx = sorted(indices)
            for target in ideal:
                if len(all_idx) >= n:
                    break
                candidate = round(target)
                # Search outward from candidate
                for offset in range(total):
                    for c in (candidate + offset, candidate - offset):
                        if 0 <= c < total and c not in indices:
                            indices.add(c)
                            all_idx = sorted(indices)
                            break
                    if len(indices) >= n:
                        break
        sorted_indices = sorted(indices)[:n]
        return [files[i] for i in sorted_indices]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    # Show current configuration and give the user a chance to abort and edit.
    print("=" * 60)
    print("Current configuration:")
    print(f"  SCAN_FOLDERS     : {SCAN_FOLDERS}")
    print(f"  FILENAME_REGEX   : {FILENAME_REGEX}")
    print(f"  SCAN_DESCENDANTS : {SCAN_DESCENDANTS}")
    print("=" * 60)
    input("Press Enter to continue, or Ctrl+C to abort and edit the script...  ")
    print()

    # Warn if FILENAME_REGEX contains an unescaped dot, which matches any character.
    unescaped_dots = [
        i for i, c in enumerate(FILENAME_REGEX)
        if c == "." and (i == 0 or FILENAME_REGEX[i - 1] != "\\")
    ]
    if unescaped_dots:
        print(
            f"WARNING: FILENAME_REGEX contains {len(unescaped_dots)} unescaped '.'"
            f" at position(s) {unescaped_dots}.\n"
            "  A bare '.' matches ANY character, not just a literal dot.\n"
            "  If a literal dot is intended, replace '.' with '\\.' in FILENAME_REGEX.\n"
        )

    pattern = re.compile(FILENAME_REGEX)

    # Collect all matched files grouped by folder
    all_folder_files: dict[str, list[tuple[Path, int]]] = {}
    for folder in SCAN_FOLDERS:
        if not os.path.isdir(folder):
            print(f"WARNING: Folder does not exist, skipping: {folder}")
            continue
        print("Collecting matched files in folder:", folder)
        matched = collect_matched_files(folder, pattern)
        all_folder_files.update(matched)

    if not all_folder_files:
        print("No matched files found in any folder.")
        return

    # Print summary
    print(f"\nFound matched files in {len(all_folder_files)} folder(s):")
    for folder, files in all_folder_files.items():
        nums = [n for _, n in files]
        print(f"\n\n  {folder}: {len(files)} files, range [{min(nums)}, {max(nums)}]")

    # Ask whether to use the same setting for all folders
    apply_same = False
    if len(all_folder_files) > 1:
        while True:
            ans = input("\nApply the same settings to all folders? (y/n): ").strip().lower()
            if ans in ("y", "n"):
                apply_same = (ans == "y")
                break
            print("Enter 'y' or 'n'.")
    else:
        apply_same = True  # Only one folder, no need to ask

    # Outer retry loop: re-ask settings if the user rejects the preview.
    while True:
        shared_strategy: str = ""
        shared_n_input_value: float = 0.0

        if apply_same:
            # Ask settings once; values are applied (scaled) per-folder below.
            print(f"\n--- Settings (applied to all folders) ---")
            while True:
                raw = input(
                    "  How many to keep in each folder?\n"
                    "    - Integer >= 1  : exact number of files to keep  (e.g. 10)\n"
                    "    - Float (0, 1)  : fraction of files to keep, result truncated with int()  (e.g. 0.2 keeps 20%)\n"
                    "  > "
                ).strip()
                try:
                    value = float(raw)
                except ValueError:
                    print("  Invalid input. Enter a number.")
                    continue
                if value <= 0:
                    raise ValueError(f"Keep count/ratio must be > 0, got {value}")
                if value >= 1:
                    if value != int(value):
                        raise ValueError(
                            f"Values >= 1 must be integers (exact file count), got {value}"
                        )
                shared_n_input_value = value
                break
            shared_strategy = ask_strategy()

        # Process each folder independently.
        files_to_delete: list[tuple[Path, int]] = []
        files_to_keep: list[tuple[Path, int]] = []

        for folder, files in all_folder_files.items():
            total = len(files)
            print(f"\n--- Folder: {folder} ({total} matched files) ---")

            if apply_same:
                if shared_n_input_value >= 1:
                    n = int(shared_n_input_value)
                else:
                    n = int(total * shared_n_input_value)
                if n < 1:
                    n = 1
                if n > total:
                    n = total
                strategy = shared_strategy
            else:
                n = ask_keep_count(total)
                strategy = ask_strategy()

            kept = select_files_to_keep(files, n, strategy)
            kept_paths_local = {p for p, _ in kept}
            to_delete = [(p, num) for p, num in files if p not in kept_paths_local]

            print(f"  Deleting {len(to_delete)}, keeping {len(kept)}")

            files_to_keep.extend(kept)
            files_to_delete.extend(to_delete)

        if not files_to_delete:
            print("\nNo files to delete.")
            return

        # Display all matched files grouped by folder, in numeric order, with status emoji.
        all_files: list[tuple[Path, int]] = files_to_keep + files_to_delete
        all_files.sort(key=lambda x: (str(x[0].parent), x[1]))
        kept_paths_set = {p for p, _ in files_to_keep}

        print()
        current_folder: str = ""
        for f, _num in all_files:
            folder_str = str(f.parent)
            if folder_str != current_folder:
                current_folder = folder_str
                print(f"\n  [{folder_str}]")
            emoji = "✅" if f in kept_paths_set else "🗑️"
            print(f"    {emoji} {f.name}")

        # Summary counts.
        print(f"\n{'='*60}")
        print(f"Total files to keep:                {len(files_to_keep)}")
        print(f"Total files to move to recycle bin: {len(files_to_delete)}")

        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm == "y":
            break
        print("\nRe-entering settings...\n")

    print(f"Sending {len(files_to_delete)} file(s) to recycle bin in one batch...")
    send2trash([str(p) for p, _ in files_to_delete])
    print(f"Done. {len(files_to_delete)} file(s) moved to recycle bin.")


if __name__ == "__main__":
    main()