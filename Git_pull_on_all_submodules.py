# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

# import sys
# import pathlib
# parent_path = str(pathlib.Path(__file__).parent.resolve())
# sys.path.insert(0,parent_path)
import os.path
import git
import hashlib
import subprocess

from My_Lib_Stock import *

if __name__ == '__main__':
    MAIN_FOLDER = r"E:\My_Program"
    BASE_LIB = r"E:\My_Program\Python_Lib"
    exclude_folders = [r"E:\My_Program\Gaussian_Toolkit_Backup"]
    project_holders = [MAIN_FOLDER,
                      os.path.join(MAIN_FOLDER,"0_Machine_Learning")]

    failed_repos = []
    for main_folder in project_holders:
        for project in list_current_folder(main_folder):
            if project in exclude_folders:
                continue
            sub_repo_folder = os.path.join(project,'Python_Lib')
            if os.path.isdir(sub_repo_folder):
                os.chdir(sub_repo_folder)
                try:
                    _ = git.Repo(".").git_dir
                except git.exc.InvalidGitRepositoryError:
                    continue
                print(sub_repo_folder)
                print("\n>>>git stash\n")
                subprocess.call('git stash')
                
                print("\n>>>git pull\n")
                ret_pull = subprocess.call('git pull')
                
                print("\n>>>git stash pop\n")
                ret_pop = subprocess.call('git stash pop')
                
                if ret_pull != 0 or ret_pop != 0:
                    failed_repos.append(sub_repo_folder)

                print("-------------Pull finished------------\n\n\n\n")

    if failed_repos:
        print("\n\n" + "="*60)
        print("Merge Conflict Handling:")
        print("1. Iterated through all repositories.")
        print("2. Recorded repositories where 'git pull' or 'git stash pop' failed (returned non-zero exit code).")
        print("3. For each failed repository, a hash is calculated based on the content of files with merge conflicts.")
        print("   - Using 'git diff --name-only --diff-filter=U' to identify conflicted files.")
        print("   - Repositories with identical conflict states (same files, same content) will share the same hash.")
        print("4. This classification allows resolving the conflict in one instance and applying the fix to others in the same group.")
        print("   (Note: Use 'git diff' or 'git status' to inspect individual conflicts)")
        print("="*60 + "\n")

        conflict_map = {}
        for repo_path in failed_repos:
            try:
                os.chdir(repo_path)
                # Find unmerged files (files with conflicts)
                unmerged_files_bytes = subprocess.check_output('git diff --name-only --diff-filter=U', shell=True)
                unmerged_files = unmerged_files_bytes.decode('utf-8', errors='ignore').strip().splitlines()
                
                hasher = hashlib.md5()
                
                if not unmerged_files:
                    # If failed but no unmerged files found, hash git status
                    status_bytes = subprocess.check_output('git status', shell=True)
                    hasher.update(status_bytes)
                else:
                    # Hash the content of the conflicted files
                    for filename in unmerged_files:
                        filename = filename.strip()
                        if not filename: continue
                        hasher.update(filename.encode('utf-8', errors='ignore')) 
                        if os.path.exists(filename):
                            with open(filename, 'rb') as f:
                                hasher.update(f.read())
                
                repo_hash = hasher.hexdigest()
                
                if repo_hash not in conflict_map:
                    conflict_map[repo_hash] = []
                conflict_map[repo_hash].append(repo_path)
                
            except Exception as e:
                print(f"Error analyzing {repo_path}: {e}")

        print("\n\n" + "="*60)
        print("Repositories grouped by conflict hash:")
        for h, paths in conflict_map.items():
            print(f"\nHash: {h} (Count: {len(paths)})")
            for p in paths:
                print(f"  {p}")
