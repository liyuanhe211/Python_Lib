# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

# import sys
# import pathlib
# parent_path = str(pathlib.Path(__file__).parent.resolve())
# sys.path.insert(0,parent_path)
import os.path
import git

from My_Lib_Stock import *

if __name__ == '__main__':
    MAIN_FOLDER = r"E:\My_Program"
    BASE_LIB = r"E:\My_Program\Python_Lib"
    exclude_folders = ["E:\My_Program\Gaussian_Toolkit_Backup"]
    project_holders = [MAIN_FOLDER,
                      os.path.join(MAIN_FOLDER,"0_Machine_Learning")]

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
                subprocess.call('git pull')
                print("\n>>>git stash pop\n")
                subprocess.call('git stash pop')
                print("-------------Pull finished------------\n\n\n\n")
