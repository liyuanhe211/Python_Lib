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
    Base_folder = r"E:\My_Program"
    Base_repo = r"E:\My_Program\Python_Lib"
    for project in list_current_folder(Base_folder):
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
