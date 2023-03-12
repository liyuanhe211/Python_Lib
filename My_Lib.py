# -*- coding: utf-8 -*-

# A backward compatibility module for older programs. Newer programs will only import some of the divided files.
# Test2

__author__ = 'LiYuanhe'

import sys

import pathlib

Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Stock import *
from My_Lib_Chemistry import *
from My_Lib_Color import *
from My_Lib_Image import *
from My_Lib_Network import *
from My_Lib_Office import *
from My_Lib_Science import *
from My_Lib_System import *
from My_Lib_Video import *

pass

if __name__ == "__main__":
    pass
