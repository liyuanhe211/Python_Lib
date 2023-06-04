# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import unittest
from My_Lib_Stock import *
from My_Lib_Office import read_xlsx

# (test_input, path, name, name_stem, append, replace_append_to('z'), replace_append_to("x.y"), insert_append("x.y"))
test_cases = read_xlsx("Filename_functions_test_cases.xlsx")[1:]
inputs = [x[0] for x in test_cases]
functions = [
    filename_parent,
    filename_name,
    filename_stem,
    filename_last_append,
    filename_remove_append,
    filename_full_append,
    lambda x:replace_last_append(x,'x.y'),
    lambda x:insert_append(x,'x.y')]

for j in inputs:
    for i in functions:
        print(j,"-->",repr(i(j)),"    ",i.__name__.removeprefix("filename_"))
    print("-----------------------------------------")


class test_filename_class_function(unittest.TestCase):
    def test_filename_class(self):
        for test_case in test_cases:
            text = test_case[0]
            filename_object = filename_class(text)
            test_functions = (filename_object.path,
                              filename_object.name,
                              filename_object.name_stem,
                              filename_object.append,
                              filename_object.only_remove_append,
                              filename_object.replace_append_to('z'),
                              filename_object.replace_append_to("x.y"),
                              filename_object.insert_append("x.y"))
            for count, function in enumerate(test_functions):
                self.assertEqual(function, test_case[count + 1])

    def test_filename_functions(self):
        pass


unittest.main()
