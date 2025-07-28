# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import pathlib
import os
import re

# Obsolete
class filename_class:
    def __init__(self, fullpath):
        fullpath = fullpath.replace('\\', '/')
        self.depth = fullpath.count('/')
        self.re_path_temp = re.match(r".+/", fullpath)
        if self.re_path_temp:
            self.path = self.re_path_temp.group(0)  # 包括最后的斜杠
        else:
            self.path = ""
        self.name = fullpath[len(self.path):]
        if self.name.rfind('.') != -1:
            self.name_stem = self.name[:self.name.rfind('.')]  # not including "."
            self.append = self.name[len(self.name_stem) - len(self.name) + 1:]
        else:
            self.name_stem = self.name
            self.append = ""

        self.only_remove_append = self.path + self.name_stem  # not including "."

    def replace_append_to(self, new_append):
        return self.only_remove_append + '.' + new_append

    def insert_append(self, append_to_be_insert):
        return self.only_remove_append + '.' + append_to_be_insert + '.' + self.append


def filename_parent(file_path):
    """
    Return "" if there is no parent, not including trailing slash
    """
    path = pathlib.Path(file_path)
    if len(path.parts)==1:
        return ""
    return str(path.parent)


def filename_last_append(file_path):
    """
    Not including dot
    """
    path = pathlib.Path(file_path)
    return path.suffix.lstrip(".")


def filename_full_append(file_path):
    """
    Not including first dot
    """
    path = pathlib.Path(file_path)
    return "".join(path.suffixes).lstrip(".")


def filename_name(file_path):
    path = pathlib.Path(file_path)
    return path.name


def filename_remove_append(file_path):
    """
    Remove all append and the one trailing dot.
    """
    path = pathlib.Path(file_path)
    ret = path.with_suffix("")
    while ret != ret.with_suffix(""):
        ret = ret.with_suffix("")
    return str(ret)


def filename_stem(file_path):
    """
    Remove path and all append
    """
    return filename_remove_append(filename_name(file_path))


def replace_last_append(file_path, new_append: str):
    """
    Parameters:
        new_append: Can be with or without the . in the front
    """
    path = pathlib.Path(file_path)
    new_append = "." + new_append if not new_append.startswith(".") else new_append
    return str(path.with_suffix(new_append))

filename_replace_last_append = replace_last_append

def insert_append(file_path, new_append):
    """
    Parameters:
        new_append: Can be with or without the . in the front
    """
    new_append = "." + new_append if not new_append.startswith(".") else new_append
    path = pathlib.Path(file_path)
    return f"{path.with_suffix('')}{new_append}{path.suffix}"


replace_append = replace_last_append


def proper_filename(input_filename, including_appendix=True, path_as_filename=False, replace_hash=True, replace_dot=True, replace_space=True):
    """

    :param input_filename:
    :param including_appendix:
    :param path_as_filename: 是否将路径转换为文件名(/home/username/file.txt --> __home__username__file.txt )
    :param replace_hash:
    :param replace_dot:
    :param replace_space:
    :return:
    """
    if path_as_filename:
        path = ""
        filename_stem = filename_class(input_filename).only_remove_append
    else:
        path = filename_class(input_filename).path
        filename_stem = filename_class(input_filename).name_stem
    append = filename_class(input_filename).append

    # remove illegal characters of filename
    forbidden_chr = "<>:\"'/\\|?*-\n. "
    if not replace_hash:
        forbidden_chr = forbidden_chr.replace('-', '')
    if not replace_space:
        forbidden_chr = forbidden_chr.replace(' ', '')
    if not replace_dot:
        forbidden_chr = forbidden_chr.replace('.', '')
    for character in forbidden_chr:
        filename_stem = filename_stem.replace(character, '__')

    if append:
        if including_appendix:
            ret = filename_stem + '.' + append
        else:
            ret = filename_stem + '_' + append
    else:
        ret = filename_stem

    while "____" in ret:
        ret = ret.replace('____', "__")

    return os.path.join(path, ret)


filename_filter = proper_filename


def walk_all_files(parent=".", glob_filter="*.*", return_pathlib_obj=False):
    """
    os.walk() wrap, return list of str for the full path
    :param parent:
    :param glob_filter:
    :param return_pathlib_obj: Whether to return a Path object, if False, return str
    """
    import pathlib
    parent_folder = pathlib.Path(parent)
    if return_pathlib_obj:
        files = [x.resolve() for x in parent_folder.rglob(glob_filter)]
    else:
        files = [str(x.resolve()) for x in parent_folder.rglob(glob_filter)]
    return files


def list_folder_content(parent=".", filter="*", return_pathlib_obj=False):
    """
    return list of str for the full path of both files and folders
    :param parent:
    :param filter:
    :param return_pathlib_obj: Whether to return a Path object, if False, return str
    """
    import pathlib
    # print(parent)
    parent_pathlib_object = pathlib.Path(parent)
    if return_pathlib_obj:
        files = [x.resolve() for x in parent_pathlib_object.glob(filter)]
    else:
        files = [str(x.resolve()) for x in parent_pathlib_object.glob(filter)]
    # print(files)
    return files


list_current_folder = list_folder_content


def file_is_busy(filepath):
    """
    Check whether a file is being used
    If it's not being used, or it doesn't exist, return False
    Else return True
    If any other exceptions occur, raise exception
    """
    import os
    if os.path.isfile(filepath):
        try:
            os.rename(filepath, filepath)
            return False
        except OSError:
            return True
    else:
        return False
