# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import importlib.util
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


def get_unused_filename(input_filename, replace_hash=False, continue_number_suffix=False):
    """
    verify whether the filename is already exist, if it is, a filename like filename_01.append; filename_02.append will be returned.
    maximum 99 files can be generated

    With continue_number_suffix=True, a stem that already ends in _<number> continues
    counting from there instead of stacking a second suffix (Job_03.gjf -> Job_04.gjf,
    not Job_03_01.gjf; an unpadded legacy Job_3.gjf also continues as Job_04.gjf),
    numbers beyond 99 simply grow to three digits and more, and the path is used
    exactly as given (no os.path.realpath). Only existence checks are performed —
    nothing is created, modified or overwritten. The default behavior is unchanged.

    :param input_filename:
    :param replace_hash
    :param continue_number_suffix:
    :return: a filename
    """

    if continue_number_suffix:
        input_filename = str(input_filename)
        if not os.path.exists(input_filename):
            return input_filename

        parent, name = os.path.split(input_filename)
        stem, extension = os.path.splitext(name)
        number_match = re.match(r"(.*)_(\d+)$", stem)
        if number_match:
            stem, number = number_match.group(1), int(number_match.group(2)) + 1
        else:
            number = 1
        while True:
            candidate = os.path.join(parent, f"{stem}_{number:02d}{extension}")
            if not os.path.exists(candidate):
                return candidate
            number += 1

    input_filename = os.path.realpath(input_filename)

    if not os.path.exists(input_filename):
        # 是新的
        return input_filename
    else:
        if os.path.isfile(input_filename):
            no_append = filename_class(input_filename).only_remove_append
            append = filename_class(input_filename).append
        else:
            no_append = input_filename
            append = ""

        number = 1
        ret = no_append + "_" + '{:0>2}'.format(number) + (('.' + append) if append else "")
        while os.path.exists(ret):
            number += 1
            ret = no_append + "_" + '{:0>2}'.format(number) + (('.' + append) if append else "")

        return ret


def format_size(num_bytes: float | None) -> str:
    """把字节数格式化成人类可读的大小字符串（B / KB / MB / GB / TB）；None 返回空字符串。"""
    if num_bytes is None:
        return ""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def find_my_program_root() -> str:
    """从本文件所在目录向上查找名为 'My_Program' 的目录，返回其路径。

    本仓库约定整体位于 My_Program 目录之下（本地工作机与集群部署皆然），
    因此从本文件出发与从仓库内任何文件出发结果相同。
    """
    current = pathlib.Path(__file__).resolve().parent
    while True:
        if current.name == "My_Program":
            return str(current)
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                "Could not find a parent directory named 'My_Program' "
                f"starting from: {pathlib.Path(__file__).resolve()}"
            )
        current = parent


# ---------------------------------------------------------------------------
# 用户级配置（<My_Program 根目录>/My_Lib_Configuration_Private.py）
# ---------------------------------------------------------------------------

USER_CONFIGURATION_FILENAME = "My_Lib_Configuration_Private.py"

_user_configuration_module = None


def load_user_configuration():
    """加载用户级配置模块 ``<My_Program 根目录>/My_Lib_Configuration_Private.py``。

    配置文件存放跟随本机而非跟随仓库的设置（例如本地 → 远程路径映射），位于
    版本控制之外；文件名带 Private 后缀，即使误入仓库也会被 .gitignore 的
    ``*[Pp][Rr][Ii][Vv][Aa][Tt][Ee]*`` 模式兜底排除。结果在进程内缓存，只加载
    一次。变量按需读取：本函数不要求所有变量都存在，调用方读不到自己需要的
    变量时自行报错。

    Raises:
        FileNotFoundError: 配置文件不存在。报错信息给出应创建的完整路径。
    """
    global _user_configuration_module
    if _user_configuration_module is not None:
        return _user_configuration_module
    configuration_path = os.path.join(find_my_program_root(), USER_CONFIGURATION_FILENAME)
    if not os.path.isfile(configuration_path):
        raise FileNotFoundError(
            f"User configuration file not found: {configuration_path}\n"
            "Create it and define the variables the caller needs "
            "(e.g. LOCAL_TO_REMOTE_PATH_MAPPINGS for local-to-remote path mapping).")
    specification = importlib.util.spec_from_file_location("My_Lib_User_Configuration",
                                                           configuration_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    _user_configuration_module = module
    return module


def local_to_remote_path_mappings() -> list[tuple[str, str, str]]:
    """用户配置的本地 → 远程路径映射表 ``LOCAL_TO_REMOTE_PATH_MAPPINGS``。

    每条映射为一个三元组 ``(本地工作路径前缀, 远程工作路径前缀, 远程 RWF 路径
    前缀)``；远程前缀使用 ``%HOME%`` 占位符，由 HPC_Lib 在提交时统一替换为各
    集群配置的 HOME_PATH。命中规则见 :func:`map_local_path_to_remote`。

    配置文件或其中的 ``LOCAL_TO_REMOTE_PATH_MAPPINGS`` 缺失时直接报错——路径
    映射决定生成的输入会写向哪里，不提供任何隐含默认值。
    """
    configuration = load_user_configuration()
    mappings = getattr(configuration, "LOCAL_TO_REMOTE_PATH_MAPPINGS", None)
    if mappings is None:
        raise AttributeError(
            "LOCAL_TO_REMOTE_PATH_MAPPINGS is not defined in the user configuration: "
            f"{configuration.__file__}")
    return mappings


def local_path_prefix_pattern(local_prefix: str) -> "re.Pattern[str]":
    """本地路径前缀 → 匹配它的正则：两种斜杠写法均可、大小写不敏感；要求前缀
    之后是路径分隔符或结尾、之前不是字母（避免把 "NOTE:" 里的 "E:" 之类
    普通文本误认成盘符）。"""
    prefix_regex = re.escape(local_prefix).replace(re.escape("\\"), r"[\\/]")
    return re.compile(r"(?<![A-Za-z])" + prefix_regex + r"(?=[\\/]|$)", re.IGNORECASE)


def map_local_path_to_remote(
    local_path: str,
    mappings: list[tuple[str, str, str]] | None = None,
) -> tuple[str, str] | None:
    """按 *mappings* 把一个本地路径映射为 ``(远程工作路径, 远程 RWF 路径)``。

    *mappings* 为 ``None``（默认）时读用户配置（见
    :func:`local_to_remote_path_mappings`，配置缺失直接报错）。命中规则：前缀
    大小写不敏感、两种斜杠写法均可命中；前缀之后必须是路径分隔符或字符串结尾
    （因此 ``D:\\Gaussian_Other`` 不会命中 ``D:\\Gaussian``）。按列表顺序取第
    一条命中的映射——更具体的前缀必须排在前面。

    没有任何前缀命中时返回 ``None``。返回的两个路径都已把反斜杠换成正斜杠。
    """
    if mappings is None:
        mappings = local_to_remote_path_mappings()
    for local_prefix, remote_work_prefix, remote_rwf_prefix in mappings:
        match = local_path_prefix_pattern(local_prefix).match(local_path)
        if match:
            remainder = local_path[match.end():].replace("\\", "/")
            return remote_work_prefix + remainder, remote_rwf_prefix + remainder
    return None


def _remote_path_prefix_pattern(remote_prefix: str) -> "re.Pattern[str]":
    """远程（POSIX）路径前缀 → 匹配它的正则：大小写敏感（Linux 文件系统），
    要求前缀之后是 ``/`` 或结尾（因此 ``~/Gaussian_Other`` 不会命中
    ``~/Gaussian``）。"""
    return re.compile(re.escape(remote_prefix.rstrip("/")) + r"(?=/|$)")


def map_remote_path_to_remote_rwf(
    remote_path: str,
    home_path: str,
    mappings: list[tuple[str, str, str]] | None = None,
) -> str | None:
    """按映射表把一个**远程**路径从工作前缀换到 RWF 前缀。

    与 :func:`map_local_path_to_remote`（本地 → 远程方向）相对的另一个方向：
    对每条映射三元组 ``(本地前缀, 远程工作前缀, 远程 RWF 前缀)``，把第 2、3
    元素里的 ``%HOME%`` 用 *home_path* 展开，再对 *remote_path* 做前缀匹配
    （大小写敏感、边界感知：前缀之后必须是 ``/`` 或结尾）。按列表顺序取第一条
    命中的映射，把工作前缀替换为 RWF 前缀返回。

    HPC_Lib 的 ORCA 提交用它决定运行（RWF）目录的位置：例如输入在
    ``<home>/Gaussian/my_project/`` 之下、映射为
    ``("D:\\Gaussian", "%HOME%/Gaussian", "%HOME%/Gaussian_RWF")`` 时，运行
    目录树落在 ``<home>/Gaussian_RWF/my_project/`` 之下。

    Args:
        remote_path:  远程侧的绝对路径（POSIX 风格）。
        home_path:    该集群的 home 目录绝对路径（用于展开 ``%HOME%``）。
        mappings:     ``None``（默认）时读用户配置（见
                      :func:`local_to_remote_path_mappings`，配置缺失直接报错）。

    Returns:
        替换前缀后的 RWF 路径；没有任何前缀命中时返回 ``None``。调用方拿到
        ``None`` 时应当自行报错——路径映射决定文件写向哪里，不提供任何隐含
        默认值。
    """
    if mappings is None:
        mappings = local_to_remote_path_mappings()
    home_path = home_path.rstrip("/")
    for _local_prefix, remote_work_prefix, remote_rwf_prefix in mappings:
        work_prefix = remote_work_prefix.replace("%HOME%", home_path)
        rwf_prefix = remote_rwf_prefix.replace("%HOME%", home_path)
        match = _remote_path_prefix_pattern(work_prefix).match(remote_path)
        if match:
            return rwf_prefix.rstrip("/") + remote_path[match.end():]
    return None


def map_remote_path_to_local(
    remote_path: str,
    home_path: str | None,
    mappings: list[tuple[str, str, str]] | None = None,
) -> str | None:
    """按映射表把一个**远程**路径映射为对应的本地（Windows）路径。

    :func:`map_local_path_to_remote` 的反方向。*remote_path* 接受三种 home
    写法：集群 home 目录的绝对路径、``~``、字面的 ``%HOME%``（后两种需要
    提供 *home_path* 才能展开）。对每条映射三元组 ``(本地前缀, 远程工作
    前缀, 远程 RWF 前缀)``，把**工作前缀与 RWF 前缀**里的 ``%HOME%`` 用
    *home_path* 展开后分别做前缀匹配（大小写敏感——Linux 文件系统；边界
    感知：前缀之后必须是 ``/`` 或结尾），同一条映射里两个前缀都命中时取
    展开后更长的那个。按列表顺序取第一条命中的映射——更具体的前缀必须
    排在前面，与本地 → 远程方向一致。

    RWF 前缀也参与反向匹配是因为 RWF 树在本地没有独立的对应目录：远程
    RWF 树与远程工作树平行、都由同一个本地前缀派生（见
    :func:`map_local_path_to_remote`），所以一个远程 RWF 路径的本地对应
    位置就是同一条映射的本地工作副本位置。例如映射
    ``("D:\\Gaussian", "%HOME%/Gaussian", "%HOME%/Gaussian_RWF")`` 下，
    ``<home>/Gaussian_RWF/sub/x.out`` 映射为 ``D:\\Gaussian\\sub\\x.out``，
    而不是落到更靠后的宽泛映射（如 ``E: ⇄ %HOME%``）算出一个并不存在的
    ``E:\\Gaussian_RWF\\...``。

    Args:
        remote_path:  远程侧路径（POSIX 风格；反斜杠会先换成正斜杠）。
        home_path:    该集群的 home 目录绝对路径；``None`` 时不展开
                      ``~`` / ``%HOME%``（工作前缀里含 ``%HOME%`` 的映射
                      条目因此不可能命中绝对路径）。
        mappings:     ``None``（默认）时读用户配置（见
                      :func:`local_to_remote_path_mappings`，配置缺失直接
                      报错）。

    Returns:
        对应的本地路径（分隔符为反斜杠）；没有任何前缀命中时返回 ``None``。
    """
    if mappings is None:
        mappings = local_to_remote_path_mappings()
    normalized = remote_path.replace("\\", "/")
    if home_path:
        home_path = home_path.rstrip("/")
        if normalized == "~" or normalized.startswith("~/"):
            normalized = home_path + normalized[1:]
        normalized = normalized.replace("%HOME%", home_path)
    for local_prefix, remote_work_prefix, remote_rwf_prefix in mappings:
        expanded_prefixes = []
        for remote_prefix in (remote_work_prefix, remote_rwf_prefix):
            if home_path:
                remote_prefix = remote_prefix.replace("%HOME%", home_path)
            expanded_prefixes.append(remote_prefix)
        # 同一条映射里两个前缀都可能命中时（例如 RWF 前缀嵌在工作前缀
        # 之下），更长（更具体）的优先。
        for remote_prefix in sorted(set(expanded_prefixes),
                                    key=len, reverse=True):
            match = _remote_path_prefix_pattern(remote_prefix).match(normalized)
            if match:
                remainder = normalized[match.end():]
                return (local_prefix.rstrip("\\/")
                        + remainder.replace("/", "\\"))
    return None
