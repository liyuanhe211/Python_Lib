# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import sys
import pathlib
Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Stock import *


def screen_capture(output_filename):
    import pyautogui
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save(output_filename)


def hide_cwd_window():
    import ctypes
    import os
    import win32process

    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd != 0:
        ctypes.windll.user32.ShowWindow(hwnd, 0)
        ctypes.windll.kernel32.CloseHandle(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        os.system('taskkill /PID ' + str(pid) + ' /f')


def addToClipBoard(text):
    import pyperclip
    pyperclip.copy(text)


def process_is_CPU_idle(pid,interval=1):
    """

    :param pid:
    :param interval: sampling time, seconds
    :return: True if is idle, False if not, None if pid not exist
    """
    import psutil
    try:
        process = psutil.Process(pid)
        cpu_percent = process.cpu_percent(interval=interval)
        if cpu_percent == 0:
            return True
        return False
    except psutil.NoSuchProcess:
        return None

    
if __name__ == '__main__':
    pass
