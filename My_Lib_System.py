# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import sys
import pathlib
Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Stock import *
import platform

# Cross-platform keyboard detection
if platform.system() == 'Windows':
    import msvcrt
else:
    import select
    import termios
    import tty

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


# Global flag to control keyboard interrupt detection
_keyboard_interrupt_enabled = True

def enable_keyboard_interrupt():
    """Enable keyboard interrupt detection."""
    global _keyboard_interrupt_enabled
    _keyboard_interrupt_enabled = True

def disable_keyboard_interrupt():
    """Disable keyboard interrupt detection (useful before calling input())."""
    global _keyboard_interrupt_enabled
    _keyboard_interrupt_enabled = False

def non_block_keyboard_interrupt(key = "C"):
    """
    Cross-platform keyboard detection for 'C' key press.
    Returns True if 'C' is pressed, False otherwise.
    Non-blocking on all platforms.
    
    Note: On Unix/Linux/Mac, this uses non-canonical mode which won't interfere with input().
    However, it's recommended to call disable_keyboard_interrupt() before input() and
    enable_keyboard_interrupt() after input() to be extra safe.
    """
    global _keyboard_interrupt_enabled
    
    if not _keyboard_interrupt_enabled:
        return False
    
    if platform.system() == 'Windows':
        # Windows implementation
        if msvcrt.kbhit():
            pressed_key = msvcrt.getch()
            return pressed_key.decode().lower() == key.lower()
        return False
    else:
        # Unix/Linux/Mac implementation
        # Check if there's input available without blocking
        if select.select([sys.stdin], [], [], 0)[0]:
            # Save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                pressed_key = sys.stdin.read(1)
                return pressed_key.lower() == key.lower()
            finally:
                # Restore terminal settings
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        return False

if __name__ == '__main__':
    pass
