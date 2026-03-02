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


def is_headless():
    """
    Detect whether the system is headless (cannot render GUI windows/plots).
    
    This checks if the system can render PyQt6 windows or matplotlib plots,
    not whether a human can see them. Even without a physical screen, if the
    system has a display server (X11, Wayland, Windows display), it's not headless.
    
    Returns:
        bool: True if the system is headless (cannot render GUI), False otherwise.
        
    Examples:
        >>> if is_headless():
        ...     print("Running in headless mode - use non-GUI backends")
        ... else:
        ...     print("GUI rendering available")
    """
    import os
    
    # Check OS-specific display availability
    system = platform.system()
    
    if system == 'Windows':
        # On Windows, check if we can get a display context
        try:
            import ctypes
            user32 = ctypes.windll.user32
            # Try to get the device context of the entire screen
            # If this returns None (0), there's no display available
            hdc = user32.GetDC(0)
            if hdc == 0:
                return True
            # Release the device context
            user32.ReleaseDC(0, hdc)
            return False
        except Exception:
            return True
            
    elif system == 'Linux' or system == 'Unix':
        # On Linux/Unix, check for DISPLAY environment variable
        display = os.environ.get('DISPLAY', '')
        wayland_display = os.environ.get('WAYLAND_DISPLAY', '')
        
        # If neither DISPLAY nor WAYLAND_DISPLAY is set, likely headless
        if not display and not wayland_display:
            return True
        
        # Try to verify X11 connection if DISPLAY is set
        if display:
            try:
                # Try to connect to X server
                import subprocess
                result = subprocess.run(
                    ['xdpyinfo'], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=2
                )
                # If xdpyinfo succeeds, X server is accessible
                if result.returncode == 0:
                    return False
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                # xdpyinfo not available or failed, try PyQt6 test
                pass
        
        # For Wayland or if xdpyinfo check didn't work, try PyQt6
        try:
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QCoreApplication
            # Check if QApplication can be instantiated
            # This will fail in true headless environments
            if not QCoreApplication.instance():
                # Create a test application (don't show any window)
                import sys
                app = QApplication(sys.argv)
                # If we got here, display is available
                app.quit()
                del app
            return False
        except Exception:
            # PyQt6 failed to initialize, likely headless
            return True
            
    elif system == 'Darwin':  # macOS
        # On macOS, check if WindowServer is available
        try:
            import subprocess
            # Check if we can connect to the WindowServer
            result = subprocess.run(
                ['pgrep', 'WindowServer'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2
            )
            if result.returncode == 0 and result.stdout:
                # WindowServer is running
                return False
            return True
        except Exception:
            # If check fails, try PyQt6 test
            try:
                from PyQt6.QtWidgets import QApplication
                from PyQt6.QtCore import QCoreApplication
                if not QCoreApplication.instance():
                    import sys
                    app = QApplication(sys.argv)
                    app.quit()
                    del app
                return False
            except Exception:
                return True
    
    # Unknown system, assume not headless to be safe
    return False


if __name__ == '__main__':
    pass
