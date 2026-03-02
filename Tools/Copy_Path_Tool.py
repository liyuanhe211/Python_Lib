import win32gui
import win32com.client
import pythoncom
import keyboard
import pyperclip
import os
import sys
import time
from datetime import datetime

# Log file path for debugging (optional, can be removed in production)
# LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "copy_tool_log.txt")

def log(message):
    pass
    # with open(LOG_FILE, "a") as f:
    #     f.write(f"{datetime.now()}: {message}\n")

def get_explorer_selection():
    try:
        pythoncom.CoInitialize()
        # Get the handle of the active window
        hwnd = win32gui.GetForegroundWindow()
        
        # Check if it is a file explorer window
        try:
            class_name = win32gui.GetClassName(hwnd)
        except Exception:
            pythoncom.CoUninitialize()
            return []

        if class_name != 'CabinetWClass' and class_name != 'ExploreWClass':
            log(f"Not Explorer window: {class_name}")
            pythoncom.CoUninitialize()
            return []

        shell = win32com.client.Dispatch("Shell.Application")
        for window in shell.Windows():
            try:
                # Compare HWNDs. Windows HWNDs can be large integers.
                if int(window.HWND) == int(hwnd):
                    selected_items = window.Document.SelectedItems()
                    paths = [item.Path for item in selected_items]
                    pythoncom.CoUninitialize()
                    return paths
            except Exception:
                # Some windows don't have Document or SelectedItems properties
                continue
        log("Window not found in Shell.Windows")
        pythoncom.CoUninitialize()
    except Exception as e:
        log(f"Error: {e}")
        try:
            pythoncom.CoUninitialize()
        except:
            pass
    return []

def copy_filenames():
    paths = get_explorer_selection()
    if paths:
        filenames = [os.path.basename(p) for p in paths]
        text_to_copy = "\n".join(filenames)
        pyperclip.copy(text_to_copy)
        log(f"Copied filenames: {text_to_copy}")

def copy_full_paths():
    paths = get_explorer_selection()
    if paths:
        text_to_copy = "\n".join(paths)
        pyperclip.copy(text_to_copy)
        log(f"Copied paths: {text_to_copy}")

def main():
    log("Started")
    # Register hotkeys
    keyboard.add_hotkey('ctrl+shift+c', copy_filenames)
    keyboard.add_hotkey('ctrl+alt+c', copy_full_paths)
    
    # Keep the script running
    keyboard.wait()

if __name__ == "__main__":
    main()
