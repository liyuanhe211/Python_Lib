import os
import sys
import win32com.client

def create_startup_shortcut():
    startup_folder = os.path.join(os.getenv('APPDATA'), r'Microsoft\Windows\Start Menu\Programs\Startup')
    shortcut_path = os.path.join(startup_folder, "Copy_Path_Tool.lnk")
    
    # Path to the python executable (pythonw.exe to hide console)
    python_exe = sys.executable.replace("python.exe", "pythonw.exe")
    
    # Path to the script
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Copy_Path_Tool.py"))
    
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.TargetPath = python_exe
    shortcut.Arguments = f'"{script_path}"'
    shortcut.WorkingDirectory = os.path.dirname(script_path)
    shortcut.Description = "Copy Path Tool"
    shortcut.IconLocation = python_exe
    shortcut.save()
    
    print(f"Shortcut created at: {shortcut_path}")
    print("The tool will start automatically on next login.")
    print("You can also double-click the shortcut in the Startup folder to start it now.")

    # Ask if user wants to start it now
    if input("Do you want to start it now? (y/n): ").strip().lower() == 'y':
        os.startfile(shortcut_path)

if __name__ == "__main__":
    create_startup_shortcut()
