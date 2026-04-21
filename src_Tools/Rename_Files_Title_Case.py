import sys
import os
from pathlib import Path

# Add library path
sys.path.append(r"E:\My_Program\Python_Lib\src")

from Python_Lib.My_Lib_Stock import get_input_with_while_cycle, title_capitalization

def main():
    print("---------------------------------------------------------")
    print("Interactive File/Folder Renamer (Title Case)")
    print("---------------------------------------------------------")
    print("Please paste the full paths of files or folders you want to rename.")
    print("Paste multiple paths (one per line).")
    print("When finished, press Enter on an empty line to start processing.")
    print("---------------------------------------------------------")
    
    paths = get_input_with_while_cycle(input_prompt="Path > ")
    
    if not paths:
        print("No paths provided.")
        return

    for path_str in paths:
        path_str = path_str.strip('"').strip("'")
        if not path_str:
            continue
            
        path = Path(path_str)
        
        if not path.exists():
            print(f"Skipping: {path} (Not found)")
            continue
        
        parent = path.parent
        name = path.name
        
        # Apply title capitalization
        new_name = title_capitalization(name)
        
        if new_name == name:
            print(f"Skipping: {name} (Already in title case)")
            continue
            
        new_path = parent / new_name
        
        # Handle collision if needed (though on Windows case-insensitive fs it might be tricky if names only differ by case)
        # On Windows, renaming "file.txt" to "File.txt" works directly usually.
        
        try:
            os.rename(path, new_path)
            print(f"Renamed: '{name}' -> '{new_name}'")
        except Exception as e:
            print(f"Error renaming '{name}': {e}")

if __name__ == "__main__":
    main()
