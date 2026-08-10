import os
import shutil

from Python_Lib.My_Lib_Stock import get_input_with_while_cycle

FOLDER_NAME_MAX_LENGTH = 100

print("Drag and drop files here (one per line, empty line to finish):")
paths = get_input_with_while_cycle(strip_quote=True)

for path in paths:
    path = path.strip()
    if not path:
        continue
    if not os.path.isfile(path):
        print(f"Skipped (not a file): {path}")
        continue

    filename = os.path.basename(path)
    name_no_ext = os.path.splitext(filename)[0]
    if len(name_no_ext) > FOLDER_NAME_MAX_LENGTH:
        truncated = name_no_ext[:FOLDER_NAME_MAX_LENGTH]
        last_space = truncated.rfind(" ")
        folder_name = (truncated[:last_space] if last_space != -1 else truncated).rstrip(". ")
    else:
        folder_name = name_no_ext

    parent_dir = os.path.dirname(path)
    dest_dir = os.path.join(parent_dir, folder_name)

    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    shutil.move(path, dest_path)
    print(f"Moved: {filename} -> {os.path.relpath(dest_path, parent_dir)}")

print("\nDone.")
