import os
import hashlib

# The paths to process
paths_str = r'''"F:\0 Send to Unraid\My_Program"
"F:\0 Send to Unraid\常用程序"'''

def calculate_hash(file_path):
    """
    Calculates the MD5 hash of a file.
    MD5 is used here as a good balance between speed and collision resistance,
    and it is available in the standard library.
    """
    hasher = hashlib.md5()
    # Read in chunks to avoid loading large files into memory
    chunk_size = 65536 # 64kb
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    # Parse the paths from the string
    # Split by newline, strip whitespace, and remove surrounding quotes
    paths = [line.strip().strip('"').strip("'") for line in paths_str.split('\n') if line.strip()]

    for folder_path in paths:
        # Normalize path separators for the current OS
        folder_path = os.path.normpath(folder_path)
        
        if not os.path.exists(folder_path):
            print(f"Skipping non-existent path: {folder_path}")
            continue

        print(f"Processing folder: {folder_path}")
        
        # Walk through the directory tree
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Skip existing .md5 files to avoid hashing the hashes
                if file.lower().endswith('.md5'):
                    continue

                file_path = os.path.join(root, file)
                md5_path = file_path + ".md5"

                # Skip if MD5 file already exists
                if os.path.exists(md5_path):
                    print(f"Skipping existing: {md5_path}")
                    continue

                # Calculate hash
                file_hash = calculate_hash(file_path)
                
                if file_hash:
                    try:
                        with open(md5_path, "w", encoding="utf-8") as f:
                            f.write(file_hash)
                        print(f"Created: {md5_path}")
                    except OSError as e:
                        print(f"Error writing {md5_path}: {e}")

if __name__ == "__main__":
    main()
