import os

def list_large_files(directory, size):
    large_files = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            try:
                file_path = os.path.join(foldername, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > size * 1024 * 1024 * 1024:  # size in bytes
                        large_files.append(file_path)
            except Exception as e:
                print(f"Error checking file {file_path}: {e}")
    return large_files

large_files = list_large_files("C:\\", 2)
for file in large_files:
    print(file)