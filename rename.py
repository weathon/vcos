import os
import glob

# Path to the root folder containing subfolders
root_folder = "davis2017"

# Iterate through all subfolders and files
for filepath in glob.glob(f"{root_folder}/**/*.jpg", recursive=True):
    folder, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    
    # Ensure the filename is numeric and has 4 digits
    if name.isdigit() and len(name) == 4:
        # Rename to 5 digits with leading zeros
        new_name = name.zfill(5) + ext
        new_filepath = os.path.join(folder, new_name)
        
        # Rename the file
        os.rename(filepath, new_filepath)
        print(f"Renamed: {filepath} -> {new_filepath}")