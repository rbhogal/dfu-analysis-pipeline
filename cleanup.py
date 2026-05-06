# cleanup.py
import os
import shutil

"""
Deletes outputs images and plots 
"""
dirs_to_delete = [
    "outputs/annotated_images",
    "outputs/plots",
]

for directory in dirs_to_delete:
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Deleted: {directory}")
    else:
        print(f"Already gone: {directory}")

print("Done — ready for next run.")
