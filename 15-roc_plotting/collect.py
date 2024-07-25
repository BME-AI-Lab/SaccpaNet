import glob
import os
import shutil
from os.path import *

all_fpr = glob.glob("**/all_fpr.npy", recursive=True)
mean_tpr = glob.glob("**/mean_tpr.npy", recursive=True)

for file in all_fpr:
    file = os.path.relpath(file)
    paths = file.split(os.sep)
    test_name = paths[-2]
    network_name = paths[1]
    print(test_name, network_name)
    target_file = f"all_fpr/{test_name}/{network_name}.npy"
    os.makedirs(dirname(target_file), exist_ok=True)
    # copy the file
    shutil.copyfile(file, target_file)

for file in mean_tpr:
    file = os.path.relpath(file)
    paths = file.split(os.sep)
    test_name = paths[-2]
    network_name = paths[1]
    print(test_name, network_name)
    target_file = f"mean_tpr/{test_name}/{network_name}.npy"
    os.makedirs(dirname(target_file), exist_ok=True)
    # copy the file
    shutil.copyfile(file, target_file)
