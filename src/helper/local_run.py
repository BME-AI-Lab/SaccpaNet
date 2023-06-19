import glob
import os
from os import path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Run all the runs locally")
    parser.add_argument("-f", "--folder", help="Folder to run", default="runs")
    parser.add_argument("-s", "--script", help="Script to run", default="run.pbs")
    args = parser.parse_args()
    folder = args.folder
    script = args.script
    # the path means "{folder group to search for}/{specific run name}/{starting script for PBS}"
    # TBD: refactor the code without python script hard coded
    for i in reversed(sorted(glob.glob(f"{folder}/*/{script}"))):
        s = f"cd {path.abspath(path.dirname(i))} && python {script}"
        print(s)
        os.system(s)
