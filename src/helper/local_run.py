"""Run all the runs locally in the folder """
import glob
import os
from os import path
import argparse
from time import sleep

from tqdm import tqdm
from .notification import get_telegram_tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Run all the runs locally")
    parser.add_argument("-f", "--folder", help="Folder to run", default="runs")
    parser.add_argument("-s", "--script", help="Script to run", default="run.pbs")
    parser.add_argument(
        "-n", "--notify", help="Notify the progress", action="store_true"
    )
    args = parser.parse_args()
    folder = args.folder
    script = args.script
    if args.notify:
        tqdm = get_telegram_tqdm()

    # the path means "{folder group to search for}/{specific run name}/{starting script for PBS}"
    # TBD: refactor the code without python script hard coded

    all_runs = list(glob.glob(f"{folder}/*/{script}"))
    all_runs.sort()

    progress = tqdm(all_runs, desc=f"{os.getlogin()}")
    if args.notify:
        progress = tqdm(all_runs, desc=f"{os.getlogin()}")
    for i in progress:
        s = f"cd {path.abspath(path.dirname(i))} && python {script}"
        progress.set_description(f"{os.getlogin()}|{path.dirname(i)}")
        print(s)
        os.system(s)
