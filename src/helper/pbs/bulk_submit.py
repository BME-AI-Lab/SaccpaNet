"""Submit all runs to PBS cluster at once"""
import glob
import os
from os import path

for i in glob.glob("runs/*/run.pbs"):
    s = f"tmux new-window -c '{path.dirname(i)}' -n '{path.dirname(i)}' 'chmod +x {path.basename(i)} ; qsub -d . -I \"{path.abspath(i)}\" -x ; bash -i'"
    print(s)
    os.system(s)
