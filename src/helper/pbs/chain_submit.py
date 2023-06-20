"""Scripts to submit jobs to the PBS cluster without overwheming the queue"""

import glob
import os
from os import path
import subprocess
from time import sleep

result = ""
for i in sorted(glob.glob("runs/*/run.pbs")):
    while True:
        try:
            result = subprocess.check_output(["qstat", "-i"])
        except Exception as e:
            print(e)
        if len(result) == 0:
            break
        sleep(30)
    s = f"tmux new-window -c '{path.dirname(i)}' -n '{path.dirname(i)}' 'chmod +x {path.basename(i)} ; qsub -d . -I \"{path.abspath(i)}\" -x ; bash -i'"
    print(s)
    os.system(s)
    sleep(10)
