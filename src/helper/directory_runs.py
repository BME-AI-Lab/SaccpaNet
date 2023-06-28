import argparse
import filecmp
import os
import pathlib
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from os.path import join
from tkinter import filedialog

from tqdm import tqdm


def get_temp_directory():
    work_dir = tempfile.mkdtemp()
    return work_dir


def sync(source_dir, work_dir):
    env = {}
    env.update(os.environ)
    subprocess.run(
        ["rclone", "sync", source_dir, work_dir], shell=True, env=env, check=True
    )


def patch(source_dir, work_dir):
    env = {}
    env.update(os.environ)
    subprocess.run(
        ["rclone", "copy", source_dir, work_dir], shell=True, env=env, check=True
    )


def copy_diff(work_dir, source_dir, stored_run):
    env = {}
    env.update(os.environ)
    subprocess.run(
        ["rclone", "copy", work_dir, stored_run, "--compare-dest", source_dir],
        shell=True,
        env=env,
        check=True,
    )


@contextmanager
def remember_cwd():
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)


def execute(work_dir, command):
    with remember_cwd():
        os.chdir(work_dir)
        env = {}
        env.update(os.environ)
        print(command)
        subprocess.run(command, shell=True, env=env, cwd=work_dir)


def ask_if_none(arg_name, default, file=False):
    if default is None or not default:
        if file:
            default = filedialog.askopenfilename(title=f"{arg_name}")
        else:  # directory
            default = filedialog.askdirectory(title=f"{arg_name}")

    assert not default is None
    return default


def patch_tqdm():
    from .notification import get_telegram_tqdm

    global tqdm
    tqdm = get_telegram_tqdm()


def resolve_arguments(args):
    arg_dict = vars(args)
    if args.notify:
        patch_tqdm()
    work_dir, source_dir, run_dir, store_dir = (
        ask_if_none(arg_name, arg_dict[arg_name])
        for arg_name in ["work_dir", "source_dir", "run_dir", "store_dir"]
    )
    command = ask_if_none("command", args.command, file=True)
    assert len(command)
    return work_dir, source_dir, run_dir, store_dir, command


def main(args):
    work_dir, source_dir, run_dir, store_dir, command = resolve_arguments(args)
    # if not work_dir is specified.
    if not work_dir:
        work_dir = get_temp_directory()
    # make dir if work_dir do not exists
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)

    # TBD:assert file path disjoint
    runs = get_run(run_dir)
    while runs:
        run = runs.pop()
        print(run)
        sync(source_dir, work_dir)
        patch(run, work_dir)
        execute(work_dir, command)
        stored_run = join(store_dir, os.path.basename(run))
        copy_diff(work_dir, source_dir, stored_run)
        add_ignore(run_dir, run)
        runs = get_run(run_dir)


def get_ignore(run_dir):
    path = os.path.join(run_dir, "ignore.txt")
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        return [i[:-1] if i[-1] == "\n" else i for i in f.readlines()]


def add_ignore(run_dir, run):
    with open(os.path.join(run_dir, "ignore.txt"), "a") as f:
        f.write(f"{os.path.basename(run)}\n")


def get_run(run_dir):
    l = list()
    all_files = os.listdir(run_dir)
    ignores = set(get_ignore(run_dir))
    filter_runs = all_files
    filter_runs = filter(lambda x: not x in ignores, filter_runs)
    filter_runs = map(lambda x: os.path.join(run_dir, x), filter_runs)
    filter_runs = filter(lambda x: os.path.isdir(x), filter_runs)
    for i in filter_runs:
        l.append(os.path.join(run_dir, i))
    return l


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize runs in directories")
    parser.add_argument("---run_dir", type=pathlib.Path)
    parser.add_argument("--source_dir", type=pathlib.Path)
    parser.add_argument("--work_dir", type=pathlib.Path)
    parser.add_argument("--store_dir", type=pathlib.Path)
    parser.add_argument("-n", "--notify", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main(args)
