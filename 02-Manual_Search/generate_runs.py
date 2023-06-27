import shutil
from configs.random_searched_params import params
from copy import deepcopy


def copy_and_write(representation, params):
    shutil.copytree("codes", f"runs/{representation}", dirs_exist_ok=True)
    with open(f"runs/{representation}/search_configs.py", "w") as f:
        f.write(params)


def search_range():
    for W0 in [48, 64, 72, 80, 88, 96, 104, 112, 120]:
        new_params = deepcopy(params)  # prevent acidentally modifying
        new_params.update({"REGNET.W0": W0})
        representation = "_".join([f"{k}{v}" for k, v in new_params.items()])
        new_params_str = f"params=" + repr(new_params)
        copy_and_write(representation, new_params_str)


if __name__ == "__main__":
    search_range()
