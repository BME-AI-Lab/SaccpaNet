import shutil
from copy import deepcopy

from configs.random_searched_params import params


def copy_and_write(representation, params):
    shutil.copytree("codes", f"runs/{representation}", dirs_exist_ok=True)
    with open(f"runs/{representation}/search_configs.py", "w") as f:
        f.write(params)


def search_range():
    for DEPTH in [12.6, 12.7, 12.8, 12.9, 13.0]:
        for WM in [2.9, 2.92, 2.94, 2.96, 2.98]:  # range(72, 250, 16)
            new_params = deepcopy(params)  # prevent acidentally modifying
            new_params.update(
                {
                    "REGNET.WM": WM,
                    "REGNET.DEPTH": DEPTH,
                }
            )
            representation = "_".join([f"{k}{v}" for k, v in new_params.items()])
            new_params_str = f"params=" + repr(new_params)
            copy_and_write(representation, new_params_str)


if __name__ == "__main__":
    search_range()
