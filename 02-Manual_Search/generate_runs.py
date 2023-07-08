import shutil
from copy import deepcopy

from configs.random_searched_params import params
from lib.modules.core.sampler import (
    check_complexity_constraints,
    check_regnet_constraints,
)


def copy_and_write(representation, params):
    shutil.copytree("codes", f"runs/{representation}", dirs_exist_ok=True)
    with open(f"runs/{representation}/search_configs.py", "w") as f:
        f.write(params)


def search_range():
    for WA in [12.6, 12.7, 12.8, 12.9, 13.0]:
        for WM in [2.9, 2.92, 2.94, 2.96, 2.98, 3]:  # range(72, 250, 16)
            new_params = deepcopy(params)  # prevent acidentally modifying
            new_params.update(
                {
                    "REGNET.WM": WM,
                    "REGNET.WA": WA,
                }
            )
            PARAM_NAME = "_".join([f"{k}{v}" for k, v in new_params.items()])
            representation = f"{PARAM_NAME}"
            params_str = f"params = {new_params}\n" + f"PARAM_NAME = '{PARAM_NAME}'\n"
            # copy_and_write(representation, new_params_str)
            if not check_regnet_constraints(params) or not check_complexity_constraints(
                params
            ):
                print(representation)
            else:
                copy_and_write(representation, params_str)


if __name__ == "__main__":
    search_range()
