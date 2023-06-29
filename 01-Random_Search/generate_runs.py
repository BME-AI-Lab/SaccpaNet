import shutil


def copy_and_write(representation, params, config_file_name="search_configs.py"):
    shutil.copytree("codes", f"runs/{representation}", dirs_exist_ok=True)
    with open(f"runs/{representation}/{config_file_name}", "w") as f:
        f.write(params)


def search_range():
    from lib.modules.core.sampler import sample_cfgs

    SEED = 0
    SAMPLE_SIZE = 32
    SAMPLES = sample_cfgs(seed=SEED, sample_size=SAMPLE_SIZE)
    for PARAM_NAME, params in SAMPLES.items():
        representation = f"{PARAM_NAME}"
        params = (
            f"seed = {SEED}\n"
            + f"params = {params}\n"
            + f"PARAM_NAME = '{PARAM_NAME}'\n"
        )
        copy_and_write(representation, params)


if __name__ == "__main__":
    search_range()
