import shutil


def copy_and_write(representation, params):

    # shutil.copytree(
    #     f"../06-Posture_Classification/retire/{representation}",
    #     f"runs/{representation}",
    #     dirs_exist_ok=True,
    # )
    shutil.copytree("codes", f"runs/{representation}", dirs_exist_ok=True)
    with open(f"runs/{representation}/search_configs.py", "w") as f:
        f.write(params)


MODELS = [
    "SACCPA_sample_ablation_A",
    "SACCPA_sample_ablation_AB",
    "SACCPA_sample_ablation_ABC",
    "SACCPA_sample_ablation_null",
]
# MODELS = ["vit_l_32", "vit_l_16", "vit_h_14", "vit_b_16", "vit_b_32"]


def search_range():
    for models in MODELS:
        representation = f"{models}"
        params_str = f"MODEL_NAME = '{models}'\n"
        copy_and_write(representation, params_str)


if __name__ == "__main__":
    search_range()
