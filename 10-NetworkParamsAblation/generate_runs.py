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
    # (
    #     "SACCPA_sample_ablation_network_structure_parameters",
    #     {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 6, 8]},
    #     3,
    # ), #original
    (
        "SACCPA_sample_ablation_network_structure_parameters",
        {"REGNET.WS": [24, 72, 200], "REGNET.DS": [2, 6, 8]},
        3,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters",
        {"REGNET.WS": [72, 200], "REGNET.DS": [6, 8]},
        3,
    ),
    # depth tune
    (
        "SACCPA_sample_ablation_network_structure_parameters",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 5, 8]},
        3,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 4, 8]},
        3,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 4, 7]},
        3,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 4, 6]},
        3,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 4, 5]},
        3,
    ),
]
# MODELS = ["vit_l_32", "vit_l_16", "vit_h_14", "vit_b_16", "vit_b_32"]


def search_range():
    for models, params, arms in MODELS:
        ds_repr = "_".join([str(i) for i in params["REGNET.DS"]])
        representation = f"ds_{ds_repr}"
        params_str = (
            f"MODEL_NAME = '{models}'\nparams={repr(params)}\nuattention_arms={arms}"
        )
        copy_and_write(representation, params_str)


if __name__ == "__main__":
    search_range()
