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
    #     "SACCPA_sample_ablation_network_structure_parameters_atrous_attention",
    #     {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 6, 8]},
    #     3,
    #     3,
    # ),
    (
        "SACCPA_sample_ablation_network_structure_parameters_atrous_attention",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 6, 8]},
        3,
        2,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters_atrous_attention",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 6, 8]},
        3,
        1,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters_atrous_attention",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 6, 8]},
        3,
        0,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters_atrous_attention",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 6, 8]},
        2,
        2,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters_atrous_attention",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 6, 8]},
        1,
        2,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters_atrous_attention",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 6, 8]},
        2,
        1,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters_atrous_attention",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 6, 8]},
        1,
        1,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters_atrous_attention",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 6, 8]},
        1,
        0,
    ),
    (
        "SACCPA_sample_ablation_network_structure_parameters_atrous_attention",
        {"REGNET.WS": [8, 24, 72, 200], "REGNET.DS": [1, 2, 6, 8]},
        2,
        0,
    ),
]
# MODELS = ["vit_l_32", "vit_l_16", "vit_h_14", "vit_b_16", "vit_b_32"]


def search_range():
    for models, params, arms, atrous in MODELS:
        ds_repr = "_".join([str(i) for i in params["REGNET.DS"]])
        representation = f"atrous_{atrous}_attentionArms_{arms}"
        params_str = f"MODEL_NAME = '{models}'\nparams={repr(params)}\nuattention_arms={arms}\nuattention_atrous={atrous}"
        copy_and_write(representation, params_str)


if __name__ == "__main__":
    search_range()
