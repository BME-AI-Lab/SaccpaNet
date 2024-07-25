import numpy as np

SACCPAs = [
    "SACCPA34",
    "SACCPA50",
    "SACCPA101",
    "SACCPA152",
]

ECAs = [
    "ECA34",
    "ECA50",
    "ECA101",
    "ECA152",
]

Resnets = [
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
]

EffincentNets = [
    "EfficientNetB0",
    "EfficientNetB2",
    "EfficientNetB4",
    "EfficientNetB7",
]

VITs = [
    ("vit_b_16", "ViT-Base/16"),
    ("vit_b_32", "ViT-Base/32"),
    ("vit_l_16", "ViT-Large/16"),
    ("vit_l_32", "ViT-Large/32"),
]

network_groups = [
    SACCPAs,
    # ECAs,
    # Resnets,
    # EffincentNets,
]
import matplotlib.pyplot as plt

figsize = None
# draw the networks in the network_groups of roc curve
for network_group in network_groups:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for network in network_group:
        all_fpr_path = f"all_fpr/TestWithQuiltTrue_ValidationFalse/{network}.npy"
        all_fpr = np.load(all_fpr_path)
        mean_tpr_path = f"mean_tpr/TestWithQuiltTrue_ValidationFalse/{network}.npy"
        mean_tpr = np.load(mean_tpr_path)

        ax.plot(
            all_fpr,
            mean_tpr,
            label=f"{network}",
            lw=2,
        )
    text_fontsize = 16
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=text_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc="lower right", fontsize=text_fontsize)
    plt.savefig(f"{network_group[0][:-2]}_roc.png", bbox_inches="tight", dpi=600)
    plt.clf()


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for network_name, network in VITs:
        all_fpr_path = f"all_fpr/TestWithQuiltTrue_ValidationFalse/{network_name}.npy"
        all_fpr = np.load(all_fpr_path)
        mean_tpr_path = f"mean_tpr/TestWithQuiltTrue_ValidationFalse/{network_name}.npy"
        mean_tpr = np.load(mean_tpr_path)

        ax.plot(
            all_fpr,
            mean_tpr,
            label=f"{network}",
            lw=2,
        )
    text_fontsize = 16
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=text_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc="lower right", fontsize=text_fontsize)
    plt.savefig(f"VITs_roc.png", bbox_inches="tight", dpi=600)
    plt.clf()
