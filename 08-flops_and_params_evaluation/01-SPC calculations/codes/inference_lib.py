from copy import deepcopy
from itertools import cycle
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from torch import nn

label2id = {"r": 0, "o": 1, "y": 2, "g": 3, "c": 4, "b": 5, "p": 6}
id2label = {v: k for k, v in label2id.items()}
posture_map = [
    "supine",
    "right log",
    "right fetal",
    "left log",
    "left fetal",
    "prone right",
    "prone left",
]
id2posture = {i: posture_map[i] for i in range(7)}


def inference_clasification_model(test_dataloader, model):
    softmax = nn.Softmax(dim=1)
    results = []

    for batch in test_dataloader:
        input, target, target_weight, meta = batch
        """The dicionary meta contains the following keys:
        "image": image_file,
        "filename": filename,
        "imgnum": imgnum,
        "joints": joints,
        "joints_vis": joints_vis,
        "center": center,
        "scale": scale,
        "rotation": rotation,
        "score": score,
        "flipped": flipped,
        "posture": posture,        
        """
        result_dict = deepcopy(meta)
        batch = (
            input.cuda().detach(),
            target.cuda().detach(),
            target_weight.cuda().detach().detach(),
            meta,
        )
        result = model.get_batch_output(batch)  # .detach()
        classify = result["classify"]
        if classify.shape[1] > 7:
            classify = classify[:, :7]  # fix num_class

        for k, v in result_dict.items():
            if isinstance(v, torch.Tensor):
                result_dict[k] = v.cpu().numpy()

        classify = softmax(classify).cpu().numpy()
        for i in range(len(classify)):
            result_dict["class_result"] = classify[i]
            result_dict["class_id"] = classify[
                i
            ].argmax()  # .cpu().numpy()  # .cpu().numpy()
            results.append(result_dict)

        """result_dict["class_result"] = classify
        result_dict["class_id"] = classify.argmax(dim=-1).cpu().numpy()
        results.append(result_dict)"""
    df = pd.DataFrame(results)
    return df


def write(filename, string):
    if not isinstance(string, str):
        string = str(string)
    with open(filename, "w") as f:
        f.write(string)


def plot_auc(PATH, ly, ly_weight):
    b = np.zeros((ly.size, ly.max() + 1))
    b[np.arange(ly.size), ly] = 1
    ly = b
    fpr, tpr, threshold = {}, {}, {}
    roc_auc = {}
    n_classes = 7
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(ly[i], ly_weight[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    # plt.plot(
    #     fpr["micro"],
    #     tpr["micro"],
    #     label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    #     color="deeppink",
    #     linestyle=":",
    #     linewidth=4,
    # )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    lw = 2
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(f"{PATH}_roc.png", bbox_inches="tight")
    plt.clf()


def write_df_result(path, df):
    y_hat = df["class_id"].values
    y = np.concatenate(df["posture"].values)
    cfm = confusion_matrix(y, y_hat)
    write(path + "_confusion_matrix.txt", repr(cfm))
    fig = sns.heatmap(cfm, annot_kws=id2posture, fmt="d", cmap="binary")
    fig.figure.savefig(path + "_confusion_matrix.png")

    cr = classification_report(y, y_hat, target_names=posture_map, digits=4)
    write(path + "_classification_report.txt", cr)
    df["class_result"].values
    array = np.stack([i for i in df["class_result"].values])
    yhat = np.zeros_like(array)
    for i in range(len(array)):
        label = df["class_id"].values[i]
        yhat[i][label] = 1       
    
    # auc = roc_auc_score(y, y_hat, multi_class="ovr", average="macro")
    # f1 = f1_score(y, y_hat, average="macro")
    result_string = f"f1:{f1}, auc:{auc}"
    write(path + "_result.txt", result_string)
    # plot_auc(path, y, y_hat)


def evaluate_cls(ALL_CONDITIONS_STRING, RESULT_DIR, df):
    PATH = f"{RESULT_DIR}/{ALL_CONDITIONS_STRING}/"
    makedirs(PATH, exist_ok=True)
    df["effect"] = sum(df["effect"].values, start=[])

    # Overall result
    write_df_result(PATH, df)
    # auc = roc_auc_score(y, y_hat, multi_class="ovr", average="macro")
    # f1 = f1_score(y, y_hat, average="macro")
    # Per quilt condition result
    id2quilt = {1: "thick", 2: "medium", 3: "thin", 4: "no_quilt"}
    for quilt in range(1, 5):
        df_quilt = df[df["effect"] == str(quilt)]
        PATH = f"{RESULT_DIR}/{ALL_CONDITIONS_STRING}_{id2quilt[quilt]}/"
        makedirs(PATH, exist_ok=True)
        write_df_result(PATH, df_quilt)
    """
    # Collapse quilt result
    df["class_id"] = df["class_id"].apply(lambda x: x if x != 6 else 5)
    df["posture"] = df["posture"].apply(lambda x: x if x[0] != 6 else [5])
    PATH = f"{RESULT_DIR}/{ALL_CONDITIONS_STRING}_pronecollapsed/"
    makedirs(PATH, exist_ok=True)
    global posture_map
    posture_map = posture_map[:6]
    write_df_result(PATH, df)
    id2quilt = {1: "thick", 2: "medium", 3: "thin", 4: "no_quilt"}
    for quilt in range(1, 5):
        df_quilt = df[df["effect"] == str(quilt)]
        PATH = f"{RESULT_DIR}/{ALL_CONDITIONS_STRING}_pronecollapsed_{id2quilt[quilt]}/"
        makedirs(PATH, exist_ok=True)
        write_df_result(PATH, df_quilt)
    """
