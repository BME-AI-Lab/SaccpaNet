import csv
import os
from copy import deepcopy
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def write_log(params, log_file):
    if os.path.isfile(log_file):
        write_header = False
    else:
        write_header = True

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, params.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(params)


def update_log(PARAM_NAME, params, x):
    params.update(x[0])
    params["key"] = PARAM_NAME


def save_roc(ly, ly_weight):
    from scipy import interp

    y_true = ly  # np.array(y_true)
    classes_to_plot = None
    classes = np.unique(y_true)
    probas = ly_weight
    classes_names = [
        "supine",
        "right log",
        "right fetal",
        "left log",
        "left fetal",
        "prone right",
        "prone left",
    ]

    if classes_to_plot is None:
        classes_to_plot = classes

    fpr_dict = dict()
    tpr_dict = dict()
    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(
            y_true, probas[:, i], pos_label=classes[i]
        )

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[x] for x in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)
    # all_fpr, mean_tpr

    return all_fpr, mean_tpr


def save_auc(RESULT_DIR, ly, ly_weight):
    import scikitplot as skplt

    # scikitplot_plot_roc(ly, ly_weight)
    os.makedirs(RESULT_DIR, exist_ok=True)
    all_fpr, mean_tpr = save_roc(ly, ly_weight)
    np.save(f"{RESULT_DIR}/all_fpr.npy", all_fpr)
    np.save(f"{RESULT_DIR}/mean_tpr.npy", mean_tpr)


def write(filename, string):
    if not isinstance(string, str):
        string = str(string)
    with open(filename, "w") as f:
        f.write(string)


def evaluate_cls(
    ALL_CONDITIONS_STRING, RESULT_DIR, ly, ly_hat, image_ids, ly_weight, input_storage
):
    posture_map = [
        "supine",
        "right log",
        "right fetal",
        "left log",
        "left fetal",
        "prone right",
        "prone left",
    ]
    cfm = confusion_matrix(ly, ly_hat)
    write(f"{RESULT_DIR}/{ALL_CONDITIONS_STRING}_confusion matrix.txt", cfm)
    cr = classification_report(ly, ly_hat, target_names=posture_map, digits=3)
    write(f"{RESULT_DIR}/{ALL_CONDITIONS_STRING}_classification report.txt", cr)
    f1 = f1_score(ly, ly_hat, average="macro")
    # print(ly_weight.shape,ly)
    auc_score = roc_auc_score(ly, ly_weight, multi_class="ovo", average="macro")
    top2 = top_k_accuracy_score(ly, ly_weight)
    result_string = f"f1:{f1}, auc:{auc_score}, top2:{top2}"
    write(f"{RESULT_DIR}/{ALL_CONDITIONS_STRING}_aux.txt", result_string)
    print(result_string)
    save_auc(f"{RESULT_DIR}/{ALL_CONDITIONS_STRING}", ly, ly_weight)
    results = []
    filter = ly != ly_hat
    dataset_positions = np.array(range(len(ly)))[filter]
    for id, predict, truth, nth, img in zip(
        image_ids[filter],
        ly_hat[filter],
        ly[filter],
        dataset_positions,
        input_storage[filter],
    ):
        results.append(
            f"id:{id} predcit:{posture_map[predict]} truth:{posture_map[truth]} nth:{nth}"
        )
        # show_id(test_dataset,nth)
        plt.imshow(img[0])
        plt.savefig(f"{RESULT_DIR}/{id}.png", bbox_inches="tight")
        plt.clf()
    results.sort()
    write(f"{RESULT_DIR}/wrong cases.txt", "\n".join(results))


def inference_model_classification_coordinate(test_dataloader, model):
    softmax = nn.Softmax(dim=1)
    ly, ly_hat = [], []
    image_ids = []
    ly_weight = []
    input_storage = []
    for batch in test_dataloader:
        input, target, target_weight, meta = batch
        meta["joints"] = meta["joints"].cuda()
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
        classify = softmax(classify).cpu()
        y = meta["posture"]
        acc = (classify.argmax(dim=-1) == y).float().mean()

        ly.append(y.cpu().numpy())
        ly_hat.append(
            classify.argmax(dim=-1).detach().cpu().numpy()
        )  # classify.argmax(dim=-1).cpu().numpy())
        image_ids.append(meta["image"].cpu().numpy())
        ly_weight.append(classify.detach().numpy())
        input_storage.append(input.cpu().numpy())

    ly, ly_hat = np.concatenate(ly), np.concatenate(ly_hat)
    image_ids = np.concatenate(image_ids)
    ly_weight = np.concatenate(ly_weight)
    input_storage = np.concatenate(input_storage)
    return ly, ly_hat, image_ids, ly_weight, input_storage


def inference_clasification_model(test_dataloader, model):
    softmax = nn.Softmax(dim=1)
    result = []
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
    id2posture = {v: posture_map[k] for k, v in label2id.items()}
    for batch in test_dataloader:
        image, target, target_weight, meta = batch
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

        classify = softmax(classify).cpu()
        result_dict["class_result"] = classify
    df = pd.DataFrame(result)
    return df
