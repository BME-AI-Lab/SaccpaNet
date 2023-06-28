import csv
import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve,
                             top_k_accuracy_score)
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


def plot_auc(RESULT_DIR, ly, ly_weight):
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
    plt.savefig(f"{RESULT_DIR}/roc.png", bbox_inches="tight")
    plt.clf()


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
    plot_auc(RESULT_DIR, ly, ly_weight)
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
        ly_weight.append(softmax(classify).detach().cpu().numpy())
        input_storage.append(input.cpu().numpy())

    ly, ly_hat = np.concatenate(ly), np.concatenate(ly_hat)
    image_ids = np.concatenate(image_ids)
    ly_weight = np.concatenate(ly_weight)
    input_storage = np.concatenate(input_storage)
    return ly, ly_hat, image_ids, ly_weight, input_storage
