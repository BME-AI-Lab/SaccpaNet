# %%
import torch
import torchvision.models
import importlib

importlib.reload(torchvision)
torch.__version__

# %%
#!pip uninstall efficientnet_pytorch -y

# %%
#!git clone https://github.com/lukemelas/EfficientNet-PyTorch

# %%
from lib.modules.dataset.SQLJointsDataset import SQLJointsDataset
from lib.modules.core.function import accuracy
from lib.modules.core.loss import JointsMSELoss
from lib.modules.modules import Network

# %%
import pytorch_lightning as pl
import torchvision.models as models
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
from models.modelA import MyLightningModule as modelA

from models.modelB import MyLightningModule as modelB
from models.modelD import MyLightningModule as modelD

# from models.modelE import MyLightningModule as modelE
from glob import glob
import numpy as np
from sklearn.metrics import *
from os.path import dirname
from itertools import cycle
import seaborn as sns


def write(filename, string):
    if not isinstance(string, str):
        string = str(string)
    with open(f"{RESULT_DIR}/{filename}", "w") as f:
        f.write(string)


import matplotlib.pyplot as plt


def plot_auc(ly, ly_weight):
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


# MODELS = ["modelD","modelA","modelC","modelH","modelE","test1","modelB","model0","modelF","test4_1","test4_2","test4_3","test4_5"]
KEYPOINT_MODELS = "segnext_sample"
CLASSIFICATION_MODELS = "ScappaClass"
ckpt_path = "log\\segnext_sample\\15-fine_tuning\\lightning_logs\\version_6\\checkpoints\\best-epoch=069-val_loss=0.344.ckpt"
params = {
    "REGNET.DEPTH": 28,
    "REGNET.W0": 104,
    "REGNET.WA": 35.7,
    "REGNET.WM": 2,
    "REGNET.GROUP_W": 40,
    "REGNET.BOT_MUL": 1,
}
# ["modelA","modelB","modelC"]
# MODELS = [i+"_co" for i in MODELS]
if True:
    BATCH_SIZE = 1
    # MODEL_NAME =#"modelD"
    default_root_dir = f"./log/{CLASSIFICATION_MODELS}"  # without_mix/
    # train_dataset = SQLJointsDataset(train=True)
    NO_QUILT_TRAIN = False
    MIX_TRAIN = True
    # assert not(MIX_TRAIN == False and NO_QUILT_TRAIN=True)
    if NO_QUILT_TRAIN:
        default_root_dir = f"./log/no_quilt/{CLASSIFICATION_MODELS}"
    elif MIX_TRAIN:
        default_root_dir = f"./log/{CLASSIFICATION_MODELS}"
    else:
        default_root_dir = f"./log/without_mix/{CLASSIFICATION_MODELS}"

    WITH_FILTERED = False
    if WITH_FILTERED == True:
        SQLJointsDataset.TABLE_NAME = "openpose_annotation_03_18_with_quilt_filtered"
    WITH_QUILT = True
    VALIDATION = True
    if WITH_QUILT:
        test_dataset = SQLJointsDataset(
            train=False, mixed=False, all_quilt=True, validation=VALIDATION
        )
        quilt_conditions = "all_quilts"
    else:
        test_dataset = SQLJointsDataset(
            train=False, mixed=False, all_quilt=False, validation=VALIDATION
        )
        quilt_conditions = "no_quilts"

    ALL_CONDITIONS_STRING = f"TrainQuilt{NO_QUILT_TRAIN}_MixTrain{MIX_TRAIN}_CoFiltered{WITH_FILTERED}_TestWithQuilt{WITH_QUILT}"
    # test_datset = RealsenseDataset(bag_file="X:\\andytam\\randomized_rotation\\save\\06202022_subject1.bag")
    # train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True, num_workers=0)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    kpt_model = importlib.import_module(f"models.{KEYPOINT_MODELS}")
    kpt_model = kpt_model.MyLightningModule(params)  # locals()[f'{MODEL_NAME}']()
    kpt_model = kpt_model.load_from_checkpoint(ckpt_path, params=params)
    model = importlib.import_module(f"models.{CLASSIFICATION_MODELS}")
    model = model.MyLightningModule(kpt_model)  # locals()[f'{MODEL_NAME}']()
    print(default_root_dir)
    # find_files = glob(f"{default_root_dir}/lightning_logs/*/checkpoints/*.ckpt")
    find_files = glob(f"{default_root_dir}/lightning_logs/*/checkpoints/last*.ckpt")
    assert len(find_files) > 0
    find_files.sort()
    model_checkpoint_file = find_files[-1]
    load = torch.load(model_checkpoint_file)
    model.load_state_dict(load["state_dict"])
    RESULT_DIR = dirname(model_checkpoint_file)
    model.cuda()
    model.eval()
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
    # %%

    ly, ly_hat = np.concatenate(ly), np.concatenate(ly_hat)
    image_ids = np.concatenate(image_ids)
    ly_weight = np.concatenate(ly_weight)
    input_storage = np.concatenate(input_storage)

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
    write(f"{ALL_CONDITIONS_STRING}_confusion matrix.txt", cfm)
    cr = classification_report(ly, ly_hat, target_names=posture_map, digits=3)
    write(f"{ALL_CONDITIONS_STRING}_classification report.txt", cr)
    f1 = f1_score(ly, ly_hat, average="macro")
    # print(ly_weight.shape,ly)
    auc_score = roc_auc_score(ly, ly_weight, multi_class="ovo", average="macro")
    top2 = top_k_accuracy_score(ly, ly_weight)
    result_string = f"f1:{f1}, auc:{auc_score}, top2:{top2}"
    write(f"{ALL_CONDITIONS_STRING}_aux.txt", result_string)
    print(result_string)
    plot_auc(ly, ly_weight)
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
    write("wrong cases.txt", "\n".join(results))
