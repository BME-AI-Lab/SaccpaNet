import csv
import gc
from glob import glob
from itertools import cycle
import os
from os.path import dirname
from 06
from torch import nn-Posture_Classification.codes.multiple_network_inference_classifcation import RESULT_DIR
import matplotlib.pyplot as plt
import numpy as np
from 06-Posture_Classification.codes.multiple_network_inference_classifcation import RESULT_DIR
import pytorch_lightning as pl
import importlib
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, top_k_accuracy_score

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from configs.manually_searched_params import params
from configs.random_searched_params import params
from lib.modules.dataset.SQLJointsDataset import SQLJointsDataset


from torch.utils.data.dataloader import DataLoader


def create_dataloader(MODEL_NAME, BATCH_SIZE):
    test_dataset = SQLJointsDataset(train=False)
    train_dataset = SQLJointsDataset(train=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=2,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=2,
    )

    return train_dataloader, test_dataloader


def load_train_eval_sample(
    MODEL_NAME, default_root_dir, train_dataloader, test_dataloader, params, epochs
):
    model = create_model_by_params(MODEL_NAME, params)  # locals()[f'{MODEL_NAME}']()

    print(f"Now training for {MODEL_NAME}")
    trainer = pl.Trainer(
        gpus=[0],
        amp_level="O2",
        accelerator="dp",
        amp_backend="apex",
        max_epochs=epochs,
        min_epochs=epochs,
        default_root_dir=default_root_dir,
    )  # gpus=1, accelerator='dp',
    trainer.tune(model, train_dataloader)
    trainer.fit(model, train_dataloader, test_dataloader)

    # %%
    # (model.net._avg_pooling.output_size)
    # %%
    x = trainer.test(model, test_dataloader, verbose=True)
    return model, trainer, x


def create_model_by_params(MODEL_NAME, params):
    model = importlib.import_module(f"models.{MODEL_NAME}")
    model = model.MyLightningModule(params)
    return model


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


def create_model_classification_with_keypoint(
    KEYPOINT_MODELS, CLASSIFICATION_MODELS, ckpt_path
):
    kpt_model = importlib.import_module(f"models.{KEYPOINT_MODELS}")
    kpt_model = kpt_model.MyLightningModule(
        params, num_classes=18
    )  # locals()[f'{MODEL_NAME}']()
    kpt_model = kpt_model.load_from_checkpoint(ckpt_path, params=params)
    kpt_model.eval()
    model = importlib.import_module(f"models.{CLASSIFICATION_MODELS}")
    model = model.MyLightningModule(kpt_model)
    return model


def train_for_model_finetune_classification(
    default_root_dir, train_dataloader, test_dataloader, model
):
    trainer = pl.Trainer(
        gpus=[0],
        amp_level="O2",
        accelerator="dp",
        amp_backend="apex",
        max_epochs=500,
        min_epochs=500,
        default_root_dir=default_root_dir,
    )  # gpus=1, accelerator='dp',
    trainer.tune(model, train_dataloader)
    trainer.fit(model, train_dataloader, test_dataloader)
    return trainer


# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
# from models.modelA import MyLightningModule as modelA

# from models.modelB import MyLightningModule as modelB
# from models.modelD import MyLightningModule as modelD
# from models.modelH import MyLightningModule as modelH


# from models.modelE import MyLightningModule as modelE
def train_and_evaluate_finetune_parameterized_by_config(
    MODEL_NAME,
    PRETRAIN_MODEL,
    model,
    default_root_dir,
    train_dataloader,
    test_dataloader,
    PARAM_NAME,
):
    state_dict = torch.load(PRETRAIN_MODEL)
    model.load_state_dict(state_dict)
    # checkpoints
    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        # dirpath=default_root_dir,
        filename="best-{epoch:03d}-{val_loss:.3f}",
    )
    checkpoint_callback_last = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        # dirpath=default_root_dir,
        filename="last-{epoch:03d}-{global_step}",
    )
    # trainig
    print(f"Now training for {MODEL_NAME}")
    trainer = pl.Trainer(
        gpus=[0],
        amp_level="O2",
        accelerator="dp",
        amp_backend="apex",
        max_epochs=300,
        min_epochs=300,
        default_root_dir=default_root_dir,
        callbacks=[checkpoint_callback_best, checkpoint_callback_last],
        resume_from_checkpoint="log\\segnext_sample\\15-fine_tuning\\lightning_logs\\version_5\\checkpoints\\best-epoch=051-val_loss=0.355.ckpt",
    )  # gpus=1, accelerator='dp',
    trainer.tune(model, train_dataloader)
    trainer.fit(model, train_dataloader, test_dataloader)

    # %%
    # (model.net._avg_pooling.output_size)
    # %%
    x = trainer.test(model, test_dataloader, verbose=True)

    # %%
    params.update(x[0])
    params["key"] = PARAM_NAME
    log_file = "log.csv"

    if os.path.isfile(log_file):
        write_header = False
    else:
        write_header = True

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, params.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(params)
        # f.write(str(params))
        # f.write("\n\n\n")

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def create_dataloader_finetune_parameterized_by_config(MODEL_NAME, BATCH_SIZE):
    test_dataset = SQLJointsDataset(train=False)

    WITH_FILTERED = True
    if MODEL_NAME == "model0":
        # test_dataset = SQLJointsDataset(train=False)
        SQLJointsDataset.TABLE_NAME = "openpose_annotation_old_data"
        train_dataset = SQLJointsDataset(train=True)
    elif WITH_FILTERED == True:
        SQLJointsDataset.TABLE_NAME = "openpose_annotation_03_18_with_quilt_filtered"
        train_dataset = SQLJointsDataset(train=True)
    else:
        train_dataset = SQLJointsDataset(train=True)
    WITHOUT_MIX = False
    # for WITHOUT_MIX in [False]: #temp script
    if WITHOUT_MIX:
        train_dataset.probability = 0
        default_root_dir = f"./log/without_mix/{MODEL_NAME}"
    else:
        default_root_dir = f"./log/{MODEL_NAME}"
        # sampler specific
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=4,
    )

    return default_root_dir, train_dataloader, test_dataloader


# ["modelA","modelB","modelC"]
# MODELS = [i+"_co" for i in MODELS]
def create_mode_coordinate_classification(KEYPOINT_MODELS, CLASSIFICATION_MODELS, ckpt_path, params, default_root_dir):
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
    return model,RESULT_DIR


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


def write(filename, string):
    if not isinstance(string, str):
        string = str(string)
    with open(f"{RESULT_DIR}/{filename}", "w") as f:
        f.write(string)


def create_dataloader_coordinate_evaluation(BATCH_SIZE, WITH_QUILT, VALIDATION):
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
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return test_dataloader


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
    # %%

    ly, ly_hat = np.concatenate(ly), np.concatenate(ly_hat)
    image_ids = np.concatenate(image_ids)
    ly_weight = np.concatenate(ly_weight)
    input_storage = np.concatenate(input_storage)
    return ly,ly_hat,image_ids,ly_weight,input_storage


def evaluate_classification_model(ALL_CONDITIONS_STRING, RESULT_DIR, ly, ly_hat, image_ids, ly_weight, input_storage):
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





