from get_classification_model import classification_models
from search_configs import CLASSIFICATION_MODELS

from configs.manually_searched_params import params
from lib.procedures import *
from lib.procedures import create_cls_kpt

if __name__ == "__main__":
    KEYPOINT_MODELS = "SACCPA_sample"
    CKPT_PATH = "../best-epoch.ckpt"
    BATCH_SIZE = 16
    default_root_dir = f"./log/{CLASSIFICATION_MODELS}"
    train_dataloader, val_dataloader = create_dataloaders(BATCH_SIZE)
    cls_model = classification_models[CLASSIFICATION_MODELS](num_classes=1000)

    model = create_cls_kpt(KEYPOINT_MODELS, cls_model, CKPT_PATH, params)
    epochs = 500
    MODEL_NAME = f"{CLASSIFICATION_MODELS}+{KEYPOINT_MODELS}"
    model, trainer, x = train_and_evaluate(
        MODEL_NAME,
        model,
        default_root_dir,
        train_dataloader,
        val_dataloader,
        epochs=epochs,
    )
    x = trainer.test(model, val_dataloader, verbose=True)
    print(x)
