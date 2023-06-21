from lib.procedures import create_cls_kpt
from lib.procedures import *

if __name__ == "__main__":
    KEYPOINT_MODELS = "segnext_sample"
    CLASSIFICATION_MODELS = "ScappaClass"
    ckpt_path = "log\\segnext_sample\\15-fine_tuning\\lightning_logs\\version_6\\checkpoints\\best-epoch=069-val_loss=0.344.ckpt"
    BATCH_SIZE = 16
    default_root_dir = f"./log/{CLASSIFICATION_MODELS}"
    train_dataloader, test_dataloader = create_dataloaders(BATCH_SIZE)

    model = create_cls_kpt(KEYPOINT_MODELS, CLASSIFICATION_MODELS, ckpt_path)
    epoch = 500
    MODEL_NAME = f"{CLASSIFICATION_MODELS}+{KEYPOINT_MODELS}"
    model, trainer, x = train_and_evluate(
        MODEL_NAME,
        model,
        default_root_dir,
        train_dataloader,
        test_dataloader,
        epoch=epoch,
    )
    x = trainer.test(model, test_dataloader, verbose=True)
    print(x)
