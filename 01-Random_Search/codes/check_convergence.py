import gc

import torch
from search_configs import *

from lib.modules.core.sampler import sample_cfgs
from lib.procedures import create_dataloaders, create_kpt, train_and_evaluate
from lib.procedures.evaluations import update_log, write_log

log_file = "log.csv"
if __name__ == "__main__":
    MODEL_NAME = "SACCPA_sample"
    BATCH_SIZE = 16
    TOTAL_EPOCH = 2
    default_root_dir = f"./log/{MODEL_NAME}"
    train_dataloader, test_dataloader = create_dataloaders(BATCH_SIZE)
    default_root_dir = f"{default_root_dir}/{PARAM_NAME}"
    model = create_kpt(MODEL_NAME, params)
    model, trainer, result = train_and_evaluate(
        MODEL_NAME,
        model,
        default_root_dir,
        train_dataloader,
        test_dataloader,
        epochs=TOTAL_EPOCH,
    )
    assert result[0]["test_joint_acc"] > 0.2
    update_log(PARAM_NAME, params, result)
    write_log(params, log_file=log_file)

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
