import torch
from lib.procedures.evaluations import update_log, write_log
from lib.procedures import (
    create_dataloaders,
    create_kpt,
    train_and_evaluate,
)
from lib.modules.core.sampler import sample_cfgs
from search_config import seed
import gc

log_file = "log.csv"
if __name__ == "__main__":
    MODEL_NAME = "saccpa_sample"
    SAMPLES = sample_cfgs(seed=seed, sample_size=1)
    BATCH_SIZE = 8
    TOTAL_EPOCH = 300
    default_root_dir = f"./log/{MODEL_NAME}"
    train_dataloader, test_dataloader = create_dataloaders(BATCH_SIZE)
    for PARAM_NAME, params in SAMPLES.items():
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
        update_log(PARAM_NAME, params, result)
        write_log(params, log_file=log_file)

        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
