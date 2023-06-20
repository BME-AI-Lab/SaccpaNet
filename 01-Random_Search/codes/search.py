# %%
import torch
import torchvision.models
import importlib

from lib.procedures import create_dataloader, load_train_eval_sample
from lib.procedures import write_log
from lib.procedures import update_log

importlib.reload(torchvision)
torch.__version__

# %%
#!pip uninstall efficientnet_pytorch -y

# %%
#!git clone https://github.com/lukemelas/EfficientNet-PyTorch

# %%
from lib.modules.core.sampler import sample_cfgs
from config import seed

# %%

# import torchvision.models as models
import gc
from config import *

log_file = "log.csv"


if __name__ == "__main__":
    import importlib

    MODEL_NAME = "segnext_sample"
    SAMPLES = sample_cfgs(seed=seed, sample_size=1)
    BATCH_SIZE = 8
    TOTAL_EPOCH = 300
    default_root_dir = f"./log/{MODEL_NAME}"
    train_dataloader, test_dataloader = create_dataloader(MODEL_NAME, BATCH_SIZE)
    for PARAM_NAME, params in SAMPLES.items():
        # MODEL_NAME = "modelD"
        default_root_dir = f"{default_root_dir}/{PARAM_NAME}"

        model, trainer, x = load_train_eval_sample(
            MODEL_NAME,
            default_root_dir,
            train_dataloader,
            test_dataloader,
            params,
            epochs=TOTAL_EPOCH,
        )

        # %%

        update_log(PARAM_NAME, params, x)
        write_log(params, log_file=log_file)

        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
