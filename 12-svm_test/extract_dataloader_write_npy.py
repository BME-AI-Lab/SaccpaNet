import numpy as np
import pandas as pd
import torch
import tqdm
from torchvision import models

from lib.procedures.dataloaders_procedure import (
    create_test_dataloader,
    create_train_dataloader,
    create_validation_dataloader,
)

model = models.resnet152(pretrained=True)
newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
train_dataloader = create_train_dataloader(1, WITH_QUILT=True)
test_dataloader = create_test_dataloader(1, WITH_QUILT=True)
validation_dataloader = create_validation_dataloader(1, WITH_QUILT=True)
newmodel = newmodel.cuda().eval()


def extract_dataloader(dataloader):
    rec = []
    for i, (input, target, target_weight, meta) in enumerate(tqdm.tqdm(dataloader)):
        # input = input.numpy()[0]
        target = target.numpy()[0]
        meta = meta
        image_id = meta["image"].numpy()[0]
        posture = meta["posture"].numpy()[0]
        input = torch.concatenate([input, input, input], dim=1).float().cuda()
        with torch.no_grad():
            feature = newmodel(input).detach().cpu().numpy()[0, :, 0, 0]
        # print(input, target, meta)
        rec.append((image_id, feature, posture))
    df = pd.DataFrame(rec, columns=["idx", "feature", "meta"])
    return df


if __name__ == "__main__":
    df = extract_dataloader(train_dataloader)
    df.to_pickle("train.pkl")

    df = extract_dataloader(test_dataloader)
    df.to_pickle("test.pkl")

    df = extract_dataloader(validation_dataloader)
    df.to_pickle("validation.pkl")
