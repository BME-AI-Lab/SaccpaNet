# model settings
from lib.modules.core.sampler import generate_regnet_full

sample_params = {
    "REGNET.DEPTH": 28,
    "REGNET.W0": 104,
    "REGNET.WA": 35.7,
    "REGNET.WM": 2,
    "REGNET.GROUP_W": 40,
    "REGNET.BOT_MUL": 1,
}
ws, ds, ss, bs, gs = generate_regnet_full(sample_params)
norm_cfg = dict(type="BN", requires_grad=True)
ham_norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)

_base_ = ["../../../_base_/default_runtime.py"]

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type="Adam",
        lr=5e-4,
    )
)

# learning policy
param_scheduler = [
    dict(
        type="LinearLR", begin=0, end=500, start_factor=0.001 / 4, by_epoch=False
    ),  # warm-up
    dict(
        type="MultiStepLR",
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True,
    ),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best="coco-wholebody/AP", rule="greater"))

# codec settings
codec = dict(type="MSRAHeatmap", input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="MSCAU",
        in_channels=3,
        embed_dims=ws,  # [32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],  # mlp ratio need
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=ds,  # [3, 3, 5, 2],
        norm_cfg=dict(type="BN", requires_grad=True),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=120,
        out_channels=133,
        deconv_out_channels=None,
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode="heatmap",
        shift_heatmap=True,
    ),
)

# base dataset settings
dataset_type = "CocoWholeBodyDataset"
data_mode = "topdown"
data_root = "D:\\dataset\\coco_posture\\data\\coco\\"

# pipelines
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=1,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/coco_wholebody_train_v1.0.json",
        data_prefix=dict(img="train2017/"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=1,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/coco_wholebody_val_v1.0.json",
        data_prefix=dict(img="val2017/"),
        test_mode=True,
        bbox_file=None,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoWholeBodyMetric",
    ann_file=data_root + "annotations/coco_wholebody_val_v1.0.json",
)
test_evaluator = val_evaluator
