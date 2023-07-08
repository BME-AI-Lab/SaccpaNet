default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco-wholebody/AP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))
custom_hooks = [dict(type='SyncBuffersHook')]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False
backend_args = dict(backend='local')
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
val_cfg = dict()
test_cfg = dict()
sample_params = dict({
    'REGNET.DEPTH': 28,
    'REGNET.W0': 104,
    'REGNET.WA': 35.7,
    'REGNET.WM': 2,
    'REGNET.GROUP_W': 40,
    'REGNET.BOT_MUL': 1
})
ws = [104, 208, 416, 832]
ds = [2, 4, 8, 14]
norm_cfg = dict(type='BN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005))
param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=500,
        start_factor=0.00025,
        by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]
auto_scale_lr = dict(base_batch_size=512)
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(24, 32), sigma=2)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='SaccpaNet',
        in_channels=3,
        embed_dims=[104, 208, 416, 832],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[2, 4, 8, 14],
        norm_cfg=dict(type='BN', requires_grad=True)),
    head=dict(
        type='HeatmapHead',
        in_channels=120,
        out_channels=133,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(24, 32),
            sigma=2)),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True))
dataset_type = 'CocoWholeBodyDataset'
data_mode = 'topdown'
data_root = 'D:\\dataset\\coco_posture\\data\\coco\\'
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(24, 32),
            sigma=2)),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=1,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoWholeBodyDataset',
        data_root='D:\\dataset\\coco_posture\\data\\coco\\',
        data_mode='topdown',
        ann_file='annotations/coco_wholebody_train_v1.0.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomHalfBody'),
            dict(type='RandomBBoxTransform'),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='MSRAHeatmap',
                    input_size=(192, 256),
                    heatmap_size=(24, 32),
                    sigma=2)),
            dict(type='PackPoseInputs')
        ]))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=1,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoWholeBodyDataset',
        data_root='D:\\dataset\\coco_posture\\data\\coco\\',
        data_mode='topdown',
        ann_file='annotations/coco_wholebody_val_v1.0.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        bbox_file=None,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(type='PackPoseInputs')
        ]))
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=1,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoWholeBodyDataset',
        data_root='D:\\dataset\\coco_posture\\data\\coco\\',
        data_mode='topdown',
        ann_file='annotations/coco_wholebody_val_v1.0.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        bbox_file=None,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(type='PackPoseInputs')
        ]))
val_evaluator = dict(
    type='CocoWholeBodyMetric',
    ann_file=
    'D:\\dataset\\coco_posture\\data\\coco\\annotations/coco_wholebody_val_v1.0.json'
)
test_evaluator = dict(
    type='CocoWholeBodyMetric',
    ann_file=
    'D:\\dataset\\coco_posture\\data\\coco\\annotations/coco_wholebody_val_v1.0.json'
)
launcher = 'none'
work_dir = './work_dirs\\saccpa_coco-wholebody-256x192'
