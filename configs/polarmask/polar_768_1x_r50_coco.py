# model settings
model = dict(
    type='PolarMask',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='PolarMask_Head',
        num_classes=4,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        # smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        iou_thr=0.5,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100))
# dataset settings
dataset_type = 'CocoSegDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,with_label=True),
    dict(type='Resize', img_scale=(1280, 768), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
val_pipeline = [ 
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True, with_mask=False,with_label=True),
    dict(type='Resize', img_scale=(1280, 768), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),]
test_pipeline = [
    dict(type='LoadImageFromFile'), 
    # dict(type='LoadAnnotations', with_bbox=True, with_mask=False,with_label=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])]

data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# evaluation = dict(metric=['bbox', 'segm'])

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,
                 paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3 /1,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=1)
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
# checkpoint='work_dirs/polar_768_1x_r50_coco/epoch_1.pth'
log_level = 'INFO'
load_from = None
resume_from = None
# resume_from = 'work_dirs/polar_768_1x_r50_coco/epoch_1.pth'
workflow = [('train', 1)]
# workflow = [('train', 1), ('val', 1)]
