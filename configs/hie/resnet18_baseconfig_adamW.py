# dataset settings
dataset_type = 'Hie_Dataset'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type='ExtractDataFromObj'),
    dict(type='NormalizeMedical', norm_type='full_volume_mean',
         instensity_min_val=0.5,
         instensity_max_val=99.5),
    dict(type='ResizeMedical', size=(80, 160, 160)),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='ConcatImage'),
    # dict(type='ImageToTensor', keys=['img']),

    dict(type='ToTensor', keys=['gt_label', 'img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type='ExtractDataFromObj'),
    dict(type='NormalizeMedical', norm_type='full_volume_mean',
         instensity_min_val=0.5,
         instensity_max_val=99.5),
    dict(type='ResizeMedical', size=(80, 160, 160)),
    dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
                    'hie_resample_0.5x0.5x0.5_niigz',
        ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/t1_zw_fse_train.txt',
        pipeline=train_pipeline,
        modes=['t1_zw']),
    val=dict(
        type=dataset_type,
        data_prefix='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
                    'hie_resample_0.5x0.5x0.5_niigz',
        ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/t1_zw_fse_val.txt',
        pipeline=test_pipeline,
        modes=['t1_zw']),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
                    'hie_resample_0.5x0.5x0.5_niigz',
        ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/t1_zw_fse_val.txt',
        pipeline=test_pipeline,
        modes=['t1_zw']))
evaluation = dict(interval=2, metric='accuracy')


norm_cfg = dict(type='BN3d', requires_grad=True)
conv_cfg = dict(type='Conv3d')
num_classes = 2
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=1,
        in_dims=3,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        init_cfg=[
             dict(type='Kaiming', layer=['Conv3d']),
             dict(
                 type='Constant',
                 val=1,
                 layer=['_BatchNorm', 'GroupNorm', 'BN3d'])
         ]
    ),
    neck=dict(type='GlobalAveragePooling', dim=3),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ))

# optimizer
optimizer = dict(type='AdamW', lr=0.003, weight_decay=0.3)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# specific to vit pretrain
# paramwise_cfg = dict(
#     custom_keys={
#         '.backbone.cls_token': dict(decay_mult=0.0),
#         '.backbone.pos_embed': dict(decay_mult=0.0)
#     })
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1e-4)
runner = dict(type='EpochBasedRunner', max_epochs=300)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
checkpoint_config = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]