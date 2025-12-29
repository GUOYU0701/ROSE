"""
ROSE Full Training Configuration
5-round complete training with optimized parameters and SSL
"""

_base_ = [
    '/home/guoyu/mmdetection3d-1.2.0/configs/_base_/schedules/cosine.py',
    '/home/guoyu/mmdetection3d-1.2.0/configs/_base_/default_runtime.py'
]

# Model settings
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# ROSE-specific model configuration with SSL disabled for debugging
model = dict(
    type='ROSEDetector',
    enable_ssl=False,  # Disable SSL for debugging
    ssl_config=dict(
        lambda_det=1.0,        # Detection loss weight
        lambda_cm=0.3,         # Cross-modal contrastive loss weight
        lambda_cons=0.2,       # Consistency loss weight
        lambda_spatial=0.1,    # Spatial contrastive loss weight
        lambda_weather=0.2,    # Weather-aware loss weight
        ema_decay=0.999,       # Teacher model EMA decay
        consistency_warmup_epochs=5,  # Warmup epochs for consistency
        enable_pseudo_labeling=True   # Enable pseudo labeling for full training
    ),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1)),
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_cfg=dict(type='BN', requires_grad=False),
        num_outs=5),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        fusion_layer=dict(
            type='PointFusion',
            img_channels=256,
            pts_channels=64,
            mid_channels=128,
            out_channels=128,
            img_levels=[0, 1, 2, 3, 4],
            align_corners=False,
            activate_out=True,
            fuse_out=False)),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=128,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -30.0, -0.6, 60, 30.0, -0.6],
                [0, -30.0, -0.6, 60, 30.0, -0.6],
                [0, -30.0, -1.78, 60, 30.0, -1.78],
            ],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.5, 1.6, 1.4]],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=True,
        diff_rad_by_sin=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.4,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50)))

# Dataset settings  
dataset_type = 'KittiDataset'
data_root = '/home/guoyu/mmdetection3d-1.2.0/data/DAIR-V2X'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# Optimized ROSE augmentation configuration 
augmentation_config = dict(
    epoch=0,
    total_epochs=50,  # Full training epochs
    weather_configs=[
        # Clear weather (baseline)
        dict(
            weather_type='clear',
            intensity=0.0,
            brightness_factor=1.0,
            contrast_factor=1.0,
            noise_level=0.0,
            blur_kernel_size=0
        ),
        # Light rain (optimized for DAIR-V2X)
        dict(
            weather_type='rain',
            intensity=0.3,
            rain_rate=2.0,
            brightness_factor=0.85,
            contrast_factor=0.9,
            noise_level=0.02,
            blur_kernel_size=1
        ),
        # Moderate rain
        dict(
            weather_type='rain',
            intensity=0.5,
            rain_rate=8.0,
            brightness_factor=0.75,
            contrast_factor=0.8,
            noise_level=0.03,
            blur_kernel_size=2
        ),
        # Heavy rain (challenging)
        dict(
            weather_type='rain',
            intensity=0.7,
            rain_rate=20.0,
            brightness_factor=0.6,
            contrast_factor=0.7,
            noise_level=0.05,
            blur_kernel_size=3
        ),
        # Light fog
        dict(
            weather_type='fog',
            intensity=0.4,
            visibility_range=46.3,
            brightness_factor=0.9,
            contrast_factor=0.6,
            noise_level=0.01,
            blur_kernel_size=2
        ),
        # Light snow 
        dict(
            weather_type='snow',
            intensity=0.3,
            snow_rate=3.0,
            brightness_factor=1.1,
            contrast_factor=0.85,
            noise_level=0.03,
            blur_kernel_size=2
        )
    ],
    weather_probabilities=[0.3, 0.2, 0.15, 0.1, 0.15, 0.1],  # Progressive difficulty
    adaptation_enabled=True,
    performance_threshold=0.75,
    intensity_adjustment_step=0.03,
    data_root=data_root,
    lisa_path='/home/guoyu/CC/ROSE-NEW/LISA-main',
    output_dir='full_training_augmented_data'
)

# Full training pipeline with augmentation
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='RandomResize', 
        scale=[(1280, 384), (1920, 576)], 
        keep_ratio=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ])
]

# Test pipeline
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 384),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', scale=0, keep_ratio=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

modality = dict(use_lidar=True, use_camera=True)

# Full training dataloader 
train_dataloader = dict(
    batch_size=1,  # Reduced batch size for debugging
    num_workers=1,  # Reduced workers for debugging
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        modality=modality,
        ann_file='kitti_infos_train.pkl',
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=train_pipeline,
        filter_empty_gt=False,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args
        ))

# Validation dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root=data_root,
        modality=modality,
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

# Optimizer settings
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.001,  # Standard learning rate
        weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Full training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # Full 50 epochs
    val_interval=5,  # Validate every 5 epochs
    dynamic_intervals=[(40, 2)])  # More frequent validation near end

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Evaluation metrics
val_evaluator = dict(
    type='KittiMetric', 
    ann_file=data_root + '/kitti_infos_val.pkl')
test_evaluator = val_evaluator

# Visualization settings
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer')

# Custom hooks for ROSE training
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='KITTI/Overall_3D_moderate',
        patience=10,
        rule='greater',
        min_delta=0.001
    )
]

# Load pretrained weights
load_from = 'https://download.openmmlab.com/mmdetection3d/pretrain_models/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth'

# Logging configuration
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5,  # Save every 5 epochs
        max_keep_ckpts=10,
        save_best='KITTI/Overall_3D_moderate',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='Det3DVisualizationHook',
        draw=True,  # Enable visualization for full training
        interval=100,
        score_thr=0.3,
        show=False,
        wait_time=0.1,
        test_out_dir='full_training_visualization'))

# Resume training settings
resume = False
auto_resume = True

# Randomness settings
randomness = dict(seed=42, deterministic=False)

# Environment settings
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)