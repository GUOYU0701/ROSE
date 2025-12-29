# ROSE配置文件 - 带可视化功能
# 基于MVX-Net的3D检测配置，集成天气增强和可视化

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']
dataset_type = 'DAIRV2XDataset'
data_root = 'data/DAIR-V2X/'

# 输入模态配置
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# 文件客户端参数（用于 Kafka 或其他分布式存储）
file_client_args = dict(backend='disk')

# 数据库采样器配置 - 平衡不同类别的训练样本
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6))

# 训练数据增强管道
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, file_client_args=file_client_args),
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, file_client_args=file_client_args),
    
    # ROSE天气增强 - 关键组件
    dict(
        type='ROSEWeatherAugmentation',
        weather_configs=[
            dict(weather_type='rain', intensity=5.0, probability=0.3),
            dict(weather_type='snow', intensity=3.0, probability=0.2), 
            dict(weather_type='fog', intensity=2.0, probability=0.3),
            dict(weather_type='clear', intensity=0.0, probability=0.2)
        ],
        enable_visualization=True,  # 启用可视化
        visualization_interval=50,  # 每50个样本保存一次可视化
        lisa_config=dict(
            wavelength=905e-9,  # 激光波长 (nm -> m)
            beam_divergence=3e-3,  # 光束发散角 (rad)
            range_accuracy=0.09,  # 距离精度 (m)
            return_mode='strongest'  # 回波模式
        )
    ),
    
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.3925, 0.3925],
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0, 0, 0]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# 测试数据管道
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, file_client_args=file_client_args),
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='GlobalRotScaleTrans',
                 rot_range=[0, 0],
                 scale_ratio_range=[1., 1.],
                 translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]

# 数据集配置
data = dict(
    samples_per_gpu=2,  # 较小的批次大小以适应内存
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

# 模型配置 - ROSE增强检测器
model = dict(
    type='ROSEDetector',
    enable_ssl=True,
    ssl_config=dict(
        lambda_det=1.0,      # 检测损失权重
        lambda_cm=0.5,       # 跨模态对比损失权重
        lambda_cons=0.3,     # 一致性损失权重
        lambda_spatial=0.2,  # 空间对比损失权重
        lambda_weather=0.4,  # 天气感知损失权重
        ema_decay=0.999,     # EMA衰减率
        consistency_warmup_epochs=5,
        enable_pseudo_labeling=True
    ),
    
    # 点云backbone
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=[0.2, 0.2, 8],
        max_voxels=(16000, 40000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 512, 512],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    # 图像backbone
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    
    # 3D检测头
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-51.2, -51.2, -1.8, 51.2, 51.2, -1.8]],
            sizes=[[4.2, 2.0, 1.6], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]],
            rotations=[0, 1.57],
            reshape_out=True),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    
    # 2D检测头（图像分支）
    img_roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    
    # 训练和测试配置
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1)
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

# 优化器配置
optimizer = dict(
    type='AdamW',
    lr=0.001,
    betas=(0.95, 0.99),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# 学习率调度
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[40, 55])

# 检查点保存
checkpoint_config = dict(interval=5)

# 日志配置
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# 运行时设置
total_epochs = 80
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# 自定义钩子 - ROSE特定功能
custom_hooks = [
    # ROSE训练钩子 - SSL和增强管理
    dict(
        type='ROSETrainingHook',
        work_dir='${work_dir}',
        ssl_enabled=True,
        augmentation_strategy=dict(
            ssl_parameters=dict(
                lambda_det=1.0,
                lambda_cm=0.5,
                lambda_cons=0.3,
                lambda_spatial=0.2,
                lambda_weather=0.4
            ),
            teacher_config=dict(
                ema_decay=0.999,
                warmup_epochs=5
            )
        ),
        save_interval=100,
        visualization_interval=200,
        performance_log_interval=1
    ),
    
    # 可视化钩子 - 增强效果和检测结果可视化
    dict(
        type='ROSEVisualizationHook',
        enable_visualization=True,
        enable_analytics=True,
        save_interval=100,
        visualization_samples=5
    )
]

# 评估配置
evaluation = dict(
    interval=5,
    pipeline=[
        dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
        dict(type='LoadImageFromFile'),
        dict(
            type='DefaultFormatBundle3D',
            class_names=class_names,
            with_label=False),
        dict(type='Collect3D', keys=['points', 'img'])
    ])