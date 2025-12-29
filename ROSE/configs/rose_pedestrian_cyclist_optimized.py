# ROSE配置文件 - 行人和骑行者检测优化版本
# 专门针对小目标（Pedestrian, Cyclist）检测性能优化

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

# 文件客户端参数
file_client_args = dict(backend='disk')

# 针对小目标优化的数据库采样器
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        # 对小目标要求更少的点数以增加样本
        filter_by_min_points=dict(Car=5, Pedestrian=3, Cyclist=3)),
    classes=class_names,
    # 大幅增加小目标的采样数量
    sample_groups=dict(Car=8, Pedestrian=20, Cyclist=20))

# 小目标优化的训练数据增强管道
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, file_client_args=file_client_args),
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, file_client_args=file_client_args),
    
    # 小目标友好的ROSE天气增强
    dict(
        type='ROSEWeatherAugmentation',
        weather_configs=[
            # 轻微雨雾，保持小目标可见性
            dict(weather_type='light_rain', intensity=0.1, probability=0.3),
            dict(weather_type='light_fog', intensity=0.05, probability=0.2),
            # 晴天概率增加，确保充足的清晰样本
            dict(weather_type='clear', intensity=0.0, probability=0.5)
        ],
        enable_visualization=True,
        visualization_interval=20,  # 更频繁的可视化
        focus_small_objects=True,   # 专注小目标增强
        lisa_config=dict(
            wavelength=905e-9,
            beam_divergence=2e-3,  # 更小的光束发散
            range_accuracy=0.05,   # 更高的距离精度
            return_mode='dual'     # 双重回波模式
        )
    ),
    
    # 增强小目标采样
    dict(type='ObjectSample', db_sampler=db_sampler),
    
    # 减少几何变换强度以保护小目标
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.3),  # 降低翻转概率
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.2, 0.2],        # 减少旋转范围
         scale_ratio_range=[0.98, 1.02], # 减少缩放范围
         translation_std=[0.1, 0.1, 0.1]),  # 减少平移
    
    # 多尺度输入训练
    dict(type='Resize', 
         img_scale=[(1280, 384), (1600, 480), (1920, 576)], 
         multiscale_mode='value',
         keep_ratio=True),
    
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
        img_scale=(1600, 480),  # 更高分辨率用于小目标
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
    samples_per_gpu=1,  # 减小批量大小以支持更高分辨率
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,  # 重复数据集以增加小目标曝光
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
            box_type_3d='LiDAR',
            # 小目标过滤设置
            filter_empty_gt=False)),  # 不过滤空GT，保留困难样本
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

# 小目标优化的ROSE检测器配置
model = dict(
    type='ROSEDetector',
    enable_ssl=True,
    ssl_config=dict(
        # 增加SSL损失权重，专注小目标
        lambda_det=1.0,
        lambda_cm=0.8,       # 增强跨模态对比
        lambda_cons=0.6,     # 增强一致性约束
        lambda_spatial=0.5,  # 增强空间对比
        lambda_weather=0.3,  # 适当降低天气损失权重
        ema_decay=0.9995,    # 更慢的教师模型更新
        consistency_warmup_epochs=3,
        enable_pseudo_labeling=True,
        # 小目标特定参数
        small_object_enhancement=True,
        problematic_class_focus=True,
        small_object_weight_multiplier=2.0
    ),
    
    # 高分辨率点云处理
    pts_voxel_layer=dict(
        max_num_points=15,  # 增加每个体素的点数
        point_cloud_range=point_cloud_range,
        voxel_size=[0.1, 0.1, 0.2],  # 更小的体素尺寸
        max_voxels=(20000, 50000)),  # 增加最大体素数
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[25, 1024, 1024],  # 更高分辨率
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[6, 6],  # 增加层数
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
    
    # 高分辨率图像backbone
    img_backbone=dict(
        type='ResNet',
        depth=101,  # 使用更深的网络
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),  # 使用可变形卷积
        stage_with_dcn=(False, True, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    
    # 小目标优化的3D检测头
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        assign_per_class=True,
        # 小目标优化的锚点生成器
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                # 车辆标准范围
                [-51.2, -51.2, -1.8, 51.2, 51.2, -1.8],
                # 行人扩展范围，重点关注近距离
                [-30.0, -30.0, -0.6, 70.0, 30.0, -0.6],
                # 骑行者扩展范围
                [-30.0, -30.0, -0.6, 70.0, 30.0, -0.6]
            ],
            sizes=[
                [4.2, 2.0, 1.6],    # Car
                [0.6, 0.8, 1.73],   # Pedestrian - 稍大尺寸以增加召回
                [1.6, 0.8, 1.73]    # Cyclist - 稍大尺寸
            ],
            rotations=[0, 1.57],
            reshape_out=True),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        
        # 小目标优化的损失函数
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=3.0,  # 增加分类损失权重
            class_weight=[1.0, 2.0, 2.0]),  # 小目标类别权重翻倍
        loss_bbox=dict(
            type='SmoothL1Loss', 
            beta=1.0 / 9.0, 
            loss_weight=4.0),  # 增加回归损失权重
        loss_dir=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.3)),
    
    # 图像检测头
    img_roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),  # 增加采样率
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
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=2.0,
                class_weight=[1.0, 1.5, 1.5]),  # 小目标类别权重
            loss_bbox=dict(type='L1Loss', loss_weight=2.0))),
    
    # 小目标优化的训练配置
    train_cfg=dict(
        pts=dict(
            # 类别特定的分配器，小目标使用更宽松的阈值
            assigner=[
                # Car - 标准阈值
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                # Pedestrian - 更宽松的阈值
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.25,  # 显著降低
                    neg_iou_thr=0.15,  # 显著降低
                    min_pos_iou=0.15,  # 显著降低
                    ignore_iof_thr=-1),
                # Cyclist - 更宽松的阈值
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.25,  # 显著降低
                    neg_iou_thr=0.15,  # 显著降低
                    min_pos_iou=0.15,  # 显著降低
                    ignore_iof_thr=-1)
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False)),
    
    # 小目标优化的测试配置
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,         # 非常低的NMS阈值
            score_thr=0.02,       # 非常低的分数阈值
            min_bbox_size=0,      # 允许极小边界框
            nms_pre=500,          # 大幅增加预选框数量
            max_num=200)))        # 大幅增加最终检测数量

# 小目标专用优化器
optimizer = dict(
    type='AdamW',
    lr=0.0005,  # 降低学习率以稳定训练
    betas=(0.9, 0.999),
    weight_decay=0.05,  # 增加权重衰减
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'pts_bbox_head.conv_cls': dict(lr_mult=2.0),  # 增加分类头学习率
            'pts_bbox_head.conv_reg': dict(lr_mult=2.0)   # 增加回归头学习率
        }))
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),  # 增加梯度裁剪阈值
    type='Fp16OptimizerHook',  # 使用半精度训练
    loss_scale=dict(init_scale=512))

# 小目标专用学习率调度
lr_config = dict(
    policy='CosineAnnealing',  # 使用余弦退火
    warmup='linear',
    warmup_iters=2000,  # 增加预热轮数
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-6)

# 检查点和日志配置
checkpoint_config = dict(interval=2, max_keep_ckpts=10)
log_config = dict(
    interval=20,  # 更频繁的日志
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='rose-small-objects'))
    ])

# 运行时设置
total_epochs = 100  # 增加训练轮数
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# 小目标专用自定义钩子
custom_hooks = [
    # ROSE训练钩子 - 小目标优化
    dict(
        type='ROSETrainingHook',
        work_dir='${work_dir}',
        ssl_enabled=True,
        augmentation_strategy=dict(
            ssl_parameters=dict(
                lambda_det=1.0,
                lambda_cm=0.8,
                lambda_cons=0.6,
                lambda_spatial=0.5,
                lambda_weather=0.3,
                small_object_enhancement=True
            ),
            teacher_config=dict(
                ema_decay=0.9995,
                warmup_epochs=3
            ),
            small_object_focus=True
        ),
        save_interval=50,
        visualization_interval=50,  # 更频繁的可视化
        performance_log_interval=1
    ),
    
    # 小目标可视化钩子
    dict(
        type='ROSEVisualizationHook',
        enable_visualization=True,
        enable_analytics=True,
        save_interval=50,
        visualization_samples=8,
        focus_classes=['Pedestrian', 'Cyclist']
    ),
    
    # 早停钩子 - 监控小目标mAP
    dict(
        type='EarlyStoppingHook',
        monitor='bbox_mAP_3d_Pedestrian',
        patience=10,
        min_delta=0.01)
]

# 小目标专用评估配置
evaluation = dict(
    interval=2,
    pipeline=test_pipeline,
    by_epoch=True,
    save_best='bbox_mAP_3d',
    rule='greater',
    # 小目标特定评估指标
    metric_options=dict(
        classwise=True,
        class_agnostic=False,
        small_object_eval=True))

# 数据预处理配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)

# 模型参数初始化
init_cfg = dict(
    type='Xavier',
    layer=['Conv1d', 'Conv2d', 'Linear'],
    distribution='uniform')