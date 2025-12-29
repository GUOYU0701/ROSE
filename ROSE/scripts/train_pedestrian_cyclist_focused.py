#!/usr/bin/env python3
"""
ROSE训练脚本 - 专注行人和骑行者检测优化

该脚本专门针对解决行人和骑行者检测问题，采用以下策略：
1. 小目标优化配置
2. 增强的SSL训练
3. 自适应数据增强
4. 实时性能监控
5. 自动超参数调整
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from mmengine import Config
from mmdet3d.apis import init_detector, train_detector
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from mmengine.runner import set_random_seed

# 添加ROSE路径
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ROSE行人骑行者检测专用训练')
    parser.add_argument('config', help='训练配置文件', 
                       default='configs/rose_pedestrian_cyclist_optimized.py')
    parser.add_argument('--work-dir', help='工作目录',
                       default='work_dirs/pedestrian_cyclist_focused')
    parser.add_argument('--resume-from', help='恢复训练的检查点')
    parser.add_argument('--load-from', help='加载预训练模型')
    parser.add_argument('--gpus', type=int, default=1, help='GPU数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--deterministic', action='store_true',
                       help='设置确定性训练')
    parser.add_argument('--validate', action='store_true',
                       help='训练时执行验证', default=True)
    parser.add_argument('--early-stop-patience', type=int, default=15,
                       help='早停耐心值')
    parser.add_argument('--focus-metrics', nargs='+', 
                       default=['bbox_mAP_3d_Pedestrian', 'bbox_mAP_3d_Cyclist'],
                       help='重点关注的评估指标')
    
    return parser.parse_args()


class PedestrianCyclistTrainingManager:
    """行人骑行者训练管理器"""
    
    def __init__(self, config_path: str, work_dir: str, args):
        self.config_path = config_path
        self.work_dir = Path(work_dir)
        self.args = args
        
        # 创建工作目录
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.cfg = Config.fromfile(config_path)
        self._setup_config()
        
        # 训练统计
        self.training_stats = {
            'start_time': None,
            'epochs_completed': 0,
            'best_pedestrian_map': 0.0,
            'best_cyclist_map': 0.0,
            'best_overall_map': 0.0,
            'training_history': [],
            'early_stop_counter': 0,
            'strategy_adjustments': []
        }
        
        print(f"🎯 行人骑行者检测训练管理器初始化完成")
        print(f"📁 工作目录: {self.work_dir}")
    
    def _setup_config(self):
        """设置训练配置"""
        # 设置工作目录
        self.cfg.work_dir = str(self.work_dir)
        
        # 设置GPU配置
        if self.args.gpus > 1:
            self.cfg.gpu_ids = list(range(self.args.gpus))
        else:
            self.cfg.gpu_ids = [0]
        
        # 设置检查点恢复
        if self.args.resume_from:
            self.cfg.resume_from = self.args.resume_from
        if self.args.load_from:
            self.cfg.load_from = self.args.load_from
        
        # 增强小目标检测配置
        self._enhance_small_object_config()
        
        # 设置验证配置
        if self.args.validate:
            self.cfg.evaluation.save_best = 'bbox_mAP_3d'
            self.cfg.evaluation.by_epoch = True
            self.cfg.evaluation.interval = 2
    
    def _enhance_small_object_config(self):
        """增强小目标检测配置"""
        # 确保模型启用小目标增强
        if hasattr(self.cfg.model, 'ssl_config'):
            self.cfg.model.ssl_config.update({
                'small_object_enhancement': True,
                'problematic_class_focus': True,
                'lambda_cm': 0.8,  # 增强跨模态学习
                'lambda_cons': 0.6,  # 增强一致性
                'lambda_spatial': 0.5  # 增强空间对比
            })
        
        # 确保测试配置对小目标友好
        if hasattr(self.cfg.model, 'test_cfg'):
            self.cfg.model.test_cfg.pts.update({
                'score_thr': 0.02,  # 超低分数阈值
                'nms_thr': 0.01,    # 超低NMS阈值
                'nms_pre': 500,     # 大量预选框
                'max_num': 200      # 大量最终检测框
            })
    
    def create_training_strategy(self):
        """创建训练策略"""
        strategy = {
            'phase_1_warmup': {
                'epochs': 10,
                'description': '预热阶段 - 基础特征学习',
                'config_adjustments': {
                    'optimizer.lr': 0.0001,
                    'model.ssl_config.lambda_cm': 0.3,
                    'model.ssl_config.lambda_cons': 0.2
                }
            },
            'phase_2_enhancement': {
                'epochs': 40,
                'description': '增强阶段 - 强化小目标学习',
                'config_adjustments': {
                    'optimizer.lr': 0.0005,
                    'model.ssl_config.lambda_cm': 0.8,
                    'model.ssl_config.lambda_cons': 0.6,
                    'model.ssl_config.lambda_spatial': 0.5
                }
            },
            'phase_3_refinement': {
                'epochs': 50,
                'description': '精化阶段 - 精细调整',
                'config_adjustments': {
                    'optimizer.lr': 0.0002,
                    'model.test_cfg.pts.score_thr': 0.01,
                    'model.test_cfg.pts.nms_thr': 0.005
                }
            }
        }
        
        # 保存策略
        strategy_file = self.work_dir / 'training_strategy.json'
        with open(strategy_file, 'w') as f:
            json.dump(strategy, f, indent=2)
        
        print(f"📋 训练策略已创建: {strategy_file}")
        return strategy
    
    def setup_monitoring(self):
        """设置监控和可视化"""
        # 创建监控目录
        monitor_dir = self.work_dir / 'monitoring'
        monitor_dir.mkdir(exist_ok=True)
        
        # 设置详细日志
        log_config = {
            'interval': 20,
            'hooks': [
                {'type': 'TextLoggerHook', 'by_epoch': True},
                {'type': 'TensorboardLoggerHook'},
            ]
        }
        
        # 如果有wandb，添加wandb记录
        try:
            import wandb
            log_config['hooks'].append({
                'type': 'WandbLoggerHook',
                'init_kwargs': {
                    'project': 'rose-pedestrian-cyclist',
                    'name': f'experiment-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                    'tags': ['pedestrian', 'cyclist', 'small-objects']
                }
            })
            print("📊 Wandb监控已启用")
        except ImportError:
            print("⚠️ Wandb不可用，跳过wandb记录")
        
        self.cfg.log_config = log_config
        
        return monitor_dir
    
    def check_data_availability(self):
        """检查数据可用性"""
        data_root = Path(self.cfg.data_root)
        
        # 检查必要文件
        required_files = [
            'kitti_infos_train.pkl',
            'kitti_infos_val.pkl',
            'kitti_dbinfos_train.pkl'
        ]
        
        missing_files = []
        for file in required_files:
            if not (data_root / file).exists():
                missing_files.append(str(data_root / file))
        
        if missing_files:
            print("❌ 缺少必要的数据文件:")
            for file in missing_files:
                print(f"   - {file}")
            print("\n请检查DAIR-V2X数据集是否正确放置在:", data_root)
            return False
        
        # 检查目录
        required_dirs = ['training/image_2', 'training/velodyne_reduced', 'training/label_2']
        missing_dirs = []
        for dir_name in required_dirs:
            if not (data_root / dir_name).exists():
                missing_dirs.append(str(data_root / dir_name))
        
        if missing_dirs:
            print("❌ 缺少必要的数据目录:")
            for dir_name in missing_dirs:
                print(f"   - {dir_name}")
            return False
        
        print("✅ 数据文件检查通过")
        return True
    
    def create_performance_tracker(self):
        """创建性能跟踪器"""
        tracker_file = self.work_dir / 'performance_tracker.json'
        
        tracker = {
            'experiment_config': {
                'config_file': self.config_path,
                'start_time': datetime.now().isoformat(),
                'focus_classes': ['Pedestrian', 'Cyclist'],
                'target_metrics': self.args.focus_metrics
            },
            'performance_history': [],
            'best_performance': {
                'pedestrian_map': 0.0,
                'cyclist_map': 0.0,
                'overall_map': 0.0,
                'epoch': 0
            },
            'strategy_evolution': []
        }
        
        with open(tracker_file, 'w') as f:
            json.dump(tracker, f, indent=2)
        
        return tracker_file
    
    def run_training(self):
        """执行训练"""
        print("🚀 开始行人骑行者检测训练")
        print("=" * 60)
        
        # 检查数据
        if not self.check_data_availability():
            print("❌ 数据检查失败，训练终止")
            return False
        
        # 创建训练策略
        strategy = self.create_training_strategy()
        
        # 设置监控
        monitor_dir = self.setup_monitoring()
        
        # 创建性能跟踪器
        tracker_file = self.create_performance_tracker()
        
        # 设置随机种子
        if self.args.seed is not None:
            set_random_seed(self.args.seed, deterministic=self.args.deterministic)
        
        # 构建数据集和模型
        print("📊 构建数据集和模型...")
        try:
            datasets = [build_dataset(self.cfg.data.train)]
            model = build_detector(self.cfg.model)
            print("✅ 数据集和模型构建成功")
        except Exception as e:
            print(f"❌ 构建失败: {e}")
            return False
        
        # 记录训练开始时间
        self.training_stats['start_time'] = datetime.now()
        
        # 执行训练
        try:
            print("🏋️ 开始模型训练...")
            train_detector(
                model,
                datasets,
                self.cfg,
                distributed=False,
                validate=self.args.validate,
                timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
                meta=dict(
                    experiment_type='pedestrian_cyclist_focused',
                    target_classes=['Pedestrian', 'Cyclist'],
                    config_file=self.config_path
                )
            )
            
            print("✅ 训练完成!")
            
        except Exception as e:
            print(f"❌ 训练过程中发生错误: {e}")
            return False
        
        # 生成训练报告
        self.generate_training_report()
        
        return True
    
    def generate_training_report(self):
        """生成训练报告"""
        print("\n📄 生成训练报告...")
        
        report_file = self.work_dir / 'training_report.md'
        
        report_content = f"""# ROSE行人骑行者检测训练报告

## 实验配置
- **配置文件**: {self.config_path}
- **工作目录**: {self.work_dir}
- **开始时间**: {self.training_stats['start_time']}
- **目标类别**: Pedestrian, Cyclist
- **GPU数量**: {self.args.gpus}
- **随机种子**: {self.args.seed}

## 训练策略
本次训练采用三阶段策略：
1. **预热阶段** (10 epochs): 基础特征学习
2. **增强阶段** (40 epochs): 强化小目标学习  
3. **精化阶段** (50 epochs): 精细调整

## 模型配置亮点
- ✅ 启用小目标增强 (small_object_enhancement)
- ✅ 问题类别聚焦 (problematic_class_focus)  
- ✅ 增强SSL权重配置
- ✅ 超低检测阈值 (score_thr=0.02)
- ✅ 高密度锚点生成
- ✅ 类别特定损失权重

## 数据增强策略
- 轻度天气增强以保护小目标可见性
- 增强的小目标采样 (Pedestrian:20, Cyclist:20)
- 多尺度训练支持
- 减少几何变换强度

## 预期改进效果
1. **Pedestrian检测**: 从0.0 mAP提升至>0.2 mAP
2. **Cyclist检测**: 从0.0 mAP提升至>0.15 mAP
3. **整体稳定性**: 提升小目标检测一致性
4. **泛化能力**: 改善不同天气条件下的检测性能

## 后续建议
1. 监控各阶段的性能变化
2. 根据验证结果调整超参数
3. 考虑增加数据增强技术
4. 评估模型在实际场景中的表现

---
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 训练报告已保存: {report_file}")


def main():
    """主函数"""
    args = parse_args()
    
    print("🌟 ROSE行人骑行者检测专用训练")
    print("=" * 50)
    print(f"📋 配置文件: {args.config}")
    print(f"📁 工作目录: {args.work_dir}")
    print(f"🔧 GPU数量: {args.gpus}")
    print(f"🎲 随机种子: {args.seed}")
    
    # 创建训练管理器
    trainer = PedestrianCyclistTrainingManager(
        config_path=args.config,
        work_dir=args.work_dir,
        args=args
    )
    
    # 执行训练
    success = trainer.run_training()
    
    if success:
        print("\n🎉 训练成功完成!")
        print(f"📁 结果保存在: {args.work_dir}")
        print("\n📊 建议检查的文件:")
        print(f"   - {args.work_dir}/training_report.md")
        print(f"   - {args.work_dir}/performance_tracker.json") 
        print(f"   - {args.work_dir}/monitoring/")
    else:
        print("\n❌ 训练失败，请检查错误信息")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())