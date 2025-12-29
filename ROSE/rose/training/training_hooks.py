"""
ROSE训练钩子
集成SSL损失计算和训练监控的自定义钩子
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import json
import yaml
from datetime import datetime

@HOOKS.register_module()
class ROSETrainingHook(Hook):
    """ROSE训练钩子"""
    
    def __init__(self,
                 work_dir: str = None,
                 augmentation_strategy: Dict = None,
                 save_interval: int = 100,
                 visualization_interval: int = 200,
                 ssl_enabled: bool = True,
                 priority: int = 'NORMAL',
                 augmentation_adaptation_interval: int = 5,
                 save_augmentation_plan: bool = True,
                 visualize_augmentation: bool = True,
                 teacher_update_interval: int = 1,
                 performance_log_interval: int = 1,
                 **kwargs):
        super().__init__()
        
        self.work_dir = Path(work_dir) if work_dir else Path('work_dirs/default')
        self.augmentation_strategy = augmentation_strategy or {}
        self.save_interval = save_interval
        self.visualization_interval = visualization_interval
        self.ssl_enabled = ssl_enabled
        self.augmentation_adaptation_interval = augmentation_adaptation_interval
        self.save_augmentation_plan = save_augmentation_plan
        self.visualize_augmentation = visualize_augmentation
        self.teacher_update_interval = teacher_update_interval
        self.performance_log_interval = performance_log_interval
        
        # SSL训练器
        self.ssl_trainer = None
        
        # 训练统计
        self.training_stats = {
            'iterations': 0,
            'ssl_losses': [],
            'detection_losses': [],
            'total_losses': [],
            'learning_rates': [],
            'batch_sizes': []
        }
        
        # 创建输出目录
        self.log_dir = self.work_dir / 'training_logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ ROSE训练钩子初始化完成")
        print(f"   工作目录: {self.work_dir}")
        print(f"   SSL启用: {self.ssl_enabled}")
    
    def before_run(self, runner) -> None:
        """训练开始前的准备"""
        print("ROSE训练钩子: 开始训练前准备...")
        
        # 初始化SSL训练器
        if self.ssl_enabled:
            from rose.ssl_training.ssl_trainer import SSLTrainer
            self.ssl_trainer = SSLTrainer(
                lambda_det=1.0,
                lambda_cm=0.5,
                lambda_cons=0.3,
                lambda_spatial=0.2,
                lambda_weather=0.4,
                ema_decay=0.999,
                consistency_warmup_epochs=5,
                enable_pseudo_labeling=True
            )
            
            # 初始化教师模型
            if hasattr(runner.model, 'module'):
                student_model = runner.model.module
            else:
                student_model = runner.model
            
            self.ssl_trainer.initialize_teacher(student_model)
        
        # 保存增强策略
        strategy_file = self.log_dir / 'augmentation_strategy.yaml'
        with open(strategy_file, 'w') as f:
            yaml.dump(self.augmentation_strategy, f, default_flow_style=False)
        
        print("✅ ROSE训练前准备完成")
    
    def before_train_iter(self, runner, batch_idx: int, data_batch: Any) -> None:
        """每次训练迭代前的处理"""
        self.training_stats['iterations'] += 1
        
        # 记录批次大小
        if hasattr(data_batch, 'get'):
            batch_size = len(data_batch.get('img_metas', []))
        elif isinstance(data_batch, dict) and 'img_metas' in data_batch:
            batch_size = len(data_batch['img_metas'])
        else:
            batch_size = 1
        
        self.training_stats['batch_sizes'].append(batch_size)
        
        # 记录学习率
        if runner.optim_wrapper is not None:
            lr = runner.optim_wrapper.get_lr()['lr'][0] if runner.optim_wrapper.get_lr() else 0
            self.training_stats['learning_rates'].append(lr)
    
    def after_train_iter(self, runner, batch_idx: int, data_batch: Any = None, outputs: Optional[Dict] = None) -> None:
        """每次训练迭代后的处理"""
        
        # 处理SSL训练
        if self.ssl_enabled and self.ssl_trainer is not None and outputs is not None:
            # 获取模型
            if hasattr(runner.model, 'module'):
                student_model = runner.model.module
            else:
                student_model = runner.model
            
            # 计算SSL损失
            try:
                # 获取教师预测
                teacher_outputs = self.ssl_trainer.get_teacher_predictions(data_batch) if self.ssl_trainer.teacher else None
                
                # 计算SSL损失
                ssl_losses = self.ssl_trainer.compute_ssl_losses(
                    outputs, teacher_outputs, data_batch, self.training_stats['iterations']
                )
                
                # 计算总损失
                detection_losses = {k: v for k, v in outputs.items() if 'loss' in k.lower()}
                total_loss, loss_dict = self.ssl_trainer.compute_total_loss(detection_losses, ssl_losses)
                
                # 更新输出损失
                outputs['loss'] = total_loss
                
                # 记录损失统计
                ssl_loss_value = sum(v.item() if isinstance(v, torch.Tensor) else v for v in ssl_losses.values()) if ssl_losses else 0
                detection_loss_value = sum(v.item() if isinstance(v, torch.Tensor) else v for v in detection_losses.values()) if detection_losses else 0
                
                self.training_stats['ssl_losses'].append(ssl_loss_value)
                self.training_stats['detection_losses'].append(detection_loss_value)
                self.training_stats['total_losses'].append(total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss)
                
                # 添加所有损失到输出日志
                outputs.update(loss_dict)
                
                # 更新教师模型
                if self.training_stats['iterations'] % self.teacher_update_interval == 0:
                    self.ssl_trainer.update_teacher(student_model, self.training_stats['iterations'])
                
            except Exception as e:
                print(f"⚠️ SSL损失计算失败: {e}")
        
        # 记录常规损失
        if outputs is not None and 'loss' in outputs:
            if len(self.training_stats['ssl_losses']) == len(self.training_stats['total_losses']):
                # SSL已记录，跳过
                pass
            else:
                # 记录常规训练损失
                self.training_stats['detection_losses'].append(outputs['loss'].item())
                self.training_stats['total_losses'].append(outputs['loss'].item())
                self.training_stats['ssl_losses'].append(0.0)
        
        # 定期保存训练状态
        if self.training_stats['iterations'] % self.save_interval == 0:
            self._save_training_progress()
        
        # 定期保存SSL状态
        if (self.ssl_enabled and self.ssl_trainer is not None and 
            self.training_stats['iterations'] % self.visualization_interval == 0):
            epoch = runner.epoch if hasattr(runner, 'epoch') else 0
            self.ssl_trainer.save_ssl_state(epoch)
    
    def after_train_epoch(self, runner) -> None:
        """每个epoch结束后的处理"""
        epoch = runner.epoch if hasattr(runner, 'epoch') else 0
        
        # 设置SSL训练器的epoch
        if self.ssl_enabled and self.ssl_trainer is not None:
            self.ssl_trainer.set_epoch(epoch)
        
        # 生成epoch报告
        self._generate_epoch_report(epoch)
        
        print(f"✅ Epoch {epoch} 训练完成")
    
    def after_run(self, runner) -> None:
        """训练结束后的清理"""
        print("ROSE训练钩子: 训练结束，生成最终报告...")
        
        # 保存最终训练统计
        self._save_final_statistics()
        
        # 生成最终可视化
        self._generate_final_visualizations()
        
        print("✅ ROSE训练钩子处理完成")
    
    def _save_training_progress(self):
        """保存训练进度"""
        progress_file = self.log_dir / 'training_progress.json'
        
        # 计算统计信息
        stats = self.training_stats.copy()
        
        if stats['total_losses']:
            stats['loss_statistics'] = {
                'mean_total_loss': float(np.mean(stats['total_losses'])),
                'mean_detection_loss': float(np.mean(stats['detection_losses'])),
                'mean_ssl_loss': float(np.mean(stats['ssl_losses'])) if stats['ssl_losses'] else 0.0,
                'latest_total_loss': float(stats['total_losses'][-1]),
                'latest_detection_loss': float(stats['detection_losses'][-1]),
                'latest_ssl_loss': float(stats['ssl_losses'][-1]) if stats['ssl_losses'] else 0.0
            }
        
        if stats['learning_rates']:
            stats['learning_rate_statistics'] = {
                'current_lr': float(stats['learning_rates'][-1]),
                'mean_lr': float(np.mean(stats['learning_rates']))
            }
        
        stats['timestamp'] = datetime.now().isoformat()
        
        with open(progress_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def _generate_epoch_report(self, epoch: int):
        """生成epoch报告"""
        if not self.training_stats['total_losses']:
            return
        
        # 计算epoch统计
        epoch_stats = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'iterations': self.training_stats['iterations'],
            'losses': {
                'mean_total': float(np.mean(self.training_stats['total_losses'][-100:])),  # 最近100次迭代
                'mean_detection': float(np.mean(self.training_stats['detection_losses'][-100:])),
                'mean_ssl': float(np.mean(self.training_stats['ssl_losses'][-100:])) if self.training_stats['ssl_losses'] else 0.0,
                'latest_total': float(self.training_stats['total_losses'][-1]),
                'latest_detection': float(self.training_stats['detection_losses'][-1]),
                'latest_ssl': float(self.training_stats['ssl_losses'][-1]) if self.training_stats['ssl_losses'] else 0.0
            },
            'ssl_enabled': self.ssl_enabled,
            'augmentation_strategy': self.augmentation_strategy
        }
        
        # SSL分析
        if self.ssl_enabled and self.ssl_trainer is not None:
            epoch_stats['ssl_analytics'] = self.ssl_trainer.get_ssl_analytics_summary()
        
        # 保存epoch报告
        epoch_file = self.log_dir / f'epoch_{epoch}_report.json'
        with open(epoch_file, 'w') as f:
            json.dump(epoch_stats, f, indent=2, default=str)
        
        print(f"✅ Epoch {epoch} 报告已生成: {epoch_file}")
    
    def _save_final_statistics(self):
        """保存最终统计信息"""
        final_stats = {
            'training_summary': {
                'total_iterations': self.training_stats['iterations'],
                'ssl_enabled': self.ssl_enabled,
                'augmentation_strategy': self.augmentation_strategy
            },
            'loss_summary': {},
            'training_curve': {
                'iterations': list(range(len(self.training_stats['total_losses']))),
                'total_losses': self.training_stats['total_losses'],
                'detection_losses': self.training_stats['detection_losses'],
                'ssl_losses': self.training_stats['ssl_losses']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 损失统计
        if self.training_stats['total_losses']:
            final_stats['loss_summary'] = {
                'total_loss': {
                    'mean': float(np.mean(self.training_stats['total_losses'])),
                    'std': float(np.std(self.training_stats['total_losses'])),
                    'min': float(np.min(self.training_stats['total_losses'])),
                    'max': float(np.max(self.training_stats['total_losses'])),
                    'final': float(self.training_stats['total_losses'][-1])
                },
                'detection_loss': {
                    'mean': float(np.mean(self.training_stats['detection_losses'])),
                    'std': float(np.std(self.training_stats['detection_losses'])),
                    'min': float(np.min(self.training_stats['detection_losses'])),
                    'max': float(np.max(self.training_stats['detection_losses'])),
                    'final': float(self.training_stats['detection_losses'][-1])
                },
                'ssl_loss': {
                    'mean': float(np.mean(self.training_stats['ssl_losses'])) if self.training_stats['ssl_losses'] else 0.0,
                    'std': float(np.std(self.training_stats['ssl_losses'])) if self.training_stats['ssl_losses'] else 0.0,
                    'min': float(np.min(self.training_stats['ssl_losses'])) if self.training_stats['ssl_losses'] else 0.0,
                    'max': float(np.max(self.training_stats['ssl_losses'])) if self.training_stats['ssl_losses'] else 0.0,
                    'final': float(self.training_stats['ssl_losses'][-1]) if self.training_stats['ssl_losses'] else 0.0
                }
            }
        
        # SSL最终分析
        if self.ssl_enabled and self.ssl_trainer is not None:
            final_stats['ssl_final_analytics'] = self.ssl_trainer.get_ssl_analytics_summary()
        
        # 保存最终统计
        final_file = self.log_dir / 'FINAL_TRAINING_STATISTICS.json'
        with open(final_file, 'w') as f:
            json.dump(final_stats, f, indent=2, default=str)
        
        print(f"✅ 最终训练统计已保存: {final_file}")
    
    def _generate_final_visualizations(self):
        """生成最终可视化"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.training_stats['total_losses']:
                print("⚠️ 没有损失数据，跳过可视化")
                return
            
            # 创建损失曲线图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            iterations = list(range(len(self.training_stats['total_losses'])))
            
            # 总损失
            axes[0, 0].plot(iterations, self.training_stats['total_losses'], 'b-', alpha=0.7)
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            
            # 检测损失
            axes[0, 1].plot(iterations, self.training_stats['detection_losses'], 'g-', alpha=0.7)
            axes[0, 1].set_title('Detection Loss')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
            
            # SSL损失
            if self.training_stats['ssl_losses'] and any(loss > 0 for loss in self.training_stats['ssl_losses']):
                axes[1, 0].plot(iterations, self.training_stats['ssl_losses'], 'r-', alpha=0.7)
                axes[1, 0].set_title('SSL Loss')
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True)
            else:
                axes[1, 0].text(0.5, 0.5, 'No SSL Loss Data', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('SSL Loss (No Data)')
            
            # 学习率
            if self.training_stats['learning_rates']:
                lr_iterations = list(range(len(self.training_stats['learning_rates'])))
                axes[1, 1].plot(lr_iterations, self.training_stats['learning_rates'], 'm-', alpha=0.7)
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Iteration')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(0.5, 0.5, 'No LR Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Learning Rate (No Data)')
            
            plt.tight_layout()
            
            # 保存可视化
            viz_file = self.log_dir / 'training_curves.png'
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 训练曲线已保存: {viz_file}")
            
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")