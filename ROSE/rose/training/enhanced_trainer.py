"""
Enhanced ROSE Trainer with Adaptive Augmentation
训练过程中实现数据增强输出、策略保存、自动验证评估和策略优化
"""

import os
import json
import yaml
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class EnhancedROSETrainingHook(Hook):
    """增强的ROSE训练钩子，实现完整的自适应增强训练流程"""
    
    def __init__(self,
                 work_dir: str,
                 save_augmented_samples: bool = True,
                 samples_per_epoch: int = 50,
                 visualization_enabled: bool = True,
                 auto_validation: bool = True,
                 strategy_adaptation: bool = True,
                 performance_analysis: bool = True,
                 priority: int = 'NORMAL',
                 **kwargs):
        super().__init__()
        
        self.work_dir = Path(work_dir)
        self.save_augmented_samples = save_augmented_samples
        self.samples_per_epoch = samples_per_epoch
        self.visualization_enabled = visualization_enabled
        self.auto_validation = auto_validation
        self.strategy_adaptation = strategy_adaptation
        self.performance_analysis = performance_analysis
        
        # 创建输出目录结构
        self.setup_directories()
        
        # 初始化统计信息
        self.training_history = {
            'epochs': [],
            'losses': [],
            'val_maps': [],
            'strategies': [],
            'performance_analysis': []
        }
        
        # 当前增强策略
        self.current_strategy = self.get_initial_strategy()
        
        # 保存的增强样本计数器
        self.saved_samples_count = 0
        
        print(f"✅ 增强ROSE训练钩子初始化完成")
        print(f"   工作目录: {self.work_dir}")
        print(f"   保存增强样本: {self.save_augmented_samples}")
        print(f"   自动验证: {self.auto_validation}")
        print(f"   策略自适应: {self.strategy_adaptation}")
    
    def setup_directories(self):
        """创建完整的输出目录结构"""
        self.aug_output_dir = self.work_dir / 'augmented_outputs'
        self.strategy_dir = self.work_dir / 'augmentation_strategies'
        self.validation_dir = self.work_dir / 'validation_results'
        self.visualization_dir = self.work_dir / 'visualizations'
        self.analysis_dir = self.work_dir / 'performance_analysis'
        
        for dir_path in [self.aug_output_dir, self.strategy_dir, 
                        self.validation_dir, self.visualization_dir, 
                        self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 为每个天气条件创建子目录
        weather_types = ['clear', 'rain', 'snow', 'fog']
        for weather in weather_types:
            (self.aug_output_dir / weather / 'images').mkdir(parents=True, exist_ok=True)
            (self.aug_output_dir / weather / 'pointclouds').mkdir(parents=True, exist_ok=True)
    
    def get_initial_strategy(self) -> Dict:
        """获取初始增强策略"""
        return {
            'epoch': 0,
            'weather_probabilities': {
                'clear': 0.4,
                'rain': 0.2,
                'snow': 0.2,  
                'fog': 0.2
            },
            'intensity_levels': {
                'rain': 0.3,
                'snow': 0.4,
                'fog': 0.5
            },
            'adaptation_params': {
                'performance_threshold': 0.65,
                'improvement_rate': 0.02,
                'min_intensity': 0.1,
                'max_intensity': 0.8
            }
        }
    
    def before_run(self, runner) -> None:
        """训练开始前的初始化"""
        print("🚀 开始增强ROSE训练流程...")
        
        # 保存初始策略
        self.save_strategy(0, self.current_strategy)
        
        # 初始化训练历史记录
        history_file = self.work_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def before_train_epoch(self, runner) -> None:
        """每个训练轮次开始前"""
        current_epoch = runner.epoch
        print(f"\\n📊 开始第 {current_epoch + 1} 轮训练")
        print(f"当前增强策略: {self.current_strategy['weather_probabilities']}")
        
        # 重置样本计数器
        self.saved_samples_count = 0
        
        # 更新策略中的轮次信息
        self.current_strategy['epoch'] = current_epoch
    
    def after_train_iter(self, runner, batch_idx: int, data_batch: Any, outputs: Dict) -> None:
        """训练迭代后处理 - 保存增强样本"""
        if not self.save_augmented_samples:
            return
            
        # 限制每轮保存的样本数量
        if self.saved_samples_count >= self.samples_per_epoch:
            return
        
        current_epoch = runner.epoch
        
        # 每20个batch保存一次样本
        if batch_idx % 20 == 0:
            self.save_augmented_sample(data_batch, current_epoch, batch_idx)
            self.saved_samples_count += 1
    
    def save_augmented_sample(self, data_batch: Any, epoch: int, batch_idx: int):
        """保存增强后的样本数据"""
        try:
            # 随机选择一个天气类型进行增强
            weather_types = list(self.current_strategy['weather_probabilities'].keys())
            weather_probs = list(self.current_strategy['weather_probabilities'].values())
            selected_weather = np.random.choice(weather_types, p=weather_probs)
            
            if selected_weather == 'clear':
                return  # 清晰天气不需要保存增强样本
            
            # 获取批次数据
            if hasattr(data_batch, 'data_samples'):
                sample = data_batch.data_samples[0] if len(data_batch.data_samples) > 0 else None
            elif isinstance(data_batch, dict):
                sample = data_batch
            else:
                return
            
            if sample is None:
                return
            
            # 保存图像数据
            if 'img' in sample:
                self.save_augmented_image(sample['img'], selected_weather, epoch, batch_idx)
            
            # 保存点云数据  
            if 'points' in sample:
                self.save_augmented_pointcloud(sample['points'], selected_weather, epoch, batch_idx)
            
            print(f"✅ 保存增强样本: Epoch{epoch}_Batch{batch_idx}_{selected_weather}")
            
        except Exception as e:
            print(f"⚠️ 保存增强样本失败: {e}")
    
    def save_augmented_image(self, img_tensor: torch.Tensor, weather: str, epoch: int, batch_idx: int):
        """保存增强后的图像"""
        try:
            # 转换tensor为numpy数组
            if isinstance(img_tensor, torch.Tensor):
                img_np = img_tensor.detach().cpu().numpy()
            else:
                img_np = img_tensor
            
            # 处理图像维度 (C, H, W) -> (H, W, C)
            if len(img_np.shape) == 3 and img_np.shape[0] in [1, 3]:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            # 归一化到0-255
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            
            # 应用天气增强效果
            augmented_img = self.apply_weather_augmentation(img_np, weather)
            
            # 保存图像
            output_path = self.aug_output_dir / weather / 'images' / f'epoch_{epoch}_batch_{batch_idx}.jpg'
            cv2.imwrite(str(output_path), augmented_img)
            
        except Exception as e:
            print(f"⚠️ 保存增强图像失败: {e}")
    
    def save_augmented_pointcloud(self, points_tensor: torch.Tensor, weather: str, epoch: int, batch_idx: int):
        """保存增强后的点云"""
        try:
            # 转换tensor为numpy数组
            if isinstance(points_tensor, torch.Tensor):
                points_np = points_tensor.detach().cpu().numpy()
            else:
                points_np = points_tensor
            
            # 应用天气增强效果 (简化版本)
            augmented_points = self.apply_pointcloud_weather_augmentation(points_np, weather)
            
            # 保存点云数据
            output_path = self.aug_output_dir / weather / 'pointclouds' / f'epoch_{epoch}_batch_{batch_idx}.npy'
            np.save(str(output_path), augmented_points)
            
        except Exception as e:
            print(f"⚠️ 保存增强点云失败: {e}")
    
    def apply_weather_augmentation(self, img: np.ndarray, weather: str) -> np.ndarray:
        """应用天气增强效果到图像"""
        augmented = img.copy()
        
        if weather == 'rain':
            # 雨天效果：降低亮度，增加噪声
            augmented = cv2.convertScaleAbs(augmented, alpha=0.8, beta=-10)
            noise = np.random.normal(0, 5, augmented.shape).astype(np.uint8)
            augmented = cv2.add(augmented, noise)
            
        elif weather == 'snow':
            # 雪天效果：增加亮度，添加雪花噪声
            augmented = cv2.convertScaleAbs(augmented, alpha=1.1, beta=20)
            # 添加雪花效果
            snow_mask = np.random.random(augmented.shape[:2]) < 0.005
            augmented[snow_mask] = 255
            
        elif weather == 'fog':
            # 雾天效果：降低对比度，增加亮度
            augmented = cv2.convertScaleAbs(augmented, alpha=0.7, beta=30)
            # 添加高斯模糊
            augmented = cv2.GaussianBlur(augmented, (3, 3), 0)
        
        return augmented
    
    def apply_pointcloud_weather_augmentation(self, points: np.ndarray, weather: str) -> np.ndarray:
        """应用天气增强效果到点云"""
        augmented = points.copy()
        
        if weather == 'rain':
            # 雨天：随机移除一些点（模拟雨滴遮挡）
            keep_ratio = 0.9
            num_points = len(augmented)
            keep_indices = np.random.choice(num_points, int(num_points * keep_ratio), replace=False)
            augmented = augmented[keep_indices]
            
        elif weather == 'snow':
            # 雪天：添加随机噪声到距离
            noise = np.random.normal(0, 0.02, (len(augmented), 3))
            augmented[:, :3] += noise
            
        elif weather == 'fog':
            # 雾天：基于距离随机移除远距离点
            distances = np.sqrt(np.sum(augmented[:, :3]**2, axis=1))
            keep_mask = np.random.random(len(augmented)) > (distances / 100.0) * 0.3
            augmented = augmented[keep_mask]
        
        return augmented
    
    def after_train_epoch(self, runner) -> None:
        """训练轮次结束后的处理"""
        current_epoch = runner.epoch
        
        print(f"\\n🏁 第 {current_epoch + 1} 轮训练完成")
        print(f"   保存的增强样本数: {self.saved_samples_count}")
        
        # 保存当前轮次的策略
        self.save_strategy(current_epoch + 1, self.current_strategy)
        
        # 执行自动验证（如果启用）
        if self.auto_validation and (current_epoch + 1) % 2 == 0:  # 每2轮验证一次
            val_results = self.run_validation(runner, current_epoch + 1)
            
            # 性能分析
            if self.performance_analysis and val_results:
                analysis = self.analyze_performance(val_results, current_epoch + 1)
                
                # 策略自适应
                if self.strategy_adaptation:
                    self.adapt_strategy(analysis, current_epoch + 1)
        
        # 更新训练历史
        self.update_training_history(runner, current_epoch + 1)
    
    def save_strategy(self, epoch: int, strategy: Dict):
        """保存增强策略"""
        strategy_file = self.strategy_dir / f'strategy_epoch_{epoch}.yaml'
        
        strategy_with_timestamp = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            **strategy
        }
        
        with open(strategy_file, 'w') as f:
            yaml.dump(strategy_with_timestamp, f, default_flow_style=False)
        
        print(f"📝 保存增强策略: {strategy_file}")
    
    def run_validation(self, runner, epoch: int) -> Optional[Dict]:
        """运行验证评估"""
        print(f"\\n🔍 开始第 {epoch} 轮验证评估...")
        
        try:
            # 运行验证循环
            val_results = runner.val_loop.run()
            
            if val_results:
                # 保存验证结果
                results_file = self.validation_dir / f'validation_epoch_{epoch}.json'
                with open(results_file, 'w') as f:
                    json.dump(val_results, f, indent=2)
                
                # 提取mAP值
                mAP = self.extract_mAP(val_results)
                print(f"✅ 验证完成 - mAP: {mAP:.4f}")
                
                # 生成可视化
                if self.visualization_enabled:
                    self.generate_validation_visualization(val_results, epoch)
                
                return val_results
            
        except Exception as e:
            print(f"⚠️ 验证评估失败: {e}")
            return None
    
    def extract_mAP(self, val_results: Dict) -> float:
        """从验证结果中提取mAP值"""
        try:
            # 查找mAP相关的键
            map_keys = [k for k in val_results.keys() if 'mAP' in k or 'Overall' in k]
            if map_keys:
                return float(val_results[map_keys[0]])
            
            # 如果没有找到mAP，尝试其他指标
            metric_keys = [k for k in val_results.keys() if isinstance(val_results[k], (int, float))]
            if metric_keys:
                return float(val_results[metric_keys[0]])
            
            return 0.0
            
        except:
            return 0.0
    
    def generate_validation_visualization(self, val_results: Dict, epoch: int):
        """生成验证结果可视化"""
        try:
            # 创建性能图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Validation Results - Epoch {epoch}', fontsize=16)
            
            # 提取各类别性能
            classes = ['Pedestrian', 'Cyclist', 'Car']
            metrics = ['Easy', 'Moderate', 'Hard']
            
            # 绘制各类别mAP
            for i, cls in enumerate(classes):
                ax = axes[i // 2, i % 2] if i < 3 else axes[1, 1]
                
                easy_key = f'KITTI/{cls}_3D_easy'
                moderate_key = f'KITTI/{cls}_3D_moderate' 
                hard_key = f'KITTI/{cls}_3D_hard'
                
                values = []
                for key in [easy_key, moderate_key, hard_key]:
                    values.append(val_results.get(key, 0))
                
                ax.bar(metrics, values, alpha=0.7)
                ax.set_title(f'{cls} Detection Performance')
                ax.set_ylabel('mAP')
                ax.set_ylim(0, 1.0)
                
                # 添加数值标签
                for j, v in enumerate(values):
                    ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # 绘制整体性能趋势
            if len(self.training_history['val_maps']) > 0:
                ax = axes[1, 1]
                epochs = self.training_history['epochs']
                maps = self.training_history['val_maps'] 
                
                ax.plot(epochs, maps, 'b-o', linewidth=2, markersize=6)
                ax.set_title('Overall mAP Trend')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('mAP')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            viz_file = self.visualization_dir / f'validation_epoch_{epoch}.png'
            plt.savefig(str(viz_file), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 生成验证可视化: {viz_file}")
            
        except Exception as e:
            print(f"⚠️ 生成可视化失败: {e}")
    
    def analyze_performance(self, val_results: Dict, epoch: int) -> Dict:
        """分析性能结果"""
        print(f"\\n📈 开始性能分析...")
        
        analysis = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'overall_mAP': self.extract_mAP(val_results),
            'class_performance': {},
            'improvement_analysis': {},
            'recommendations': []
        }
        
        # 分析各类别性能
        classes = ['Pedestrian', 'Cyclist', 'Car']
        for cls in classes:
            easy_key = f'KITTI/{cls}_3D_easy'
            moderate_key = f'KITTI/{cls}_3D_moderate'
            hard_key = f'KITTI/{cls}_3D_hard'
            
            class_perf = {
                'easy': val_results.get(easy_key, 0),
                'moderate': val_results.get(moderate_key, 0),
                'hard': val_results.get(hard_key, 0),
                'average': np.mean([val_results.get(k, 0) for k in [easy_key, moderate_key, hard_key]])
            }
            
            analysis['class_performance'][cls] = class_perf
        
        # 改进分析
        if len(self.training_history['val_maps']) > 0:
            prev_mAP = self.training_history['val_maps'][-1]
            current_mAP = analysis['overall_mAP']
            improvement = current_mAP - prev_mAP
            
            analysis['improvement_analysis'] = {
                'previous_mAP': prev_mAP,
                'current_mAP': current_mAP,
                'improvement': improvement,
                'improvement_rate': improvement / max(prev_mAP, 0.001)
            }
        
        # 生成建议
        analysis['recommendations'] = self.generate_recommendations(analysis)
        
        # 保存分析结果
        analysis_file = self.analysis_dir / f'analysis_epoch_{epoch}.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"📋 性能分析完成 - 整体mAP: {analysis['overall_mAP']:.4f}")
        
        return analysis
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """基于性能分析生成改进建议"""
        recommendations = []
        
        # 基于整体mAP给出建议
        overall_mAP = analysis['overall_mAP']
        if overall_mAP < 0.3:
            recommendations.append("整体性能较低，建议增加基础数据增强强度")
        elif overall_mAP < 0.5:
            recommendations.append("性能中等，建议优化困难样本的增强策略")
        else:
            recommendations.append("性能良好，建议维持当前策略并进行微调")
        
        # 基于类别性能给出建议
        for cls, perf in analysis['class_performance'].items():
            avg_perf = perf['average']
            if avg_perf < 0.2:
                recommendations.append(f"{cls}类别性能很低，需要增加针对性增强")
            elif avg_perf < 0.4:
                recommendations.append(f"{cls}类别性能偏低，建议调整增强参数")
        
        # 基于改进趋势给出建议
        if 'improvement_analysis' in analysis:
            improvement = analysis['improvement_analysis']['improvement']
            if improvement < -0.05:
                recommendations.append("性能下降明显，建议降低增强强度")
            elif improvement > 0.05:
                recommendations.append("性能提升显著，可以适当增加增强难度")
        
        return recommendations
    
    def adapt_strategy(self, analysis: Dict, epoch: int):
        """基于性能分析自适应调整增强策略"""
        print(f"\\n🔄 开始策略自适应调整...")
        
        old_strategy = self.current_strategy.copy()
        
        # 基于整体性能调整
        overall_mAP = analysis['overall_mAP']
        threshold = self.current_strategy['adaptation_params']['performance_threshold']
        
        if overall_mAP < threshold * 0.7:  # 性能很差
            # 降低增强强度
            for weather in ['rain', 'snow', 'fog']:
                if weather in self.current_strategy['intensity_levels']:
                    self.current_strategy['intensity_levels'][weather] *= 0.9
                    
        elif overall_mAP > threshold * 1.2:  # 性能很好
            # 增加增强强度
            for weather in ['rain', 'snow', 'fog']:
                if weather in self.current_strategy['intensity_levels']:
                    self.current_strategy['intensity_levels'][weather] *= 1.1
        
        # 基于类别性能调整天气概率
        class_performances = analysis['class_performance']
        
        # 如果Pedestrian和Cyclist性能较差，增加雨雾天气概率（这些天气对小目标检测更有挑战）
        small_object_perf = (class_performances['Pedestrian']['average'] + 
                            class_performances['Cyclist']['average']) / 2
        
        if small_object_perf < 0.3:
            self.current_strategy['weather_probabilities']['rain'] *= 1.2
            self.current_strategy['weather_probabilities']['fog'] *= 1.2
            self.current_strategy['weather_probabilities']['clear'] *= 0.8
        
        # 规范化概率
        total_prob = sum(self.current_strategy['weather_probabilities'].values())
        for weather in self.current_strategy['weather_probabilities']:
            self.current_strategy['weather_probabilities'][weather] /= total_prob
        
        # 限制强度范围
        min_intensity = self.current_strategy['adaptation_params']['min_intensity']
        max_intensity = self.current_strategy['adaptation_params']['max_intensity']
        
        for weather in self.current_strategy['intensity_levels']:
            intensity = self.current_strategy['intensity_levels'][weather]
            self.current_strategy['intensity_levels'][weather] = np.clip(intensity, min_intensity, max_intensity)
        
        print(f"🎯 策略调整完成:")
        print(f"   旧策略天气概率: {old_strategy['weather_probabilities']}")
        print(f"   新策略天气概率: {self.current_strategy['weather_probabilities']}")
        print(f"   强度调整: {self.current_strategy['intensity_levels']}")
    
    def update_training_history(self, runner, epoch: int):
        """更新训练历史记录"""
        # 获取当前训练损失
        current_loss = getattr(runner.message_hub, 'train_loss', 0)
        
        self.training_history['epochs'].append(epoch)
        self.training_history['losses'].append(current_loss)
        self.training_history['strategies'].append(self.current_strategy.copy())
        
        # 如果有验证结果，添加到历史中
        if hasattr(self, '_last_val_mAP'):
            self.training_history['val_maps'].append(self._last_val_mAP)
        
        # 保存历史记录
        history_file = self.work_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
    
    def after_run(self, runner) -> None:
        """训练完成后的最终处理"""
        print(f"\\n🎉 训练流程完成！")
        
        # 生成最终报告
        self.generate_final_report()
        
        print(f"📊 完整训练报告已保存到: {self.work_dir}")
        print(f"   - 增强样本输出: {self.aug_output_dir}")
        print(f"   - 增强策略历史: {self.strategy_dir}")
        print(f"   - 验证结果: {self.validation_dir}")
        print(f"   - 可视化结果: {self.visualization_dir}")
        print(f"   - 性能分析: {self.analysis_dir}")
    
    def generate_final_report(self):
        """生成最终训练报告"""
        try:
            report = {
                'training_summary': {
                    'total_epochs': len(self.training_history['epochs']),
                    'final_loss': self.training_history['losses'][-1] if self.training_history['losses'] else None,
                    'final_mAP': self.training_history['val_maps'][-1] if self.training_history['val_maps'] else None,
                    'best_mAP': max(self.training_history['val_maps']) if self.training_history['val_maps'] else None
                },
                'strategy_evolution': self.training_history['strategies'],
                'performance_trend': {
                    'epochs': self.training_history['epochs'],
                    'losses': self.training_history['losses'],
                    'val_maps': self.training_history['val_maps']
                },
                'final_recommendations': self.generate_final_recommendations()
            }
            
            # 保存最终报告
            report_file = self.work_dir / 'final_training_report.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # 生成最终可视化
            self.generate_final_visualization()
            
        except Exception as e:
            print(f"⚠️ 生成最终报告失败: {e}")
    
    def generate_final_recommendations(self) -> List[str]:
        """生成最终建议"""
        recommendations = [
            "基于完整训练过程的建议:",
            "1. 根据保存的增强策略历史，选择性能最好的策略用于后续训练",
            "2. 分析各类别检测性能，针对性优化数据增强参数",
            "3. 考虑结合多种天气条件进行混合增强",
            "4. 根据可视化结果调整验证频率和增强强度"
        ]
        
        if self.training_history['val_maps']:
            best_epoch = self.training_history['epochs'][np.argmax(self.training_history['val_maps'])]
            recommendations.append(f"5. 最佳性能出现在第{best_epoch}轮，建议参考该轮的增强策略")
        
        return recommendations
    
    def generate_final_visualization(self):
        """生成最终可视化图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Complete Training Analysis', fontsize=16)
            
            # 训练损失趋势
            if self.training_history['losses']:
                axes[0, 0].plot(self.training_history['epochs'], self.training_history['losses'], 'b-o')
                axes[0, 0].set_title('Training Loss Trend')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 验证mAP趋势
            if self.training_history['val_maps']:
                axes[0, 1].plot(self.training_history['epochs'][:len(self.training_history['val_maps'])], 
                               self.training_history['val_maps'], 'g-o')
                axes[0, 1].set_title('Validation mAP Trend')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('mAP')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 增强策略演化 - 天气概率
            if self.training_history['strategies']:
                weather_types = ['rain', 'snow', 'fog']
                for weather in weather_types:
                    probs = [s['weather_probabilities'][weather] for s in self.training_history['strategies']]
                    axes[1, 0].plot(self.training_history['epochs'], probs, label=weather, marker='o')
                
                axes[1, 0].set_title('Weather Probability Evolution')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Probability')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # 增强强度演化
            if self.training_history['strategies']:
                weather_types = ['rain', 'snow', 'fog']
                for weather in weather_types:
                    intensities = [s.get('intensity_levels', {}).get(weather, 0) for s in self.training_history['strategies']]
                    axes[1, 1].plot(self.training_history['epochs'], intensities, label=weather, marker='s')
                
                axes[1, 1].set_title('Intensity Level Evolution')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Intensity')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存最终可视化
            final_viz_file = self.visualization_dir / 'final_training_analysis.png'
            plt.savefig(str(final_viz_file), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 生成最终可视化: {final_viz_file}")
            
        except Exception as e:
            print(f"⚠️ 生成最终可视化失败: {e}")


class ROSEEnhancedTrainer:
    """ROSE增强训练器主类"""
    
    def __init__(self, config_path: str, work_dir: str):
        self.config_path = config_path
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.cfg = Config.fromfile(config_path)
        
        # 添加增强训练钩子
        self.add_enhanced_hook()
    
    def add_enhanced_hook(self):
        """添加增强训练钩子到配置"""
        enhanced_hook = {
            'type': 'EnhancedROSETrainingHook',
            'work_dir': str(self.work_dir),
            'save_augmented_samples': True,
            'samples_per_epoch': 50,
            'visualization_enabled': True,
            'auto_validation': True,
            'strategy_adaptation': True,
            'performance_analysis': True,
            'priority': 'NORMAL'
        }
        
        if not hasattr(self.cfg, 'custom_hooks'):
            self.cfg.custom_hooks = []
        
        self.cfg.custom_hooks.append(enhanced_hook)
        
        # 更新工作目录
        self.cfg.work_dir = str(self.work_dir)
    
    def train(self):
        """开始增强训练流程"""
        print("🚀 启动ROSE增强训练系统...")
        
        # 创建并启动训练器
        runner = Runner.from_cfg(self.cfg)
        runner.train()
        
        print("✅ ROSE增强训练完成！")
        
        return runner


if __name__ == '__main__':
    # 使用示例
    trainer = ROSEEnhancedTrainer(
        config_path='configs/rose_simple_no_ssl.py',
        work_dir='work_dirs/enhanced_rose_training'
    )
    
    trainer.train()