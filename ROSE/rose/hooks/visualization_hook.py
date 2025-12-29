"""
ROSE可视化和分析钩子
集成检测结果可视化和训练分析
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class ROSEVisualizationHook(Hook):
    """ROSE可视化钩子"""
    
    def __init__(self,
                 enable_visualization: bool = True,
                 enable_analytics: bool = True,
                 save_interval: int = 100,
                 visualization_samples: int = 5):
        """
        初始化可视化钩子
        
        Args:
            enable_visualization: 是否启用可视化
            enable_analytics: 是否启用分析
            save_interval: 保存间隔
            visualization_samples: 可视化样本数
        """
        self.enable_visualization = enable_visualization
        self.enable_analytics = enable_analytics
        self.save_interval = save_interval
        self.visualization_samples = visualization_samples
        
        self.visualizer = None
        self.analytics = None
        
    def before_run(self, runner):
        """训练开始前初始化"""
        if not (self.enable_visualization or self.enable_analytics):
            return
            
        work_dir = runner.work_dir
        
        # 初始化可视化器
        if self.enable_visualization:
            try:
                from rose.visualization.augmentation_visualizer import AugmentationVisualizer
                self.visualizer = AugmentationVisualizer(
                    save_dir=os.path.join(work_dir, 'detection_visualization'),
                    enabled=True
                )
                print("✅ 检测结果可视化器已初始化")
            except ImportError as e:
                print(f"⚠️ 可视化器导入失败: {e}")
                
        # 初始化分析器
        if self.enable_analytics:
            try:
                from rose.analysis.training_analytics import ROSETrainingAnalytics
                self.analytics = ROSETrainingAnalytics(
                    work_dir=work_dir,
                    enabled=True
                )
                print("✅ 训练分析器已初始化")
            except ImportError as e:
                print(f"⚠️ 分析器导入失败: {e}")
    
    def after_train_iter(self, runner, batch_idx: int, data_batch: Dict, outputs: Dict):
        """训练迭代后处理"""
        if not self.enable_visualization or self.visualizer is None:
            return
            
        # 每隔一定间隔保存可视化结果
        if batch_idx % self.save_interval == 0:
            self._save_training_visualization(runner, batch_idx, data_batch, outputs)
    
    def after_val_epoch(self, runner, metrics: Optional[Dict] = None):
        """验证epoch后处理"""
        current_epoch = runner.epoch
        
        # 收集检测结果进行可视化
        if self.enable_visualization and self.visualizer is not None:
            self._save_validation_results(runner, current_epoch, metrics)
        
        # 记录训练数据用于分析
        if self.enable_analytics and self.analytics is not None:
            self._record_epoch_analytics(runner, current_epoch, metrics)
    
    def after_run(self, runner):
        """训练结束后处理"""
        # 生成最终报告
        if self.enable_visualization and self.visualizer is not None:
            try:
                self.visualizer.save_summary_report()
                print("✅ 可视化总结报告已生成")
            except Exception as e:
                print(f"❌ 可视化报告生成失败: {e}")
        
        if self.enable_analytics and self.analytics is not None:
            try:
                self.analytics.create_performance_visualizations()
                self.analytics.generate_comprehensive_report()
                self.analytics.save_training_data()
                print("✅ 训练分析报告已生成")
            except Exception as e:
                print(f"❌ 分析报告生成失败: {e}")
    
    def _save_training_visualization(self, runner, batch_idx: int, data_batch: Dict, outputs: Dict):
        """保存训练过程可视化"""
        try:
            # 获取数据
            inputs = data_batch.get('inputs', {})
            data_samples = data_batch.get('data_samples', [])
            
            # 限制可视化样本数
            num_samples = min(len(data_samples), self.visualization_samples)
            
            for i in range(num_samples):
                sample_id = f"train_epoch_{runner.epoch}_iter_{batch_idx}_sample_{i}"
                
                # 获取图像和点云
                if 'imgs' in inputs and len(inputs['imgs']) > i:
                    image = inputs['imgs'][i]
                    if hasattr(image, 'cpu'):
                        image = image.cpu().numpy()
                    if hasattr(image, 'transpose'):
                        image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
                    image = (image * 255).astype(np.uint8)
                else:
                    image = None
                
                if 'points' in inputs and len(inputs['points']) > i:
                    points = inputs['points'][i]
                    if hasattr(points, 'cpu'):
                        points = points.cpu().numpy()
                else:
                    points = None
                
                # 获取检测结果
                detections = []
                if 'pred_instances_3d' in outputs and len(outputs['pred_instances_3d']) > i:
                    pred_instances = outputs['pred_instances_3d'][i]
                    
                    if hasattr(pred_instances, 'bboxes_3d'):
                        bboxes_3d = pred_instances.bboxes_3d
                        scores_3d = pred_instances.scores_3d if hasattr(pred_instances, 'scores_3d') else None
                        labels_3d = pred_instances.labels_3d if hasattr(pred_instances, 'labels_3d') else None
                        
                        for j in range(len(bboxes_3d)):
                            detection = {
                                'bbox_3d': bboxes_3d[j].tensor.cpu().numpy() if hasattr(bboxes_3d[j], 'tensor') else bboxes_3d[j],
                                'score': float(scores_3d[j]) if scores_3d is not None else 0.0,
                                'label': int(labels_3d[j]) if labels_3d is not None else 0
                            }
                            detections.append(detection)
                
                # 保存可视化
                if image is not None and points is not None:
                    weather_type = 'unknown'  # 可以从data_sample中获取
                    self.visualizer.save_detection_visualization(
                        sample_id=sample_id,
                        image=image,
                        points=points,
                        detections=detections,
                        weather_type=weather_type
                    )
        
        except Exception as e:
            print(f"训练可视化保存失败: {e}")
    
    def _save_validation_results(self, runner, epoch: int, metrics: Optional[Dict]):
        """保存验证结果可视化"""
        try:
            # 从验证集中获取一些样本进行可视化
            val_dataloader = runner.val_dataloader
            if val_dataloader is None:
                return
            
            # 获取前几个样本
            sample_count = 0
            for batch_idx, data_batch in enumerate(val_dataloader):
                if sample_count >= self.visualization_samples:
                    break
                
                inputs = data_batch.get('inputs', {})
                data_samples = data_batch.get('data_samples', [])
                
                # 运行推理
                with runner.model.eval():
                    outputs = runner.model.test_step(data_batch)
                
                for i, data_sample in enumerate(data_samples):
                    if sample_count >= self.visualization_samples:
                        break
                        
                    sample_id = f"val_epoch_{epoch}_batch_{batch_idx}_sample_{i}"
                    
                    # 处理图像和点云
                    image = None
                    points = None
                    
                    if 'imgs' in inputs and len(inputs['imgs']) > i:
                        image = inputs['imgs'][i]
                        if hasattr(image, 'cpu'):
                            image = image.cpu().numpy()
                        if len(image.shape) == 3 and image.shape[0] == 3:
                            image = np.transpose(image, (1, 2, 0))
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                    
                    if 'points' in inputs and len(inputs['points']) > i:
                        points = inputs['points'][i]
                        if hasattr(points, 'cpu'):
                            points = points.cpu().numpy()
                    
                    # 获取检测结果
                    detections = []
                    if outputs and len(outputs) > i:
                        pred_instances = outputs[i].get('pred_instances_3d')
                        if pred_instances is not None:
                            # 简化的检测结果处理
                            if hasattr(pred_instances, 'bboxes_3d'):
                                bboxes = pred_instances.bboxes_3d
                                scores = pred_instances.scores_3d if hasattr(pred_instances, 'scores_3d') else None
                                labels = pred_instances.labels_3d if hasattr(pred_instances, 'labels_3d') else None
                                
                                for j in range(len(bboxes)):
                                    detection = {
                                        'bbox_3d': bboxes[j] if isinstance(bboxes[j], (list, np.ndarray)) else [0,0,0,1,1,1,0],
                                        'score': float(scores[j]) if scores is not None and j < len(scores) else 0.5,
                                        'label': runner.model.dataset_meta.get('classes', ['Unknown'])[int(labels[j])] if labels is not None and j < len(labels) else 'Unknown'
                                    }
                                    detections.append(detection)
                    
                    # 保存验证可视化
                    if image is not None and points is not None:
                        weather_type = data_sample.get('weather_type', 'clear') if isinstance(data_sample, dict) else 'clear'
                        self.visualizer.save_detection_visualization(
                            sample_id=sample_id,
                            image=image,
                            points=points,
                            detections=detections,
                            weather_type=weather_type
                        )
                    
                    sample_count += 1
        
        except Exception as e:
            print(f"验证结果可视化保存失败: {e}")
    
    def _record_epoch_analytics(self, runner, epoch: int, metrics: Optional[Dict]):
        """记录epoch分析数据"""
        try:
            epoch_data = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
            }
            
            # 记录训练损失
            if hasattr(runner, 'log_buffer'):
                log_buffer = runner.log_buffer
                epoch_data.update({
                    'train_loss': log_buffer.val.get('loss', 0),
                    'detection_loss': log_buffer.val.get('loss_cls', 0) + log_buffer.val.get('loss_bbox', 0),
                    'ssl_loss': log_buffer.val.get('ssl_loss', 0),
                    'total_loss': log_buffer.val.get('loss', 0)
                })
            
            # 记录验证指标
            if metrics:
                epoch_data['detection_results'] = {}
                
                # 处理KITTI指标
                for metric_name, metric_value in metrics.items():
                    if 'recall' in metric_name.lower():
                        class_name = metric_name.split('_')[0] if '_' in metric_name else 'overall'
                        if class_name not in epoch_data['detection_results']:
                            epoch_data['detection_results'][class_name] = {}
                        epoch_data['detection_results'][class_name]['recall'] = float(metric_value)
                    
                    elif 'precision' in metric_name.lower():
                        class_name = metric_name.split('_')[0] if '_' in metric_name else 'overall'
                        if class_name not in epoch_data['detection_results']:
                            epoch_data['detection_results'][class_name] = {}
                        epoch_data['detection_results'][class_name]['precision'] = float(metric_value)
                    
                    elif 'f1' in metric_name.lower() or 'map' in metric_name.lower():
                        class_name = metric_name.split('_')[0] if '_' in metric_name else 'overall'
                        if class_name not in epoch_data['detection_results']:
                            epoch_data['detection_results'][class_name] = {}
                        epoch_data['detection_results'][class_name]['f1'] = float(metric_value)
            
            # 记录增强策略信息（如果可用）
            train_dataset = runner.train_dataloader.dataset
            if hasattr(train_dataset, 'get_augmentation_stats'):
                aug_stats = train_dataset.get_augmentation_stats()
                epoch_data['augmentation_strategy'] = {
                    'weather_distribution': aug_stats.get('weather_distribution', {}),
                    'augmentation_rate': aug_stats.get('augmentation_rate', 0)
                }
            
            # 记录数据到分析器
            self.analytics.record_epoch_data(epoch, epoch_data)
        
        except Exception as e:
            print(f"分析数据记录失败: {e}")