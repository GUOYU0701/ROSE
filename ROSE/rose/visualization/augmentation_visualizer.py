"""
ROSE数据增强可视化器
处理增强前后的图像和点云数据对比可视化
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

class AugmentationVisualizer:
    """数据增强可视化器"""
    
    def __init__(self, save_dir: str, enabled: bool = True):
        """
        初始化可视化器
        
        Args:
            save_dir: 保存目录
            enabled: 是否启用
        """
        self.save_dir = save_dir
        self.enabled = enabled
        
        if self.enabled:
            os.makedirs(save_dir, exist_ok=True)
            
            # 创建子目录
            self.comparison_dir = os.path.join(save_dir, 'before_after_comparisons')
            self.detection_dir = os.path.join(save_dir, 'detection_results')
            self.statistics_dir = os.path.join(save_dir, 'statistics')
            
            os.makedirs(self.comparison_dir, exist_ok=True)
            os.makedirs(self.detection_dir, exist_ok=True)
            os.makedirs(self.statistics_dir, exist_ok=True)
            
        self.visualization_count = 0
        self.saved_samples = []
    
    def save_augmented_comparison(self,
                                 sample_id: str,
                                 original_img: np.ndarray,
                                 augmented_img: np.ndarray,
                                 original_points: np.ndarray,
                                 augmented_points: np.ndarray,
                                 weather_type: str,
                                 intensity: float,
                                 metadata: Optional[Dict] = None):
        """
        保存增强前后对比图
        
        Args:
            sample_id: 样本ID
            original_img: 原始图像
            augmented_img: 增强后图像
            original_points: 原始点云
            augmented_points: 增强后点云
            weather_type: 天气类型
            intensity: 强度
            metadata: 元数据
        """
        if not self.enabled:
            return
        
        try:
            # 创建2x2子图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 原始图像
            axes[0, 0].imshow(original_img)
            axes[0, 0].set_title(f'Original Image - Sample {sample_id}')
            axes[0, 0].axis('off')
            
            # 增强后图像
            axes[0, 1].imshow(augmented_img)
            axes[0, 1].set_title(f'Augmented Image - {weather_type} (intensity: {intensity:.2f})')
            axes[0, 1].axis('off')
            
            # 原始点云鸟瞰图
            self._plot_point_cloud_bev(axes[1, 0], original_points, f'Original Point Cloud')
            
            # 增强后点云鸟瞰图
            self._plot_point_cloud_bev(axes[1, 1], augmented_points, f'Augmented Point Cloud - {weather_type}')
            
            plt.tight_layout()
            
            # 保存图像
            save_path = os.path.join(self.comparison_dir, f'{sample_id}_comparison.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 保存元数据
            if metadata:
                metadata_path = os.path.join(self.comparison_dir, f'{sample_id}_metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'sample_id': sample_id,
                        'weather_type': weather_type,
                        'intensity': intensity,
                        'timestamp': datetime.now().isoformat(),
                        'metadata': metadata
                    }, f, indent=2, ensure_ascii=False)
            
            self.visualization_count += 1
            self.saved_samples.append({
                'sample_id': sample_id,
                'weather_type': weather_type,
                'intensity': intensity,
                'save_path': save_path
            })
            
        except Exception as e:
            print(f"保存增强对比失败 {sample_id}: {e}")
    
    def save_detection_visualization(self,
                                   sample_id: str,
                                   image: np.ndarray,
                                   points: np.ndarray,
                                   detections: List[Dict],
                                   weather_type: str = 'unknown'):
        """
        保存检测结果可视化
        
        Args:
            sample_id: 样本ID
            image: 图像
            points: 点云
            detections: 检测结果
            weather_type: 天气类型
        """
        if not self.enabled:
            return
        
        try:
            # 创建3D检测结果可视化
            fig = plt.figure(figsize=(20, 12))
            
            # 子图1：图像上的2D检测框
            ax1 = plt.subplot(2, 3, 1)
            ax1.imshow(image)
            ax1.set_title(f'Image Detection - {weather_type}')
            ax1.axis('off')
            
            # 子图2：点云鸟瞰图with检测框
            ax2 = plt.subplot(2, 3, 2)
            self._plot_point_cloud_bev_with_detections(ax2, points, detections, 'Point Cloud BEV with Detections')
            
            # 子图3：点云3D视图
            ax3 = plt.subplot(2, 3, 3, projection='3d')
            self._plot_point_cloud_3d_with_detections(ax3, points, detections, 'Point Cloud 3D with Detections')
            
            # 子图4：检测统计
            ax4 = plt.subplot(2, 3, 4)
            self._plot_detection_statistics(ax4, detections, 'Detection Statistics')
            
            # 子图5：距离分析
            ax5 = plt.subplot(2, 3, 5)
            self._plot_distance_analysis(ax5, points, detections, 'Distance Distribution')
            
            # 子图6：置信度分析
            ax6 = plt.subplot(2, 3, 6)
            self._plot_confidence_analysis(ax6, detections, 'Confidence Distribution')
            
            plt.tight_layout()
            
            # 保存图像
            save_path = os.path.join(self.detection_dir, f'{sample_id}_detection.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 保存检测数据
            detection_data = {
                'sample_id': sample_id,
                'weather_type': weather_type,
                'num_detections': len(detections),
                'detections': detections,
                'timestamp': datetime.now().isoformat()
            }
            
            data_path = os.path.join(self.detection_dir, f'{sample_id}_detection_data.json')
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(detection_data, f, indent=2, default=str, ensure_ascii=False)
                
        except Exception as e:
            print(f"保存检测可视化失败 {sample_id}: {e}")
    
    def _plot_point_cloud_bev(self, ax, points: np.ndarray, title: str):
        """绘制点云鸟瞰图"""
        if len(points) > 0:
            x, y = points[:, 0], points[:, 1]
            ax.scatter(x, y, c=points[:, 2] if points.shape[1] > 2 else 'blue', 
                      s=1, cmap='viridis', alpha=0.6)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_point_cloud_bev_with_detections(self, ax, points: np.ndarray, 
                                            detections: List[Dict], title: str):
        """绘制带检测框的点云鸟瞰图"""
        # 绘制点云
        if len(points) > 0:
            x, y = points[:, 0], points[:, 1]
            ax.scatter(x, y, c='lightblue', s=1, alpha=0.5)
        
        # 绘制检测框
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, detection in enumerate(detections):
            bbox_3d = detection.get('bbox_3d', [0, 0, 0, 1, 1, 1, 0])
            if len(bbox_3d) >= 7:
                center_x, center_y, center_z = bbox_3d[0], bbox_3d[1], bbox_3d[2]
                width, length, height, rotation = bbox_3d[3], bbox_3d[4], bbox_3d[5], bbox_3d[6]
                
                color = colors[i % len(colors)]
                
                # 简化的矩形框绘制
                rect = patches.Rectangle((center_x - length/2, center_y - width/2), 
                                       length, width, linewidth=2, 
                                       edgecolor=color, facecolor='none', alpha=0.8)
                ax.add_patch(rect)
                
                # 添加标签
                label = detection.get('label', 'Unknown')
                score = detection.get('score', 0.0)
                ax.text(center_x, center_y, f'{label}\n{score:.2f}', 
                       color=color, fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_point_cloud_3d_with_detections(self, ax, points: np.ndarray, 
                                           detections: List[Dict], title: str):
        """绘制3D点云和检测框"""
        # 绘制点云（采样以提高性能）
        if len(points) > 0:
            sample_size = min(5000, len(points))
            indices = np.random.choice(len(points), sample_size, replace=False)
            sampled_points = points[indices]
            
            ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
                      c=sampled_points[:, 2], s=1, alpha=0.3, cmap='viridis')
        
        # 绘制3D检测框（简化版本）
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, detection in enumerate(detections):
            bbox_3d = detection.get('bbox_3d', [0, 0, 0, 1, 1, 1, 0])
            if len(bbox_3d) >= 6:
                center_x, center_y, center_z = bbox_3d[0], bbox_3d[1], bbox_3d[2]
                width, length, height = bbox_3d[3], bbox_3d[4], bbox_3d[5]
                
                color = colors[i % len(colors)]
                
                # 绘制简化的3D框架
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                # 创建立方体顶点
                vertices = np.array([
                    [center_x - length/2, center_y - width/2, center_z - height/2],
                    [center_x + length/2, center_y - width/2, center_z - height/2],
                    [center_x + length/2, center_y + width/2, center_z - height/2],
                    [center_x - length/2, center_y + width/2, center_z - height/2],
                    [center_x - length/2, center_y - width/2, center_z + height/2],
                    [center_x + length/2, center_y - width/2, center_z + height/2],
                    [center_x + length/2, center_y + width/2, center_z + height/2],
                    [center_x - length/2, center_y + width/2, center_z + height/2]
                ])
                
                # 只绘制框架边缘
                edges = [
                    [vertices[0], vertices[1]], [vertices[1], vertices[2]], 
                    [vertices[2], vertices[3]], [vertices[3], vertices[0]],
                    [vertices[4], vertices[5]], [vertices[5], vertices[6]], 
                    [vertices[6], vertices[7]], [vertices[7], vertices[4]],
                    [vertices[0], vertices[4]], [vertices[1], vertices[5]], 
                    [vertices[2], vertices[6]], [vertices[3], vertices[7]]
                ]
                
                for edge in edges:
                    ax.plot([edge[0][0], edge[1][0]], 
                           [edge[0][1], edge[1][1]], 
                           [edge[0][2], edge[1][2]], color=color, linewidth=2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
    
    def _plot_detection_statistics(self, ax, detections: List[Dict], title: str):
        """绘制检测统计"""
        if not detections:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # 统计各类别数量
        label_counts = {}
        for detection in detections:
            label = detection.get('label', 'Unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
        
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        
        ax.bar(labels, counts, color=['skyblue', 'lightgreen', 'lightcoral'][:len(labels)])
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_distance_analysis(self, ax, points: np.ndarray, detections: List[Dict], title: str):
        """绘制距离分析"""
        distances = []
        for detection in detections:
            bbox_3d = detection.get('bbox_3d', [0, 0, 0, 1, 1, 1, 0])
            if len(bbox_3d) >= 3:
                center_x, center_y = bbox_3d[0], bbox_3d[1]
                distance = np.sqrt(center_x**2 + center_y**2)
                distances.append(distance)
        
        if distances:
            ax.hist(distances, bins=10, color='lightblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Count')
        else:
            ax.text(0.5, 0.5, 'No distance data', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(title)
    
    def _plot_confidence_analysis(self, ax, detections: List[Dict], title: str):
        """绘制置信度分析"""
        scores = [detection.get('score', 0.0) for detection in detections]
        
        if scores:
            ax.hist(scores, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Count')
            ax.set_xlim(0, 1)
        else:
            ax.text(0.5, 0.5, 'No confidence data', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(title)
    
    def save_summary_report(self):
        """保存总结报告"""
        if not self.enabled:
            return
        
        try:
            report = f"""# 数据增强可视化总结报告

## 基本统计
- 总可视化数量: {self.visualization_count}
- 保存样本数量: {len(self.saved_samples)}
- 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 天气类型分布
"""
            
            # 统计天气类型分布
            weather_counts = {}
            for sample in self.saved_samples:
                weather_type = sample.get('weather_type', 'unknown')
                weather_counts[weather_type] = weather_counts.get(weather_type, 0) + 1
            
            for weather_type, count in weather_counts.items():
                percentage = (count / len(self.saved_samples)) * 100 if self.saved_samples else 0
                report += f"- {weather_type}: {count} 个样本 ({percentage:.1f}%)\n"
            
            report += f"""

## 文件目录结构
```
{self.save_dir}/
├── before_after_comparisons/    # 增强前后对比图
├── detection_results/           # 检测结果可视化
└── statistics/                  # 统计数据
```

## 可视化功能
1. ✅ 数据增强前后对比可视化
2. ✅ 3D检测框结果可视化
3. ✅ 点云和图像联合可视化
4. ✅ 检测统计和分析图表

---
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            report_path = os.path.join(self.save_dir, 'AUGMENTATION_VISUALIZATION_SUMMARY.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"可视化总结报告已保存: {report_path}")
            
        except Exception as e:
            print(f"保存总结报告失败: {e}")
    
    def get_statistics(self) -> Dict:
        """获取可视化统计信息"""
        return {
            'total_visualizations': self.visualization_count,
            'saved_samples': len(self.saved_samples),
            'weather_distribution': {
                sample.get('weather_type', 'unknown'): 1 
                for sample in self.saved_samples
            }
        }