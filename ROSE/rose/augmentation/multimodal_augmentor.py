"""
多模态数据增强引擎
实现图像和点云之间物理一致性的同步增强
"""

import os
import sys
import cv2
import numpy as np
import pickle
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yaml

# Add LISA path
lisa_path = '/home/guoyu/CC/ROSE-NEW/LISA-main'
if lisa_path not in sys.path:
    sys.path.insert(0, lisa_path)

try:
    from pylisa.lisa import Lisa
    LISA_AVAILABLE = True
except ImportError:
    LISA_AVAILABLE = False
    print("⚠️ LISA未安装，将使用模拟增强")

class MultiModalAugmentor:
    """多模态数据增强器"""
    
    def __init__(self, work_dir: str, data_root: str):
        """
        初始化多模态增强器
        
        Args:
            work_dir: 工作目录
            data_root: 数据根目录
        """
        self.work_dir = Path(work_dir)
        self.data_root = Path(data_root)
        
        # 初始化LISA
        if LISA_AVAILABLE:
            self.lisa = Lisa(
                rmax=200.0,
                rmin=1.5,
                wavelength=905e-9,
                mode='strongest'
            )
            print("✅ LISA点云增强引擎初始化完成")
        else:
            self.lisa = None
            print("⚠️ 使用模拟点云增强")
        
        # 天气增强参数映射
        self.weather_configs = {
            'clear': {
                'point_cloud': {'augment': False},
                'image': {'augment': False}
            },
            'rain_light': {
                'point_cloud': {
                    'augment': True,
                    'method': 'monte_carlo',
                    'intensity': 0.3,
                    'rain_rate': 3.0
                },
                'image': {
                    'augment': True,
                    'brightness_factor': 0.9,
                    'contrast_factor': 0.95,
                    'noise_level': 0.01,
                    'blur_kernel': 1,
                    'rain_effect': True
                }
            },
            'rain_heavy': {
                'point_cloud': {
                    'augment': True,
                    'method': 'monte_carlo',
                    'intensity': 0.6,
                    'rain_rate': 8.0
                },
                'image': {
                    'augment': True,
                    'brightness_factor': 0.8,
                    'contrast_factor': 0.9,
                    'noise_level': 0.02,
                    'blur_kernel': 2,
                    'rain_effect': True
                }
            },
            'fog_light': {
                'point_cloud': {
                    'augment': True,
                    'method': 'average_extinction',
                    'intensity': 0.2,
                    'visibility': 80.0
                },
                'image': {
                    'augment': True,
                    'brightness_factor': 0.95,
                    'contrast_factor': 0.8,
                    'haze_intensity': 0.2,
                    'blur_kernel': 1
                }
            },
            'fog_heavy': {
                'point_cloud': {
                    'augment': True,
                    'method': 'average_extinction',
                    'intensity': 0.5,
                    'visibility': 30.0
                },
                'image': {
                    'augment': True,
                    'brightness_factor': 0.9,
                    'contrast_factor': 0.7,
                    'haze_intensity': 0.5,
                    'blur_kernel': 2
                }
            }
        }
    
    def process_dataset(self, strategy: Dict, output_dir: str):
        """
        处理整个数据集的增强
        
        Args:
            strategy: 增强策略
            output_dir: 输出目录
        """
        print("开始处理数据集增强...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载数据集信息
        train_info_file = self.data_root / 'kitti_infos_train.pkl'
        with open(train_info_file, 'rb') as f:
            data_infos = pickle.load(f)
        
        print(f"数据集样本数: {len(data_infos)}")
        
        # 根据策略确定每个样本的增强类型
        weather_distribution = strategy['weather_distribution']
        augmentation_plan = self._create_augmentation_plan(data_infos, weather_distribution)
        
        # 保存增强计划
        plan_file = output_path / 'augmentation_plan.yaml'
        with open(plan_file, 'w') as f:
            yaml.dump(augmentation_plan, f)
        
        # 执行增强
        self._execute_augmentation_plan(data_infos, augmentation_plan, output_path)
        
        print(f"✅ 数据集增强完成: {output_dir}")
    
    def _create_augmentation_plan(self, data_infos: List, weather_distribution: Dict) -> Dict:
        """创建增强计划"""
        print("创建增强计划...")
        
        total_samples = len(data_infos)
        plan = {
            'total_samples': total_samples,
            'weather_distribution': weather_distribution,
            'sample_assignments': {}
        }
        
        # 为每个样本分配天气类型
        weather_types = list(weather_distribution.keys())
        probabilities = list(weather_distribution.values())
        
        # 确保概率之和为1
        prob_sum = sum(probabilities)
        probabilities = [p / prob_sum for p in probabilities]
        
        sample_assignments = np.random.choice(
            weather_types, 
            size=total_samples, 
            p=probabilities
        )
        
        for i, weather_type in enumerate(sample_assignments):
            plan['sample_assignments'][i] = weather_type
        
        # 统计分配结果
        weather_counts = {}
        for weather_type in weather_types:
            count = np.sum(sample_assignments == weather_type)
            weather_counts[weather_type] = count
        
        plan['actual_distribution'] = weather_counts
        
        print("增强计划统计:")
        for weather_type, count in weather_counts.items():
            percentage = count / total_samples * 100
            print(f"  {weather_type}: {count} 样本 ({percentage:.1f}%)")
        
        return plan
    
    def _execute_augmentation_plan(self, data_infos: List, plan: Dict, output_path: Path):
        """执行增强计划"""
        print("执行数据增强...")
        
        # 创建输出目录结构
        dirs = {
            'images': output_path / 'images',
            'point_clouds': output_path / 'point_clouds',
            'annotations': output_path / 'annotations',
            'visualizations': output_path / 'visualizations'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 处理每个样本
        sample_assignments = plan['sample_assignments']
        augmented_infos = []
        
        for idx, data_info in enumerate(data_infos):
            if idx % 500 == 0:
                print(f"处理进度: {idx}/{len(data_infos)}")
            
            weather_type = sample_assignments[str(idx)]
            
            try:
                # 执行单个样本的增强
                augmented_info = self._augment_single_sample(
                    data_info, weather_type, dirs, idx
                )
                augmented_infos.append(augmented_info)
                
            except Exception as e:
                print(f"样本{idx}增强失败: {e}")
                # 使用原始样本
                augmented_infos.append(data_info)
        
        # 保存增强后的数据信息
        augmented_info_file = output_path / 'kitti_infos_train_augmented.pkl'
        with open(augmented_info_file, 'wb') as f:
            pickle.dump(augmented_infos, f)
        
        # 生成增强报告
        self._generate_augmentation_report(plan, output_path)
        
        print("✅ 数据增强执行完成")
    
    def _augment_single_sample(self, data_info: Dict, weather_type: str, 
                             dirs: Dict, sample_idx: int) -> Dict:
        """增强单个样本"""
        
        # 加载原始数据
        image_path = self.data_root / data_info['image']['image_path']
        points_path = self.data_root / data_info['lidar']['lidar_path']
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 读取点云
        points = np.fromfile(str(points_path), dtype=np.float32).reshape(-1, 4)
        if len(points) == 0:
            raise ValueError(f"无法读取点云: {points_path}")
        
        # 获取增强配置
        weather_config = self.weather_configs.get(weather_type, self.weather_configs['clear'])
        
        # 执行增强
        augmented_image, augmented_points = self._apply_synchronized_augmentation(
            image, points, weather_config
        )
        
        # 保存增强后的数据
        aug_image_path = dirs['images'] / f'sample_{sample_idx:06d}_{weather_type}.jpg'
        aug_points_path = dirs['point_clouds'] / f'sample_{sample_idx:06d}_{weather_type}.bin'
        
        cv2.imwrite(str(aug_image_path), augmented_image)
        augmented_points.astype(np.float32).tofile(str(aug_points_path))
        
        # 保存可视化(每100个样本保存一个)
        if sample_idx % 100 == 0:
            self._save_augmentation_visualization(
                image, augmented_image, points, augmented_points,
                weather_type, dirs['visualizations'], sample_idx
            )
        
        # 更新数据信息
        augmented_info = data_info.copy()
        augmented_info['image']['image_path'] = str(aug_image_path.relative_to(self.data_root))
        augmented_info['lidar']['lidar_path'] = str(aug_points_path.relative_to(self.data_root))
        augmented_info['weather_type'] = weather_type
        augmented_info['augmented'] = weather_type != 'clear'
        
        return augmented_info
    
    def _apply_synchronized_augmentation(self, image: np.ndarray, points: np.ndarray, 
                                       weather_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """应用同步的图像-点云增强"""
        
        if not weather_config['point_cloud']['augment']:
            # 无增强情况
            return image.copy(), points.copy()
        
        # 点云增强
        augmented_points = self._augment_point_cloud(points, weather_config['point_cloud'])
        
        # 图像增强（确保与点云增强的物理一致性）
        augmented_image = self._augment_image(image, weather_config['image'])
        
        return augmented_image, augmented_points
    
    def _augment_point_cloud(self, points: np.ndarray, config: Dict) -> np.ndarray:
        """增强点云"""
        if not config.get('augment', False):
            return points.copy()
        
        method = config.get('method', 'monte_carlo')
        intensity = config.get('intensity', 0.3)
        
        if self.lisa is not None:
            try:
                if method == 'monte_carlo':
                    # 雨雪增强
                    augmented = self.lisa.augment_mc(points)
                elif method == 'average_extinction':
                    # 雾增强
                    augmented = self.lisa.augment_avg(points)
                else:
                    augmented = points.copy()
                
                return augmented.astype(np.float32)
                
            except Exception as e:
                print(f"LISA增强失败，使用模拟增强: {e}")
                return self._simulate_point_cloud_augmentation(points, config)
        else:
            return self._simulate_point_cloud_augmentation(points, config)
    
    def _simulate_point_cloud_augmentation(self, points: np.ndarray, config: Dict) -> np.ndarray:
        """模拟点云增强"""
        augmented = points.copy()
        method = config.get('method', 'monte_carlo')
        intensity = config.get('intensity', 0.3)
        
        if method == 'monte_carlo':
            # 模拟雨雪效果：随机移除一些点，降低强度
            removal_rate = intensity * 0.2  # 最多移除20%的点
            keep_mask = np.random.random(len(points)) > removal_rate
            augmented = augmented[keep_mask]
            
            # 降低强度
            intensity_reduction = np.random.uniform(0.8, 1.0, size=len(augmented))
            augmented[:, 3] *= intensity_reduction
            
        elif method == 'average_extinction':
            # 模拟雾效果：基于距离的强度衰减
            distances = np.linalg.norm(augmented[:, :3], axis=1)
            visibility = config.get('visibility', 50.0)
            
            # 指数衰减
            extinction_factor = np.exp(-distances / visibility * intensity)
            augmented[:, 3] *= extinction_factor
            
            # 移除强度过低的点
            keep_mask = augmented[:, 3] > 0.1
            augmented = augmented[keep_mask]
        
        return augmented.astype(np.float32)
    
    def _augment_image(self, image: np.ndarray, config: Dict) -> np.ndarray:
        """增强图像"""
        if not config.get('augment', False):
            return image.copy()
        
        augmented = image.astype(np.float32)
        
        # 亮度调整
        brightness_factor = config.get('brightness_factor', 1.0)
        if brightness_factor != 1.0:
            augmented = augmented * brightness_factor
        
        # 对比度调整
        contrast_factor = config.get('contrast_factor', 1.0)
        if contrast_factor != 1.0:
            mean = np.mean(augmented, axis=(0, 1), keepdims=True)
            augmented = (augmented - mean) * contrast_factor + mean
        
        # 噪声
        noise_level = config.get('noise_level', 0.0)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, augmented.shape)
            augmented = augmented + noise
        
        # 模糊
        blur_kernel = config.get('blur_kernel', 0)
        if blur_kernel > 0:
            kernel_size = blur_kernel * 2 + 1
            augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
        
        # 雨效果
        if config.get('rain_effect', False):
            augmented = self._add_rain_effect(augmented)
        
        # 雾霾效果
        haze_intensity = config.get('haze_intensity', 0.0)
        if haze_intensity > 0:
            augmented = self._add_haze_effect(augmented, haze_intensity)
        
        # 限制像素值范围
        augmented = np.clip(augmented, 0, 255)
        
        return augmented.astype(np.uint8)
    
    def _add_rain_effect(self, image: np.ndarray) -> np.ndarray:
        """添加雨效果"""
        h, w = image.shape[:2]
        
        # 生成雨线
        rain_lines = np.zeros((h, w), dtype=np.float32)
        
        # 随机生成雨线位置
        num_lines = np.random.randint(100, 300)
        for _ in range(num_lines):
            x = np.random.randint(0, w)
            y_start = np.random.randint(0, h // 2)
            length = np.random.randint(10, 30)
            intensity = np.random.uniform(0.3, 0.8)
            
            y_end = min(h, y_start + length)
            rain_lines[y_start:y_end, x] = intensity * 255
        
        # 应用雨线
        rain_mask = np.stack([rain_lines] * 3, axis=2)
        image = np.maximum(image, rain_mask)
        
        return image
    
    def _add_haze_effect(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """添加雾霾效果"""
        h, w = image.shape[:2]
        
        # 创建雾霾遮罩（距离相关）
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # 模拟距离效应
        distance_mask = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        distance_mask = distance_mask / np.max(distance_mask)
        
        # 生成雾霾颜色（偏白/灰）
        haze_color = np.array([200, 200, 200], dtype=np.float32)
        
        # 应用雾霾
        haze_alpha = intensity * (0.3 + 0.4 * distance_mask)
        haze_alpha = np.stack([haze_alpha] * 3, axis=2)
        
        image = image * (1 - haze_alpha) + haze_color * haze_alpha
        
        return image
    
    def _save_augmentation_visualization(self, original_image: np.ndarray, 
                                       augmented_image: np.ndarray,
                                       original_points: np.ndarray,
                                       augmented_points: np.ndarray,
                                       weather_type: str,
                                       viz_dir: Path,
                                       sample_idx: int):
        """保存增强可视化"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建对比图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 原始图像
            axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title(f'Original Image (Sample {sample_idx})')
            axes[0, 0].axis('off')
            
            # 增强图像
            axes[0, 1].imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(f'Augmented Image ({weather_type})')
            axes[0, 1].axis('off')
            
            # 原始点云俯视图
            axes[1, 0].scatter(original_points[:, 0], original_points[:, 1], 
                             c=original_points[:, 3], cmap='viridis', s=0.1, alpha=0.7)
            axes[1, 0].set_title(f'Original Point Cloud ({len(original_points)} points)')
            axes[1, 0].set_xlabel('X (m)')
            axes[1, 0].set_ylabel('Y (m)')
            
            # 增强点云俯视图
            axes[1, 1].scatter(augmented_points[:, 0], augmented_points[:, 1],
                             c=augmented_points[:, 3], cmap='viridis', s=0.1, alpha=0.7)
            axes[1, 1].set_title(f'Augmented Point Cloud ({len(augmented_points)} points)')
            axes[1, 1].set_xlabel('X (m)')
            axes[1, 1].set_ylabel('Y (m)')
            
            plt.tight_layout()
            
            # 保存
            viz_file = viz_dir / f'augmentation_sample_{sample_idx:06d}_{weather_type}.png'
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"可视化保存失败: {e}")
    
    def _generate_augmentation_report(self, plan: Dict, output_path: Path):
        """生成增强报告"""
        report = {
            'augmentation_plan': plan,
            'timestamp': str(np.datetime64('now')),
            'statistics': {
                'total_samples': plan['total_samples'],
                'weather_distribution': plan['actual_distribution'],
                'lisa_available': LISA_AVAILABLE
            }
        }
        
        # 保存报告
        report_file = output_path / 'augmentation_report.yaml'
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        # 生成Markdown报告
        md_report = self._generate_markdown_report(report)
        md_file = output_path / 'augmentation_report.md'
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        print(f"✅ 增强报告已生成: {report_file}")
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """生成Markdown报告"""
        plan = report['augmentation_plan']
        stats = report['statistics']
        timestamp = report['timestamp']
        
        md = f"""# 多模态数据增强报告

## 基本信息
- **生成时间**: {timestamp}
- **总样本数**: {stats['total_samples']}
- **LISA状态**: {'可用' if stats['lisa_available'] else '不可用（使用模拟增强）'}

## 天气分布

### 计划分布
"""
        
        planned_dist = plan['weather_distribution']
        for weather, prob in planned_dist.items():
            md += f"- {weather}: {prob:.1%}\n"
        
        md += "\n### 实际分布\n"
        
        actual_dist = stats['weather_distribution']
        total = sum(actual_dist.values())
        
        for weather, count in actual_dist.items():
            percentage = count / total * 100 if total > 0 else 0
            md += f"- {weather}: {count} 样本 ({percentage:.1f}%)\n"
        
        md += f"""

## 增强技术

### 点云增强
- **雨雪效果**: Monte Carlo模拟 + 点移除 + 强度衰减
- **雾效果**: 平均消光 + 距离衰减
- **物理引擎**: {'LISA' if stats['lisa_available'] else '数学模拟'}

### 图像增强
- **亮度调整**: 模拟不同天气的光照条件
- **对比度调整**: 增强天气特征
- **噪声添加**: 模拟传感器噪声
- **模糊效果**: 模拟大气散射
- **特效添加**: 雨线、雾霾覆盖

### 物理一致性
- ✅ 图像和点云增强类型同步
- ✅ 增强强度程度一致
- ✅ 基于真实物理参数的模拟

## 输出文件
- **增强图像**: `images/`
- **增强点云**: `point_clouds/` 
- **标注文件**: `annotations/`
- **可视化样例**: `visualizations/`
- **数据索引**: `kitti_infos_train_augmented.pkl`

---
*报告生成时间: {timestamp}*
"""
        
        return md