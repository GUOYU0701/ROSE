"""
Enhanced ROSE Dataset
集成了增强功能和SSL支持的数据集，兼容MMDetection3D 1.2.0
"""

import os
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from mmdet3d.datasets import KittiDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class EnhancedROSEDataset(KittiDataset):
    """增强的ROSE数据集，兼容MMDetection3D 1.2.0"""
    
    def __init__(self, 
                 augmentation_config: Optional[Dict] = None,
                 augmentation_prob: float = 0.8,
                 save_augmented_data: bool = False,
                 **kwargs):
        """
        初始化增强ROSE数据集
        
        Args:
            augmentation_config: 增强配置
            augmentation_prob: 增强概率
            save_augmented_data: 是否保存增强数据
            **kwargs: 其他参数，传递给KittiDataset
        """
        # 存储ROSE特定参数
        self.augmentation_config = augmentation_config or {}
        self.augmentation_prob = augmentation_prob
        self.save_augmented_data = save_augmented_data
        
        # 移除不被基类识别的参数
        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['augmentation_config', 'augmentation_prob', 'save_augmented_data']}
        
        # 初始化基类
        super(EnhancedROSEDataset, self).__init__(**base_kwargs)
        
        # 初始化增强功能
        self.augmentor = None
        if self.augmentation_config:
            self._init_augmentor()
        
        print(f"✅ 增强ROSE数据集初始化完成")
        print(f"   数据样本数: {len(self.data_list) if hasattr(self, 'data_list') else 'N/A'}")
        print(f"   增强配置: {'已启用' if self.augmentation_config else '未启用'}")
    
    def _init_augmentor(self):
        """初始化数据增强器"""
        try:
            from rose.augmentation.config import AugmentationConfig, WeatherConfig
            from rose.augmentation.weather_augmentor import WeatherAugmentor
            
            # 构建天气配置
            weather_configs = []
            for config in self.augmentation_config.get('weather_configs', []):
                weather_config = WeatherConfig(
                    weather_type=config['weather_type'],
                    intensity=config['intensity'],
                    rain_rate=config.get('rain_rate', 0.0),
                    fog_type=config.get('fog_type', 'no_fog'),
                    visibility=config.get('visibility_range', 1000.0),
                    brightness_factor=config.get('brightness_factor', 1.0),
                    contrast_factor=config.get('contrast_factor', 1.0),
                    noise_level=config.get('noise_level', 0.0),
                    blur_kernel_size=config.get('blur_kernel_size', 0)
                )
                weather_configs.append(weather_config)
            
            # 构建增强配置
            augmentation_config = AugmentationConfig(
                weather_configs=weather_configs,
                weather_probabilities=[config.get('probability', 1.0) for config in self.augmentation_config.get('weather_configs', [])],
                enable_visualization=True,
                visualization_dir='work_dirs/augmentation_visualizations',
                save_frequency=10,
                adaptation_enabled=False,
                performance_threshold=0.65,
                epoch=0
            )
            
            # 创建增强器
            self.augmentor = WeatherAugmentor(
                config=augmentation_config,
                enable_visualization=True,
                visualization_dir='work_dirs/augmentation_visualizations'
            )
            
            print("✅ 数据增强器初始化成功")
            
        except Exception as e:
            print(f"⚠️ 数据增强器初始化失败: {e}")
            self.augmentor = None
    
    def get_data_info(self, index: int) -> Dict:
        """获取数据信息"""
        data_info = super().get_data_info(index)
        
        # 添加ROSE特定信息
        data_info['enable_augmentation'] = self.augmentor is not None
        data_info['sample_index'] = index
        
        return data_info
    
    def __getitem__(self, idx: int) -> Dict:
        """获取数据样本，支持数据增强"""
        # 获取基础数据
        data = super().__getitem__(idx)
        
        # 如果启用了增强并且是训练模式，根据概率决定是否增强
        if self.augmentor is not None and not self.test_mode and np.random.rand() < self.augmentation_prob:
            try:
                # 从数据中提取图像和点云
                if 'img' in data and 'points' in data:
                    image = self._extract_image_array(data['img'])
                    points = self._extract_points_array(data['points'])
                    
                    if image is not None and points is not None:
                        # 执行增强
                        augmented_image, augmented_points, augmentation_info = self.augmentor.augment_sample(
                            image=image,
                            points=points,
                            calibration_info=None
                        )
                        
                        # 更新数据
                        data['img'] = self._update_image_data(data['img'], augmented_image)
                        data['points'] = self._update_points_data(data['points'], augmented_points)
                        
                        # 添加增强信息
                        data['augmentation_info'] = augmentation_info
                        
            except Exception as e:
                print(f"⚠️ 数据增强失败 (idx={idx}): {e}")
        
        return data
    
    def _extract_image_array(self, img_data) -> Optional[np.ndarray]:
        """从MMDet3D的img数据结构中提取图像数组"""
        try:
            if hasattr(img_data, 'data'):
                # 如果是tensor，转换为numpy
                if torch.is_tensor(img_data.data):
                    image = img_data.data.cpu().numpy()
                    # 调整维度 (C, H, W) -> (H, W, C)
                    if image.ndim == 3 and image.shape[0] == 3:
                        image = image.transpose(1, 2, 0)
                    # 从[0,1]范围转换到[0,255]
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    return image
            elif isinstance(img_data, np.ndarray):
                return img_data
            else:
                return None
        except Exception as e:
            print(f"⚠️ 图像提取失败: {e}")
            return None
    
    def _extract_points_array(self, points_data) -> Optional[np.ndarray]:
        """从MMDet3D的points数据结构中提取点云数组"""
        try:
            if hasattr(points_data, 'tensor'):
                # 如果是BasePoints类型
                points = points_data.tensor.cpu().numpy()
                return points
            elif hasattr(points_data, 'data'):
                # 如果是tensor
                if torch.is_tensor(points_data.data):
                    points = points_data.data.cpu().numpy()
                    return points
            elif isinstance(points_data, np.ndarray):
                return points_data
            else:
                return None
        except Exception as e:
            print(f"⚠️ 点云提取失败: {e}")
            return None
    
    def _update_image_data(self, original_img_data, new_image: np.ndarray):
        """更新图像数据"""
        try:
            if hasattr(original_img_data, 'data'):
                # 调整维度 (H, W, C) -> (C, H, W)
                if new_image.ndim == 3:
                    new_image = new_image.transpose(2, 0, 1)
                # 转换到[0,1]范围
                if new_image.max() > 1.0:
                    new_image = new_image.astype(np.float32) / 255.0
                
                # 更新tensor
                if torch.is_tensor(original_img_data.data):
                    device = original_img_data.data.device
                    original_img_data.data = torch.from_numpy(new_image).to(device)
                
            return original_img_data
        except Exception as e:
            print(f"⚠️ 图像数据更新失败: {e}")
            return original_img_data
    
    def _update_points_data(self, original_points_data, new_points: np.ndarray):
        """更新点云数据"""
        try:
            if hasattr(original_points_data, 'tensor'):
                # 更新BasePoints的tensor
                device = original_points_data.tensor.device
                original_points_data.tensor = torch.from_numpy(new_points).to(device)
            elif hasattr(original_points_data, 'data'):
                # 更新tensor
                if torch.is_tensor(original_points_data.data):
                    device = original_points_data.data.device
                    original_points_data.data = torch.from_numpy(new_points).to(device)
            
            return original_points_data
        except Exception as e:
            print(f"⚠️ 点云数据更新失败: {e}")
            return original_points_data
    
    def get_augmentation_statistics(self) -> Dict:
        """获取增强统计信息"""
        if self.augmentor:
            return self.augmentor.get_augmentation_statistics()
        return {'augmentor_enabled': False}
    
    def save_augmentation_report(self, output_path: str):
        """保存增强报告"""
        if self.augmentor and hasattr(self.augmentor, 'visualizer'):
            self.augmentor.visualizer.save_summary_report()
            print(f"✅ 增强报告已保存")
        else:
            print("⚠️ 没有增强器或可视化器可用")


@DATASETS.register_module() 
class ROSEDatasetSimple(KittiDataset):
    """简化的ROSE数据集，只接受基本参数"""
    
    def __init__(self, **kwargs):
        # 过滤掉所有不被基类识别的参数
        filtered_kwargs = {}
        
        # 基类KittiDataset的已知参数
        valid_params = [
            'data_root', 'ann_file', 'pipeline', 'modality', 'box_type_3d',
            'filter_empty_gt', 'test_mode', 'metainfo', 'data_prefix',
            'backend_args', 'indices', 'serialize_data', 'lazy_init'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params:
                filtered_kwargs[key] = value
            else:
                print(f"⚠️ 忽略未知参数: {key}")
        
        # 初始化基类
        super(ROSEDatasetSimple, self).__init__(**filtered_kwargs)
        
        print(f"✅ 简化ROSE数据集初始化完成，样本数: {len(self.data_list) if hasattr(self, 'data_list') else 'N/A'}")


if __name__ == "__main__":
    # 测试数据集创建
    try:
        from mmdet3d.registry import DATASETS
        print("测试数据集注册...")
        
        # 查看已注册的数据集
        print("已注册的数据集:")
        for name in DATASETS.module_dict.keys():
            if 'ROSE' in name or 'Kitti' in name:
                print(f"  - {name}")
        
        print("✅ 数据集模块测试完成")
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")