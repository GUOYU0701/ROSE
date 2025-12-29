"""
ROSE数据集类
集成增强数据和SSL训练支持的数据集
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
class ROSEDataset(KittiDataset):
    """ROSE增强数据集"""
    
    def __init__(self, 
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Dict],
                 augmented_data_root: str = None,
                 augmentation_config: Optional[Dict] = None,
                 augmentation_prob: float = 0.8,
                 save_augmented_data: bool = True,
                 augmentation_strategy: Optional[Dict] = None,
                 enable_ssl: bool = True,
                 weather_mapping: Optional[Dict] = None,
                 **kwargs):
        """
        初始化ROSE数据集
        
        Args:
            data_root: 原始数据根目录
            ann_file: 标注文件
            pipeline: 数据处理管道
            augmented_data_root: 增强数据根目录
            augmentation_config: 增强配置
            augmentation_prob: 增强概率
            save_augmented_data: 是否保存增强数据
            augmentation_strategy: 增强策略
            enable_ssl: 是否启用SSL
            weather_mapping: 天气类型映射
        """
        # 保存ROSE特定参数
        self.augmented_data_root = Path(augmented_data_root) if augmented_data_root else Path('augmented_data')
        self.augmentation_config = augmentation_config or {}
        self.augmentation_prob = augmentation_prob
        self.save_augmented_data = save_augmented_data
        self.augmentation_strategy = augmentation_strategy or {}
        self.enable_ssl = enable_ssl
        self.weather_mapping = weather_mapping or {
            'clear': 0, 'rain_light': 1, 'rain_heavy': 2, 
            'fog_light': 3, 'fog_heavy': 4
        }
        
        # 过滤掉KittiDataset不认识的参数
        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['augmented_data_root', 'augmentation_config', 
                                  'augmentation_prob', 'save_augmented_data',
                                  'augmentation_strategy', 'enable_ssl', 'weather_mapping']}
        
        # 初始化基类
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            **base_kwargs
        )
        
        # 加载增强数据信息
        self._load_augmented_data_info()
        
        print(f"✅ ROSE数据集初始化完成")
        data_list = getattr(self, 'data_list', getattr(self, 'data_infos', []))
        print(f"   原始样本数: {len(data_list)}")
        print(f"   增强样本数: {len(self.augmented_infos) if hasattr(self, 'augmented_infos') else 0}")
        print(f"   SSL启用: {self.enable_ssl}")
    
    def _load_augmented_data_info(self):
        """加载增强数据信息"""
        augmented_info_file = self.augmented_data_root / 'kitti_infos_train_augmented.pkl'
        
        if augmented_info_file.exists():
            with open(augmented_info_file, 'rb') as f:
                self.augmented_infos = pickle.load(f)
            
            # 合并原始数据和增强数据
            combined_infos = []
            
            # 添加原始数据(标记为clear天气)
            data_list = getattr(self, 'data_list', getattr(self, 'data_infos', []))
            for info in data_list:
                info_copy = info.copy()
                info_copy['weather_type'] = 'clear'
                info_copy['augmented'] = False
                combined_infos.append(info_copy)
            
            # 添加增强数据
            for info in self.augmented_infos:
                info_copy = info.copy()
                if 'weather_type' not in info_copy:
                    info_copy['weather_type'] = 'clear'
                if 'augmented' not in info_copy:
                    info_copy['augmented'] = True
                combined_infos.append(info_copy)
            
            # 更新数据信息
            if hasattr(self, 'data_list'):
                self.data_list = combined_infos
            else:
                self.data_infos = combined_infos
            
            print(f"✅ 增强数据已合并: {len(self.augmented_infos)} 个增强样本")
        else:
            print(f"⚠️ 未找到增强数据文件: {augmented_info_file}")
            self.augmented_infos = []
            
            # 为原始数据添加天气标记，保持原始数据不变
            data_list = getattr(self, 'data_list', getattr(self, 'data_infos', []))
            modified_infos = []
            for info in data_list:
                info_copy = info.copy()
                info_copy['weather_type'] = 'clear'
                info_copy['augmented'] = False
                modified_infos.append(info_copy)
            
            # 更新数据信息
            if hasattr(self, 'data_list'):
                self.data_list = modified_infos
            else:
                self.data_infos = modified_infos
    
    def get_data_info(self, index: int) -> Dict:
        """获取数据信息"""
        data_list = getattr(self, 'data_list', getattr(self, 'data_infos', []))
        
        # 确保索引在有效范围内
        if len(data_list) == 0:
            raise ValueError("Data list is empty. Please check dataset initialization.")
        if index >= len(data_list):
            print(f"Warning: Index {index} out of range for data_list of size {len(data_list)}")
            index = index % len(data_list)  # 使用模运算防止越界
            
        info = data_list[index].copy()
        
        # 添加SSL相关信息
        if self.enable_ssl:
            info['enable_ssl'] = True
            info['weather_type_id'] = self.weather_mapping.get(
                info.get('weather_type', 'clear'), 0
            )
            
            # 添加增强配置信息
            if info.get('augmented', False):
                info['augmentation_config'] = self.augmentation_strategy
        
        return info
    
    def __getitem__(self, idx: int) -> Dict:
        """获取数据样本"""
        data_info = self.get_data_info(idx)
        
        # 使用管道处理数据
        data = self.pipeline(data_info)
        
        # 添加SSL相关信息到数据中
        if self.enable_ssl:
            data['weather_types'] = torch.tensor([data_info.get('weather_type_id', 0)], 
                                               dtype=torch.long)
            data['augmented'] = data_info.get('augmented', False)
            
            # 如果是增强数据，添加原始数据信息用于对比学习
            if data_info.get('augmented', False):
                data['is_augmented'] = True
                data['weather_type'] = data_info.get('weather_type', 'clear')
            else:
                data['is_augmented'] = False
                data['weather_type'] = 'clear'
        
        return data
    
    def evaluate(self, results: List, metric: str = 'mAP', **kwargs) -> Dict:
        """评估函数"""
        # 分别评估原始数据和增强数据的结果
        original_results = []
        augmented_results = []
        
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            if data_info.get('augmented', False):
                augmented_results.append(result)
            else:
                original_results.append(result)
        
        # 评估原始数据
        if original_results:
            original_eval = super().evaluate(original_results, metric, **kwargs)
        else:
            original_eval = {}
        
        # 综合评估结果
        eval_results = {
            'overall': super().evaluate(results, metric, **kwargs),
            'original_data': original_eval
        }
        
        if augmented_results:
            # 这里简化处理增强数据的评估
            eval_results['augmented_data'] = {
                'sample_count': len(augmented_results)
            }
        
        return eval_results
    
    def get_weather_distribution(self) -> Dict:
        """获取天气分布统计"""
        weather_counts = {}
        
        for info in self.data_infos:
            weather_type = info.get('weather_type', 'clear')
            weather_counts[weather_type] = weather_counts.get(weather_type, 0) + 1
        
        total = len(self.data_infos)
        weather_distribution = {
            weather: count / total 
            for weather, count in weather_counts.items()
        }
        
        return {
            'counts': weather_counts,
            'distribution': weather_distribution,
            'total_samples': total
        }
    
    def get_augmentation_statistics(self) -> Dict:
        """获取增强统计信息"""
        if not hasattr(self, 'augmented_infos'):
            return {'augmented_samples': 0, 'original_samples': len(self.data_infos)}
        
        original_count = len(self.data_infos) - len(self.augmented_infos)
        augmented_count = len(self.augmented_infos)
        
        return {
            'original_samples': original_count,
            'augmented_samples': augmented_count,
            'total_samples': len(self.data_infos),
            'augmentation_ratio': augmented_count / original_count if original_count > 0 else 0
        }