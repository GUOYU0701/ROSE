"""
ROSE检测器
基于MVXNet的增强3D目标检测模型，集成SSL训练
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
# build_model is deprecated in MMDet3D 1.2.0, use MODELS.build() instead
from mmdet3d.registry import MODELS
from mmengine import Config
from mmdet3d.structures import Det3DDataSample


from mmdet3d.models.detectors import MVXTwoStageDetector

@MODELS.register_module()
class ROSEDetector(MVXTwoStageDetector):
    """ROSE增强检测器 - 基于MVXTwoStageDetector"""
    
    def __init__(self, 
                 ssl_config: Optional[Dict] = None,
                 enable_ssl: bool = True,
                 **kwargs):
        """
        初始化ROSE检测器
        
        Args:
            ssl_config: SSL配置参数
            enable_ssl: 是否启用SSL训练
            **kwargs: 其他参数，包括所有标准MMDet3D模型参数
        """
        # 提取标准模型参数
        model_kwargs = {k: v for k, v in kwargs.items() if k not in ['type', 'ssl_config', 'enable_ssl']}
        
        # 调用父类初始化
        super(ROSEDetector, self).__init__(**model_kwargs)
        
        self.enable_ssl = enable_ssl
        self.ssl_config = ssl_config or {}
        
        # 添加cfg属性以兼容MMDetection3D框架
        if not hasattr(self, 'cfg'):
            # Create proper Config object for MMDetection3D compatibility
            cfg_dict = {
                'type': 'ROSEDetector',
                'ssl_config': self.ssl_config,
                'enable_ssl': self.enable_ssl,
                **model_kwargs
            }
            self.cfg = Config(cfg_dict)
            
        # Store original constructor arguments for teacher model creation
        self._init_args = model_kwargs
        self._ssl_config = ssl_config
        self._enable_ssl = enable_ssl
        
        # SSL相关配置
        if self.enable_ssl:
            self.ssl_feature_extractors = self._build_ssl_feature_extractors()
        
        print(f"✅ ROSE检测器初始化完成")
        print(f"   SSL启用: {self.enable_ssl}")
    
    def _build_ssl_feature_extractors(self) -> nn.ModuleDict:
        """构建SSL特征提取器"""
        extractors = nn.ModuleDict()
        
        # 图像特征提取器适配器
        extractors['img_feature_adapter'] = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),  # 适配不同backbone输出
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 点云特征提取器适配器
        extractors['pts_feature_adapter'] = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),  # 适配不同backbone输出
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        
        return extractors
    
    def forward(self, 
                batch_inputs: Dict,
                data_samples: Optional[List[Det3DDataSample]] = None,
                mode: str = 'tensor') -> Any:
        """
        前向传播
        
        Args:
            batch_inputs: 批次输入数据
            data_samples: 数据样本
            mode: 推理模式 ('tensor', 'predict', 'loss')
            
        Returns:
            根据模式返回不同结果
        """
        if mode == 'loss':
            return self.loss(batch_inputs, data_samples)
        elif mode == 'predict':
            return self.predict(batch_inputs, data_samples)
        else:
            return self._forward(batch_inputs)
    
    def loss(self, 
             batch_inputs: Dict,
             data_samples: List[Det3DDataSample]) -> Dict:
        """
        计算损失
        
        Args:
            batch_inputs: 批次输入数据
            data_samples: 数据样本
            
        Returns:
            损失字典
        """
        # 基础检测损失 - 调用父类的loss方法
        detection_losses = super(ROSEDetector, self).loss(batch_inputs, data_samples)
        
        # SSL增强
        if self.enable_ssl:
            # 获取中间特征
            features = self._extract_features(batch_inputs)
            
            # 将SSL特征添加到检测输出中
            if isinstance(detection_losses, dict):
                detection_losses.update({
                    'img_features': features.get('img_features'),
                    'pts_features': features.get('pts_features')
                })
        
        return detection_losses
    
    def predict(self,
                batch_inputs: Dict,
                data_samples: List[Det3DDataSample],
                rescale: bool = True) -> List[Det3DDataSample]:
        """
        预测
        
        Args:
            batch_inputs: 批次输入数据
            data_samples: 数据样本
            rescale: 是否重缩放
            
        Returns:
            预测结果
        """
        return super(ROSEDetector, self).predict(batch_inputs, data_samples, rescale)
    
    def _forward(self, batch_inputs: Dict) -> Tuple[List]:
        """
        特征提取前向传播
        
        Args:
            batch_inputs: 批次输入数据
            
        Returns:
            特征元组
        """
        return super(ROSEDetector, self)._forward(batch_inputs)
    
    def _extract_features(self, batch_inputs: Dict) -> Dict:
        """
        提取SSL所需特征
        
        Args:
            batch_inputs: 批次输入数据
            
        Returns:
            特征字典
        """
        features = {}
        
        try:
            # 调用自身的backbone进行特征提取
            if hasattr(self, 'extract_feat'):
                # 对于有extract_feat方法的模型
                img = batch_inputs.get('imgs', None)
                points = batch_inputs.get('points', None)
                
                if img is not None and hasattr(self, 'img_backbone'):
                    img_feats = self.img_backbone(img)
                    if isinstance(img_feats, (list, tuple)):
                        img_feats = img_feats[-1]  # 使用最后一层特征
                    
                    # 适配特征维度
                    if self.enable_ssl:
                        img_feats_adapted = self.ssl_feature_extractors['img_feature_adapter'](img_feats)
                        features['img_features'] = img_feats_adapted
                
                if points is not None and hasattr(self, 'pts_backbone'):
                    # 简化点云特征提取
                    batch_size = len(points)
                    pts_features_list = []
                    
                    for i in range(batch_size):
                        if len(points[i]) > 0:
                            # 简单的点云特征提取
                            pts_feat = points[i][:, :3].mean(dim=0)  # 简单平均
                            pts_features_list.append(pts_feat)
                        else:
                            pts_features_list.append(torch.zeros(3, device=points[i].device))
                    
                    if pts_features_list:
                        pts_feats = torch.stack(pts_features_list)
                        # 扩展维度以适配适配器
                        pts_feats = pts_feats.unsqueeze(-1)  # (B, 3, 1)
                        
                        if self.enable_ssl:
                            # 先调整维度以适配Conv1d
                            if pts_feats.size(1) == 3:  # 如果特征维度是3
                                # 扩展到128维
                                pts_feats = torch.cat([pts_feats] * 42 + [pts_feats[:, :2, :]], dim=1)  # 3*42+2=128
                            
                            pts_feats_adapted = self.ssl_feature_extractors['pts_feature_adapter'](pts_feats)
                            features['pts_features'] = pts_feats_adapted
            
        except Exception as e:
            print(f"⚠️ SSL特征提取失败: {e}")
            # 提供默认特征
            if self.enable_ssl:
                batch_size = len(batch_inputs.get('points', [torch.zeros(1, 4)]))
                device = next(self.parameters()).device
                features['img_features'] = torch.zeros(batch_size, 256, device=device)
                features['pts_features'] = torch.zeros(batch_size, 256, device=device)
        
        return features
    
    def extract_feat(self, batch_inputs: Dict) -> Any:
        """提取特征（兼容接口）"""
        return super(ROSEDetector, self).extract_feat(batch_inputs)
    
    def train_step(self, data: Dict, optimizer: torch.optim.Optimizer) -> Dict:
        """
        训练步骤
        
        Args:
            data: 训练数据
            optimizer: 优化器
            
        Returns:
            损失字典
        """
        # 获取输入数据
        batch_inputs = data.get('inputs', {})
        data_samples = data.get('data_samples', [])
        
        # 前向传播计算损失
        losses = self.loss(batch_inputs, data_samples)
        
        # 反向传播
        if isinstance(losses, dict) and 'loss' in losses:
            total_loss = losses['loss']
        else:
            total_loss = sum(losses.values()) if isinstance(losses, dict) else losses
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return losses
    
    def val_step(self, data: Dict) -> List[Det3DDataSample]:
        """
        验证步骤
        
        Args:
            data: 验证数据
            
        Returns:
            预测结果
        """
        batch_inputs = data.get('inputs', {})
        data_samples = data.get('data_samples', [])
        
        with torch.no_grad():
            predictions = self.predict(batch_inputs, data_samples)
        
        return predictions
    
    def test_step(self, data: Dict) -> List[Det3DDataSample]:
        """
        测试步骤
        
        Args:
            data: 测试数据
            
        Returns:
            预测结果
        """
        return self.val_step(data)


class ROSEModelBuilder:
    """ROSE模型构建器"""
    
    @staticmethod
    def create_rose_detector(base_config_path: str,
                            ssl_config: Optional[Dict] = None,
                            enable_ssl: bool = True) -> ROSEDetector:
        """
        创建ROSE检测器
        
        Args:
            base_config_path: 基础配置文件路径
            ssl_config: SSL配置
            enable_ssl: 是否启用SSL
            
        Returns:
            ROSE检测器实例
        """
        # 加载基础配置
        base_cfg = Config.fromfile(base_config_path)
        
        # 修改模型配置以支持SSL
        model_config = base_cfg.model.copy()
        
        # 添加SSL配置
        if enable_ssl:
            model_config.update({
                'ssl_config': ssl_config or {},
                'enable_ssl': enable_ssl
            })
        
        # 创建ROSE检测器
        rose_detector = ROSEDetector(
            base_model_config=model_config,
            ssl_config=ssl_config,
            enable_ssl=enable_ssl
        )
        
        return rose_detector
    
    @staticmethod
    def adapt_config_for_rose(base_config_path: str,
                             work_dir: str,
                             data_root: str,
                             augmented_data_root: str,
                             strategy: Dict,
                             epochs: int = 10) -> Config:
        """
        适配配置以支持ROSE训练
        
        Args:
            base_config_path: 基础配置路径
            work_dir: 工作目录
            data_root: 数据根目录
            augmented_data_root: 增强数据根目录
            strategy: 增强策略
            epochs: 训练轮数
            
        Returns:
            适配后的配置
        """
        # 加载基础配置
        cfg = Config.fromfile(base_config_path)
        
        # 修改基本配置
        cfg.work_dir = work_dir
        cfg.train_cfg.max_epochs = epochs
        
        # 修改模型配置
        cfg.model = dict(
            type='ROSEDetector',
            base_model_config=cfg.model,
            ssl_config=strategy.get('ssl_parameters', {}),
            enable_ssl=True
        )
        
        # 修改数据集配置
        cfg.train_dataloader.dataset = dict(
            type='ROSEDataset',
            data_root=data_root,
            augmented_data_root=augmented_data_root,
            ann_file=cfg.train_dataloader.dataset.ann_file,
            pipeline=cfg.train_dataloader.dataset.pipeline,
            metainfo=cfg.train_dataloader.dataset.metainfo,
            augmentation_strategy=strategy,
            enable_ssl=True
        )
        
        # 添加自定义钩子
        if not hasattr(cfg, 'custom_hooks'):
            cfg.custom_hooks = []
        
        cfg.custom_hooks.append(
            dict(
                type='ROSETrainingHook',
                work_dir=work_dir,
                augmentation_strategy=strategy,
                save_interval=100,
                visualization_interval=200,
                ssl_enabled=True
            )
        )
        
        # 配置可视化
        if hasattr(cfg, 'default_hooks') and hasattr(cfg.default_hooks, 'visualization'):
            cfg.default_hooks.visualization.update({
                'draw': True,
                'interval': 200,
                'test_out_dir': f'{work_dir}/visualizations'
            })
        
        return cfg