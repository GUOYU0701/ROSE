"""
ROSE SSL训练器
实现跨模态对比学习和师生一致性约束的自监督学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import yaml
import json

class CrossModalContrastiveLoss(nn.Module):
    """跨模态对比损失"""
    
    def __init__(self, temperature: float = 0.1, proj_dim: int = 256):
        super().__init__()
        self.temperature = temperature
        self.proj_dim = proj_dim
        
        # 投影头
        self.image_projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        self.lidar_projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim),
            nn.LayerNorm(proj_dim)
        )
    
    def forward(self, image_features: torch.Tensor, 
                lidar_features: torch.Tensor) -> torch.Tensor:
        """
        计算跨模态对比损失
        
        Args:
            image_features: (B, 256) 图像特征
            lidar_features: (B, 256) 点云特征
            
        Returns:
            对比损失值
        """
        # 投影到低维空间
        image_proj = F.normalize(self.image_projector(image_features), dim=-1)
        lidar_proj = F.normalize(self.lidar_projector(lidar_features), dim=-1)
        
        batch_size = image_proj.shape[0]
        
        # 计算相似性矩阵
        logits = torch.mm(image_proj, lidar_proj.t()) / self.temperature
        
        # 构造标签(对角线为正样本)
        labels = torch.arange(batch_size).to(logits.device)
        
        # 计算对比损失
        loss_i2l = F.cross_entropy(logits, labels)
        loss_l2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2l + loss_l2i) / 2

class ConsistencyLoss(nn.Module):
    """师生一致性损失"""
    
    def __init__(self, consistency_type: str = 'mse'):
        super().__init__()
        self.consistency_type = consistency_type
    
    def forward(self, student_outputs: Dict, teacher_outputs: Dict) -> torch.Tensor:
        """
        计算师生一致性损失
        
        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出
            
        Returns:
            一致性损失值
        """
        consistency_loss = 0.0
        count = 0
        
        # 检测头一致性
        if 'cls_scores' in student_outputs and 'cls_scores' in teacher_outputs:
            student_cls = student_outputs['cls_scores']
            teacher_cls = teacher_outputs['cls_scores']
            
            if isinstance(student_cls, (list, tuple)):
                for s_cls, t_cls in zip(student_cls, teacher_cls):
                    if self.consistency_type == 'mse':
                        consistency_loss += F.mse_loss(s_cls, t_cls.detach())
                    elif self.consistency_type == 'kl':
                        consistency_loss += F.kl_div(
                            F.log_softmax(s_cls, dim=-1),
                            F.softmax(t_cls.detach(), dim=-1),
                            reduction='batchmean'
                        )
                    count += 1
            else:
                if self.consistency_type == 'mse':
                    consistency_loss += F.mse_loss(student_cls, teacher_cls.detach())
                elif self.consistency_type == 'kl':
                    consistency_loss += F.kl_div(
                        F.log_softmax(student_cls, dim=-1),
                        F.softmax(teacher_cls.detach(), dim=-1),
                        reduction='batchmean'
                    )
                count += 1
        
        # 回归头一致性
        if 'bbox_preds' in student_outputs and 'bbox_preds' in teacher_outputs:
            student_bbox = student_outputs['bbox_preds']
            teacher_bbox = teacher_outputs['bbox_preds']
            
            if isinstance(student_bbox, (list, tuple)):
                for s_bbox, t_bbox in zip(student_bbox, teacher_bbox):
                    consistency_loss += F.smooth_l1_loss(s_bbox, t_bbox.detach())
                    count += 1
            else:
                consistency_loss += F.smooth_l1_loss(student_bbox, teacher_bbox.detach())
                count += 1
        
        return consistency_loss / max(count, 1)

class EMATeacher(nn.Module):
    """指数移动平均教师模型"""
    
    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999):
        super().__init__()
        self.ema_decay = ema_decay
        self.teacher_model = self._create_teacher_copy(student_model)
        self.global_step = 0
    
    def _create_teacher_copy(self, student_model: nn.Module) -> nn.Module:
        """创建教师模型副本"""
        import copy
        try:
            # 尝试直接深拷贝
            teacher = copy.deepcopy(student_model)
        except Exception as e:
            print(f"⚠️ 深拷贝失败，使用参数拷贝方式: {e}")
            # 如果深拷贝失败，创建新实例并复制参数
            if hasattr(student_model, 'cfg'):
                teacher = type(student_model)(**student_model.cfg)
            else:
                # 如果没有cfg，创建一个简化的教师模型
                print("⚠️ 学生模型没有cfg属性，创建简化教师模型")
                teacher = copy.deepcopy(student_model)
        
        # 冻结教师模型参数
        for param in teacher.parameters():
            param.requires_grad = False
        
        return teacher
    
    def update_teacher(self, student_model: nn.Module):
        """更新教师模型参数"""
        self.global_step += 1
        
        # 动态调整EMA衰减率
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_decay)
        
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(), 
                student_model.parameters()
            ):
                teacher_param.data.mul_(alpha).add_(
                    student_param.data, alpha=1 - alpha
                )
    
    def forward(self, *args, **kwargs):
        """教师模型前向传播"""
        return self.teacher_model(*args, **kwargs)

class WeatherAwareLoss(nn.Module):
    """天气感知损失"""
    
    def __init__(self, num_weather_types: int = 5):
        super().__init__()
        self.num_weather_types = num_weather_types
        
        # 天气类型编码器
        self.weather_encoder = nn.Embedding(num_weather_types, 64)
        
        # 天气感知权重网络
        self.weight_network = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, outputs: Dict, weather_types: torch.Tensor) -> torch.Tensor:
        """
        计算天气感知损失
        
        Args:
            outputs: 模型输出
            weather_types: 天气类型标签 (B,)
            
        Returns:
            天气感知损失值
        """
        # 编码天气类型
        weather_embed = self.weather_encoder(weather_types)  # (B, 64)
        
        # 计算天气相关权重
        weather_weights = self.weight_network(weather_embed)  # (B, 1)
        
        # 计算基础损失
        base_loss = 0.0
        if 'cls_scores' in outputs:
            cls_scores = outputs['cls_scores']
            if isinstance(cls_scores, (list, tuple)):
                for cls_score in cls_scores:
                    # 这里简化为分类一致性损失
                    entropy = -torch.sum(F.softmax(cls_score, dim=-1) * 
                                       F.log_softmax(cls_score, dim=-1), dim=-1)
                    base_loss += torch.mean(entropy)
            else:
                entropy = -torch.sum(F.softmax(cls_scores, dim=-1) * 
                                   F.log_softmax(cls_scores, dim=-1), dim=-1)
                base_loss += torch.mean(entropy)
        
        # 应用天气权重
        weighted_loss = base_loss * torch.mean(weather_weights)
        
        return weighted_loss

class SpatialContrastiveLoss(nn.Module):
    """空间对比损失"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, 
                spatial_coords: torch.Tensor) -> torch.Tensor:
        """
        计算空间对比损失
        
        Args:
            features: (B, D) 特征向量
            spatial_coords: (B, 3) 空间坐标
            
        Returns:
            空间对比损失值
        """
        batch_size = features.shape[0]
        
        # 计算空间距离
        spatial_dist = torch.cdist(spatial_coords, spatial_coords)  # (B, B)
        
        # 计算特征相似性
        features_norm = F.normalize(features, dim=-1)
        feature_sim = torch.mm(features_norm, features_norm.t())  # (B, B)
        
        # 构造正负样本标签
        # 空间距离小的为正样本，距离大的为负样本
        spatial_threshold = torch.quantile(spatial_dist, 0.3)  # 前30%为正样本
        positive_mask = (spatial_dist < spatial_threshold).float()
        
        # 去除对角线
        eye_mask = torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask * (1 - eye_mask)
        
        # 计算对比损失
        logits = feature_sim / self.temperature
        
        # InfoNCE损失
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
        
        # 计算正样本的平均log概率
        positive_log_prob = log_prob * positive_mask
        positive_count = torch.sum(positive_mask, dim=1)
        positive_count = torch.clamp(positive_count, min=1)  # 避免除零
        
        loss = -torch.sum(positive_log_prob, dim=1) / positive_count
        
        return torch.mean(loss)

class ROSESSLTrainer:
    """ROSE SSL训练器"""
    
    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir)
        self.ssl_dir = self.work_dir / 'ssl_training'
        self.ssl_dir.mkdir(parents=True, exist_ok=True)
        
        # SSL损失组件
        self.cross_modal_loss = CrossModalContrastiveLoss()
        self.consistency_loss = ConsistencyLoss()
        self.weather_aware_loss = WeatherAwareLoss()
        self.spatial_loss = SpatialContrastiveLoss()
        
        # 教师模型
        self.teacher_model = None
        
        # 训练统计
        self.ssl_stats = {
            'cross_modal_losses': [],
            'consistency_losses': [],
            'weather_aware_losses': [],
            'spatial_losses': [],
            'total_ssl_losses': []
        }
        
        print("✅ SSL训练器初始化完成")
    
    def initialize_teacher(self, student_model: nn.Module, ema_decay: float = 0.999):
        """初始化教师模型"""
        self.teacher_model = EMATeacher(student_model, ema_decay)
        print("✅ EMA教师模型初始化完成")
    
    def compute_ssl_loss(self, 
                        student_outputs: Dict,
                        data_batch: Dict,
                        ssl_config: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        计算SSL损失
        
        Args:
            student_outputs: 学生模型输出
            data_batch: 数据批次
            ssl_config: SSL配置参数
            
        Returns:
            SSL总损失和损失详情
        """
        ssl_loss = 0.0
        loss_dict = {}
        
        # 获取损失权重
        lambda_cm = ssl_config.get('lambda_cm', 0.5)
        lambda_cons = ssl_config.get('lambda_cons', 0.3)
        lambda_spatial = ssl_config.get('lambda_spatial', 0.2)
        lambda_weather = ssl_config.get('lambda_weather', 0.4)
        
        # 1. 跨模态对比损失
        if 'img_features' in student_outputs and 'pts_features' in student_outputs:
            cm_loss = self.cross_modal_loss(
                student_outputs['img_features'],
                student_outputs['pts_features']
            )
            ssl_loss += lambda_cm * cm_loss
            loss_dict['cross_modal_loss'] = cm_loss.item()
            self.ssl_stats['cross_modal_losses'].append(cm_loss.item())
        
        # 2. 师生一致性损失
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    points=data_batch.get('points'),
                    img=data_batch.get('img'),
                    img_metas=data_batch.get('img_metas')
                )
            
            cons_loss = self.consistency_loss(student_outputs, teacher_outputs)
            ssl_loss += lambda_cons * cons_loss
            loss_dict['consistency_loss'] = cons_loss.item()
            self.ssl_stats['consistency_losses'].append(cons_loss.item())
        
        # 3. 空间对比损失
        if 'pts_features' in student_outputs and 'points' in data_batch:
            points = data_batch['points'][0]  # 取第一个batch的点云
            if len(points) > 0:
                # 随机采样一些点用于空间对比
                num_samples = min(512, len(points))
                sampled_indices = torch.randperm(len(points))[:num_samples]
                sampled_points = points[sampled_indices, :3]  # 取xyz坐标
                
                # 使用对应的特征(这里简化为使用平均特征)
                pts_features = student_outputs['pts_features']
                if pts_features.dim() > 2:
                    pts_features = pts_features.mean(dim=(2, 3))  # 空间维度平均
                
                if len(pts_features) >= num_samples:
                    sampled_features = pts_features[:num_samples]
                    
                    spatial_loss = self.spatial_loss(sampled_features, sampled_points)
                    ssl_loss += lambda_spatial * spatial_loss
                    loss_dict['spatial_loss'] = spatial_loss.item()
                    self.ssl_stats['spatial_losses'].append(spatial_loss.item())
        
        # 4. 天气感知损失
        if 'weather_types' in data_batch:
            weather_loss = self.weather_aware_loss(
                student_outputs, 
                data_batch['weather_types']
            )
            ssl_loss += lambda_weather * weather_loss
            loss_dict['weather_aware_loss'] = weather_loss.item()
            self.ssl_stats['weather_aware_losses'].append(weather_loss.item())
        
        # 记录总SSL损失
        loss_dict['total_ssl_loss'] = ssl_loss.item()
        self.ssl_stats['total_ssl_losses'].append(ssl_loss.item())
        
        return ssl_loss, loss_dict
    
    def update_teacher(self, student_model: nn.Module):
        """更新教师模型"""
        if self.teacher_model is not None:
            self.teacher_model.update_teacher(student_model)
    
    def get_ssl_analytics_summary(self) -> Dict:
        """获取SSL分析总结"""
        summary = {}
        
        for loss_name, losses in self.ssl_stats.items():
            if losses:
                summary[loss_name] = {
                    'mean': float(np.mean(losses)),
                    'std': float(np.std(losses)),
                    'latest': float(losses[-1]),
                    'count': len(losses)
                }
            else:
                summary[loss_name] = {
                    'mean': 0.0, 'std': 0.0, 'latest': 0.0, 'count': 0
                }
        
        return summary
    
    def save_ssl_state(self, epoch: int, save_path: Optional[str] = None):
        """保存SSL训练状态"""
        if save_path is None:
            save_path = self.ssl_dir / f'ssl_state_epoch_{epoch}.pth'
        
        ssl_state = {
            'epoch': epoch,
            'cross_modal_loss_state': self.cross_modal_loss.state_dict(),
            'consistency_loss_state': self.consistency_loss.state_dict(),
            'weather_aware_loss_state': self.weather_aware_loss.state_dict(),
            'spatial_loss_state': self.spatial_loss.state_dict(),
            'ssl_stats': self.ssl_stats
        }
        
        if self.teacher_model is not None:
            ssl_state['teacher_model_state'] = self.teacher_model.teacher_model.state_dict()
            ssl_state['teacher_global_step'] = self.teacher_model.global_step
        
        torch.save(ssl_state, save_path)
        print(f"✅ SSL状态已保存: {save_path}")
    
    def load_ssl_state(self, load_path: str):
        """加载SSL训练状态"""
        ssl_state = torch.load(load_path, map_location='cpu')
        
        self.cross_modal_loss.load_state_dict(ssl_state['cross_modal_loss_state'])
        self.consistency_loss.load_state_dict(ssl_state['consistency_loss_state'])
        self.weather_aware_loss.load_state_dict(ssl_state['weather_aware_loss_state'])
        self.spatial_loss.load_state_dict(ssl_state['spatial_loss_state'])
        self.ssl_stats = ssl_state['ssl_stats']
        
        if 'teacher_model_state' in ssl_state and self.teacher_model is not None:
            self.teacher_model.teacher_model.load_state_dict(ssl_state['teacher_model_state'])
            self.teacher_model.global_step = ssl_state.get('teacher_global_step', 0)
        
        print(f"✅ SSL状态已加载: {load_path}")
    
    def generate_ssl_report(self, epoch: int) -> Dict:
        """生成SSL训练报告"""
        summary = self.get_ssl_analytics_summary()
        
        report = {
            'epoch': epoch,
            'ssl_performance': summary,
            'training_progress': {
                'total_iterations': len(self.ssl_stats['total_ssl_losses']),
                'teacher_model_active': self.teacher_model is not None,
                'teacher_updates': self.teacher_model.global_step if self.teacher_model else 0
            }
        }
        
        # 保存报告
        report_file = self.ssl_dir / f'ssl_report_epoch_{epoch}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def create_ssl_config_template(self) -> Dict:
        """创建SSL配置模板"""
        config = {
            'ssl_parameters': {
                'lambda_det': 1.0,      # 检测损失权重
                'lambda_cm': 0.5,       # 跨模态对比损失权重
                'lambda_cons': 0.3,     # 一致性损失权重
                'lambda_spatial': 0.2,  # 空间对比损失权重
                'lambda_weather': 0.4   # 天气感知损失权重
            },
            'teacher_config': {
                'ema_decay': 0.999,     # EMA衰减率
                'warmup_epochs': 5      # 预热轮数
            },
            'contrastive_config': {
                'temperature': 0.1,     # 对比学习温度
                'proj_dim': 256        # 投影维度
            }
        }
        
        return config