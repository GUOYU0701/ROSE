"""
SSL Trainer for ROSE framework
Combines detection losses with self-supervised learning objectives
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict

from .contrastive_loss import (
    CrossModalContrastiveLoss, 
    SpatialContrastiveLoss,
    WeatherAwareContrastiveLoss
)
from .consistency_loss import ConsistencyLoss, PseudoLabelConsistency
from .ema_teacher import EMATeacher
# Import moved to avoid circular import - imported when needed


class SSLTrainer:
    """
    Self-Supervised Learning Trainer for 3D Detection
    Integrates multiple SSL objectives with detection training
    """
    
    def __init__(self, 
                 lambda_det: float = 1.0,
                 lambda_cm: float = 0.5,  # Cross-modal contrastive
                 lambda_cons: float = 0.3,  # Consistency 
                 lambda_spatial: float = 0.2,  # Spatial contrastive
                 lambda_weather: float = 0.4,  # Weather-aware
                 ema_decay: float = 0.999,
                 consistency_warmup_epochs: int = 5,
                 enable_pseudo_labeling: bool = True):
        
        self.lambda_det = lambda_det
        self.lambda_cm = lambda_cm
        self.lambda_cons = lambda_cons
        self.lambda_spatial = lambda_spatial
        self.lambda_weather = lambda_weather
        
        self.consistency_warmup_epochs = consistency_warmup_epochs
        self.enable_pseudo_labeling = enable_pseudo_labeling
        self.current_epoch = 0
        
        # Initialize SSL loss components
        self.cross_modal_loss = CrossModalContrastiveLoss(
            temperature=0.07, 
            lambda_weight=lambda_cm
        )
        
        self.consistency_loss = ConsistencyLoss(
            consistency_weight=lambda_cons,
            temperature=4.0,
            consistency_type='kl'
        )
        
        self.spatial_contrastive_loss = SpatialContrastiveLoss(
            temperature=0.07,
            spatial_threshold=2.0
        )
        
        self.weather_contrastive_loss = WeatherAwareContrastiveLoss(
            temperature=0.07,
            weather_weight=0.5
        )
        
        # Flag to track device placement
        self.device_initialized = False
        
        if enable_pseudo_labeling:
            self.pseudo_label_loss = PseudoLabelConsistency(
                confidence_threshold=0.7,
                pseudo_label_weight=0.5
            )
        
        # Teacher model (will be initialized with student model)
        self.teacher = None
        self.ema_decay = ema_decay
        
        # Loss tracking
        self.loss_history = defaultdict(list)
        
        # Enhanced analytics tracking
        self.ssl_analytics = {
            'feature_alignment_scores': [],
            'cross_modal_similarities': [],
            'weather_adaptation_scores': [],
            'consistency_convergence': [],
            'class_specific_performance': defaultdict(list)
        }
        
    def initialize_teacher(self, student_model: nn.Module):
        """Initialize EMA teacher model"""
        self.teacher = EMATeacher(
            student_model, 
            ema_decay=self.ema_decay,
            warmup_steps=1000
        )
        # Move teacher to same device as student
        if next(student_model.parameters()).is_cuda:
            self.teacher = self.teacher.cuda()
            
        # Initialize SSL components on correct device
        self._initialize_device(next(student_model.parameters()).device)
            
    def _initialize_device(self, device: torch.device):
        """Initialize SSL components on correct device"""
        if not self.device_initialized:
            self.cross_modal_loss = self.cross_modal_loss.to(device)
            self.consistency_loss = self.consistency_loss.to(device)
            self.spatial_contrastive_loss = self.spatial_contrastive_loss.to(device)
            self.weather_contrastive_loss = self.weather_contrastive_loss.to(device)
            
            if hasattr(self, 'pseudo_label_loss'):
                self.pseudo_label_loss = self.pseudo_label_loss.to(device)
                
            self.device_initialized = True
        
    def compute_ssl_losses(self, 
                          student_outputs: Dict,
                          teacher_outputs: Optional[Dict],
                          batch_data: Dict,
                          step: int) -> Dict[str, torch.Tensor]:
        """
        Compute all SSL losses
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs (optional)
            batch_data: Batch data containing images, points, labels, etc.
            step: Current training step
            
        Returns:
            Dictionary of SSL losses
        """
        ssl_losses = {}
        
        # Cross-modal contrastive loss
        if self.lambda_cm > 0 and self._has_multimodal_features(student_outputs):
            cm_losses = self._compute_cross_modal_loss(student_outputs, batch_data)
            ssl_losses.update(cm_losses)
            
        # Teacher-student consistency loss
        if (self.lambda_cons > 0 and teacher_outputs is not None and 
            self.current_epoch >= self.consistency_warmup_epochs):
            consistency_losses = self._compute_consistency_loss(
                student_outputs, teacher_outputs, batch_data
            )
            ssl_losses.update(consistency_losses)
            
        # Spatial contrastive loss
        if self.lambda_spatial > 0 and self._has_spatial_info(batch_data):
            spatial_loss = self._compute_spatial_contrastive_loss(
                student_outputs, batch_data
            )
            ssl_losses['spatial_contrastive'] = spatial_loss
            
        # Weather-aware contrastive loss
        if self.lambda_weather > 0 and self._has_weather_info(batch_data):
            weather_losses = self._compute_weather_contrastive_loss(
                student_outputs, batch_data
            )
            ssl_losses.update(weather_losses)
            
        # Pseudo-labeling loss
        if (self.enable_pseudo_labeling and teacher_outputs is not None and
            self.current_epoch >= self.consistency_warmup_epochs):
            pseudo_losses = self.pseudo_label_loss(
                student_outputs, teacher_outputs, batch_data.get('gt_labels_3d')
            )
            ssl_losses.update(pseudo_losses)
            
        # Class-specific SSL enhancement for problematic classes
        if self._has_problematic_classes(batch_data):
            class_specific_losses = self._compute_class_specific_ssl_loss(
                student_outputs, teacher_outputs, batch_data
            )
            ssl_losses.update(class_specific_losses)
        
        # Track SSL analytics
        self._update_ssl_analytics(ssl_losses, student_outputs, teacher_outputs, batch_data)
        
        return ssl_losses
    
    def _compute_cross_modal_loss(self, outputs: Dict, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """Compute cross-modal contrastive loss"""
        img_features = outputs.get('img_features')
        pts_features = outputs.get('pts_features')
        
        if img_features is not None and pts_features is not None:
            # Pool features if needed (e.g., from multi-level features)
            if isinstance(img_features, (list, tuple)):
                img_features = self._pool_multilevel_features(img_features)
            if isinstance(pts_features, (list, tuple)):
                pts_features = self._pool_multilevel_features(pts_features)
                
            roi_coords = batch_data.get('roi_coordinates')
            return self.cross_modal_loss(img_features, pts_features, roi_coords)
        
        return {}
    
    def _compute_consistency_loss(self, student_outputs: Dict, 
                                teacher_outputs: Dict, 
                                batch_data: Dict) -> Dict[str, torch.Tensor]:
        """Compute teacher-student consistency loss"""
        # Create confidence mask for reliable teacher predictions
        mask = self._create_confidence_mask(teacher_outputs, batch_data)
        
        return self.consistency_loss(student_outputs, teacher_outputs, mask)
    
    def _compute_spatial_contrastive_loss(self, outputs: Dict, batch_data: Dict) -> torch.Tensor:
        """Compute spatial contrastive loss"""
        # Extract object features, bboxes, and labels
        obj_features = outputs.get('object_features')  # Features for detected objects
        gt_bboxes_3d = batch_data.get('gt_bboxes_3d')
        gt_labels_3d = batch_data.get('gt_labels_3d')
        
        if obj_features is not None and gt_bboxes_3d is not None and gt_labels_3d is not None:
            # Convert to tensors if needed
            if isinstance(gt_bboxes_3d, (list, tuple)):
                gt_bboxes_3d = torch.cat(gt_bboxes_3d, dim=0)
            if isinstance(gt_labels_3d, (list, tuple)):
                gt_labels_3d = torch.cat(gt_labels_3d, dim=0)
                
            return self.spatial_contrastive_loss(obj_features, gt_bboxes_3d, gt_labels_3d)
        
        return torch.tensor(0.0, device=next(iter(outputs.values())).device)
    
    def _compute_weather_contrastive_loss(self, outputs: Dict, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """Compute weather-aware contrastive loss"""
        clean_features = outputs.get('clean_features')
        augmented_features = outputs.get('augmented_features') 
        weather_types = batch_data.get('weather_types')
        
        if (clean_features is not None and augmented_features is not None and 
            weather_types is not None):
            return self.weather_contrastive_loss(
                clean_features, augmented_features, weather_types
            )
        
        return {}
    
    def _pool_multilevel_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Pool multi-level features into single representation"""
        # Simple approach: use highest resolution features
        if len(features) == 1:
            return features[0]
        
        # Use the feature map with largest spatial dimensions
        max_size = max(feat.numel() for feat in features)
        for feat in features:
            if feat.numel() == max_size:
                return feat
                
        return features[0]  # Fallback
    
    def _create_confidence_mask(self, teacher_outputs: Dict, batch_data: Dict,
                               confidence_threshold: float = 0.7) -> Optional[torch.Tensor]:
        """Create mask for confident teacher predictions"""
        if 'cls_scores' not in teacher_outputs:
            return None
            
        cls_scores = teacher_outputs['cls_scores']
        probs = torch.softmax(cls_scores, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        
        confidence_mask = max_probs > confidence_threshold
        return confidence_mask
    
    def _has_multimodal_features(self, outputs: Dict) -> bool:
        """Check if outputs contain multi-modal features"""
        return ('img_features' in outputs and 'pts_features' in outputs and
                outputs['img_features'] is not None and outputs['pts_features'] is not None)
    
    def _has_spatial_info(self, batch_data: Dict) -> bool:
        """Check if batch contains spatial information for spatial contrastive loss"""
        return ('gt_bboxes_3d' in batch_data and 'gt_labels_3d' in batch_data)
    
    def _has_weather_info(self, batch_data: Dict) -> bool:
        """Check if batch contains weather information"""
        return ('weather_types' in batch_data or 'clean_features' in batch_data)
    
    def update_teacher(self, student_model: nn.Module, step: int):
        """Update teacher model using EMA"""
        if self.teacher is not None:
            self.teacher.update_teacher(student_model, step)
    
    def get_teacher_predictions(self, batch_inputs: Dict) -> Optional[Dict]:
        """Get predictions from teacher model"""
        if self.teacher is None:
            return None
            
        with torch.no_grad():
            self.teacher.eval()
            # Extract features only, not full forward pass
            try:
                # Try to call extract_feat method directly
                img_feats, pts_feats = self.teacher.teacher_model.extract_feat(batch_inputs)
                teacher_outputs = {
                    'img_feats': img_feats,
                    'pts_feats': pts_feats
                }
            except Exception as e:
                # Fallback: return None if teacher prediction fails
                return None
            
        return teacher_outputs
    
    def compute_total_loss(self, 
                          detection_losses: Dict[str, torch.Tensor],
                          ssl_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss combining detection and SSL objectives
        
        Args:
            detection_losses: Standard detection losses
            ssl_losses: SSL losses
            
        Returns:
            Tuple of (total_loss, loss_dict_for_logging)
        """
        # Detection losses - handle different loss formats
        det_loss_total = 0.0
        for loss_name, loss_value in detection_losses.items():
            if isinstance(loss_value, (list, tuple)):
                # Handle multiple loss values (from different heads/stages)
                det_loss_total += sum(loss_value)
            else:
                # Single loss value
                det_loss_total += loss_value
        det_loss_total = det_loss_total * self.lambda_det
        
        # SSL losses with weights
        ssl_loss_total = 0.0
        weighted_ssl_losses = {}
        
        # Get device from detection losses
        device = torch.device('cpu')
        if detection_losses:
            for loss_value in detection_losses.values():
                if isinstance(loss_value, torch.Tensor):
                    device = loss_value.device
                    break
                elif isinstance(loss_value, (list, tuple)) and len(loss_value) > 0:
                    if isinstance(loss_value[0], torch.Tensor):
                        device = loss_value[0].device
                        break
        
        for loss_name, loss_value in ssl_losses.items():
            if 'cross_modal' in loss_name:
                weighted_loss = loss_value * self.lambda_cm
            elif 'consistency' in loss_name:
                weighted_loss = loss_value * self.lambda_cons
            elif 'spatial' in loss_name:
                weighted_loss = loss_value * self.lambda_spatial
            elif 'weather' in loss_name:
                weighted_loss = loss_value * self.lambda_weather
            else:
                weighted_loss = loss_value
                
            # Ensure weighted_loss is a scalar tensor
            if isinstance(weighted_loss, torch.Tensor):
                if weighted_loss.dim() > 0:
                    weighted_loss = weighted_loss.mean()
                ssl_loss_total += weighted_loss
            else:
                ssl_loss_total += torch.tensor(weighted_loss, dtype=torch.float32, device=device)
            weighted_ssl_losses[loss_name] = weighted_loss
        
        # Ensure all losses are tensors 
        if isinstance(ssl_loss_total, (int, float)):
            ssl_loss_total = torch.tensor(ssl_loss_total, device=det_loss_total.device, dtype=det_loss_total.dtype)
        
        # Total loss
        total_loss = det_loss_total + ssl_loss_total
        
        # Loss dictionary for logging (only include tensor losses)
        loss_dict = {}
        
        # Add detection losses
        for key, value in detection_losses.items():
            if isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    loss_dict[f"{key}_{i}"] = v
            else:
                loss_dict[key] = value
        
        # Add SSL losses
        loss_dict['ssl_loss'] = ssl_loss_total
        for key, value in weighted_ssl_losses.items():
            loss_dict[key] = value
        
        # Update loss history
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.loss_history[key].append(value.detach().cpu().item())
        
        return total_loss, loss_dict
    
    def set_epoch(self, epoch: int):
        """Set current epoch for loss scheduling"""
        self.current_epoch = epoch
    
    def get_loss_statistics(self) -> Dict[str, Dict]:
        """Get loss statistics for monitoring"""
        stats = {}
        for loss_name, values in self.loss_history.items():
            if values:
                stats[loss_name] = {
                    'mean': np.mean(values[-100:]),  # Last 100 values
                    'std': np.std(values[-100:]),
                    'min': np.min(values[-100:]),
                    'max': np.max(values[-100:]),
                    'count': len(values)
                }
        return stats
    
    def reset_loss_history(self):
        """Reset loss history"""
        self.loss_history = defaultdict(list)
    
    def _has_problematic_classes(self, batch_data: Dict) -> bool:
        """Check if batch contains problematic classes (Pedestrian, Cyclist)"""
        gt_labels = batch_data.get('gt_labels_3d')
        if gt_labels is None:
            return False
        
        # Assuming class indices: 0=Car, 1=Pedestrian, 2=Cyclist
        problematic_class_ids = [1, 2]  # Pedestrian, Cyclist
        
        if isinstance(gt_labels, (list, tuple)):
            for label_batch in gt_labels:
                if torch.any(torch.isin(label_batch, torch.tensor(problematic_class_ids, device=label_batch.device))):
                    return True
        elif isinstance(gt_labels, torch.Tensor):
            return torch.any(torch.isin(gt_labels, torch.tensor(problematic_class_ids, device=gt_labels.device)))
        
        return False
    
    def _compute_class_specific_ssl_loss(self, student_outputs: Dict, 
                                       teacher_outputs: Optional[Dict],
                                       batch_data: Dict) -> Dict[str, torch.Tensor]:
        """Compute enhanced SSL losses for problematic classes"""
        class_ssl_losses = {}
        
        gt_labels = batch_data.get('gt_labels_3d')
        if gt_labels is None:
            return class_ssl_losses
        
        # Extract features and predictions for Pedestrian and Cyclist classes
        pred_scores = student_outputs.get('cls_scores', [])
        pred_bboxes = student_outputs.get('bbox_preds', [])
        
        # Enhanced cross-modal alignment for small objects
        if 'img_features' in student_outputs and 'pts_features' in student_outputs:
            img_feats = student_outputs['img_features']
            pts_feats = student_outputs['pts_features']
            
            # Focus on small object regions
            small_obj_loss = self._compute_small_object_alignment_loss(
                img_feats, pts_feats, gt_labels, batch_data.get('gt_bboxes_3d')
            )
            if small_obj_loss > 0:
                class_ssl_losses['small_object_alignment'] = small_obj_loss
        
        # Enhanced consistency for problematic classes
        if teacher_outputs is not None:
            problematic_consistency_loss = self._compute_problematic_class_consistency_loss(
                student_outputs, teacher_outputs, gt_labels
            )
            if problematic_consistency_loss > 0:
                class_ssl_losses['problematic_class_consistency'] = problematic_consistency_loss
        
        # Augmentation-aware contrastive learning for challenging cases
        if self._has_augmentation_info(batch_data):
            aug_aware_loss = self._compute_augmentation_aware_loss(
                student_outputs, batch_data, focus_classes=[1, 2]  # Pedestrian, Cyclist
            )
            if aug_aware_loss > 0:
                class_ssl_losses['augmentation_aware'] = aug_aware_loss
        
        return class_ssl_losses
    
    def _compute_small_object_alignment_loss(self, img_features: torch.Tensor,
                                           pts_features: torch.Tensor,
                                           gt_labels: torch.Tensor,
                                           gt_bboxes: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute alignment loss focused on small objects"""
        device = img_features.device
        
        if gt_bboxes is None:
            return torch.tensor(0.0, device=device)
        
        # Identify small objects (Pedestrian=1, Cyclist=2)
        # Handle both list and tensor inputs
        if isinstance(gt_labels, list):
            if not gt_labels:
                return torch.tensor(0.0, device=device)
            gt_labels_tensor = torch.cat(gt_labels, dim=0) if len(gt_labels) > 1 else gt_labels[0]
        else:
            gt_labels_tensor = gt_labels
        
        small_object_mask = torch.isin(gt_labels_tensor, torch.tensor([1, 2], device=device))
        
        if not torch.any(small_object_mask):
            return torch.tensor(0.0, device=device)
        
        # Extract features for small objects
        # This is a simplified version - in practice, you'd need proper RoI extraction
        if img_features.dim() > 2:
            img_features = torch.mean(img_features, dim=[-2, -1])  # Global pool
        if pts_features.dim() > 2:
            pts_features = torch.mean(pts_features, dim=[-2, -1])  # Global pool
        
        # Compute alignment loss for small objects
        if img_features.shape[0] > 0 and pts_features.shape[0] > 0:
            # Cosine similarity loss
            img_norm = torch.nn.functional.normalize(img_features, dim=-1)
            pts_norm = torch.nn.functional.normalize(pts_features, dim=-1)
            
            similarity = torch.mm(img_norm, pts_norm.t())
            alignment_loss = 1.0 - torch.mean(torch.diag(similarity))
            
            return alignment_loss * 2.0  # Increase weight for small objects
        
        return torch.tensor(0.0, device=device)
    
    def _compute_problematic_class_consistency_loss(self, student_outputs: Dict,
                                                  teacher_outputs: Dict,
                                                  gt_labels: torch.Tensor) -> torch.Tensor:
        """Enhanced consistency loss for problematic classes"""
        device = next(iter(student_outputs.values())).device
        
        student_scores = student_outputs.get('cls_scores', [])
        teacher_scores = teacher_outputs.get('cls_scores', [])
        
        if not student_scores or not teacher_scores:
            return torch.tensor(0.0, device=device)
        
        consistency_loss = torch.tensor(0.0, device=device)
        
        # Process each level of predictions
        for s_score, t_score in zip(student_scores, teacher_scores):
            if s_score.shape != t_score.shape:
                continue
            
            # Focus on problematic classes (indices 1, 2)
            problematic_indices = torch.tensor([1, 2], device=device)
            
            # Extract scores for problematic classes
            s_prob_scores = s_score[:, problematic_indices]
            t_prob_scores = t_score[:, problematic_indices]
            
            # Enhanced consistency loss with higher temperature for stability
            temperature = 8.0
            s_soft = torch.softmax(s_prob_scores / temperature, dim=-1)
            t_soft = torch.softmax(t_prob_scores / temperature, dim=-1)
            
            # KL divergence loss
            kl_loss = torch.nn.functional.kl_div(
                torch.log(s_soft + 1e-8), t_soft, reduction='batchmean'
            )
            
            consistency_loss += kl_loss * 1.5  # Higher weight for problematic classes
        
        return consistency_loss
    
    def _compute_augmentation_aware_loss(self, student_outputs: Dict,
                                       batch_data: Dict,
                                       focus_classes: List[int]) -> torch.Tensor:
        """Compute loss that adapts to augmentation intensity"""
        device = next(iter(student_outputs.values())).device
        
        weather_info = batch_data.get('augmentation_info', [])
        if not weather_info:
            return torch.tensor(0.0, device=device)
        
        # Get augmentation intensities
        intensities = []
        for info in weather_info:
            if isinstance(info, dict):
                intensities.append(info.get('intensity', 0.0))
            else:
                intensities.append(0.0)
        
        if not intensities:
            return torch.tensor(0.0, device=device)
        
        avg_intensity = sum(intensities) / len(intensities)
        
        # Adaptive loss weight based on augmentation intensity
        # Higher intensity -> higher SSL weight to maintain robustness
        adaptive_weight = 1.0 + avg_intensity * 2.0
        
        # Focus on maintaining feature quality under augmentation
        cls_scores = student_outputs.get('cls_scores', [])
        if not cls_scores:
            return torch.tensor(0.0, device=device)
        
        # Compute feature stability loss
        stability_loss = torch.tensor(0.0, device=device)
        for score in cls_scores:
            if score.dim() > 1:
                # Focus on problematic classes
                focus_scores = score[:, focus_classes] if score.shape[1] > max(focus_classes) else score
                
                # Entropy regularization to maintain confidence under augmentation
                softmax_scores = torch.softmax(focus_scores, dim=-1)
                entropy = -torch.sum(softmax_scores * torch.log(softmax_scores + 1e-8), dim=-1)
                
                # Lower entropy (higher confidence) is better
                stability_loss += torch.mean(entropy) * adaptive_weight
        
        return stability_loss
    
    def _has_augmentation_info(self, batch_data: Dict) -> bool:
        """Check if batch contains augmentation information"""
        aug_info = batch_data.get('augmentation_info', [])
        return len(aug_info) > 0
    
    def _update_ssl_analytics(self, ssl_losses: Dict, student_outputs: Dict,
                            teacher_outputs: Optional[Dict], batch_data: Dict):
        """Update SSL analytics for monitoring"""
        # Feature alignment score
        if 'cross_modal' in str(ssl_losses.keys()):
            for loss_name, loss_value in ssl_losses.items():
                if 'cross_modal' in loss_name and isinstance(loss_value, torch.Tensor):
                    # Convert loss to alignment score (lower loss = higher alignment)
                    alignment_score = torch.exp(-loss_value).detach().cpu().item()
                    self.ssl_analytics['feature_alignment_scores'].append(alignment_score)
        
        # Consistency convergence
        if 'consistency' in str(ssl_losses.keys()):
            for loss_name, loss_value in ssl_losses.items():
                if 'consistency' in loss_name and isinstance(loss_value, torch.Tensor):
                    consistency_score = loss_value.detach().cpu().item()
                    self.ssl_analytics['consistency_convergence'].append(consistency_score)
        
        # Weather adaptation tracking
        if self._has_augmentation_info(batch_data):
            weather_info = batch_data.get('augmentation_info', [])
            avg_intensity = np.mean([
                info.get('intensity', 0.0) for info in weather_info 
                if isinstance(info, dict)
            ])
            self.ssl_analytics['weather_adaptation_scores'].append(avg_intensity)
        
        # Class-specific performance tracking
        if 'problematic_class_consistency' in ssl_losses:
            consistency_loss = ssl_losses['problematic_class_consistency']
            if isinstance(consistency_loss, torch.Tensor):
                for class_idx in [1, 2]:  # Pedestrian, Cyclist
                    class_name = ['Car', 'Pedestrian', 'Cyclist'][class_idx]
                    self.ssl_analytics['class_specific_performance'][class_name].append(
                        consistency_loss.detach().cpu().item()
                    )
    
    def get_ssl_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of SSL analytics"""
        summary = {}
        
        for metric_name, values in self.ssl_analytics.items():
            if values:
                if isinstance(values, dict):
                    summary[metric_name] = {}
                    for sub_name, sub_values in values.items():
                        if sub_values:
                            summary[metric_name][sub_name] = {
                                'mean': np.mean(sub_values[-50:]),  # Last 50 values
                                'std': np.std(sub_values[-50:]),
                                'trend': self._compute_trend(sub_values[-20:]) if len(sub_values) >= 20 else 0
                            }
                else:
                    summary[metric_name] = {
                        'mean': np.mean(values[-50:]) if values else 0,
                        'std': np.std(values[-50:]) if values else 0,
                        'trend': self._compute_trend(values[-20:]) if len(values) >= 20 else 0,
                        'latest': values[-1] if values else 0
                    }
        
        return summary
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend of values using linear regression"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0.0
        return float(slope)
    
    def save_ssl_state(self, epoch: int):
        """Save SSL state for monitoring"""
        try:
            import json
            from pathlib import Path
            
            # Save SSL analytics
            ssl_state = {
                'epoch': epoch,
                'ssl_analytics': self.get_ssl_analytics_summary(),
                'loss_history': dict(self.loss_history),
                'current_epoch': self.current_epoch,
                'lambda_weights': {
                    'det': self.lambda_det,
                    'cm': self.lambda_cm,
                    'cons': self.lambda_cons,
                    'spatial': self.lambda_spatial,
                    'weather': self.lambda_weather
                }
            }
            
            # Create SSL state directory
            ssl_dir = Path(f'work_dirs/ssl_states')
            ssl_dir.mkdir(parents=True, exist_ok=True)
            
            # Save state
            state_file = ssl_dir / f'ssl_state_epoch_{epoch}.json'
            with open(state_file, 'w') as f:
                json.dump(ssl_state, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Failed to save SSL state: {e}")