"""
Consistency losses for teacher-student training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for teacher-student training
    Ensures consistent predictions between teacher and student models
    """
    
    def __init__(self, consistency_weight: float = 1.0, 
                 temperature: float = 4.0,
                 consistency_type: str = 'kl'):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.temperature = temperature
        self.consistency_type = consistency_type
        
    def forward(self, student_outputs: Dict, teacher_outputs: Dict,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute consistency loss between student and teacher predictions
        
        Args:
            student_outputs: Student model predictions
            teacher_outputs: Teacher model predictions  
            mask: Optional mask for valid predictions
            
        Returns:
            Dictionary containing consistency loss components
        """
        losses = {}
        
        # Classification consistency
        if 'cls_scores' in student_outputs and 'cls_scores' in teacher_outputs:
            cls_loss = self._compute_classification_consistency(
                student_outputs['cls_scores'], 
                teacher_outputs['cls_scores'],
                mask
            )
            losses['cls_consistency'] = cls_loss
        
        # Regression consistency
        if 'bbox_preds' in student_outputs and 'bbox_preds' in teacher_outputs:
            reg_loss = self._compute_regression_consistency(
                student_outputs['bbox_preds'],
                teacher_outputs['bbox_preds'], 
                mask
            )
            losses['reg_consistency'] = reg_loss
        
        # Direction consistency (for 3D detection)
        if 'dir_cls_preds' in student_outputs and 'dir_cls_preds' in teacher_outputs:
            dir_loss = self._compute_classification_consistency(
                student_outputs['dir_cls_preds'],
                teacher_outputs['dir_cls_preds'],
                mask
            )
            losses['dir_consistency'] = dir_loss
        
        # Feature consistency
        if 'features' in student_outputs and 'features' in teacher_outputs:
            feat_loss = self._compute_feature_consistency(
                student_outputs['features'],
                teacher_outputs['features'],
                mask
            )
            losses['feature_consistency'] = feat_loss
        
        # Total consistency loss
        total_loss = sum(losses.values()) * self.consistency_weight
        losses['total_consistency'] = total_loss
        
        return losses
    
    def _compute_classification_consistency(self, student_logits: torch.Tensor,
                                          teacher_logits: torch.Tensor,
                                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute classification consistency loss"""
        if self.consistency_type == 'kl':
            # KL divergence between softmax distributions
            student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            loss = F.kl_div(student_probs, teacher_probs, reduction='none').sum(-1)
            
        elif self.consistency_type == 'mse':
            # MSE between logits
            loss = F.mse_loss(student_logits, teacher_logits, reduction='none').mean(-1)
            
        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def _compute_regression_consistency(self, student_preds: torch.Tensor,
                                      teacher_preds: torch.Tensor,
                                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute regression consistency loss"""
        # Smooth L1 loss for regression
        loss = F.smooth_l1_loss(student_preds, teacher_preds, reduction='none', beta=1.0)
        loss = loss.mean(-1)
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def _compute_feature_consistency(self, student_features: torch.Tensor,
                                   teacher_features: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute feature-level consistency loss"""
        # Cosine similarity loss
        student_norm = F.normalize(student_features, dim=-1)
        teacher_norm = F.normalize(teacher_features, dim=-1)
        
        cosine_sim = (student_norm * teacher_norm).sum(-1)
        loss = 1 - cosine_sim  # Convert similarity to loss
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss


class PseudoLabelConsistency(nn.Module):
    """
    Pseudo-label consistency for semi-supervised learning
    Uses confident teacher predictions as pseudo-labels for student
    """
    
    def __init__(self, confidence_threshold: float = 0.7,
                 pseudo_label_weight: float = 0.5,
                 class_balanced: bool = True):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.pseudo_label_weight = pseudo_label_weight
        self.class_balanced = class_balanced
        
    def forward(self, student_outputs: Dict, teacher_outputs: Dict,
                gt_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate pseudo-labels and compute consistency loss
        
        Args:
            student_outputs: Student predictions
            teacher_outputs: Teacher predictions
            gt_labels: Ground truth labels (if available)
            
        Returns:
            Dictionary with pseudo-label losses
        """
        losses = {}
        
        if 'cls_scores' in teacher_outputs:
            # Generate pseudo-labels from teacher predictions
            teacher_probs = F.softmax(teacher_outputs['cls_scores'], dim=-1)
            max_probs, pseudo_labels = torch.max(teacher_probs, dim=-1)
            
            # Filter by confidence threshold
            confident_mask = max_probs > self.confidence_threshold
            
            if confident_mask.sum() > 0:
                # Compute pseudo-label classification loss
                student_logits = student_outputs['cls_scores'][confident_mask]
                pseudo_targets = pseudo_labels[confident_mask]
                
                pseudo_cls_loss = F.cross_entropy(student_logits, pseudo_targets)
                losses['pseudo_cls_loss'] = pseudo_cls_loss * self.pseudo_label_weight
                
                # Statistics
                losses['pseudo_label_ratio'] = confident_mask.float().mean()
            else:
                losses['pseudo_cls_loss'] = torch.tensor(0.0, device=teacher_outputs['cls_scores'].device)
                losses['pseudo_label_ratio'] = torch.tensor(0.0, device=teacher_outputs['cls_scores'].device)
        
        return losses


class MultiScaleConsistency(nn.Module):
    """
    Multi-scale consistency for 3D detection
    Ensures consistent predictions across different scales/resolutions
    """
    
    def __init__(self, scales: List[float] = [0.8, 1.0, 1.2],
                 consistency_weight: float = 0.3):
        super().__init__()
        self.scales = scales
        self.consistency_weight = consistency_weight
        
    def forward(self, multi_scale_outputs: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale consistency loss
        
        Args:
            multi_scale_outputs: List of outputs from different scales
            
        Returns:
            Multi-scale consistency loss
        """
        if len(multi_scale_outputs) < 2:
            return {'multiscale_consistency': torch.tensor(0.0)}
        
        losses = []
        
        # Compare each scale with the reference scale (typically 1.0)
        ref_idx = len(multi_scale_outputs) // 2  # Middle scale as reference
        ref_outputs = multi_scale_outputs[ref_idx]
        
        for i, scale_outputs in enumerate(multi_scale_outputs):
            if i == ref_idx:
                continue
                
            # Feature consistency between scales
            if 'features' in scale_outputs and 'features' in ref_outputs:
                # Resize features to match reference scale
                scale_features = scale_outputs['features']
                ref_features = ref_outputs['features']
                
                if scale_features.shape != ref_features.shape:
                    scale_features = F.interpolate(
                        scale_features, 
                        size=ref_features.shape[-2:],
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Compute cosine similarity loss
                scale_norm = F.normalize(scale_features, dim=1)
                ref_norm = F.normalize(ref_features, dim=1)
                
                cosine_sim = (scale_norm * ref_norm).sum(1).mean()
                scale_loss = 1 - cosine_sim
                
                losses.append(scale_loss)
        
        if losses:
            total_loss = torch.stack(losses).mean() * self.consistency_weight
        else:
            total_loss = torch.tensor(0.0)
        
        return {'multiscale_consistency': total_loss}


class TemporalConsistency(nn.Module):
    """
    Temporal consistency for sequential frames (if available)
    """
    
    def __init__(self, temporal_weight: float = 0.2,
                 max_temporal_distance: float = 10.0):
        super().__init__()
        self.temporal_weight = temporal_weight
        self.max_temporal_distance = max_temporal_distance
        
    def forward(self, current_outputs: Dict, previous_outputs: Dict,
                temporal_distance: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute temporal consistency loss
        
        Args:
            current_outputs: Current frame predictions
            previous_outputs: Previous frame predictions
            temporal_distance: Time distance between frames
            
        Returns:
            Temporal consistency loss
        """
        # Weight by temporal distance (closer frames should be more consistent)
        temporal_weights = torch.exp(-temporal_distance / self.max_temporal_distance)
        
        losses = {}
        
        # Feature temporal consistency
        if 'features' in current_outputs and 'features' in previous_outputs:
            curr_feat = current_outputs['features']
            prev_feat = previous_outputs['features']
            
            # Cosine similarity between temporal features
            curr_norm = F.normalize(curr_feat, dim=-1)
            prev_norm = F.normalize(prev_feat, dim=-1)
            
            temporal_sim = (curr_norm * prev_norm).sum(-1)
            temporal_loss = (1 - temporal_sim) * temporal_weights
            
            losses['temporal_consistency'] = temporal_loss.mean() * self.temporal_weight
        
        return losses