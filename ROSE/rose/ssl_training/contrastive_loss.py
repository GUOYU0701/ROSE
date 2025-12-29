"""
Contrastive learning losses for cross-modal consistency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class ContrastiveLoss(nn.Module):
    """
    Basic contrastive loss for self-supervised learning
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, features_1: torch.Tensor, features_2: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss between two feature sets
        
        Args:
            features_1: First set of features (N, D)
            features_2: Second set of features (N, D)
            labels: Optional similarity labels (N,)
        
        Returns:
            Contrastive loss
        """
        # Normalize features
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features_1, features_2.T) / self.temperature
        
        if labels is None:
            # Use identity as positive pairs (same sample different modality)
            labels = torch.eye(similarity.size(0), device=similarity.device)
        
        # InfoNCE loss
        exp_sim = torch.exp(similarity)
        pos_sim = torch.sum(exp_sim * labels, dim=1)
        all_sim = torch.sum(exp_sim, dim=1)
        
        loss = -torch.log(pos_sim / all_sim).mean()
        
        return loss


class CrossModalContrastiveLoss(nn.Module):
    """
    Cross-modal contrastive loss for image and point cloud features
    Designed for multi-modal 3D detection
    """
    
    def __init__(self, temperature: float = 0.07, lambda_weight: float = 1.0,
                 projection_dim: int = 128):
        super().__init__()
        self.temperature = temperature
        self.lambda_weight = lambda_weight
        self.projection_dim = projection_dim
        
        # Projection heads will be created dynamically with proper device placement
        self.img_projector = None
        self.pts_projector = None
        self.projectors_initialized = False
        
    def _initialize_projectors(self, img_dim: int, pts_dim: int, device: torch.device):
        """Initialize projectors with correct dimensions and device placement"""
        if self.projectors_initialized:
            return
            
        self.img_projector = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        ).to(device)
        
        self.pts_projector = nn.Sequential(
            nn.Linear(pts_dim, 128),
            nn.ReLU(), 
            nn.Linear(128, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        ).to(device)
        
        self.projectors_initialized = True
    
    def forward(self, img_features: torch.Tensor, pts_features: torch.Tensor,
                roi_coords: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute cross-modal contrastive loss
        
        Args:
            img_features: Image features (N, D_img)
            pts_features: Point cloud features (N, D_pts)  
            roi_coords: Optional RoI coordinates for spatial alignment
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = img_features.size(0)
        device = img_features.device
        
        # Initialize projectors if needed
        if not self.projectors_initialized:
            img_dim = img_features.size(1)
            pts_dim = pts_features.size(1)
            self._initialize_projectors(img_dim, pts_dim, device)
        
        # Project features to common space
        img_proj = self.img_projector(img_features)
        pts_proj = self.pts_projector(pts_features)
        
        # Normalize projected features
        img_proj = F.normalize(img_proj, dim=1)
        pts_proj = F.normalize(pts_proj, dim=1)
        
        # Compute cross-modal similarity
        cross_similarity = torch.matmul(img_proj, pts_proj.T) / self.temperature
        
        # Create positive pair labels (same sample index)
        labels = torch.eye(batch_size, device=img_features.device)
        
        # Image to point cloud contrastive loss
        img_to_pts_loss = self._compute_infonce_loss(cross_similarity, labels)
        
        # Point cloud to image contrastive loss  
        pts_to_img_loss = self._compute_infonce_loss(cross_similarity.T, labels)
        
        # Total cross-modal loss
        total_loss = (img_to_pts_loss + pts_to_img_loss) / 2.0
        
        return {
            'cross_modal_loss': total_loss * self.lambda_weight,
            'img_to_pts_loss': img_to_pts_loss,
            'pts_to_img_loss': pts_to_img_loss,
            'cross_similarity': cross_similarity.detach()
        }
    
    def _compute_infonce_loss(self, similarity: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss"""
        exp_sim = torch.exp(similarity)
        pos_sim = torch.sum(exp_sim * labels, dim=1)
        all_sim = torch.sum(exp_sim, dim=1)
        
        # Add small epsilon for numerical stability
        loss = -torch.log(pos_sim / (all_sim + 1e-8)).mean()
        
        return loss


class SpatialContrastiveLoss(nn.Module):
    """
    Spatial-aware contrastive loss for 3D detection
    Considers spatial relationships between objects
    """
    
    def __init__(self, temperature: float = 0.07, spatial_threshold: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.spatial_threshold = spatial_threshold
    
    def forward(self, features: torch.Tensor, bbox_3d: torch.Tensor,
                labels_3d: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial contrastive loss
        
        Args:
            features: Object features (N, D)
            bbox_3d: 3D bounding boxes (N, 7) - x,y,z,w,h,l,r
            labels_3d: Object class labels (N,)
            
        Returns:
            Spatial contrastive loss
        """
        if features.size(0) <= 1:
            return torch.tensor(0.0, device=features.device)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute spatial distances between object centers
        centers = bbox_3d[:, :3]  # x, y, z centers
        spatial_dist = torch.cdist(centers, centers, p=2)
        
        # Create similarity matrix based on class and spatial proximity
        class_sim = (labels_3d.unsqueeze(0) == labels_3d.unsqueeze(1)).float()
        spatial_sim = (spatial_dist < self.spatial_threshold).float()
        
        # Positive pairs: same class and spatially close
        positive_mask = class_sim * spatial_sim
        # Remove self-similarity
        positive_mask.fill_diagonal_(0)
        
        # Negative pairs: different class or spatially far
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)
        
        # Compute feature similarity
        feature_sim = torch.matmul(features, features.T) / self.temperature
        
        # Contrastive loss
        exp_sim = torch.exp(feature_sim)
        pos_sum = torch.sum(exp_sim * positive_mask, dim=1)
        neg_sum = torch.sum(exp_sim * negative_mask, dim=1)
        
        # Only compute loss for samples with positive pairs
        valid_samples = pos_sum > 0
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        loss = -torch.log(pos_sum[valid_samples] / (pos_sum[valid_samples] + neg_sum[valid_samples] + 1e-8)).mean()
        
        return loss


class WeatherAwareContrastiveLoss(nn.Module):
    """
    Weather-aware contrastive loss for augmented samples
    Ensures consistent representations across weather conditions
    """
    
    def __init__(self, temperature: float = 0.07, weather_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.weather_weight = weather_weight
    
    def forward(self, clean_features: torch.Tensor, 
                augmented_features: torch.Tensor,
                weather_types: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute weather-aware contrastive loss
        
        Args:
            clean_features: Features from clean samples (N, D)
            augmented_features: Features from augmented samples (N, D)
            weather_types: Weather condition labels (N,)
            
        Returns:
            Dictionary containing loss components
        """
        # Normalize features
        clean_features = F.normalize(clean_features, dim=1)
        augmented_features = F.normalize(augmented_features, dim=1)
        
        # Cross-weather similarity matrix
        similarity = torch.matmul(clean_features, augmented_features.T) / self.temperature
        
        # Create positive pairs (same sample, different weather)
        positive_mask = torch.eye(similarity.size(0), device=similarity.device)
        
        # Weather consistency loss
        weather_consistency_loss = self._compute_infonce_loss(similarity, positive_mask)
        
        # Same weather type should be more similar
        same_weather_mask = (weather_types.unsqueeze(0) == weather_types.unsqueeze(1)).float()
        same_weather_mask.fill_diagonal_(0)
        
        if same_weather_mask.sum() > 0:
            same_weather_sim = torch.matmul(augmented_features, augmented_features.T) / self.temperature
            weather_grouping_loss = self._compute_infonce_loss(same_weather_sim, same_weather_mask)
        else:
            weather_grouping_loss = torch.tensor(0.0, device=similarity.device)
        
        total_loss = weather_consistency_loss + self.weather_weight * weather_grouping_loss
        
        return {
            'weather_contrastive_loss': total_loss,
            'weather_consistency_loss': weather_consistency_loss,
            'weather_grouping_loss': weather_grouping_loss
        }
    
    def _compute_infonce_loss(self, similarity: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss with positive mask"""
        exp_sim = torch.exp(similarity)
        pos_sum = torch.sum(exp_sim * positive_mask, dim=1)
        all_sum = torch.sum(exp_sim, dim=1)
        
        # Only compute loss for samples with positive pairs
        valid_samples = pos_sum > 0
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=similarity.device)
        
        loss = -torch.log(pos_sum[valid_samples] / (all_sum[valid_samples] + 1e-8)).mean()
        
        return loss