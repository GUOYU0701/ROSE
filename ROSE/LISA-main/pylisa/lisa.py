"""
Temporary Mock LISA Implementation for Testing
This is a simplified version for testing the enhanced functionality.
"""

import numpy as np
from typing import Dict, Optional


class Lisa:
    """Mock LISA class for weather augmentation testing"""
    
    def __init__(self, rmax=200.0, rmin=1.5, wavelength=905e-9, mode='strongest'):
        """Initialize LISA with physics parameters"""
        self.rmax = rmax
        self.rmin = rmin
        self.wavelength = wavelength
        self.mode = mode
        print(f"Mock LISA initialized with range [{rmin}, {rmax}]m, wavelength {wavelength*1e9}nm")
    
    def augment_mc(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Mock Monte Carlo augmentation for rain/snow
        
        Args:
            points: (N, 4) point cloud array [x, y, z, intensity]
            **kwargs: Additional parameters (ignored in mock)
            
        Returns:
            Augmented point cloud with simulated weather effects
        """
        if points is None or len(points) == 0:
            return points
            
        augmented = points.copy()
        
        # Simple mock augmentation: reduce intensity and add noise
        # Simulate rain/snow particle scattering effects
        intensity_reduction = np.random.uniform(0.8, 0.95, size=len(points))
        augmented[:, 3] *= intensity_reduction  # Reduce intensity
        
        # Add small random noise to position (simulate scattering)
        position_noise = np.random.normal(0, 0.02, size=(len(points), 3))
        augmented[:, :3] += position_noise
        
        # Remove some points (simulate occlusion)
        removal_rate = 0.05  # Remove 5% of points
        keep_mask = np.random.random(len(points)) > removal_rate
        augmented = augmented[keep_mask]
        
        return augmented.astype(np.float32)
    
    def augment_avg(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Mock average extinction augmentation for fog
        
        Args:
            points: (N, 4) point cloud array [x, y, z, intensity]
            **kwargs: Additional parameters (ignored in mock)
            
        Returns:
            Augmented point cloud with fog effects
        """
        if points is None or len(points) == 0:
            return points
            
        augmented = points.copy()
        
        # Simple mock fog: distance-based intensity reduction
        distances = np.linalg.norm(augmented[:, :3], axis=1)
        max_distance = np.max(distances)
        
        # Fog extinction: exponential decay with distance
        fog_intensity = 0.02  # Fog extinction coefficient
        extinction_factor = np.exp(-fog_intensity * distances)
        
        # Apply extinction to intensity
        augmented[:, 3] *= extinction_factor
        
        # Remove points with very low intensity
        min_intensity_threshold = 0.1
        keep_mask = augmented[:, 3] > min_intensity_threshold
        augmented = augmented[keep_mask]
        
        return augmented.astype(np.float32)
    
    def haze_point_cloud(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Mock haze modeling
        
        Args:
            points: (N, 4) point cloud array [x, y, z, intensity]
            **kwargs: Additional parameters (ignored in mock)
            
        Returns:
            Augmented point cloud with haze effects
        """
        return self.augment_avg(points, **kwargs)  # Use same logic as fog
