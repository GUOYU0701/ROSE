"""
Point cloud augmentation using LISA framework
Enhanced for ROSE with calibration-aware processing
"""
import numpy as np
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Add LISA to path
lisa_path = Path(__file__).parent.parent.parent / "LISA-main" / "pylisa"
sys.path.insert(0, str(lisa_path))

try:
    from lisa import Lisa
    LISA_AVAILABLE = True
except ImportError:
    print("Warning: LISA not available. Point cloud augmentation will be disabled.")
    LISA_AVAILABLE = False

from .config import WeatherConfig


class PointCloudAugmentor:
    """Point cloud augmentation using LISA physics-based simulation"""
    
    def __init__(self, point_cloud_range: list = [0, -40, -3, 70.4, 40, 1]):
        self.point_cloud_range = point_cloud_range
        self.lisa_instances = {}  # Cache LISA instances for different weather types
        
        if LISA_AVAILABLE:
            self._initialize_lisa_instances()
        
    def _initialize_lisa_instances(self):
        """Initialize LISA instances for different weather types"""
        # Rain model
        self.lisa_instances['rain'] = Lisa(
            rmax=200, rmin=1.5,
            wavelength=905e-9,
            mode='strongest'
        )
        
        # Snow model  
        self.lisa_instances['snow'] = Lisa(
            rmax=200, rmin=1.5,
            wavelength=905e-9,
            mode='strongest'
        )
        
        # Fog models
        self.lisa_instances['fog_chu_hogg'] = Lisa(
            rmax=200, rmin=1.5,
            wavelength=905e-9,
            mode='strongest'
        )
        
        self.lisa_instances['fog_strong_advection'] = Lisa(
            rmax=200, rmin=1.5,
            wavelength=905e-9,
            mode='strongest'
        )
        
        self.lisa_instances['fog_moderate_advection'] = Lisa(
            rmax=200, rmin=1.5,
            wavelength=905e-9,
            mode='strongest'
        )
    
    def augment_point_cloud(self, points: np.ndarray, weather_config: WeatherConfig,
                          calibration_info: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Augment point cloud with weather effects
        
        Args:
            points: Input point cloud (N, 4) - x, y, z, intensity
            weather_config: Weather configuration
            calibration_info: Calibration info for coordinate transforms
            
        Returns:
            Tuple of (augmented_points, augmentation_info)
        """
        if not LISA_AVAILABLE:
            print("Warning: LISA not available, returning original points")
            return points, {'weather_type': 'clear', 'num_lost': 0, 'num_scattered': 0}
        
        # Prepare point cloud for LISA (ensure correct format)
        if points.shape[1] < 4:
            # Add dummy intensity if not present
            intensity = np.ones((points.shape[0], 1)) * 0.5
            pc_input = np.hstack([points, intensity])
        else:
            pc_input = points[:, :4].copy()
            
        # Normalize reflectivity/intensity to [0, 1]
        if pc_input[:, 3].max() > 1.0:
            pc_input[:, 3] = pc_input[:, 3] / pc_input[:, 3].max()
        
        # Apply weather-specific augmentation
        augmentation_info = {
            'weather_type': weather_config.weather_type,
            'intensity': weather_config.intensity,
            'num_lost': 0,
            'num_scattered': 0,
            'num_original': 0
        }
        
        if weather_config.weather_type == 'clear':
            augmented_pc = pc_input
        elif weather_config.weather_type == 'rain':
            augmented_pc = self._augment_rain(pc_input, weather_config)
        elif weather_config.weather_type == 'snow':
            augmented_pc = self._augment_snow(pc_input, weather_config)
        elif weather_config.weather_type == 'fog':
            augmented_pc = self._augment_fog(pc_input, weather_config)
        else:
            augmented_pc = pc_input
        
        # Calculate statistics if augmentation was applied
        if augmented_pc.shape[1] > 4:  # LISA adds label column
            labels = augmented_pc[:, 4]
            augmentation_info['num_lost'] = np.sum(labels == 0)
            augmentation_info['num_scattered'] = np.sum(labels == 1) 
            augmentation_info['num_original'] = np.sum(labels == 2)
            
            # Remove label column for output
            augmented_pc = augmented_pc[:, :4]
        
        # Filter points within range
        augmented_pc = self._filter_point_cloud_range(augmented_pc)
        
        return augmented_pc, augmentation_info
    
    def _augment_rain(self, points: np.ndarray, config: WeatherConfig) -> np.ndarray:
        """Apply rain augmentation using LISA"""
        lisa_rain = self.lisa_instances['rain']
        
        # Scale rain rate based on intensity if not specified
        rain_rate = config.rain_rate if config.rain_rate is not None else config.intensity * 20.0
        
        # Apply LISA augmentation
        augmented_pc = lisa_rain.augment_mc(points)
        
        return augmented_pc
        
    def _augment_snow(self, points: np.ndarray, config: WeatherConfig) -> np.ndarray:
        """Apply snow augmentation using LISA"""
        lisa_snow = self.lisa_instances['snow']
        
        # Scale snow rate based on intensity
        snow_rate = config.rain_rate if config.rain_rate is not None else config.intensity * 15.0
        
        # Apply LISA augmentation
        augmented_pc = lisa_snow.augment_mc(points)
        
        return augmented_pc
    
    def _augment_fog(self, points: np.ndarray, config: WeatherConfig) -> np.ndarray:
        """Apply fog augmentation using LISA"""
        # Select fog type
        fog_type = config.fog_type if config.fog_type else 'moderate_advection_fog'
        
        if fog_type == 'chu_hogg_fog':
            lisa_fog = self.lisa_instances['fog_chu_hogg']
        elif fog_type == 'strong_advection_fog':
            lisa_fog = self.lisa_instances['fog_strong_advection']
        else:  # moderate_advection_fog
            lisa_fog = self.lisa_instances['fog_moderate_advection']
        
        # Apply LISA fog augmentation (uses average effects method)
        augmented_pc = lisa_fog.augment_avg(points)
        
        # Scale effect based on intensity
        if config.intensity != 1.0 and augmented_pc.shape[0] == points.shape[0]:
            # Interpolate between original and fully augmented only if same shape
            alpha = config.intensity
            augmented_pc = (1 - alpha) * points + alpha * augmented_pc
        else:
            # If shapes don't match, apply intensity by random sampling
            if config.intensity < 1.0 and augmented_pc.shape[0] < points.shape[0]:
                # Randomly keep some original points for reduced intensity
                n_keep = int((1 - config.intensity) * (points.shape[0] - augmented_pc.shape[0]))
                if n_keep > 0:
                    remaining_indices = np.random.choice(points.shape[0], n_keep, replace=False)
                    additional_points = points[remaining_indices]
                    augmented_pc = np.vstack([augmented_pc, additional_points])
        
        return augmented_pc
    
    def _filter_point_cloud_range(self, points: np.ndarray) -> np.ndarray:
        """Filter points within specified range"""
        if len(self.point_cloud_range) != 6:
            return points
            
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        
        return points[mask]
    
    def estimate_effective_range(self, weather_config: WeatherConfig) -> float:
        """
        Estimate effective detection range after weather augmentation
        For consistency with image augmentation visibility estimation
        """
        if weather_config.weather_type == 'clear':
            return 200.0
        
        # Use similar logic as image augmentor
        if weather_config.weather_type == 'rain':
            rain_rate = weather_config.rain_rate if weather_config.rain_rate else weather_config.intensity * 20
            return max(50, 200 - rain_rate * 8)
        elif weather_config.weather_type == 'snow':
            snow_rate = weather_config.rain_rate if weather_config.rain_rate else weather_config.intensity * 15
            return max(30, 200 - snow_rate * 12)
        elif weather_config.weather_type == 'fog':
            if weather_config.visibility:
                return weather_config.visibility
            else:
                return max(20, 200 - weather_config.intensity * 150)
        
        return 200.0
    
    def get_augmentation_statistics(self, points_original: np.ndarray, 
                                  points_augmented: np.ndarray,
                                  augmentation_info: Dict) -> Dict[str, Any]:
        """Get detailed statistics about the augmentation process"""
        stats = {
            'original_points': len(points_original),
            'augmented_points': len(points_augmented),
            'reduction_ratio': len(points_augmented) / len(points_original),
            'weather_type': augmentation_info['weather_type'],
            'intensity': augmentation_info['intensity']
        }
        
        if 'num_lost' in augmentation_info:
            stats.update({
                'lost_points': augmentation_info['num_lost'],
                'scattered_points': augmentation_info['num_scattered'],
                'original_detected': augmentation_info['num_original']
            })
        
        return stats