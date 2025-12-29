"""
Unified weather augmentor for synchronized image and point cloud augmentation
"""
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import yaml
import random

from .config import AugmentationConfig, WeatherConfig
from .image_augment import ImageAugmentor
from .point_cloud_augment import PointCloudAugmentor


class WeatherAugmentor:
    """
    Unified weather augmentation system for ROSE framework
    Ensures physical consistency between image and point cloud augmentation
    """
    
    def __init__(self, config: AugmentationConfig, 
                 enable_visualization: bool = False, 
                 visualization_dir: Optional[str] = None):
        self.config = config
        self.image_augmentor = ImageAugmentor()
        self.point_cloud_augmentor = PointCloudAugmentor()
        
        # Visualization setup
        self.enable_visualization = enable_visualization
        self.visualizer = None
        if self.enable_visualization and visualization_dir:
            try:
                from rose.visualization.augmentation_visualizer import AugmentationVisualizer
                self.visualizer = AugmentationVisualizer(save_dir=visualization_dir, enabled=True)
                print("✅ Weather augmentation visualization enabled")
            except ImportError as e:
                print(f"⚠️ Could not enable visualization: {e}")
                self.enable_visualization = False
        
        # Augmentation statistics
        self.augmentation_stats = {
            'total_samples': 0,
            'weather_distribution': {cfg.weather_type: 0 for cfg in config.weather_configs},
            'visualized_samples': 0
        }
    
    def augment_sample(self, image: np.ndarray, points: np.ndarray,
                      calibration_info: Optional[Dict] = None,
                      force_weather: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Augment a single sample with synchronized weather effects
        
        Args:
            image: Input image (H, W, C)
            points: Input point cloud (N, 4) - x, y, z, intensity
            calibration_info: Camera calibration information
            force_weather: Force specific weather type (for testing)
            
        Returns:
            Tuple of (augmented_image, augmented_points, augmentation_info)
        """
        # Select weather configuration
        if force_weather:
            weather_config = self._get_weather_config(force_weather)
        else:
            weather_config = self._sample_weather_config()
        
        # Ensure physical consistency between modalities
        weather_config = self._ensure_physical_consistency(weather_config, calibration_info)
        
        # Augment image
        augmented_image = self.image_augmentor.augment_image(
            image, weather_config, calibration_info
        )
        
        # Augment point cloud
        augmented_points, pc_info = self.point_cloud_augmentor.augment_point_cloud(
            points, weather_config, calibration_info
        )
        
        # Collect augmentation information
        augmentation_info = {
            'weather_type': weather_config.weather_type,
            'intensity': weather_config.intensity,
            'image_visibility': self.image_augmentor.estimate_visibility(weather_config),
            'pc_effective_range': self.point_cloud_augmentor.estimate_effective_range(weather_config),
            'point_cloud_stats': pc_info,
            'weather_config': weather_config
        }
        
        # Update statistics
        self._update_statistics(weather_config)
        
        # Save visualization if enabled
        if self.enable_visualization and self.visualizer is not None:
            # Save comparison every few samples to avoid too many files
            if self.augmentation_stats['total_samples'] % 10 == 0:  # Save every 10th sample
                sample_id = f"sample_{self.augmentation_stats['total_samples']}"
                try:
                    self.visualizer.save_augmented_comparison(
                        sample_id=sample_id,
                        original_img=image,
                        augmented_img=augmented_image,
                        original_points=points,
                        augmented_points=augmented_points,
                        weather_type=weather_config.weather_type,
                        intensity=weather_config.intensity,
                        metadata=augmentation_info
                    )
                    self.augmentation_stats['visualized_samples'] += 1
                except Exception as e:
                    print(f"⚠️ Visualization failed for sample {sample_id}: {e}")
        
        return augmented_image, augmented_points, augmentation_info
    
    def _sample_weather_config(self) -> WeatherConfig:
        """Sample weather configuration based on probabilities"""
        if len(self.config.weather_configs) == 1:
            return self.config.weather_configs[0]
        
        # Sample based on probabilities
        weather_idx = np.random.choice(
            len(self.config.weather_configs),
            p=self.config.weather_probabilities
        )
        
        return self.config.weather_configs[weather_idx]
    
    def _get_weather_config(self, weather_type: str) -> WeatherConfig:
        """Get specific weather configuration by type"""
        for config in self.config.weather_configs:
            if config.weather_type == weather_type:
                return config
        
        # Return clear weather as fallback
        return WeatherConfig('clear', 0.0)
    
    def _ensure_physical_consistency(self, weather_config: WeatherConfig,
                                   calibration_info: Optional[Dict]) -> WeatherConfig:
        """
        Ensure physical consistency between image and point cloud augmentation
        Adjust parameters based on sensor setup and environmental constraints
        """
        # Create a copy to avoid modifying original config
        consistent_config = WeatherConfig(
            weather_type=weather_config.weather_type,
            intensity=weather_config.intensity,
            rain_rate=weather_config.rain_rate,
            fog_type=weather_config.fog_type,
            visibility=weather_config.visibility,
            brightness_factor=weather_config.brightness_factor,
            contrast_factor=weather_config.contrast_factor,
            noise_level=weather_config.noise_level,
            blur_kernel_size=weather_config.blur_kernel_size
        )
        
        # Adjust parameters for physical consistency
        if weather_config.weather_type in ['rain', 'snow']:
            # Ensure rain/snow rate is consistent with intensity
            if consistent_config.rain_rate is None:
                base_rate = 20.0 if weather_config.weather_type == 'rain' else 15.0
                consistent_config.rain_rate = weather_config.intensity * base_rate
            
            # Adjust brightness based on precipitation intensity
            if consistent_config.brightness_factor == 1.0:
                # Reduce brightness with precipitation
                consistent_config.brightness_factor = max(0.6, 1.0 - weather_config.intensity * 0.4)
                
        elif weather_config.weather_type == 'fog':
            # Ensure visibility and blur are consistent
            if consistent_config.visibility is None:
                consistent_config.visibility = max(20, 200 - weather_config.intensity * 150)
            
            # Adjust blur based on fog intensity
            if consistent_config.blur_kernel_size == 0:
                consistent_config.blur_kernel_size = int(weather_config.intensity * 5)
            
            # Reduce contrast in fog
            if consistent_config.contrast_factor == 1.0:
                consistent_config.contrast_factor = max(0.5, 1.0 - weather_config.intensity * 0.5)
        
        return consistent_config
    
    def _update_statistics(self, weather_config: WeatherConfig):
        """Update augmentation statistics"""
        self.augmentation_stats['total_samples'] += 1
        self.augmentation_stats['weather_distribution'][weather_config.weather_type] += 1
    
    def update_config_from_performance(self, performance_results: Dict[str, float]):
        """Update augmentation configuration based on validation performance"""
        self.config.update_weather_probabilities(performance_results)
    
    def save_augmentation_plan(self, output_path: str, epoch: int):
        """Save current augmentation plan to YAML file"""
        plan_path = Path(output_path) / f'augmentation_plan_epoch_{epoch}.yaml'
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update epoch in config
        self.config.epoch = epoch
        
        # Save configuration
        self.config.save_yaml(plan_path)
        
        # Save statistics
        stats_path = Path(output_path) / f'augmentation_stats_epoch_{epoch}.yaml'
        with open(stats_path, 'w') as f:
            yaml.dump(self.augmentation_stats, f, default_flow_style=False)
    
    def load_augmentation_plan(self, plan_path: str):
        """Load augmentation plan from YAML file"""
        self.config = AugmentationConfig.from_yaml(plan_path)
        
        # Reinitialize augmentors if needed
        self.image_augmentor = ImageAugmentor()
        self.point_cloud_augmentor = PointCloudAugmentor()
    
    def get_augmentation_statistics(self) -> Dict[str, Any]:
        """Get current augmentation statistics"""
        stats = self.augmentation_stats.copy()
        
        if stats['total_samples'] > 0:
            # Add percentage distribution
            stats['weather_percentages'] = {
                weather: count / stats['total_samples'] * 100
                for weather, count in stats['weather_distribution'].items()
            }
        
        return stats
    
    def reset_statistics(self):
        """Reset augmentation statistics"""
        self.augmentation_stats = {
            'total_samples': 0,
            'weather_distribution': {cfg.weather_type: 0 for cfg in self.config.weather_configs}
        }
    
    def create_batch_augmentation_plan(self, batch_size: int, epoch: int) -> list:
        """
        Create augmentation plan for a batch
        Ensures diverse weather conditions within each batch
        """
        plan = []
        
        # Calculate how many samples of each weather type
        samples_per_weather = batch_size // len(self.config.weather_configs)
        remaining_samples = batch_size % len(self.config.weather_configs)
        
        # Create weather assignment
        weather_assignments = []
        for i, config in enumerate(self.config.weather_configs):
            count = samples_per_weather
            if i < remaining_samples:  # Distribute remaining samples
                count += 1
            weather_assignments.extend([config.weather_type] * count)
        
        # Shuffle to randomize order within batch
        random.shuffle(weather_assignments)
        
        return weather_assignments[:batch_size]