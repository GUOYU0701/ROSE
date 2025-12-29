"""
Configuration for weather augmentation
"""
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class WeatherConfig:
    """Configuration for a specific weather type"""
    weather_type: str  # 'rain', 'snow', 'fog', 'clear'
    intensity: float   # 0.0 (clear) to 1.0 (heavy)
    
    # Weather-specific parameters
    rain_rate: Optional[float] = None      # mm/hr for rain/snow
    fog_type: Optional[str] = None         # 'chu_hogg_fog', 'strong_advection_fog', 'moderate_advection_fog'
    visibility: Optional[float] = None     # meters for fog
    
    # Image augmentation parameters
    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    noise_level: float = 0.0
    blur_kernel_size: int = 0


@dataclass  
class AugmentationConfig:
    """Main configuration for ROSE data augmentation"""
    
    # Training strategy
    epoch: int = 0
    total_epochs: int = 80
    
    # Weather mix strategy
    weather_configs: List[WeatherConfig] = None
    weather_probabilities: List[float] = None
    
    # Adaptive parameters based on validation performance
    adaptation_enabled: bool = True
    performance_threshold: float = 0.65  # mAP threshold for adaptation
    intensity_adjustment_step: float = 0.1
    
    # File paths
    data_root: str = "data/DAIR-V2X"
    lisa_path: str = "LISA-main"
    output_dir: str = "augmented_data"
    
    def __post_init__(self):
        if self.weather_configs is None:
            self.weather_configs = [
                WeatherConfig('clear', 0.0),
                WeatherConfig('rain', 0.3, rain_rate=5.0, 
                            brightness_factor=0.8, noise_level=0.02),
                WeatherConfig('snow', 0.4, rain_rate=3.0,
                            brightness_factor=0.9, contrast_factor=1.1),
                WeatherConfig('fog', 0.5, fog_type='moderate_advection_fog',
                            contrast_factor=0.7, blur_kernel_size=3)
            ]
            
        if self.weather_probabilities is None:
            # Start with balanced probabilities
            n_weathers = len(self.weather_configs)
            self.weather_probabilities = [1.0/n_weathers] * n_weathers

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'AugmentationConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert weather configs
        if 'weather_configs' in config_dict:
            config_dict['weather_configs'] = [
                WeatherConfig(**cfg) for cfg in config_dict['weather_configs']
            ]
        
        return cls(**config_dict)
    
    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        config_dict = asdict(self)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def update_weather_probabilities(self, performance_results: Dict[str, float]) -> None:
        """Update weather probabilities based on performance"""
        if not self.adaptation_enabled:
            return
            
        # Simple adaptation strategy: increase probability for challenging conditions
        # if performance is above threshold
        overall_map = performance_results.get('mAP', 0.0)
        
        if overall_map > self.performance_threshold:
            # Increase challenging weather probabilities
            for i, config in enumerate(self.weather_configs):
                if config.weather_type != 'clear':
                    if config.intensity < 1.0:
                        config.intensity = min(1.0, 
                                             config.intensity + self.intensity_adjustment_step)
        else:
            # Decrease challenging weather probabilities
            for i, config in enumerate(self.weather_configs):
                if config.weather_type != 'clear':
                    if config.intensity > 0.1:
                        config.intensity = max(0.1,
                                             config.intensity - self.intensity_adjustment_step)


def create_default_config(epoch: int = 0, total_epochs: int = 80) -> AugmentationConfig:
    """Create default augmentation configuration"""
    return AugmentationConfig(epoch=epoch, total_epochs=total_epochs)