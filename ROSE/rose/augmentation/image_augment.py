"""
Image augmentation for weather simulation
Physical consistency with point cloud augmentation
"""
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import torch
from PIL import Image, ImageEnhance, ImageFilter
from .config import WeatherConfig


class ImageAugmentor:
    """Image augmentation for weather effects"""
    
    def __init__(self, image_size: Tuple[int, int] = (1280, 384)):
        self.image_size = image_size
        
    def augment_image(self, image: np.ndarray, weather_config: WeatherConfig,
                     calibration_info: Optional[Dict] = None) -> np.ndarray:
        """
        Apply weather augmentation to image
        
        Args:
            image: Input image (H, W, C) in BGR format
            weather_config: Weather configuration
            calibration_info: Camera calibration for physical consistency
        
        Returns:
            Augmented image
        """
        # Convert to PIL for easier manipulation
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            
        # Apply weather-specific augmentation
        if weather_config.weather_type == 'rain':
            pil_image = self._add_rain_effects(pil_image, weather_config)
        elif weather_config.weather_type == 'snow':
            pil_image = self._add_snow_effects(pil_image, weather_config)
        elif weather_config.weather_type == 'fog':
            pil_image = self._add_fog_effects(pil_image, weather_config)
        elif weather_config.weather_type == 'clear':
            pass  # No augmentation for clear weather
            
        # Apply general atmospheric effects
        pil_image = self._apply_atmospheric_effects(pil_image, weather_config)
        
        # Convert back to numpy array
        result = np.array(pil_image)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        return result.astype(np.float32) / 255.0 if image.dtype != np.uint8 else result
    
    def _add_rain_effects(self, image: Image.Image, config: WeatherConfig) -> Image.Image:
        """Add rain effects to image"""
        img_array = np.array(image)
        h, w, c = img_array.shape
        
        # Rain intensity based on rain rate
        if config.rain_rate is not None:
            # Scale rain intensity with rain rate (mm/hr)
            rain_density = min(1.0, config.rain_rate / 20.0)  # Normalize to [0,1]
        else:
            rain_density = config.intensity
            
        # Generate rain streaks
        num_streaks = int(rain_density * 1000 * (w * h) / (1280 * 384))
        
        if num_streaks > 0:
            # Create rain mask
            rain_mask = np.zeros((h, w), dtype=np.uint8)
            
            for _ in range(num_streaks):
                # Random starting point
                x = np.random.randint(0, w)
                y = np.random.randint(0, h // 2)  # Start from upper half
                
                # Rain streak parameters
                length = np.random.randint(10, 50)
                angle = np.random.normal(90, 10)  # Mostly vertical with some variation
                thickness = np.random.randint(1, 3)
                
                # Draw rain streak
                end_x = int(x + length * np.sin(np.radians(angle)))
                end_y = int(y + length * np.cos(np.radians(angle)))
                
                if 0 <= end_x < w and 0 <= end_y < h:
                    cv2.line(rain_mask, (x, y), (end_x, end_y), 255, thickness)
            
            # Apply rain to image
            rain_intensity = int(255 * rain_density * 0.5)  # Semi-transparent
            img_array = img_array.astype(np.float32)
            
            for c_idx in range(c):
                img_array[:, :, c_idx] = np.where(
                    rain_mask > 0,
                    np.clip(img_array[:, :, c_idx] + rain_intensity, 0, 255),
                    img_array[:, :, c_idx]
                )
            
            img_array = img_array.astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _add_snow_effects(self, image: Image.Image, config: WeatherConfig) -> Image.Image:
        """Add snow effects to image"""
        img_array = np.array(image)
        h, w, c = img_array.shape
        
        # Snow intensity 
        if config.rain_rate is not None:
            snow_density = min(1.0, config.rain_rate / 15.0)  # Snow typically lighter than rain
        else:
            snow_density = config.intensity
            
        # Generate snow particles
        num_flakes = int(snow_density * 2000 * (w * h) / (1280 * 384))
        
        if num_flakes > 0:
            snow_mask = np.zeros((h, w), dtype=np.uint8)
            
            for _ in range(num_flakes):
                # Random position
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                
                # Snowflake size (typically larger than rain)
                size = np.random.randint(2, 8)
                
                # Draw snowflake
                cv2.circle(snow_mask, (x, y), size, 255, -1)
            
            # Apply snow to image (brighter than rain)
            snow_intensity = int(255 * snow_density * 0.8)
            img_array = img_array.astype(np.float32)
            
            for c_idx in range(c):
                img_array[:, :, c_idx] = np.where(
                    snow_mask > 0,
                    np.clip(img_array[:, :, c_idx] + snow_intensity, 0, 255),
                    img_array[:, :, c_idx]
                )
            
            img_array = img_array.astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _add_fog_effects(self, image: Image.Image, config: WeatherConfig) -> Image.Image:
        """Add fog effects to image"""
        # Fog reduces contrast and adds uniform haze
        
        # Apply Gaussian blur for fog scattering effect
        if config.blur_kernel_size > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=config.blur_kernel_size))
        
        # Add uniform fog overlay
        fog_overlay = Image.new('RGB', image.size, (200, 200, 200))  # Light gray
        fog_alpha = config.intensity * 0.3  # Fog transparency
        
        # Blend with fog overlay
        image = Image.blend(image, fog_overlay, fog_alpha)
        
        return image
    
    def _apply_atmospheric_effects(self, image: Image.Image, config: WeatherConfig) -> Image.Image:
        """Apply general atmospheric effects"""
        
        # Brightness adjustment
        if config.brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(config.brightness_factor)
        
        # Contrast adjustment
        if config.contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(config.contrast_factor)
        
        # Add noise
        if config.noise_level > 0:
            img_array = np.array(image).astype(np.float32)
            noise = np.random.normal(0, config.noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
            image = Image.fromarray(img_array.astype(np.uint8))
        
        return image
    
    def estimate_visibility(self, weather_config: WeatherConfig) -> float:
        """
        Estimate visibility distance for consistency with point cloud augmentation
        
        Returns:
            Visibility distance in meters
        """
        if weather_config.weather_type == 'clear':
            return 200.0  # Maximum detection range
        elif weather_config.weather_type == 'rain':
            # Rain reduces visibility based on rain rate
            if weather_config.rain_rate is not None:
                # Empirical formula for rain visibility
                return max(50, 200 - weather_config.rain_rate * 8)
            else:
                return max(50, 200 - weather_config.intensity * 100)
        elif weather_config.weather_type == 'snow':
            # Snow typically reduces visibility more than rain
            if weather_config.rain_rate is not None:
                return max(30, 200 - weather_config.rain_rate * 12)
            else:
                return max(30, 200 - weather_config.intensity * 120)
        elif weather_config.weather_type == 'fog':
            # Fog can severely reduce visibility
            if weather_config.visibility is not None:
                return weather_config.visibility
            else:
                return max(20, 200 - weather_config.intensity * 150)
        
        return 200.0