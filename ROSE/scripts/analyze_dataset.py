#!/usr/bin/env python
"""
DAIR-V2X Dataset Analysis Script
Analyze dataset characteristics for optimizing weather augmentation parameters
"""

import os
import numpy as np
import cv2
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_images(data_root, num_samples=100):
    """Analyze image characteristics"""
    image_dir = Path(data_root) / "training" / "image_2"
    image_files = list(image_dir.glob("*.jpg"))[:num_samples]
    
    brightness_values = []
    contrast_values = []
    resolutions = []
    
    print(f"Analyzing {len(image_files)} images...")
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Brightness (mean pixel value)
            brightness = np.mean(gray)
            brightness_values.append(brightness)
            
            # Contrast (standard deviation)
            contrast = np.std(gray)
            contrast_values.append(contrast)
            
            # Resolution
            h, w = img.shape[:2]
            resolutions.append((w, h))
    
    return {
        'brightness': {
            'mean': np.mean(brightness_values),
            'std': np.std(brightness_values),
            'min': np.min(brightness_values),
            'max': np.max(brightness_values)
        },
        'contrast': {
            'mean': np.mean(contrast_values),
            'std': np.std(contrast_values),
            'min': np.min(contrast_values),
            'max': np.max(contrast_values)
        },
        'common_resolution': max(set(resolutions), key=resolutions.count)
    }

def analyze_point_clouds(data_root, num_samples=50):
    """Analyze point cloud characteristics"""
    pc_dir = Path(data_root) / "training" / "velodyne_reduced"
    pc_files = list(pc_dir.glob("*.bin"))[:num_samples]
    
    point_counts = []
    range_distributions = []
    intensity_distributions = []
    
    print(f"Analyzing {len(pc_files)} point clouds...")
    
    for pc_file in pc_files:
        # Load point cloud (x, y, z, intensity)
        points = np.fromfile(str(pc_file), dtype=np.float32).reshape(-1, 4)
        
        # Point count
        point_counts.append(len(points))
        
        # Range distribution
        ranges = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))
        range_distributions.extend(ranges)
        
        # Intensity distribution
        intensity_distributions.extend(points[:, 3])
    
    return {
        'point_count': {
            'mean': np.mean(point_counts),
            'std': np.std(point_counts),
            'min': np.min(point_counts),
            'max': np.max(point_counts)
        },
        'range': {
            'mean': np.mean(range_distributions),
            'std': np.std(range_distributions),
            'percentile_95': np.percentile(range_distributions, 95),
            'percentile_99': np.percentile(range_distributions, 99)
        },
        'intensity': {
            'mean': np.mean(intensity_distributions),
            'std': np.std(intensity_distributions),
            'min': np.min(intensity_distributions),
            'max': np.max(intensity_distributions)
        }
    }

def analyze_annotations(data_root):
    """Analyze annotation statistics"""
    try:
        # Load training annotations
        train_infos_path = Path(data_root) / "kitti_infos_train.pkl"
        with open(train_infos_path, 'rb') as f:
            train_infos = pickle.load(f)
        
        object_counts = defaultdict(int)
        object_sizes = defaultdict(list)
        
        for info in train_infos:
            if 'annos' in info:
                annos = info['annos']
                names = annos.get('name', [])
                dimensions = annos.get('dimensions', [])
                
                for name in names:
                    object_counts[name] += 1
                
                if len(dimensions) > 0:
                    for i, name in enumerate(names):
                        if i < len(dimensions):
                            # dimensions: [height, width, length]
                            size = np.prod(dimensions[i])  # Volume
                            object_sizes[name].append(size)
        
        # Calculate average sizes
        avg_sizes = {}
        for obj_type, sizes in object_sizes.items():
            avg_sizes[obj_type] = {
                'mean_volume': np.mean(sizes),
                'std_volume': np.std(sizes)
            }
        
        return {
            'object_counts': dict(object_counts),
            'object_sizes': avg_sizes,
            'total_samples': len(train_infos)
        }
        
    except Exception as e:
        print(f"Warning: Could not analyze annotations: {e}")
        return None

def recommend_augmentation_parameters(image_stats, pc_stats, anno_stats):
    """Recommend optimized augmentation parameters based on dataset analysis"""
    
    # Base brightness and contrast of the dataset
    base_brightness = image_stats['brightness']['mean']
    base_contrast = image_stats['contrast']['mean']
    
    # Point cloud characteristics
    avg_range = pc_stats['range']['percentile_95']
    avg_intensity = pc_stats['intensity']['mean']
    
    recommendations = {
        'clear': {
            'weather_type': 'clear',
            'intensity': 0.0,
            'brightness_factor': 1.0,
            'contrast_factor': 1.0,
            'noise_level': 0.0,
            'blur_kernel_size': 0
        },
        'light_rain': {
            'weather_type': 'rain',
            'intensity': 0.3,
            'rain_rate': 2.0,  # mm/hr - light rain
            'brightness_factor': 0.85,  # Slightly darker
            'contrast_factor': 0.9,     # Slightly less contrast
            'noise_level': 0.02,
            'blur_kernel_size': 1
        },
        'moderate_rain': {
            'weather_type': 'rain', 
            'intensity': 0.5,
            'rain_rate': 8.0,  # mm/hr - moderate rain
            'brightness_factor': 0.75,  # More significant darkening
            'contrast_factor': 0.8,     # Reduced contrast
            'noise_level': 0.03,
            'blur_kernel_size': 2
        },
        'heavy_rain': {
            'weather_type': 'rain',
            'intensity': 0.7,
            'rain_rate': 20.0,  # mm/hr - heavy rain
            'brightness_factor': 0.6,   # Significant darkening
            'contrast_factor': 0.7,     # Much reduced contrast
            'noise_level': 0.05,
            'blur_kernel_size': 3
        },
        'light_snow': {
            'weather_type': 'snow',
            'intensity': 0.3,
            'snow_rate': 3.0,   # mm/hr equivalent
            'brightness_factor': 1.1,   # Snow can increase brightness
            'contrast_factor': 0.85,    # Reduced contrast
            'noise_level': 0.03,
            'blur_kernel_size': 2
        },
        'fog': {
            'weather_type': 'fog',
            'intensity': 0.4,
            'visibility_range': avg_range * 0.5,  # Reduce visibility to 50%
            'brightness_factor': 0.9,
            'contrast_factor': 0.6,     # Significant contrast reduction
            'noise_level': 0.01,
            'blur_kernel_size': 2
        }
    }
    
    # Adjust parameters based on dataset characteristics
    if base_brightness < 100:  # Dark dataset
        for config in recommendations.values():
            config['brightness_factor'] *= 1.1  # Increase brightness factors
    
    if base_contrast < 30:  # Low contrast dataset
        for config in recommendations.values():
            config['contrast_factor'] = min(1.0, config['contrast_factor'] * 1.1)
    
    return recommendations

def main():
    data_root = "/home/guoyu/mmdetection3d-1.2.0/data/DAIR-V2X"
    
    print("DAIR-V2X Dataset Analysis")
    print("=" * 50)
    
    # Analyze images
    print("\n1. Analyzing Images...")
    image_stats = analyze_images(data_root)
    
    # Analyze point clouds
    print("\n2. Analyzing Point Clouds...")
    pc_stats = analyze_point_clouds(data_root)
    
    # Analyze annotations
    print("\n3. Analyzing Annotations...")
    anno_stats = analyze_annotations(data_root)
    
    # Print results
    print("\n" + "=" * 50)
    print("DATASET ANALYSIS RESULTS")
    print("=" * 50)
    
    print(f"\nImage Statistics:")
    print(f"  Average Brightness: {image_stats['brightness']['mean']:.1f} ± {image_stats['brightness']['std']:.1f}")
    print(f"  Average Contrast: {image_stats['contrast']['mean']:.1f} ± {image_stats['contrast']['std']:.1f}")
    print(f"  Common Resolution: {image_stats['common_resolution']}")
    
    print(f"\nPoint Cloud Statistics:")
    print(f"  Average Points per Cloud: {pc_stats['point_count']['mean']:.0f} ± {pc_stats['point_count']['std']:.0f}")
    print(f"  Average Range: {pc_stats['range']['mean']:.1f}m")
    print(f"  95th Percentile Range: {pc_stats['range']['percentile_95']:.1f}m")
    print(f"  Average Intensity: {pc_stats['intensity']['mean']:.2f}")
    
    if anno_stats:
        print(f"\nAnnotation Statistics:")
        print(f"  Total Training Samples: {anno_stats['total_samples']}")
        print(f"  Object Distribution:")
        for obj_type, count in anno_stats['object_counts'].items():
            print(f"    {obj_type}: {count}")
    
    # Generate recommendations
    print("\n" + "=" * 50)
    print("RECOMMENDED AUGMENTATION PARAMETERS")
    print("=" * 50)
    
    recommendations = recommend_augmentation_parameters(image_stats, pc_stats, anno_stats)
    
    for weather_name, config in recommendations.items():
        print(f"\n{weather_name.upper()}:")
        for param, value in config.items():
            print(f"  {param}: {value}")
    
    # Save recommendations to file
    output_path = "/home/guoyu/CC/ROSE-NEW/optimized_augmentation_config.py"
    with open(output_path, 'w') as f:
        f.write('"""\\nOptimized Weather Augmentation Configuration\\n')
        f.write(f'Generated based on DAIR-V2X dataset analysis\\n"""\\n\\n')
        f.write('optimized_augmentation_config = {\\n')
        f.write(f'    "epoch": 0,\\n')
        f.write(f'    "total_epochs": 5,\\n')
        f.write(f'    "weather_configs": [\\n')
        
        for weather_name, config in recommendations.items():
            f.write(f'        # {weather_name}\\n')
            f.write(f'        dict(\\n')
            for param, value in config.items():
                if isinstance(value, str):
                    f.write(f'            {param}=\\"{value}\\",\\n')
                else:
                    f.write(f'            {param}={value},\\n')
            f.write(f'        ),\\n')
        
        f.write(f'    ],\\n')
        f.write(f'    "weather_probabilities": [0.3, 0.2, 0.15, 0.1, 0.15, 0.1],  # Clear, light_rain, mod_rain, heavy_rain, snow, fog\\n')
        f.write(f'    "adaptation_enabled": True,\\n')
        f.write(f'    "performance_threshold": 0.75,\\n')
        f.write(f'    "intensity_adjustment_step": 0.03,\\n')
        f.write(f'    "data_root": "{data_root}",\\n')
        f.write(f'    "lisa_path": "/home/guoyu/CC/ROSE-NEW/LISA-main",\\n')
        f.write(f'    "output_dir": "optimized_augmented_data"\\n')
        f.write(f'}}\\n')
    
    print(f"\n✓ Saved optimized configuration to: {output_path}")
    
    return recommendations

if __name__ == "__main__":
    recommendations = main()