#!/usr/bin/env python3
"""
Demo script for ROSE augmentation visualization
Demonstrates weather augmentation and visualization functionality
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add ROSE to path
sys.path.insert(0, str(Path(__file__).parent))

from rose.augmentation.config import AugmentationConfig, WeatherConfig
from rose.augmentation.weather_augmentor import WeatherAugmentor
from rose.visualization.augmentation_visualizer import AugmentationVisualizer


def create_demo_data():
    """Create demo image and point cloud data"""
    # Create a demo image (640x480, RGB)
    demo_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some simple geometric shapes to make it more interesting
    cv2.rectangle(demo_image, (100, 100), (300, 200), (0, 255, 0), 2)
    cv2.circle(demo_image, (450, 300), 50, (255, 0, 0), -1)
    cv2.putText(demo_image, 'ROSE Demo', (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Create demo point cloud (1000 points, x,y,z,intensity format)
    num_points = 1000
    demo_points = np.random.randn(num_points, 4).astype(np.float32)
    demo_points[:, 0] *= 50  # x: -50 to 50m
    demo_points[:, 1] *= 25  # y: -25 to 25m  
    demo_points[:, 2] *= 3   # z: -3 to 3m
    demo_points[:, 3] = np.random.uniform(0, 255, num_points)  # intensity
    
    # Add some structured points (simulating a car)
    car_points = np.array([
        [10, 0, -1, 200], [12, 2, -1, 200], [12, -2, -1, 200], [8, 0, -1, 200],
        [10, 0, 0, 180], [12, 2, 0, 180], [12, -2, 0, 180], [8, 0, 0, 180],
    ], dtype=np.float32)
    
    demo_points = np.vstack([demo_points, car_points])
    
    return demo_image, demo_points


def setup_augmentation_config():
    """Setup augmentation configuration with different weather types"""
    
    # Rain configuration
    rain_config = WeatherConfig(
        weather_type='rain',
        intensity=5.0,  # mm/hr
        particle_size_params={'mean': 1.5, 'std': 0.8},
        atmospheric_params={'visibility': 500}
    )
    
    # Snow configuration  
    snow_config = WeatherConfig(
        weather_type='snow',
        intensity=3.0,
        particle_size_params={'mean': 3.0, 'std': 1.2},
        atmospheric_params={'visibility': 300}
    )
    
    # Fog configuration
    fog_config = WeatherConfig(
        weather_type='fog',
        intensity=2.0,
        particle_size_params={'mean': 10.0, 'std': 5.0},
        atmospheric_params={'visibility': 100}
    )
    
    # Create augmentation config
    aug_config = AugmentationConfig(
        weather_configs=[rain_config, snow_config, fog_config],
        weather_probabilities=[0.4, 0.3, 0.3]
    )
    
    return aug_config


def main():
    """Main demonstration function"""
    
    print("🌦️ ROSE Augmentation Visualization Demo")
    print("=" * 50)
    
    # Setup output directory
    output_dir = Path("demo_augmentation_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Create demo data
    print("🎯 Creating demo data...")
    demo_image, demo_points = create_demo_data()
    print(f"   - Image shape: {demo_image.shape}")
    print(f"   - Point cloud shape: {demo_points.shape}")
    
    # Setup augmentation configuration
    print("⚙️ Setting up weather augmentation...")
    aug_config = setup_augmentation_config()
    weather_types = [cfg.weather_type for cfg in aug_config.weather_configs]
    print(f"   - Weather types: {weather_types}")
    
    # Initialize weather augmentor with visualization
    augmentor = WeatherAugmentor(
        config=aug_config,
        enable_visualization=True,
        visualization_dir=str(output_dir)
    )
    
    print("🔄 Running augmentation demonstrations...")
    
    # Test each weather type
    for i, weather_type in enumerate(weather_types, 1):
        print(f"\n   {i}. Testing {weather_type} augmentation...")
        
        try:
            # Apply augmentation
            aug_image, aug_points, aug_info = augmentor.augment_sample(
                image=demo_image.copy(),
                points=demo_points.copy(),
                force_weather=weather_type
            )
            
            print(f"      ✅ {weather_type} augmentation completed")
            print(f"         - Weather intensity: {aug_info['intensity']:.2f}")
            print(f"         - Original points: {demo_points.shape[0]}")
            print(f"         - Augmented points: {aug_points.shape[0]}")
            print(f"         - Image visibility: {aug_info.get('image_visibility', 'N/A')}")
            print(f"         - PC effective range: {aug_info.get('pc_effective_range', 'N/A')}")
            
        except Exception as e:
            print(f"      ❌ {weather_type} augmentation failed: {e}")
    
    # Test random augmentation (using probability distribution)
    print(f"\n   4. Testing random weather selection...")
    try:
        aug_image, aug_points, aug_info = augmentor.augment_sample(
            image=demo_image.copy(),
            points=demo_points.copy()
        )
        print(f"      ✅ Random augmentation completed")
        print(f"         - Selected weather: {aug_info['weather_type']}")
        print(f"         - Weather intensity: {aug_info['intensity']:.2f}")
        
    except Exception as e:
        print(f"      ❌ Random augmentation failed: {e}")
    
    # Print augmentation statistics
    stats = augmentor.augmentation_stats
    print(f"\n📊 Augmentation Statistics:")
    print(f"   - Total samples processed: {stats['total_samples']}")
    print(f"   - Visualized samples: {stats['visualized_samples']}")
    print(f"   - Weather distribution: {stats['weather_distribution']}")
    
    # List generated visualization files
    viz_files = list(output_dir.rglob("*.png"))
    print(f"\n🖼️ Generated visualization files ({len(viz_files)} total):")
    for viz_file in sorted(viz_files):
        rel_path = viz_file.relative_to(output_dir)
        print(f"   - {rel_path}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📁 Check {output_dir.absolute()} for visualization results")
    
    # Simple test of direct visualizer
    print(f"\n🧪 Testing direct visualizer...")
    try:
        from rose.visualization.augmentation_visualizer import AugmentationVisualizer
        direct_visualizer = AugmentationVisualizer(
            save_dir=str(output_dir / "direct_test"),
            enabled=True
        )
        
        # Create a simple before/after comparison
        direct_visualizer.save_augmented_comparison(
            sample_id="direct_test",
            original_img=demo_image,
            augmented_img=aug_image,
            original_points=demo_points,
            augmented_points=aug_points,
            weather_type=aug_info['weather_type'],
            intensity=aug_info['intensity'],
            metadata=aug_info
        )
        print("   ✅ Direct visualizer test passed")
        
    except Exception as e:
        print(f"   ❌ Direct visualizer test failed: {e}")
        
    print("\n🎉 All demonstrations completed!")


if __name__ == "__main__":
    main()