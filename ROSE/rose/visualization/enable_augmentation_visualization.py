"""
启用数据增强可视化的配置和示例脚本
"""

import os
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any

from rose.augmentation.config import AugmentationConfig, WeatherConfig
from rose.augmentation.weather_augmentor import WeatherAugmentor
from rose.visualization.augmentation_visualizer import AugmentationVisualizer


def create_visualization_enabled_config(work_dir: str = 'work_dirs/visualization_test') -> AugmentationConfig:
    """创建启用可视化的增强配置"""
    
    # 创建天气配置
    weather_configs = [
        WeatherConfig(
            weather_type='clear',
            intensity=0.0,
            rain_rate=0.0,
            fog_type='no_fog',
            visibility=1000,
            brightness_factor=1.0,
            contrast_factor=1.0,
            noise_level=0.0,
            blur_kernel_size=0
        ),
        WeatherConfig(
            weather_type='rain_light',
            intensity=0.3,
            rain_rate=5.0,
            fog_type='no_fog',
            visibility=500,
            brightness_factor=0.8,
            contrast_factor=0.9,
            noise_level=0.02,
            blur_kernel_size=1
        ),
        WeatherConfig(
            weather_type='rain_heavy',
            intensity=0.7,
            rain_rate=15.0,
            fog_type='no_fog', 
            visibility=200,
            brightness_factor=0.6,
            contrast_factor=0.7,
            noise_level=0.05,
            blur_kernel_size=2
        ),
        WeatherConfig(
            weather_type='fog_light',
            intensity=0.4,
            rain_rate=0.0,
            fog_type='moderate_advection_fog',
            visibility=100,
            brightness_factor=0.7,
            contrast_factor=0.6,
            noise_level=0.01,
            blur_kernel_size=3
        ),
        WeatherConfig(
            weather_type='fog_heavy',
            intensity=0.8,
            rain_rate=0.0,
            fog_type='strong_advection_fog',
            visibility=30,
            brightness_factor=0.5,
            contrast_factor=0.4,
            noise_level=0.03,
            blur_kernel_size=5
        )
    ]
    
    # 创建增强配置
    config = AugmentationConfig(
        weather_configs=weather_configs,
        weather_probabilities=[0.2, 0.25, 0.15, 0.25, 0.15],  # 均匀分布，更多样化
        enable_visualization=True,  # 启用可视化
        visualization_dir=os.path.join(work_dir, 'augmentation_visualizations'),
        save_frequency=5,  # 每5个样本保存一次可视化
        adaptation_enabled=True,
        performance_threshold=0.65,
        epoch=0
    )
    
    return config


def setup_augmentation_with_visualization(work_dir: str = 'work_dirs/visualization_test') -> WeatherAugmentor:
    """设置带可视化的增强器"""
    
    # 创建工作目录
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    
    # 创建可视化目录
    viz_dir = work_path / 'augmentation_visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建配置
    config = create_visualization_enabled_config(work_dir)
    
    # 创建启用可视化的增强器
    augmentor = WeatherAugmentor(
        config=config,
        enable_visualization=True,
        visualization_dir=str(viz_dir)
    )
    
    # 保存配置
    config_path = work_path / 'augmentation_config_with_viz.yaml'
    config.save_yaml(str(config_path))
    
    print(f"✅ 增强器已设置，可视化启用")
    print(f"   工作目录: {work_dir}")
    print(f"   可视化目录: {viz_dir}")
    print(f"   配置文件: {config_path}")
    
    return augmentor


def test_augmentation_visualization(sample_image_path: str = None, 
                                  sample_points_path: str = None,
                                  work_dir: str = 'work_dirs/visualization_test'):
    """测试增强可视化功能"""
    
    # 设置增强器
    augmentor = setup_augmentation_with_visualization(work_dir)
    
    # 创建测试数据（如果没有提供真实数据）
    if sample_image_path is None or not os.path.exists(sample_image_path):
        print("创建模拟测试数据...")
        # 创建模拟图像 (640x480x3)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 创建模拟点云 (1000点x4维)
        test_points = np.random.randn(1000, 4).astype(np.float32)
        test_points[:, :3] *= 20  # 缩放xyz坐标到合理范围
        test_points[:, 3] = np.random.rand(1000)  # 强度值0-1
        
        print("✅ 模拟数据已创建")
    else:
        # 加载真实数据
        import cv2
        test_image = cv2.imread(sample_image_path)
        test_points = np.load(sample_points_path) if sample_points_path else np.random.randn(1000, 4).astype(np.float32)
        print("✅ 真实数据已加载")
    
    # 测试不同天气条件下的增强
    test_weather_types = ['clear', 'rain_light', 'rain_heavy', 'fog_light', 'fog_heavy']
    
    print(f"开始测试 {len(test_weather_types)} 种天气条件...")
    
    for i, weather_type in enumerate(test_weather_types):
        print(f"测试天气类型: {weather_type} ({i+1}/{len(test_weather_types)})")
        
        try:
            # 执行增强
            augmented_image, augmented_points, augmentation_info = augmentor.augment_sample(
                image=test_image,
                points=test_points,
                calibration_info=None,
                force_weather=weather_type
            )
            
            print(f"  ✅ {weather_type} 增强成功")
            print(f"     强度: {augmentation_info['intensity']:.2f}")
            print(f"     点云有效范围: {augmentation_info['pc_effective_range']:.1f}m")
            
        except Exception as e:
            print(f"  ❌ {weather_type} 增强失败: {e}")
    
    # 生成统计报告
    stats = augmentor.get_augmentation_statistics()
    print(f"\n📊 增强统计:")
    print(f"   总样本数: {stats['total_samples']}")
    print(f"   可视化样本数: {stats.get('visualized_samples', 0)}")
    
    if 'weather_percentages' in stats:
        print(f"   天气分布:")
        for weather, percentage in stats['weather_percentages'].items():
            print(f"     {weather}: {percentage:.1f}%")
    
    # 保存可视化总结
    if augmentor.visualizer:
        augmentor.visualizer.save_summary_report()
        print(f"✅ 可视化总结报告已保存")
    
    print(f"\n🎉 测试完成！请查看可视化结果: {work_dir}/augmentation_visualizations/")


def create_training_config_with_visualization(base_config_path: str, 
                                            output_config_path: str,
                                            work_dir: str = 'work_dirs/enhanced_train'):
    """创建启用可视化的训练配置"""
    
    # 读取基础配置
    if base_config_path.endswith('.py'):
        # Python配置文件
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", base_config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # 修改配置以启用可视化
        if hasattr(config_module, 'augmentation_config'):
            config_module.augmentation_config['enable_visualization'] = True
            config_module.augmentation_config['visualization_dir'] = f'{work_dir}/augmentation_visualizations'
        
        if hasattr(config_module, 'custom_hooks'):
            # 添加可视化钩子
            viz_hook = {
                'type': 'ROSETrainingHook',
                'work_dir': work_dir,
                'visualize_augmentation': True,
                'save_augmentation_plan': True,
                'visualization_interval': 100
            }
            config_module.custom_hooks.append(viz_hook)
        
        # 保存修改后的配置
        with open(output_config_path, 'w') as f:
            f.write("# Enhanced ROSE config with visualization enabled\n")
            f.write(f"# Generated from: {base_config_path}\n\n")
            
            # 写入修改后的配置
            for attr_name in dir(config_module):
                if not attr_name.startswith('_'):
                    attr_value = getattr(config_module, attr_name)
                    f.write(f"{attr_name} = {repr(attr_value)}\n")
    
    print(f"✅ 可视化训练配置已创建: {output_config_path}")


if __name__ == "__main__":
    # 测试增强可视化
    test_augmentation_visualization()