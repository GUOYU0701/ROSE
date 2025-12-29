#!/usr/bin/env python
"""
ROSE Installation Verification Script
Verify that all components are properly installed and working
"""
import sys
import os
from pathlib import Path
import traceback

# Add ROSE to Python path
rose_root = Path(__file__).parent
sys.path.insert(0, str(rose_root))

def test_basic_imports():
    """Test basic Python imports"""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        import torch
        import yaml
        print("✓ Basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Basic import failed: {e}")
        return False

def test_mmdet3d_imports():
    """Test MMDetection3D imports"""
    print("Testing MMDetection3D imports...")
    
    try:
        # Add MMDetection3D to path
        mmdet3d_path = "/home/guoyu/mmdetection3d-1.2.0"
        if mmdet3d_path not in sys.path:
            sys.path.insert(0, mmdet3d_path)
        
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmdet3d.datasets import KittiDataset
        from mmdet3d.models import DynamicMVXFasterRCNN
        print("✓ MMDetection3D imports successful")
        return True
    except ImportError as e:
        print(f"✗ MMDetection3D import failed: {e}")
        return False

def test_lisa_integration():
    """Test LISA framework integration"""
    print("Testing LISA integration...")
    
    try:
        lisa_path = rose_root / "LISA-main" / "pylisa"
        if str(lisa_path) not in sys.path:
            sys.path.insert(0, str(lisa_path))
        
        from lisa import Lisa
        
        # Test LISA initialization
        lisa_instance = Lisa(
            m=1.328,
            lam=905,
            rmax=200,
            rmin=1.5,
            atm_model='rain'
        )
        
        print("✓ LISA integration successful")
        return True
    except Exception as e:
        print(f"✗ LISA integration failed: {e}")
        print("  Note: This is expected if LISA C library is not compiled")
        print("  Run: cd LISA-main/pylisa && ./build.sh")
        return False

def test_rose_imports():
    """Test ROSE framework imports"""
    print("Testing ROSE framework imports...")
    
    try:
        from rose import WeatherAugmentor, AugmentationConfig
        from rose.ssl_training import SSLTrainer
        from rose.training import ROSETrainer
        from rose.visualization import Visualizer
        print("✓ ROSE framework imports successful")
        return True
    except ImportError as e:
        print(f"✗ ROSE import failed: {e}")
        return False

def test_augmentation_functionality():
    """Test augmentation functionality"""
    print("Testing augmentation functionality...")
    
    try:
        from rose.augmentation import WeatherAugmentor, AugmentationConfig
        
        # Create test config
        config = AugmentationConfig()
        augmentor = WeatherAugmentor(config)
        
        # Create dummy data
        import numpy as np
        dummy_image = np.random.randint(0, 255, (384, 1280, 3), dtype=np.uint8)
        dummy_points = np.random.randn(1000, 4).astype(np.float32)
        dummy_points[:, :3] *= 20  # Scale to reasonable range
        dummy_points[:, 3] = np.random.rand(1000)  # Intensity
        
        # Test augmentation (without LISA, will fall back to image-only)
        try:
            aug_img, aug_points, aug_info = augmentor.augment_sample(
                dummy_image, dummy_points, force_weather='rain'
            )
            print("✓ Augmentation functionality working")
            return True
        except Exception as e:
            print(f"✗ Augmentation test failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Augmentation setup failed: {e}")
        return False

def test_ssl_functionality():
    """Test SSL training components"""
    print("Testing SSL components...")
    
    try:
        from rose.ssl_training import SSLTrainer, ContrastiveLoss, ConsistencyLoss
        
        # Test SSL trainer initialization
        ssl_trainer = SSLTrainer()
        
        # Test loss components
        contrastive_loss = ContrastiveLoss()
        consistency_loss = ConsistencyLoss()
        
        print("✓ SSL components working")
        return True
    except Exception as e:
        print(f"✗ SSL test failed: {e}")
        return False

def test_visualization():
    """Test visualization components"""
    print("Testing visualization components...")
    
    try:
        from rose.visualization import Visualizer
        
        # Test visualizer initialization
        visualizer = Visualizer(show_plots=False, save_plots=False)
        
        print("✓ Visualization components working")
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    
    try:
        config_path = rose_root / "configs" / "rose_mvxnet_dair_v2x.py"
        
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False
        
        # Try to load config (basic syntax check)
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Basic validation
        if 'ROSEDetector' in config_content and 'ROSEDataset' in config_content:
            print("✓ Configuration file format valid")
            return True
        else:
            print("✗ Configuration file missing key components")
            return False
            
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def check_data_paths():
    """Check if data paths exist"""
    print("Checking data paths...")
    
    data_root = Path("/home/guoyu/mmdetection3d-1.2.0/data/DAIR-V2X")
    
    required_paths = [
        data_root / "training" / "image_2",
        data_root / "training" / "velodyne_reduced",
        data_root / "training" / "label_2",
        data_root / "training" / "calib"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not path.exists():
            missing_paths.append(str(path))
    
    if missing_paths:
        print("✗ Missing data paths:")
        for path in missing_paths:
            print(f"    {path}")
        return False
    else:
        print("✓ All required data paths exist")
        return True

def run_verification():
    """Run all verification tests"""
    print("="*60)
    print("ROSE Installation Verification")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("MMDetection3D", test_mmdet3d_imports),
        ("ROSE Framework", test_rose_imports),
        ("Augmentation", test_augmentation_functionality),
        ("SSL Training", test_ssl_functionality), 
        ("Visualization", test_visualization),
        ("Configuration", test_config_loading),
        ("Data Paths", check_data_paths),
        ("LISA Integration", test_lisa_integration)  # Last as it's optional
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! ROSE is ready to use.")
    elif passed >= total - 1:  # Allow LISA to fail
        print("\n⚠️  Almost ready! Only LISA integration failed (optional).")
        print("   To enable LISA: cd LISA-main/pylisa && ./build.sh")
    else:
        print(f"\n❌ {total - passed} tests failed. Please check installation.")
        print("\nNext steps:")
        print("1. Check Python environment and dependencies")
        print("2. Verify MMDetection3D installation")
        print("3. Check data paths")
        print("4. Compile LISA C library (optional)")
    
    return passed == total

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)