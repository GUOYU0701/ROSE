"""
ROSE (Roadside Oversight-guided Scenario Enhancement) Framework
路侧多模态感知的场景增强系统
"""

__version__ = "1.0.0"
__author__ = "ROSE Team"
__description__ = "Roadside Multi-modal Perception Enhancement Framework"

# 导入核心组件
from .core.rose_framework import ROSEFramework
from .augmentation.multimodal_augmentor import MultiModalAugmentor
from .augmentation.config import AugmentationConfig
from .ssl.ssl_trainer import ROSESSLTrainer
from .training.rose_dataset import ROSEDataset
from .training.rose_detector import ROSEDetector, ROSEModelBuilder
from .training.rose_trainer import ROSETrainer
from .training.training_hooks import ROSETrainingHook

__all__ = [
    'ROSEFramework',
    'MultiModalAugmentor', 
    'AugmentationConfig',
    'ROSESSLTrainer',
    'ROSEDataset',
    'ROSEDetector',
    'ROSEModelBuilder',
    'ROSETrainer',
    'ROSETrainingHook'
]