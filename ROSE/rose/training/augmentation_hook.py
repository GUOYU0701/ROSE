"""
Augmentation Hook for ROSE Training
Custom training hook to save augmented data during training
"""

import os
import numpy as np
from typing import Dict, Any, Optional
from mmengine.hooks import Hook
from mmengine.runner import Runner
import mmcv


class AugmentationSaverHook(Hook):
    """Hook to save augmented data during training"""
    
    def __init__(self, 
                 saver,
                 save_original: bool = True,
                 priority: int = 50):
        """
        Initialize augmentation saver hook
        
        Args:
            saver: AugmentationDataSaver instance
            save_original: Whether to save original data alongside augmented
            priority: Hook priority
        """
        super().__init__()
        self.saver = saver
        self.save_original = save_original
        self.priority = priority
        
        # Track current batch info
        self.current_epoch = 0
        self.current_iter = 0
        
    def before_train_epoch(self, runner: Runner):
        """Called before each training epoch"""
        self.current_epoch = runner.epoch
        self.saver.set_current_epoch(self.current_epoch)
        
    def before_train_iter(self, 
                         runner: Runner, 
                         batch_idx: int, 
                         data_batch: Dict[str, Any]):
        """Called before each training iteration"""
        self.current_iter = runner.iter
        
        # Check if we should save this batch
        if not self.saver.should_save_sample():
            return
        
        # Extract data from batch
        try:
            self._save_batch_data(data_batch, batch_idx)
        except Exception as e:
            print(f"Warning: Failed to save augmented data for batch {batch_idx}: {e}")
    
    def _save_batch_data(self, data_batch: Dict[str, Any], batch_idx: int):
        """
        Save augmented data from current batch
        
        Args:
            data_batch: Current training batch
            batch_idx: Batch index
        """
        # Extract batch data
        inputs = data_batch.get('inputs', {})
        data_samples = data_batch.get('data_samples', [])
        
        if not data_samples:
            return
        
        # Process each sample in the batch
        for sample_idx, data_sample in enumerate(data_samples):
            try:
                self._save_single_sample(inputs, data_sample, batch_idx, sample_idx)
            except Exception as e:
                print(f"Warning: Failed to save sample {sample_idx} in batch {batch_idx}: {e}")
    
    def _save_single_sample(self, 
                           inputs: Dict[str, Any], 
                           data_sample, 
                           batch_idx: int, 
                           sample_idx: int):
        """
        Save a single augmented sample
        
        Args:
            inputs: Input data (images, points)
            data_sample: Data sample with metadata
            batch_idx: Batch index
            sample_idx: Sample index within batch
        """
        # Create unique sample ID
        sample_id = f"epoch_{self.current_epoch}_batch_{batch_idx}_sample_{sample_idx}"
        
        # Extract image data
        images = inputs.get('imgs', None)
        if images is not None:
            if isinstance(images, list):
                image = images[sample_idx] if sample_idx < len(images) else images[0]
            else:
                image = images[sample_idx] if len(images.shape) > 3 else images
                
            # Convert tensor to numpy if needed
            if hasattr(image, 'cpu'):
                image = image.cpu().numpy()
                
            # Ensure correct format (H, W, C)
            if len(image.shape) == 3 and image.shape[0] in [1, 3]:  # (C, H, W)
                image = np.transpose(image, (1, 2, 0))
                
            # Normalize to 0-255 range if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        else:
            # Create dummy image if not available
            image = np.zeros((384, 1280, 3), dtype=np.uint8)
        
        # Extract point cloud data
        points = inputs.get('points', None)
        if points is not None:
            if isinstance(points, list):
                point_cloud = points[sample_idx] if sample_idx < len(points) else points[0]
            else:
                point_cloud = points[sample_idx] if len(points.shape) > 2 else points
                
            # Convert tensor to numpy if needed
            if hasattr(point_cloud, 'cpu'):
                point_cloud = point_cloud.cpu().numpy()
                
            # Ensure correct format (N, 4)
            if len(point_cloud.shape) == 3:  # (B, N, 4)
                point_cloud = point_cloud[0] if sample_idx == 0 else point_cloud[sample_idx]
        else:
            # Create dummy point cloud if not available
            point_cloud = np.random.randn(1000, 4).astype(np.float32)
        
        # Get metadata
        metadata = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'batch_idx': batch_idx,
            'sample_idx': sample_idx,
        }
        
        # Add sample-specific metadata if available
        if hasattr(data_sample, 'img_path'):
            metadata['img_path'] = data_sample.img_path
        if hasattr(data_sample, 'lidar_path'):
            metadata['lidar_path'] = data_sample.lidar_path
        
        # Determine augmentation info (simplified - in real implementation this would come from augmentation pipeline)
        weather_type = self._infer_weather_type(image, point_cloud)
        intensity = self._estimate_augmentation_intensity(image, point_cloud)
        
        # For now, use original as both original and augmented
        # In full implementation, this would be properly tracked through the augmentation pipeline
        self.saver.save_augmented_sample(
            sample_id=sample_id,
            original_img=image,
            augmented_img=image,  # In real scenario, this would be the augmented version
            original_points=point_cloud,
            augmented_points=point_cloud,  # In real scenario, this would be the augmented version
            weather_type=weather_type,
            intensity=intensity,
            metadata=metadata
        )
    
    def _infer_weather_type(self, image: np.ndarray, points: np.ndarray) -> str:
        """
        Infer weather type from image/point characteristics
        
        This is a simplified implementation. In practice, this information
        should be tracked from the augmentation pipeline.
        """
        # Simple heuristics based on image properties
        mean_brightness = np.mean(image)
        
        if mean_brightness < 100:
            return 'fog'
        elif mean_brightness > 180:
            return 'snow'
        elif len(points) < 800:  # Reduced points might indicate rain/fog
            return 'rain'
        else:
            return 'clear'
    
    def _estimate_augmentation_intensity(self, image: np.ndarray, points: np.ndarray) -> float:
        """
        Estimate augmentation intensity from data characteristics
        
        This is a simplified implementation. In practice, this information
        should be tracked from the augmentation pipeline.
        """
        # Simple heuristics
        brightness_var = np.var(image)
        point_density = len(points) / 1000.0  # Normalize by typical point count
        
        # Combine metrics to estimate intensity
        estimated_intensity = max(0.0, min(1.0, (1.0 - point_density) * 0.5 + (1.0 - brightness_var / 10000.0) * 0.5))
        
        return estimated_intensity
    
    def after_train_epoch(self, runner: Runner):
        """Called after each training epoch"""
        # Create visualization for this epoch
        self.saver.create_comparison_visualization(self.current_epoch)
        
        # Save epoch strategy if available
        if hasattr(runner.model, 'augmentation_config'):
            augmentation_config = runner.model.augmentation_config
            self.saver.save_epoch_strategy(
                epoch=self.current_epoch,
                strategy=augmentation_config
            )
        
        print(f"Completed data saving for epoch {self.current_epoch}")
    
    def after_train(self, runner: Runner):
        """Called after training completion"""
        summary = self.saver.get_strategy_summary()
        
        print(f"\nAugmentation Data Saving Summary:")
        print(f"  Total epochs: {summary.get('total_epochs', 0)}")
        print(f"  Total samples saved: {summary.get('total_saved_samples', 0)}")
        print(f"  Save location: {summary.get('save_locations', {}).get('augmented_data', 'N/A')}")