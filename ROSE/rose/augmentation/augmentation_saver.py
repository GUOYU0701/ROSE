"""
Augmentation Data Saver for ROSE Training
Saves augmented images and point clouds during training
"""

import os
import json
import numpy as np
import cv2
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
import mmcv


class AugmentationDataSaver:
    """Save augmented data and strategies during training"""
    
    def __init__(self, 
                 work_dir: str,
                 save_interval: int = 100,
                 max_samples_per_epoch: int = 50):
        """
        Initialize the augmentation data saver
        
        Args:
            work_dir: Working directory for saving data
            save_interval: Save data every N iterations
            max_samples_per_epoch: Maximum samples to save per epoch
        """
        self.work_dir = work_dir
        self.save_interval = save_interval
        self.max_samples_per_epoch = max_samples_per_epoch
        
        # Create save directories
        self.augmented_data_dir = os.path.join(work_dir, 'augmented_data')
        self.strategy_dir = os.path.join(work_dir, 'augmentation_strategies')
        
        for dir_path in [self.augmented_data_dir, self.strategy_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Initialize counters
        self.current_epoch = 0
        self.saved_samples_count = 0
        self.iteration_count = 0
        
        # Strategy tracking
        self.current_strategy = None
        self.strategy_history = []
        
    def set_current_epoch(self, epoch: int):
        """Set current training epoch"""
        self.current_epoch = epoch
        self.saved_samples_count = 0  # Reset counter for new epoch
        
        # Create epoch-specific directories
        epoch_img_dir = os.path.join(self.augmented_data_dir, f'epoch_{epoch}', 'images')
        epoch_pts_dir = os.path.join(self.augmented_data_dir, f'epoch_{epoch}', 'point_clouds')
        
        for dir_path in [epoch_img_dir, epoch_pts_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
    def set_augmentation_strategy(self, strategy: Dict[str, Any]):
        """Set current augmentation strategy"""
        self.current_strategy = strategy.copy()
        
    def should_save_sample(self) -> bool:
        """Determine if current sample should be saved"""
        self.iteration_count += 1
        
        # Check interval and max samples constraints
        return (self.iteration_count % self.save_interval == 0 and 
                self.saved_samples_count < self.max_samples_per_epoch)
    
    def save_augmented_sample(self, 
                            sample_id: str,
                            original_img: np.ndarray,
                            augmented_img: np.ndarray,
                            original_points: np.ndarray,
                            augmented_points: np.ndarray,
                            weather_type: str,
                            intensity: float,
                            metadata: Optional[Dict] = None):
        """
        Save a single augmented sample
        
        Args:
            sample_id: Unique identifier for the sample
            original_img: Original image array (H, W, 3)
            augmented_img: Augmented image array (H, W, 3)
            original_points: Original point cloud (N, 4)
            augmented_points: Augmented point cloud (N, 4)
            weather_type: Type of weather augmentation applied
            intensity: Augmentation intensity
            metadata: Additional metadata
        """
        if not self.should_save_sample():
            return
            
        epoch_dir = os.path.join(self.augmented_data_dir, f'epoch_{self.current_epoch}')
        img_dir = os.path.join(epoch_dir, 'images')
        pts_dir = os.path.join(epoch_dir, 'point_clouds')
        
        # Save images
        original_img_path = os.path.join(img_dir, f'{sample_id}_original.jpg')
        augmented_img_path = os.path.join(img_dir, f'{sample_id}_augmented_{weather_type}_{intensity:.2f}.jpg')
        
        # Ensure images are in correct format for saving
        if original_img.dtype != np.uint8:
            original_img = (original_img * 255).astype(np.uint8) if original_img.max() <= 1.0 else original_img.astype(np.uint8)
        if augmented_img.dtype != np.uint8:
            augmented_img = (augmented_img * 255).astype(np.uint8) if augmented_img.max() <= 1.0 else augmented_img.astype(np.uint8)
            
        cv2.imwrite(original_img_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(augmented_img_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
        
        # Save point clouds
        original_pts_path = os.path.join(pts_dir, f'{sample_id}_original.bin')
        augmented_pts_path = os.path.join(pts_dir, f'{sample_id}_augmented_{weather_type}_{intensity:.2f}.bin')
        
        original_points.astype(np.float32).tofile(original_pts_path)
        augmented_points.astype(np.float32).tofile(augmented_pts_path)
        
        # Save sample metadata
        sample_metadata = {
            'sample_id': sample_id,
            'epoch': self.current_epoch,
            'iteration': self.iteration_count,
            'weather_type': weather_type,
            'intensity': intensity,
            'original_img_path': original_img_path,
            'augmented_img_path': augmented_img_path,
            'original_pts_path': original_pts_path,
            'augmented_pts_path': augmented_pts_path,
            'augmentation_strategy': self.current_strategy,
            'custom_metadata': metadata or {}
        }
        
        metadata_path = os.path.join(epoch_dir, f'{sample_id}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f, indent=2)
            
        self.saved_samples_count += 1
        print(f"Saved augmented sample {sample_id} for epoch {self.current_epoch} "
              f"({self.saved_samples_count}/{self.max_samples_per_epoch})")
    
    def save_epoch_strategy(self, 
                           epoch: int, 
                           strategy: Dict[str, Any],
                           performance_metrics: Optional[Dict] = None):
        """
        Save augmentation strategy for an epoch
        
        Args:
            epoch: Training epoch number
            strategy: Augmentation strategy configuration
            performance_metrics: Performance metrics from previous epoch
        """
        import time
        strategy_data = {
            'epoch': epoch,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': strategy,
            'performance_metrics': performance_metrics or {},
            'iteration_count': self.iteration_count,
            'saved_samples': self.saved_samples_count
        }
        
        # Save individual epoch strategy
        epoch_strategy_path = os.path.join(self.strategy_dir, f'strategy_epoch_{epoch}.json')
        with open(epoch_strategy_path, 'w') as f:
            json.dump(strategy_data, f, indent=2)
            
        # Update strategy history
        self.strategy_history.append(strategy_data)
        
        # Save complete strategy history
        history_path = os.path.join(self.strategy_dir, 'strategy_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.strategy_history, f, indent=2)
            
        print(f"Saved augmentation strategy for epoch {epoch}")
    
    def create_comparison_visualization(self, epoch: int):
        """Create comparison visualizations for saved samples"""
        epoch_dir = os.path.join(self.augmented_data_dir, f'epoch_{epoch}')
        img_dir = os.path.join(epoch_dir, 'images')
        vis_dir = os.path.join(epoch_dir, 'visualizations')
        
        os.makedirs(vis_dir, exist_ok=True)
        
        # Find all sample pairs
        sample_ids = set()
        for filename in os.listdir(img_dir):
            if filename.endswith('_original.jpg'):
                sample_id = filename.replace('_original.jpg', '')
                sample_ids.add(sample_id)
        
        # Create comparison images
        for sample_id in list(sample_ids)[:10]:  # Limit to first 10 samples
            original_path = os.path.join(img_dir, f'{sample_id}_original.jpg')
            
            if os.path.exists(original_path):
                original_img = cv2.imread(original_path)
                
                # Find corresponding augmented images
                augmented_files = [f for f in os.listdir(img_dir) 
                                 if f.startswith(f'{sample_id}_augmented_') and f.endswith('.jpg')]
                
                for aug_file in augmented_files:
                    aug_path = os.path.join(img_dir, aug_file)
                    augmented_img = cv2.imread(aug_path)
                    
                    if augmented_img is not None:
                        # Create side-by-side comparison
                        comparison = np.hstack([original_img, augmented_img])
                        
                        # Add text labels
                        cv2.putText(comparison, 'Original', (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(comparison, 'Augmented', (original_img.shape[1] + 10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # Save comparison
                        comparison_filename = aug_file.replace('_augmented_', '_comparison_')
                        comparison_path = os.path.join(vis_dir, comparison_filename)
                        cv2.imwrite(comparison_path, comparison)
                        
        print(f"Created comparison visualizations for epoch {epoch}")
    
    def get_strategy_summary(self) -> Dict:
        """Get summary of all augmentation strategies used"""
        return {
            'total_epochs': len(self.strategy_history),
            'total_iterations': self.iteration_count,
            'total_saved_samples': sum(s.get('saved_samples', 0) for s in self.strategy_history),
            'strategy_evolution': self.strategy_history,
            'save_locations': {
                'augmented_data': self.augmented_data_dir,
                'strategies': self.strategy_dir
            }
        }