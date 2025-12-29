"""
Main ROSE Trainer - Orchestrates the complete training pipeline
"""
import os
import yaml
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import torch
import numpy as np
from mmengine import Config
from mmengine.runner import Runner
from mmengine.registry import RUNNERS

from ..augmentation import AugmentationConfig, WeatherAugmentor, create_default_config
from ..visualization import Visualizer


class ROSETrainer:
    """
    Main ROSE Training Coordinator
    Manages the complete training pipeline including:
    - Adaptive data augmentation
    - SSL training 
    - Model training and validation
    - Results visualization
    """
    
    def __init__(self,
                 config_path: str,
                 work_dir: str,
                 augmentation_config: Optional[AugmentationConfig] = None,
                 resume_from: Optional[str] = None,
                 load_from: Optional[str] = None):
        
        self.config_path = config_path
        self.work_dir = work_dir
        self.resume_from = resume_from
        self.load_from = load_from
        
        # Load training configuration
        self.cfg = Config.fromfile(config_path)
        
        # Initialize augmentation configuration
        if augmentation_config is not None:
            self.aug_config = augmentation_config
        else:
            self.aug_config = create_default_config(
                total_epochs=self.cfg.get('train_cfg', {}).get('max_epochs', 80)
            )
        
        # Create work directory
        os.makedirs(work_dir, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = Visualizer()
        
        # Training history
        self.training_history = {
            'epoch_results': [],
            'augmentation_configs': [],
            'performance_metrics': []
        }
    
    def setup_training(self):
        """Setup training configuration and data"""
        # Update config with ROSE-specific settings
        self._update_config_for_rose()
        
        # Create augmentation plan directory
        aug_plan_dir = os.path.join(self.work_dir, 'augmentation_plans')
        os.makedirs(aug_plan_dir, exist_ok=True)
        
        # Save initial augmentation config
        initial_plan_path = os.path.join(aug_plan_dir, 'initial_augmentation_config.yaml')
        self.aug_config.save_yaml(initial_plan_path)
        
        print(f"Training setup completed. Work directory: {self.work_dir}")
        print(f"Initial augmentation config saved to: {initial_plan_path}")
    
    def train(self):
        """Execute the complete ROSE training pipeline"""
        print("Starting ROSE training pipeline...")
        
        # Setup training
        self.setup_training()
        
        total_epochs = self.cfg.train_cfg.max_epochs
        
        for epoch in range(total_epochs):
            print(f"\n{'='*60}")
            print(f"Starting Epoch {epoch + 1}/{total_epochs}")
            print(f"{'='*60}")
            
            # Update augmentation config for current epoch
            self._update_augmentation_for_epoch(epoch)
            
            # Run epoch training
            epoch_results = self._run_epoch(epoch)
            
            # Store results
            self.training_history['epoch_results'].append(epoch_results)
            self.training_history['augmentation_configs'].append(
                self.aug_config.__dict__.copy()
            )
            
            # Adapt augmentation config based on results
            if epoch > 0 and epoch % 5 == 0:  # Every 5 epochs
                self._adapt_augmentation_config(epoch_results)
            
            # Generate visualizations
            if epoch % 10 == 0:  # Every 10 epochs
                self._generate_epoch_visualizations(epoch, epoch_results)
            
            # Save checkpoint and progress
            self._save_training_progress(epoch, epoch_results)
            
            print(f"Epoch {epoch + 1} completed. mAP: {epoch_results.get('mAP', 0):.4f}")
        
        # Training completed
        self._finalize_training()
    
    def _update_config_for_rose(self):
        """Update MMDetection3D config for ROSE training"""
        # Update dataset to use ROSEDataset
        self.cfg.train_dataloader.dataset.type = 'ROSEDataset'
        self.cfg.train_dataloader.dataset.augmentation_config = self.aug_config.__dict__
        self.cfg.train_dataloader.dataset.augmentation_prob = 0.8
        self.cfg.train_dataloader.dataset.save_augmented_data = True
        
        # Update model to use ROSEDetector
        self.cfg.model.type = 'ROSEDetector'
        self.cfg.model.enable_ssl = True
        self.cfg.model.ssl_config = {
            'lambda_det': 1.0,
            'lambda_cm': 0.5,
            'lambda_cons': 0.3,
            'lambda_spatial': 0.2,
            'lambda_weather': 0.4,
            'ema_decay': 0.999
        }
        
        # Add ROSE-specific hooks
        if 'custom_hooks' not in self.cfg:
            self.cfg.custom_hooks = []
        
        self.cfg.custom_hooks.extend([
            dict(type='ROSETrainingHook', priority='NORMAL'),
            dict(type='WeatherAugmentationHook', priority='NORMAL'),
            dict(type='SSLSchedulerHook', priority='NORMAL')
        ])
        
        # Set work directory
        self.cfg.work_dir = self.work_dir
        
        # Set resume/load options
        if self.resume_from:
            self.cfg.resume = True
            self.cfg.load_from = self.resume_from
        elif self.load_from:
            self.cfg.load_from = self.load_from
    
    def _update_augmentation_for_epoch(self, epoch: int):
        """Update augmentation configuration for current epoch"""
        self.aug_config.epoch = epoch
        
        # Save current epoch's augmentation plan
        aug_plan_path = os.path.join(
            self.work_dir, 'augmentation_plans', f'augmentation_plan_epoch_{epoch}.yaml'
        )
        self.aug_config.save_yaml(aug_plan_path)
    
    def _run_epoch(self, epoch: int) -> Dict[str, Any]:
        """Run a single training epoch"""
        # Create runner for this epoch
        runner = Runner.from_cfg(self.cfg)
        
        # Set epoch-specific configuration
        if hasattr(runner.model, 'set_ssl_epoch'):
            runner.model.set_ssl_epoch(epoch)
        
        # Run training for one epoch
        runner.train()
        
        # Get validation results
        val_results = runner.test()
        
        # Extract metrics
        epoch_results = {
            'epoch': epoch,
            'timestamp': time.time(),
            'val_results': val_results
        }
        
        # Extract specific metrics
        if val_results and len(val_results) > 0:
            metrics = val_results[0]  # Assuming single dataset
            epoch_results.update({
                'mAP': metrics.get('KITTI/Overall_3D_moderate', 0.0),
                'Car_3D': metrics.get('KITTI/Car_3D_moderate', 0.0),
                'Pedestrian_3D': metrics.get('KITTI/Pedestrian_3D_moderate', 0.0),
                'Cyclist_3D': metrics.get('KITTI/Cyclist_3D_moderate', 0.0)
            })
        
        return epoch_results
    
    def _adapt_augmentation_config(self, epoch_results: Dict[str, Any]):
        """Adapt augmentation configuration based on performance"""
        current_map = epoch_results.get('mAP', 0.0)
        
        # Get performance history for comparison
        if len(self.training_history['epoch_results']) >= 2:
            prev_results = self.training_history['epoch_results'][-2]
            prev_map = prev_results.get('mAP', 0.0)
            
            performance_change = current_map - prev_map
            
            # Simple adaptation strategy
            if performance_change > 0.01:  # Good improvement
                # Slightly increase weather intensity
                for weather_config in self.aug_config.weather_configs:
                    if weather_config.weather_type != 'clear':
                        weather_config.intensity = min(1.0, weather_config.intensity + 0.05)
                        
            elif performance_change < -0.01:  # Performance drop
                # Decrease weather intensity
                for weather_config in self.aug_config.weather_configs:
                    if weather_config.weather_type != 'clear':
                        weather_config.intensity = max(0.1, weather_config.intensity - 0.05)
        
        # Update weather probabilities
        performance_results = {'mAP': current_map}
        self.aug_config.update_weather_probabilities(performance_results)
    
    def _generate_epoch_visualizations(self, epoch: int, epoch_results: Dict[str, Any]):
        """Generate visualizations for the epoch"""
        viz_dir = os.path.join(self.work_dir, 'visualizations', f'epoch_{epoch}')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot training progress
        if len(self.training_history['epoch_results']) > 1:
            map_history = [r.get('mAP', 0) for r in self.training_history['epoch_results']]
            self.visualizer.plot_training_progress(
                map_history,
                output_path=os.path.join(viz_dir, 'training_progress.png'),
                title=f'ROSE Training Progress - Epoch {epoch}'
            )
        
        # Plot augmentation statistics
        weather_stats = {}
        for aug_config in self.training_history['augmentation_configs']:
            for weather_cfg in aug_config.get('weather_configs', []):
                weather_type = weather_cfg['weather_type']
                if weather_type not in weather_stats:
                    weather_stats[weather_type] = []
                weather_stats[weather_type].append(weather_cfg['intensity'])
        
        if weather_stats:
            self.visualizer.plot_augmentation_evolution(
                weather_stats,
                output_path=os.path.join(viz_dir, 'augmentation_evolution.png'),
                title=f'Weather Augmentation Evolution - Epoch {epoch}'
            )
    
    def _save_training_progress(self, epoch: int, epoch_results: Dict[str, Any]):
        """Save training progress and checkpoints"""
        # Save training history
        history_path = os.path.join(self.work_dir, 'training_history.yaml')
        with open(history_path, 'w') as f:
            yaml.dump(self.training_history, f, default_flow_style=False)
        
        # Save current augmentation config
        current_aug_path = os.path.join(self.work_dir, 'current_augmentation_config.yaml')
        self.aug_config.save_yaml(current_aug_path)
        
        # Save epoch summary
        epoch_summary = {
            'epoch': epoch,
            'results': epoch_results,
            'augmentation_config': self.aug_config.__dict__
        }
        
        epoch_summary_path = os.path.join(self.work_dir, f'epoch_{epoch}_summary.yaml')
        with open(epoch_summary_path, 'w') as f:
            yaml.dump(epoch_summary, f, default_flow_style=False)
    
    def _finalize_training(self):
        """Finalize training and generate final reports"""
        print("\n" + "="*60)
        print("Training Completed - Generating Final Reports")
        print("="*60)
        
        # Generate final visualizations
        final_viz_dir = os.path.join(self.work_dir, 'final_visualizations')
        os.makedirs(final_viz_dir, exist_ok=True)
        
        # Final training progress plot
        if self.training_history['epoch_results']:
            map_history = [r.get('mAP', 0) for r in self.training_history['epoch_results']]
            self.visualizer.plot_training_progress(
                map_history,
                output_path=os.path.join(final_viz_dir, 'final_training_progress.png'),
                title='ROSE Training - Final Results'
            )
        
        # Generate training summary report
        self._generate_final_report()
        
        print(f"Training completed! Results saved in: {self.work_dir}")
        
        # Print best performance
        if self.training_history['epoch_results']:
            best_epoch = max(self.training_history['epoch_results'], 
                           key=lambda x: x.get('mAP', 0))
            print(f"Best Performance: Epoch {best_epoch['epoch']}, "
                  f"mAP = {best_epoch.get('mAP', 0):.4f}")
    
    def _generate_final_report(self):
        """Generate comprehensive final training report"""
        # Calculate training statistics
        results = self.training_history['epoch_results']
        if not results:
            return
        
        map_values = [r.get('mAP', 0) for r in results]
        car_values = [r.get('Car_3D', 0) for r in results]
        ped_values = [r.get('Pedestrian_3D', 0) for r in results]
        cyc_values = [r.get('Cyclist_3D', 0) for r in results]
        
        report = {
            'training_summary': {
                'total_epochs': len(results),
                'best_mAP': max(map_values) if map_values else 0,
                'final_mAP': map_values[-1] if map_values else 0,
                'average_mAP': np.mean(map_values) if map_values else 0,
                'best_Car_3D': max(car_values) if car_values else 0,
                'best_Pedestrian_3D': max(ped_values) if ped_values else 0,
                'best_Cyclist_3D': max(cyc_values) if cyc_values else 0
            },
            'augmentation_summary': {
                'total_configs_tested': len(self.training_history['augmentation_configs']),
                'final_config': self.aug_config.__dict__
            },
            'detailed_results': results
        }
        
        # Save final report
        report_path = os.path.join(self.work_dir, 'final_training_report.yaml')
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        print(f"Final training report saved to: {report_path}")
    
    def evaluate_on_test_set(self, test_config_path: Optional[str] = None):
        """Evaluate best model on test set"""
        print("Evaluating on test set...")
        
        # Load best checkpoint
        checkpoints_dir = os.path.join(self.work_dir, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            # Find best checkpoint based on validation results
            best_epoch = max(self.training_history['epoch_results'], 
                           key=lambda x: x.get('mAP', 0))
            best_checkpoint = os.path.join(checkpoints_dir, f"epoch_{best_epoch['epoch']}.pth")
            
            if os.path.exists(best_checkpoint):
                # Update config for testing
                test_cfg = self.cfg.copy()
                test_cfg.load_from = best_checkpoint
                
                # Run test
                runner = Runner.from_cfg(test_cfg)
                test_results = runner.test()
                
                # Save test results
                test_results_path = os.path.join(self.work_dir, 'test_results.yaml')
                with open(test_results_path, 'w') as f:
                    yaml.dump(test_results, f, default_flow_style=False)
                
                print(f"Test results saved to: {test_results_path}")
                return test_results
        
        print("No suitable checkpoint found for testing.")
        return None