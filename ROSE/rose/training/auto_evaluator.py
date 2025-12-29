"""
Auto Evaluator for ROSE Training
Handles automatic validation and testing after each training round
"""

import os
import json
import subprocess
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import mmcv
from mmengine import Config
from mmengine.runner import Runner


class ROSEAutoEvaluator:
    """Automatic evaluation system for ROSE training"""
    
    def __init__(self, 
                 config_file: str,
                 work_dir: str,
                 data_root: str):
        """
        Initialize the auto evaluator
        
        Args:
            config_file: Path to training configuration file
            work_dir: Working directory for training outputs
            data_root: Root directory of dataset
        """
        self.config_file = config_file
        self.work_dir = work_dir
        self.data_root = data_root
        
        # Create evaluation directories
        self.eval_dir = os.path.join(work_dir, 'evaluations')
        self.vis_dir = os.path.join(work_dir, 'test_visualizations')
        
        for dir_path in [self.eval_dir, self.vis_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Load base configuration
        self.cfg = Config.fromfile(config_file)
        
        # Evaluation history
        self.evaluation_history = []
        
    def run_validation(self, 
                      checkpoint_path: str, 
                      round_num: int) -> Dict[str, float]:
        """
        Run validation on trained model
        
        Args:
            checkpoint_path: Path to model checkpoint
            round_num: Training round number
            
        Returns:
            Dictionary of validation metrics
        """
        print(f"\nRunning validation for round {round_num}...")
        
        # Configure for validation
        cfg = self.cfg.copy()
        cfg.work_dir = os.path.join(self.eval_dir, f'round_{round_num}_validation')
        os.makedirs(cfg.work_dir, exist_ok=True)
        
        # Load checkpoint
        cfg.load_from = checkpoint_path
        
        # Disable training-specific components
        cfg.model.enable_ssl = False
        
        # Initialize runner for validation
        runner = Runner.from_cfg(cfg)
        
        try:
            # Run validation
            val_metrics = runner.val_loop.run()
            
            # Extract key metrics
            validation_results = {
                'round': round_num,
                'checkpoint': checkpoint_path,
                'metrics': val_metrics,
                'overall_3d_moderate': val_metrics.get('KITTI/Overall_3D_moderate', 0.0),
                'car_3d_moderate': val_metrics.get('KITTI/Car_3D_moderate', 0.0),
                'pedestrian_3d_moderate': val_metrics.get('KITTI/Pedestrian_3D_moderate', 0.0),
                'cyclist_3d_moderate': val_metrics.get('KITTI/Cyclist_3D_moderate', 0.0),
            }
            
            # Save validation results
            val_results_path = os.path.join(cfg.work_dir, 'validation_results.json')
            with open(val_results_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
                
            print(f"Validation completed. mAP: {validation_results['overall_3d_moderate']:.4f}")
            return validation_results
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return {
                'round': round_num,
                'checkpoint': checkpoint_path,
                'error': str(e),
                'overall_3d_moderate': 0.0
            }
    
    def run_test_with_visualization(self, 
                                  checkpoint_path: str, 
                                  round_num: int,
                                  num_samples: int = 20) -> Dict[str, Any]:
        """
        Run test with visualization enabled
        
        Args:
            checkpoint_path: Path to model checkpoint
            round_num: Training round number
            num_samples: Number of samples to visualize
            
        Returns:
            Dictionary containing test results and visualization info
        """
        print(f"\nRunning test with visualization for round {round_num}...")
        
        # Configure for testing
        cfg = self.cfg.copy()
        cfg.work_dir = os.path.join(self.eval_dir, f'round_{round_num}_test')
        os.makedirs(cfg.work_dir, exist_ok=True)
        
        # Test-specific visualization directory
        test_vis_dir = os.path.join(self.vis_dir, f'round_{round_num}')
        os.makedirs(test_vis_dir, exist_ok=True)
        
        # Enable detailed visualization
        cfg.default_hooks.visualization = dict(
            type='Det3DVisualizationHook',
            draw=True,
            interval=1,  # Visualize every sample
            score_thr=0.1,  # Lower threshold to see more detections
            show=False,
            wait_time=0.0,
            test_out_dir=test_vis_dir
        )
        
        # Load checkpoint
        cfg.load_from = checkpoint_path
        cfg.model.enable_ssl = False
        
        # Limit test samples for faster evaluation
        if hasattr(cfg, 'test_dataloader'):
            # Create a subset of test data
            cfg.test_dataloader.dataset.pipeline.append(dict(type='Collect3D', keys=['points', 'img']))
            
        try:
            # Initialize runner for testing
            runner = Runner.from_cfg(cfg)
            
            # Run testing
            test_metrics = runner.test_loop.run()
            
            # Process test results
            test_results = {
                'round': round_num,
                'checkpoint': checkpoint_path,
                'metrics': test_metrics,
                'visualization_dir': test_vis_dir,
                'overall_3d_moderate': test_metrics.get('KITTI/Overall_3D_moderate', 0.0),
                'car_3d_moderate': test_metrics.get('KITTI/Car_3D_moderate', 0.0),
                'pedestrian_3d_moderate': test_metrics.get('KITTI/Pedestrian_3D_moderate', 0.0),
                'cyclist_3d_moderate': test_metrics.get('KITTI/Cyclist_3D_moderate', 0.0),
            }
            
            # Save test results
            test_results_path = os.path.join(cfg.work_dir, 'test_results.json')
            with open(test_results_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            # Create performance summary visualization
            self._create_performance_visualization(test_results, round_num)
            
            print(f"Test completed. mAP: {test_results['overall_3d_moderate']:.4f}")
            print(f"Visualizations saved to: {test_vis_dir}")
            
            return test_results
            
        except Exception as e:
            print(f"Test with visualization failed: {e}")
            return {
                'round': round_num,
                'checkpoint': checkpoint_path,
                'error': str(e),
                'overall_3d_moderate': 0.0,
                'visualization_dir': test_vis_dir
            }
    
    def _create_performance_visualization(self, test_results: Dict, round_num: int):
        """Create performance visualization charts"""
        import matplotlib.pyplot as plt
        
        # Extract metrics for visualization
        metrics = {
            'Overall': test_results.get('overall_3d_moderate', 0.0),
            'Car': test_results.get('car_3d_moderate', 0.0),
            'Pedestrian': test_results.get('pedestrian_3d_moderate', 0.0),
            'Cyclist': test_results.get('cyclist_3d_moderate', 0.0)
        }
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics.keys(), metrics.values())
        
        # Customize chart
        ax.set_ylabel('mAP (3D Moderate)')
        ax.set_title(f'Detection Performance - Round {round_num}')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.annotate(f'{value:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        # Color bars based on performance
        colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in metrics.values()]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(self.vis_dir, f'round_{round_num}', 'performance_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance chart saved to: {chart_path}")
    
    def evaluate_round(self, 
                      checkpoint_path: str, 
                      round_num: int) -> Dict[str, Any]:
        """
        Complete evaluation for a training round
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            round_num: Training round number
            
        Returns:
            Complete evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Starting Evaluation for Round {round_num}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        # Run validation
        val_results = self.run_validation(checkpoint_path, round_num)
        
        # Run test with visualization
        test_results = self.run_test_with_visualization(checkpoint_path, round_num)
        
        # Combine results
        evaluation_results = {
            'round': round_num,
            'checkpoint': checkpoint_path,
            'validation': val_results,
            'test': test_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save complete evaluation results
        eval_results_path = os.path.join(self.eval_dir, f'round_{round_num}_complete_evaluation.json')
        with open(eval_results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Update evaluation history
        self.evaluation_history.append(evaluation_results)
        
        # Save evaluation history
        history_path = os.path.join(self.eval_dir, 'evaluation_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        
        print(f"\nRound {round_num} Evaluation Summary:")
        print(f"  Validation mAP: {val_results.get('overall_3d_moderate', 0.0):.4f}")
        print(f"  Test mAP: {test_results.get('overall_3d_moderate', 0.0):.4f}")
        print(f"  Results saved to: {eval_results_path}")
        
        return evaluation_results
    
    def get_performance_trend(self) -> Dict[str, List[float]]:
        """Get performance trend across all evaluated rounds"""
        if not self.evaluation_history:
            return {}
            
        trends = {
            'rounds': [],
            'validation_map': [],
            'test_map': [],
            'car_map': [],
            'pedestrian_map': [],
            'cyclist_map': []
        }
        
        for eval_result in self.evaluation_history:
            round_num = eval_result['round']
            val_metrics = eval_result['validation']
            test_metrics = eval_result['test']
            
            trends['rounds'].append(round_num)
            trends['validation_map'].append(val_metrics.get('overall_3d_moderate', 0.0))
            trends['test_map'].append(test_metrics.get('overall_3d_moderate', 0.0))
            trends['car_map'].append(test_metrics.get('car_3d_moderate', 0.0))
            trends['pedestrian_map'].append(test_metrics.get('pedestrian_3d_moderate', 0.0))
            trends['cyclist_map'].append(test_metrics.get('cyclist_3d_moderate', 0.0))
            
        return trends
    
    def create_trend_visualization(self):
        """Create performance trend visualization"""
        trends = self.get_performance_trend()
        
        if not trends or len(trends['rounds']) < 2:
            print("Not enough data for trend visualization")
            return
            
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall performance trend
        ax1.plot(trends['rounds'], trends['validation_map'], 'b-o', label='Validation mAP')
        ax1.plot(trends['rounds'], trends['test_map'], 'r-o', label='Test mAP')
        ax1.set_xlabel('Training Round')
        ax1.set_ylabel('Overall mAP (3D Moderate)')
        ax1.set_title('Overall Performance Trend')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Class-specific performance trend
        ax2.plot(trends['rounds'], trends['car_map'], 'g-o', label='Car')
        ax2.plot(trends['rounds'], trends['pedestrian_map'], 'b-o', label='Pedestrian')
        ax2.plot(trends['rounds'], trends['cyclist_map'], 'r-o', label='Cyclist')
        ax2.set_xlabel('Training Round')
        ax2.set_ylabel('Class mAP (3D Moderate)')
        ax2.set_title('Class-wise Performance Trend')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save trend chart
        trend_chart_path = os.path.join(self.eval_dir, 'performance_trend.png')
        plt.savefig(trend_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance trend chart saved to: {trend_chart_path}")
        
        return trend_chart_path