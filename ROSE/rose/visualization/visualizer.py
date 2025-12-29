"""
Main visualizer for ROSE framework
Combines detection and augmentation visualization capabilities
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from .detection_visualizer import DetectionVisualizer
from .augmentation_visualizer import AugmentationVisualizer


class Visualizer:
    """
    Main ROSE visualization coordinator
    Provides unified interface for all visualization needs
    """
    
    def __init__(self, 
                 output_dir: Optional[str] = None,
                 show_plots: bool = False,
                 save_plots: bool = True):
        
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.save_plots = save_plots
        
        # Initialize sub-visualizers
        self.detection_visualizer = DetectionVisualizer()
        default_save_dir = output_dir or "visualizations"
        self.augmentation_visualizer = AugmentationVisualizer(save_dir=default_save_dir)
        
        # Set matplotlib backend for non-interactive use
        if not show_plots:
            plt.switch_backend('Agg')
    
    def visualize_augmented_batch(self, 
                                batch_data: Dict[str, Any],
                                output_dir: Optional[str] = None,
                                epoch: Optional[int] = None,
                                max_samples: int = 4) -> None:
        """
        Visualize a batch of augmented data
        
        Args:
            batch_data: Batch data from dataloader
            output_dir: Output directory for visualizations
            epoch: Current epoch (for filename)
            max_samples: Maximum number of samples to visualize
        """
        output_dir = output_dir or self.output_dir or "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract samples from batch
        if 'inputs' in batch_data:
            images = batch_data['inputs'].get('img', [])
            points = batch_data['inputs'].get('points', [])
        else:
            # Handle different batch formats
            images = batch_data.get('img', [])
            points = batch_data.get('points', [])
        
        # Limit number of samples
        num_samples = min(len(images), max_samples) if isinstance(images, list) else min(images.size(0), max_samples)
        
        for i in range(num_samples):
            # Extract sample data
            if isinstance(images, list):
                sample_img = images[i]
                sample_points = points[i] if i < len(points) else None
            else:
                sample_img = images[i]
                sample_points = points[i] if points is not None else None
            
            # Get augmentation info if available
            aug_info = None
            if 'aug_info' in batch_data:
                aug_info = batch_data['aug_info'][i] if isinstance(batch_data['aug_info'], list) else batch_data['aug_info']
            
            # Visualize sample
            sample_output_dir = os.path.join(output_dir, f'sample_{i}')
            self._visualize_single_sample(
                sample_img, sample_points, aug_info, 
                sample_output_dir, epoch, i
            )
    
    def _visualize_single_sample(self,
                               image: Union[np.ndarray, 'torch.Tensor'],
                               points: Union[np.ndarray, 'torch.Tensor'],
                               aug_info: Optional[Dict],
                               output_dir: str,
                               epoch: Optional[int],
                               sample_idx: int) -> None:
        """Visualize a single sample with augmentation effects"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert tensors to numpy if needed
        if hasattr(image, 'numpy'):
            image = image.numpy()
        if hasattr(points, 'numpy'):
            points = points.numpy()
        
        # Visualize augmented image
        if image is not None:
            img_path = os.path.join(output_dir, f'augmented_image_epoch_{epoch}_sample_{sample_idx}.png')
            self.augmentation_visualizer.visualize_augmented_image(
                image, aug_info, img_path
            )
        
        # Visualize augmented point cloud
        if points is not None:
            pc_path = os.path.join(output_dir, f'augmented_pointcloud_epoch_{epoch}_sample_{sample_idx}.png')
            self.augmentation_visualizer.visualize_augmented_pointcloud(
                points, aug_info, pc_path
            )
        
        # Create combined visualization
        if image is not None and points is not None:
            combined_path = os.path.join(output_dir, f'combined_epoch_{epoch}_sample_{sample_idx}.png')
            self.augmentation_visualizer.create_combined_visualization(
                image, points, aug_info, combined_path
            )
    
    def visualize_detection_results(self,
                                  images: List[np.ndarray],
                                  point_clouds: List[np.ndarray], 
                                  predictions: List[Dict],
                                  ground_truths: Optional[List[Dict]] = None,
                                  output_dir: Optional[str] = None,
                                  class_names: Optional[List[str]] = None) -> None:
        """
        Visualize 3D detection results
        
        Args:
            images: List of input images
            point_clouds: List of point clouds
            predictions: List of model predictions
            ground_truths: Optional ground truth annotations
            output_dir: Output directory
            class_names: Class names for visualization
        """
        output_dir = output_dir or self.output_dir or "detection_results"
        os.makedirs(output_dir, exist_ok=True)
        
        class_names = class_names or ['Pedestrian', 'Cyclist', 'Car']
        
        for i, (img, pc, pred) in enumerate(zip(images, point_clouds, predictions)):
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            
            # Visualize on image
            img_with_boxes = self.detection_visualizer.draw_3d_boxes_on_image(
                img, pred, gt, class_names
            )
            img_path = os.path.join(output_dir, f'detection_image_{i}.png')
            cv2.imwrite(img_path, img_with_boxes)
            
            # Visualize point cloud with boxes
            pc_path = os.path.join(output_dir, f'detection_pointcloud_{i}.png')
            self.detection_visualizer.visualize_3d_boxes_on_pointcloud(
                pc, pred, gt, pc_path, class_names
            )
    
    def plot_training_progress(self,
                             performance_history: List[float],
                             output_path: Optional[str] = None,
                             title: str = "Training Progress",
                             metric_name: str = "mAP") -> None:
        """
        Plot training progress over epochs
        
        Args:
            performance_history: List of performance values
            output_path: Output file path
            title: Plot title
            metric_name: Name of the metric being plotted
        """
        plt.figure(figsize=(12, 6))
        
        epochs = list(range(1, len(performance_history) + 1))
        plt.plot(epochs, performance_history, 'b-', linewidth=2, marker='o', markersize=4)
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add best performance annotation
        if performance_history:
            best_value = max(performance_history)
            best_epoch = performance_history.index(best_value) + 1
            plt.annotate(f'Best: {best_value:.4f}', 
                        xy=(best_epoch, best_value),
                        xytext=(best_epoch + len(epochs)*0.1, best_value),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red')
        
        plt.tight_layout()
        
        if output_path and self.save_plots:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_augmentation_evolution(self,
                                  weather_stats: Dict[str, List[float]],
                                  output_path: Optional[str] = None,
                                  title: str = "Augmentation Evolution") -> None:
        """
        Plot evolution of augmentation intensities over epochs
        
        Args:
            weather_stats: Dictionary mapping weather types to intensity histories
            output_path: Output file path
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (weather_type, intensities) in enumerate(weather_stats.items()):
            if weather_type == 'clear':
                continue  # Skip clear weather (always 0 intensity)
                
            epochs = list(range(1, len(intensities) + 1))
            color = colors[i % len(colors)]
            
            plt.plot(epochs, intensities, color=color, linewidth=2, 
                    marker='o', markersize=4, label=weather_type.capitalize())
        
        plt.xlabel('Epoch')
        plt.ylabel('Weather Intensity')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        
        if output_path and self.save_plots:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def create_training_summary_report(self,
                                     training_history: Dict[str, Any],
                                     output_dir: str) -> None:
        """
        Create comprehensive training summary report with visualizations
        
        Args:
            training_history: Complete training history
            output_dir: Output directory for report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for visualization
        epoch_results = training_history.get('epoch_results', [])
        aug_configs = training_history.get('augmentation_configs', [])
        
        if not epoch_results:
            print("No training results found for visualization")
            return
        
        # Performance metrics over time
        map_history = [r.get('mAP', 0) for r in epoch_results]
        car_history = [r.get('Car_3D', 0) for r in epoch_results]
        ped_history = [r.get('Pedestrian_3D', 0) for r in epoch_results]
        cyc_history = [r.get('Cyclist_3D', 0) for r in epoch_results]
        
        # Plot overall performance
        self._plot_multi_metric_progress(
            {
                'Overall mAP': map_history,
                'Car 3D': car_history,
                'Pedestrian 3D': ped_history,
                'Cyclist 3D': cyc_history
            },
            os.path.join(output_dir, 'performance_summary.png'),
            "ROSE Training Performance Summary"
        )
        
        # Plot augmentation evolution if available
        if aug_configs:
            weather_evolution = self._extract_weather_evolution(aug_configs)
            self.plot_augmentation_evolution(
                weather_evolution,
                os.path.join(output_dir, 'augmentation_summary.png'),
                "Weather Augmentation Evolution Summary"
            )
        
        # Create performance statistics visualization
        self._create_performance_stats_plot(
            epoch_results,
            os.path.join(output_dir, 'performance_statistics.png')
        )
        
        print(f"Training summary report created in: {output_dir}")
    
    def _plot_multi_metric_progress(self,
                                  metrics: Dict[str, List[float]],
                                  output_path: str,
                                  title: str) -> None:
        """Plot multiple metrics on the same graph"""
        plt.figure(figsize=(14, 8))
        
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            if not values:
                continue
                
            epochs = list(range(1, len(values) + 1))
            color = colors[i % len(colors)]
            
            plt.plot(epochs, values, color=color, linewidth=2,
                    marker='o', markersize=3, label=metric_name)
        
        plt.xlabel('Epoch')
        plt.ylabel('Performance Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if not self.show_plots:
            plt.close()
    
    def _extract_weather_evolution(self, aug_configs: List[Dict]) -> Dict[str, List[float]]:
        """Extract weather intensity evolution from augmentation configs"""
        weather_evolution = {}
        
        for config in aug_configs:
            weather_configs = config.get('weather_configs', [])
            
            for weather_cfg in weather_configs:
                weather_type = weather_cfg.get('weather_type', 'unknown')
                intensity = weather_cfg.get('intensity', 0.0)
                
                if weather_type not in weather_evolution:
                    weather_evolution[weather_type] = []
                weather_evolution[weather_type].append(intensity)
        
        return weather_evolution
    
    def _create_performance_stats_plot(self, epoch_results: List[Dict], output_path: str) -> None:
        """Create performance statistics visualization"""
        if not epoch_results:
            return
        
        # Extract metrics for statistics
        metrics = ['mAP', 'Car_3D', 'Pedestrian_3D', 'Cyclist_3D']
        stats_data = {}
        
        for metric in metrics:
            values = [r.get(metric, 0) for r in epoch_results if r.get(metric, 0) > 0]
            if values:
                stats_data[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': np.max(values),
                    'min': np.min(values)
                }
        
        if not stats_data:
            return
        
        # Create bar plot for statistics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Mean performance
        metric_names = list(stats_data.keys())
        means = [stats_data[m]['mean'] for m in metric_names]
        stds = [stats_data[m]['std'] for m in metric_names]
        
        ax1.bar(metric_names, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
        ax1.set_title('Average Performance with Standard Deviation')
        ax1.set_ylabel('Performance Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Min/Max performance
        mins = [stats_data[m]['min'] for m in metric_names]
        maxs = [stats_data[m]['max'] for m in metric_names]
        
        x_pos = np.arange(len(metric_names))
        width = 0.35
        
        ax2.bar(x_pos - width/2, mins, width, label='Min', alpha=0.7, color='lightcoral')
        ax2.bar(x_pos + width/2, maxs, width, label='Max', alpha=0.7, color='lightgreen')
        
        ax2.set_title('Min/Max Performance')
        ax2.set_ylabel('Performance Score')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metric_names)
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if not self.show_plots:
            plt.close()