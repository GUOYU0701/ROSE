#!/usr/bin/env python
"""
ROSE Testing Script
Test trained model on DAIR-V2X dataset with visualization
"""
import argparse
import os
import sys
from pathlib import Path

# Add ROSE to Python path
rose_root = Path(__file__).parent
sys.path.insert(0, str(rose_root))

# Add MMDetection3D to Python path
mmdet3d_root = Path("/home/guoyu/mmdetection3d-1.2.0")
sys.path.insert(0, str(mmdet3d_root))

import torch
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmdet3d.registry import DATASETS

from rose.visualization import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Test ROSE model')
    
    # Basic arguments
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--work-dir', help='Working directory for test results')
    
    # Testing options
    parser.add_argument('--test-set', default='val', choices=['val', 'test'],
                       help='Test on validation or test set')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for testing')
    
    # Augmentation testing
    parser.add_argument('--test-augmentation', action='store_true',
                       help='Test on augmented data')
    parser.add_argument('--weather-type', choices=['rain', 'snow', 'fog', 'clear'],
                       help='Specific weather type for testing')
    parser.add_argument('--weather-intensity', type=float, default=0.5,
                       help='Weather intensity for augmented testing')
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization results')
    parser.add_argument('--vis-samples', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--vis-score-thr', type=float, default=0.3,
                       help='Score threshold for visualization')
    
    # Output options
    parser.add_argument('--save-results', action='store_true',
                       help='Save detection results')
    parser.add_argument('--result-format', default='pkl',
                       choices=['pkl', 'json', 'csv'],
                       help='Result file format')
    
    # Advanced options
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                       help='Override config options')
    
    return parser.parse_args()


def setup_test_config(cfg, args):
    """Setup configuration for testing"""
    
    # Set work directory
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = './test_results'
    
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Update test configuration
    cfg.test_dataloader.batch_size = args.batch_size
    
    # Select test dataset
    if args.test_set == 'test':
        cfg.test_dataloader.dataset.ann_file = 'kitti_infos_test.pkl'
    else:
        cfg.test_dataloader.dataset.ann_file = 'kitti_infos_val.pkl'
    
    # Setup augmented testing if requested
    if args.test_augmentation:
        # Switch to ROSE dataset for augmented testing
        cfg.test_dataloader.dataset.type = 'ROSEDataset'
        cfg.test_dataloader.dataset.augmentation_prob = 1.0  # Always augment
        
        # Create specific weather config for testing
        if args.weather_type:
            weather_config = {
                'weather_type': args.weather_type,
                'intensity': args.weather_intensity
            }
            
            if args.weather_type == 'rain':
                weather_config.update({
                    'rain_rate': args.weather_intensity * 20.0,
                    'brightness_factor': 0.8,
                    'noise_level': 0.02
                })
            elif args.weather_type == 'snow':
                weather_config.update({
                    'rain_rate': args.weather_intensity * 15.0,
                    'brightness_factor': 0.9,
                    'contrast_factor': 1.1,
                    'noise_level': 0.01
                })
            elif args.weather_type == 'fog':
                weather_config.update({
                    'fog_type': 'moderate_advection_fog',
                    'visibility': max(20, 200 - args.weather_intensity * 150),
                    'contrast_factor': 0.7,
                    'blur_kernel_size': int(args.weather_intensity * 5)
                })
            
            cfg.test_dataloader.dataset.augmentation_config = {
                'weather_configs': [weather_config],
                'weather_probabilities': [1.0]
            }
    
    # Apply command line config options
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    
    return cfg


def test_model(cfg, checkpoint_path, args):
    """Test the model and return results"""
    
    print("Initializing model and loading checkpoint...")
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(checkpoint_path)
    
    print("Running evaluation...")
    results = runner.test()
    
    return results


def visualize_results(cfg, checkpoint_path, args):
    """Generate visualizations of test results"""
    
    print("Generating visualizations...")
    
    # Create visualizer
    visualizer = Visualizer(
        output_dir=os.path.join(args.work_dir, 'visualizations'),
        save_plots=True,
        show_plots=False
    )
    
    # Load model
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(checkpoint_path)
    model = runner.model
    model.eval()
    
    # Get test dataset
    test_dataset = DATASETS.build(cfg.test_dataloader.dataset)
    
    # Visualize sample predictions
    vis_count = 0
    for i, data_sample in enumerate(test_dataset):
        if vis_count >= args.vis_samples:
            break
        
        # Prepare input data
        data = test_dataset.pipeline(test_dataset.get_data_info(i))
        
        # Run inference
        with torch.no_grad():
            results = model.test_step(data)
        
        # Extract data for visualization
        img = data['inputs']['img'].numpy()
        points = data['inputs']['points'].numpy()
        
        # Extract predictions
        pred_instances = results[0].pred_instances_3d
        predictions = {
            'bboxes_3d': pred_instances.bboxes_3d,
            'scores_3d': pred_instances.scores_3d,
            'labels_3d': pred_instances.labels_3d
        }
        
        # Extract ground truth if available
        ground_truth = None
        if hasattr(results[0], 'gt_instances_3d'):
            gt_instances = results[0].gt_instances_3d
            ground_truth = {
                'bboxes_3d': gt_instances.bboxes_3d,
                'labels_3d': gt_instances.labels_3d
            }
        
        # Generate visualization
        visualizer.visualize_detection_results(
            [img], [points], [predictions], [ground_truth] if ground_truth else None,
            output_dir=os.path.join(args.work_dir, 'visualizations', f'sample_{i}'),
            class_names=cfg.metainfo['classes']
        )
        
        vis_count += 1
        if i % 5 == 0:
            print(f"Visualized {vis_count}/{args.vis_samples} samples")
    
    print(f"Visualizations saved in: {os.path.join(args.work_dir, 'visualizations')}")


def save_test_results(results, args):
    """Save test results to file"""
    
    if not args.save_results:
        return
    
    import pickle
    import json
    import pandas as pd
    
    results_file = os.path.join(args.work_dir, f'test_results.{args.result_format}')
    
    if args.result_format == 'pkl':
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
    elif args.result_format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results[0].items():  # Assuming single dataset results
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    elif args.result_format == 'csv':
        # Convert results to DataFrame for CSV
        df_data = []
        for key, value in results[0].items():
            df_data.append({'Metric': key, 'Value': value})
        
        df = pd.DataFrame(df_data)
        df.to_csv(results_file, index=False)
    
    print(f"Results saved to: {results_file}")


def print_test_summary(results, args):
    """Print test results summary"""
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    if results and len(results) > 0:
        metrics = results[0]  # Assuming single dataset
        
        print(f"Dataset: {args.test_set}")
        if args.test_augmentation:
            print(f"Weather Augmentation: {args.weather_type or 'Mixed'}")
            if args.weather_type:
                print(f"Weather Intensity: {args.weather_intensity}")
        
        print(f"\nDetection Performance:")
        
        # Print main metrics
        main_metrics = ['KITTI/Overall_3D_moderate', 'KITTI/Car_3D_moderate', 
                       'KITTI/Pedestrian_3D_moderate', 'KITTI/Cyclist_3D_moderate']
        
        for metric in main_metrics:
            if metric in metrics:
                print(f"  {metric.split('/')[-1]}: {metrics[metric]:.4f}")
        
        # Print all other metrics
        print(f"\nAll Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    
    print("="*60)


def main():
    """Main testing function"""
    args = parse_args()
    
    print("="*60)
    print("ROSE Model Testing")
    print("="*60)
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # Setup test configuration
    cfg = setup_test_config(cfg, args)
    
    print(f"Testing Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test Set: {args.test_set}")
    print(f"  Work Directory: {cfg.work_dir}")
    print(f"  Batch Size: {cfg.test_dataloader.batch_size}")
    
    if args.test_augmentation:
        print(f"  Augmentation Testing: Enabled")
        print(f"  Weather Type: {args.weather_type or 'Mixed'}")
        if args.weather_type:
            print(f"  Weather Intensity: {args.weather_intensity}")
    
    if args.visualize:
        print(f"  Visualization: Enabled ({args.vis_samples} samples)")
    
    print("\n" + "="*60)
    
    try:
        # Run testing
        results = test_model(cfg, args.checkpoint, args)
        
        # Save results
        save_test_results(results, args)
        
        # Generate visualizations
        if args.visualize:
            visualize_results(cfg, args.checkpoint, args)
        
        # Print summary
        print_test_summary(results, args)
        
        print(f"\nTesting completed successfully!")
        print(f"Results saved in: {cfg.work_dir}")
        
    except Exception as e:
        print(f"\nTesting failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()