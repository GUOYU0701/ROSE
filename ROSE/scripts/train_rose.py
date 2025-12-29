#!/usr/bin/env python
"""
ROSE Training Script
Train 3D detection model with weather augmentation and SSL
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

# Add ROSE to Python path
rose_root = Path(__file__).parent
sys.path.insert(0, str(rose_root))

# Add MMDetection3D to Python path
mmdet3d_root = Path("/home/guoyu/mmdetection3d-1.2.0")
sys.path.insert(0, str(mmdet3d_root))

import torch
import mmcv
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.logging import print_log

from rose import ROSETrainer, AugmentationConfig

# Suppress warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train ROSE 3D detection model')
    
    # Basic training arguments
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--work-dir', help='Working directory to save logs and models')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--load-from', help='Load model weights from checkpoint')
    
    # Augmentation arguments
    parser.add_argument('--aug-config', help='Augmentation config file path')
    parser.add_argument('--aug-prob', type=float, default=0.8, 
                       help='Augmentation probability (default: 0.8)')
    parser.add_argument('--disable-aug', action='store_true', 
                       help='Disable weather augmentation')
    
    # SSL arguments
    parser.add_argument('--disable-ssl', action='store_true', 
                       help='Disable SSL training')
    parser.add_argument('--ssl-warmup', type=int, default=5,
                       help='SSL warmup epochs (default: 5)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=80, help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    
    # Evaluation arguments
    parser.add_argument('--val-interval', type=int, default=2, 
                       help='Validation interval')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='Model save interval')
    
    # Visualization arguments
    parser.add_argument('--vis-interval', type=int, default=10,
                       help='Visualization interval')
    parser.add_argument('--disable-vis', action='store_true',
                       help='Disable visualization')
    
    # Advanced options
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                       help='Enable deterministic training')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                       help='Override config options')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup training environment"""
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Set deterministic mode
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Setup CUDA
    if torch.cuda.is_available():
        print(f"CUDA available. Using {args.gpus} GPU(s)")
        if args.gpus > torch.cuda.device_count():
            print(f"Warning: Requested {args.gpus} GPUs but only {torch.cuda.device_count()} available")
            args.gpus = torch.cuda.device_count()
    else:
        print("CUDA not available. Training on CPU")
        args.gpus = 0


def update_config(cfg, args):
    """Update configuration with command line arguments"""
    
    # Update work directory
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not hasattr(cfg, 'work_dir'):
        cfg.work_dir = './work_dirs/rose_training'
    
    # Create work directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Update training configuration
    if args.epochs:
        cfg.train_cfg.max_epochs = args.epochs
        if hasattr(cfg, 'augmentation_config'):
            cfg.augmentation_config['total_epochs'] = args.epochs
    
    if args.batch_size:
        cfg.train_dataloader.batch_size = args.batch_size
    
    if args.lr:
        cfg.optim_wrapper.optimizer.lr = args.lr
    
    if args.val_interval:
        cfg.train_cfg.val_interval = args.val_interval
    
    # Update augmentation settings
    if args.disable_aug:
        if 'augmentation_config' in cfg.train_dataloader.dataset:
            cfg.train_dataloader.dataset.augmentation_prob = 0.0
        print("Weather augmentation disabled")
    else:
        if 'augmentation_config' in cfg.train_dataloader.dataset:
            cfg.train_dataloader.dataset.augmentation_prob = args.aug_prob
        print(f"Weather augmentation probability: {args.aug_prob}")
    
    # Update SSL settings
    if args.disable_ssl:
        if hasattr(cfg.model, 'enable_ssl'):
            cfg.model.enable_ssl = False
        print("SSL training disabled")
    else:
        if hasattr(cfg.model, 'ssl_config'):
            # Update SSL warmup epochs
            for hook in cfg.custom_hooks:
                if hook.get('type') == 'SSLSchedulerHook':
                    hook['ssl_warmup_epochs'] = args.ssl_warmup
        print(f"SSL training enabled with {args.ssl_warmup} warmup epochs")
    
    # Update visualization settings
    if args.disable_vis:
        # Remove visualization hooks
        cfg.custom_hooks = [h for h in cfg.custom_hooks 
                           if h.get('type') != 'ROSETrainingHook' or not h.get('visualize_augmentation')]
        print("Visualization disabled")
    else:
        # Update visualization interval
        for hook in cfg.custom_hooks:
            if hook.get('type') == 'ROSETrainingHook':
                hook['visualization_interval'] = args.vis_interval
        print(f"Visualization enabled with interval: {args.vis_interval}")
    
    # Update checkpoint settings
    for hook_cfg in cfg.default_hooks.values():
        if hasattr(hook_cfg, 'type') and hook_cfg.type == 'CheckpointHook':
            hook_cfg.interval = args.save_interval
    
    # Apply command line config options
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    
    # Update resume/load settings
    if args.resume:
        cfg.resume = True
        cfg.auto_resume = True
    
    if args.load_from:
        cfg.load_from = args.load_from
    
    return cfg


def main():
    """Main training function"""
    args = parse_args()
    
    print("="*60)
    print("ROSE: Roadside Oversight-guided Scenario Enhancement")
    print("3D Detection Training with Weather Augmentation & SSL")
    print("="*60)
    
    # Setup environment
    setup_environment(args)
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # Update configuration with arguments
    cfg = update_config(cfg, args)
    
    # Print configuration summary
    print(f"\nTraining Configuration:")
    print(f"  Work Directory: {cfg.work_dir}")
    print(f"  Total Epochs: {cfg.train_cfg.max_epochs}")
    print(f"  Batch Size: {cfg.train_dataloader.batch_size}")
    print(f"  Learning Rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"  GPUs: {args.gpus}")
    
    # Print augmentation summary
    aug_enabled = cfg.train_dataloader.dataset.get('augmentation_prob', 0) > 0
    print(f"  Weather Augmentation: {'Enabled' if aug_enabled else 'Disabled'}")
    if aug_enabled:
        print(f"    Augmentation Probability: {cfg.train_dataloader.dataset.get('augmentation_prob')}")
        
    ssl_enabled = cfg.model.get('enable_ssl', False)
    print(f"  SSL Training: {'Enabled' if ssl_enabled else 'Disabled'}")
    
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    try:
        # Initialize runner
        runner = Runner.from_cfg(cfg)
        
        # Start training
        runner.train()
        
        print("\n" + "="*60)
        print("Training Completed Successfully!")
        print("="*60)
        
        # Print final results
        if hasattr(runner, 'message_hub'):
            final_results = runner.message_hub.get_info('val')
            if final_results:
                print(f"Final Validation Results:")
                for metric, value in final_results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
        
        print(f"\nResults saved in: {cfg.work_dir}")
        
        # Generate final report if using ROSE trainer
        if aug_enabled or ssl_enabled:
            print("Generating training report...")
            try:
                from rose.visualization import Visualizer
                visualizer = Visualizer()
                
                # Create final visualization directory
                final_viz_dir = os.path.join(cfg.work_dir, 'final_report')
                os.makedirs(final_viz_dir, exist_ok=True)
                
                print(f"Final report saved in: {final_viz_dir}")
                
            except Exception as e:
                print(f"Warning: Could not generate final report: {e}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()