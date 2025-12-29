#!/usr/bin/env python
"""
Enhanced ROSE Training Example with Comprehensive Analytics
Demonstrates usage of improved detection, SSL, and data augmentation analytics
"""

import os
import sys
import warnings
from pathlib import Path
import numpy as np
import torch
from typing import Dict, Any

# Add ROSE to Python path
rose_root = Path(__file__).parent
sys.path.insert(0, str(rose_root))

# Suppress warnings
warnings.filterwarnings('ignore')

# Import ROSE components
from rose.training.training_analytics import ROSETrainingAnalytics, DetectionMetricsAnalyzer
from rose.ssl_training.ssl_trainer import SSLTrainer
from rose.visualization.detection_visualizer import DetectionVisualizer
from rose.augmentation.weather_augmentor import WeatherAugmentor
from rose.augmentation.config import AugmentationConfig, WeatherConfig


def create_enhanced_training_pipeline(work_dir: str, config_path: str):
    """
    Create enhanced training pipeline with comprehensive analytics
    
    Args:
        work_dir: Working directory for training
        config_path: Path to training configuration
    """
    
    print("🚀 Initializing Enhanced ROSE Training Pipeline")
    
    # 1. Initialize Training Analytics
    print("📊 Setting up training analytics...")
    analytics = ROSETrainingAnalytics(work_dir=work_dir, enabled=True)
    
    # 2. Initialize SSL Trainer with enhanced features
    print("🔗 Initializing SSL trainer with problematic class focus...")
    ssl_trainer = SSLTrainer(
        lambda_det=1.0,
        lambda_cm=0.6,      # Increased for better cross-modal alignment
        lambda_cons=0.4,    # Increased for small object consistency
        lambda_spatial=0.3, # Increased for spatial relationships
        lambda_weather=0.5, # Enhanced weather adaptation
        ema_decay=0.999,
        consistency_warmup_epochs=3,  # Earlier start for problematic classes
        enable_pseudo_labeling=True
    )
    
    # 3. Initialize Detection Visualizer with enhanced analysis
    print("🎯 Setting up enhanced detection visualization...")
    detector_visualizer = DetectionVisualizer()
    
    # 4. Setup Enhanced Weather Augmentation
    print("🌦️ Configuring weather augmentation for better pedestrian/cyclist detection...")
    augmentation_config = create_enhanced_augmentation_config()
    weather_augmentor = WeatherAugmentor(augmentation_config)
    
    # 5. Initialize Detection Metrics Analyzer
    print("📈 Setting up detection metrics analyzer...")
    metrics_analyzer = DetectionMetricsAnalyzer()
    
    return {
        'analytics': analytics,
        'ssl_trainer': ssl_trainer,
        'visualizer': detector_visualizer,
        'weather_augmentor': weather_augmentor,
        'metrics_analyzer': metrics_analyzer
    }


def create_enhanced_augmentation_config() -> AugmentationConfig:
    """Create augmentation config optimized for pedestrian/cyclist detection"""
    
    weather_configs = [
        # Enhanced rain configuration for pedestrians
        WeatherConfig(
            weather_type='rain',
            intensity=0.4,
            rain_rate=6.0,  # Increased for better augmentation
            enabled=True
        ),
        # Snow configuration for cyclists  
        WeatherConfig(
            weather_type='snow',
            intensity=0.3,
            rain_rate=4.0,
            enabled=True
        ),
        # Fog for challenging visibility conditions
        WeatherConfig(
            weather_type='fog',
            intensity=0.5,
            fog_type='moderate_advection_fog',
            enabled=True
        ),
        # Clear weather as control
        WeatherConfig(
            weather_type='clear',
            intensity=0.0,
            enabled=True
        )
    ]
    
    return AugmentationConfig(
        weather_configs=weather_configs,
        weather_probabilities=[0.3, 0.25, 0.25, 0.2],  # Higher probability for weather
        adaptation_enabled=True,
        performance_threshold=0.6,  # Lower threshold for more adaptation
        total_epochs=80
    )


def demonstrate_enhanced_analytics(components: Dict[str, Any], 
                                 sample_predictions: Dict[str, Any],
                                 sample_ground_truth: Dict[str, Any],
                                 sample_batch_data: Dict[str, Any]):
    """
    Demonstrate enhanced analytics capabilities
    """
    print("\n🔍 Demonstrating Enhanced Analytics Capabilities")
    
    analytics = components['analytics']
    ssl_trainer = components['ssl_trainer'] 
    visualizer = components['visualizer']
    metrics_analyzer = components['metrics_analyzer']
    
    # 1. Log Augmentation Statistics
    print("1️⃣ Logging augmentation batch statistics...")
    augmentation_info = {
        'batch_size': 4,
        'augmentation_info': [
            {'weather_type': 'rain', 'intensity': 0.4, 'effectiveness_score': 0.75},
            {'weather_type': 'snow', 'intensity': 0.3, 'effectiveness_score': 0.68},
            {'weather_type': 'fog', 'intensity': 0.5, 'effectiveness_score': 0.82},
            {'weather_type': 'clear', 'intensity': 0.0, 'effectiveness_score': 1.0}
        ]
    }
    analytics.log_augmentation_batch(augmentation_info)
    
    # 2. Log SSL Metrics
    print("2️⃣ Logging SSL training metrics...")
    ssl_metrics = {
        'cross_modal_loss': 0.45,
        'spatial_loss': 0.32, 
        'weather_aware_loss': 0.28,
        'consistency_loss': 0.38,
        'teacher_student_divergence': 0.15,
        'feature_alignment': 0.73
    }
    analytics.log_ssl_metrics(ssl_metrics)
    
    # 3. Log Detection Performance
    print("3️⃣ Logging detection performance...")
    detection_metrics = {
        'mAP': 0.68,
        'mAP_easy': 0.75,
        'mAP_moderate': 0.64,
        'mAP_hard': 0.52,
        'Car_AP': 0.82, 'Car_AP_easy': 0.87, 'Car_AP_moderate': 0.79, 'Car_AP_hard': 0.71,
        'Pedestrian_AP': 0.54, 'Pedestrian_AP_easy': 0.63, 'Pedestrian_AP_moderate': 0.48, 'Pedestrian_AP_hard': 0.31,
        'Cyclist_AP': 0.48, 'Cyclist_AP_easy': 0.58, 'Cyclist_AP_moderate': 0.41, 'Cyclist_AP_hard': 0.28,
        'bbox_loss': 0.25,
        'cls_loss': 0.34,
        'ssl_loss': 0.18
    }
    analytics.log_detection_metrics(epoch=10, metrics=detection_metrics)
    
    # 4. Analyze Detection Performance (Focus on Problematic Classes)
    print("4️⃣ Analyzing detection performance with focus on problematic classes...")
    detection_analysis = visualizer.analyze_detection_performance(
        sample_predictions, sample_ground_truth, 
        class_names=['Pedestrian', 'Cyclist', 'Car']
    )
    
    print("📊 Detection Analysis Results:")
    for class_name, stats in detection_analysis['class_specific_stats'].items():
        print(f"  {class_name}:")
        print(f"    Detection Rate: {stats.get('detection_rate', 0):.3f}")
        print(f"    Avg Confidence: {stats.get('avg_confidence', 0):.3f}")
        print(f"    Pred Count: {stats.get('prediction_count', 0)}")
        print(f"    GT Count: {stats.get('ground_truth_count', 0)}")
    
    # 5. Generate Improvement Suggestions
    print("5️⃣ Generating improvement suggestions...")
    improvement_suggestions = metrics_analyzer.generate_improvement_suggestions(
        detection_analysis['class_specific_stats']
    )
    
    print("💡 Improvement Suggestions:")
    for suggestion in improvement_suggestions:
        print(f"  • {suggestion}")
    
    # 6. Get SSL Analytics Summary
    print("6️⃣ SSL Analytics Summary:")
    ssl_summary = ssl_trainer.get_ssl_analytics_summary()
    for metric_name, values in ssl_summary.items():
        if isinstance(values, dict):
            print(f"  {metric_name}:")
            for sub_name, sub_vals in values.items():
                if isinstance(sub_vals, dict) and 'mean' in sub_vals:
                    print(f"    {sub_name}: mean={sub_vals['mean']:.3f}, trend={sub_vals.get('trend', 0):.3f}")
        else:
            print(f"  {metric_name}: {values}")
    
    # 7. Generate Comprehensive Training Report
    print("7️⃣ Generating comprehensive training report...")
    training_report = analytics.generate_training_report()
    
    print("📋 Training Report Summary:")
    print(f"  Total Samples Processed: {training_report.get('total_samples_processed', 0)}")
    
    aug_analysis = training_report.get('augmentation_analysis', {})
    weather_dist = aug_analysis.get('weather_distribution', {})
    if weather_dist:
        print(f"  Weather Diversity Index: {weather_dist.get('diversity_index', 0):.3f}")
        print(f"  Most Common Weather: {weather_dist.get('most_common', 'N/A')}")
    
    # Key insights
    insights = training_report.get('insights', {})
    print("🔍 Key Insights:")
    for insight_type, insight_list in insights.items():
        if insight_list:
            print(f"  {insight_type.replace('_', ' ').title()}:")
            for insight in insight_list[:3]:  # Show top 3
                print(f"    • {insight}")
    
    # Recommendations
    recommendations = training_report.get('recommendations', [])
    print("🎯 Recommendations:")
    for rec in recommendations[:5]:  # Show top 5
        print(f"  • {rec}")
    
    return training_report, detection_analysis


def simulate_training_with_enhanced_analytics():
    """
    Simulate a training session with enhanced analytics
    """
    print("\n🎮 Simulating Enhanced Training Session")
    
    work_dir = "/tmp/rose_enhanced_training"
    config_path = "configs/rose_mvxnet_dair_v2x.py"
    
    # Create directories
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    components = create_enhanced_training_pipeline(work_dir, config_path)
    
    # Create sample data for demonstration
    sample_predictions = create_sample_predictions()
    sample_ground_truth = create_sample_ground_truth()
    sample_batch_data = create_sample_batch_data()
    
    # Demonstrate analytics
    training_report, detection_analysis = demonstrate_enhanced_analytics(
        components, sample_predictions, sample_ground_truth, sample_batch_data
    )
    
    # Save analytics report
    print("8️⃣ Saving analytics reports...")
    components['analytics'].save_analytics_report(epoch=10)
    
    # Create visualizations
    print("9️⃣ Creating performance visualizations...")
    components['visualizer'].create_performance_summary_plot(
        detection_analysis, 
        os.path.join(work_dir, "performance_summary.png")
    )
    
    # Create analytics visualizations
    components['analytics'].create_visualizations()
    
    print("✅ Enhanced training simulation completed!")
    print(f"📁 Results saved in: {work_dir}")
    
    return components, training_report, detection_analysis


def create_sample_predictions() -> Dict[str, Any]:
    """Create sample predictions for demonstration"""
    # Simulate predictions with low performance for Pedestrian and Cyclist
    return {
        'bboxes_3d': np.array([
            [10, 5, 0, 2, 2, 5, 0.1],   # Car
            [15, -3, 0, 1.8, 1.8, 4.5, 0.2],  # Car
            [8, 2, 0, 0.6, 0.6, 1.7, 0.0],    # Pedestrian (small)
            [12, -1, 0, 0.8, 0.8, 1.8, 0.1],  # Pedestrian  
            [20, 3, 0, 1.2, 1.2, 1.8, 0.0],   # Cyclist (small)
        ]),
        'scores_3d': np.array([0.85, 0.78, 0.42, 0.38, 0.35]),  # Low scores for Pedestrian/Cyclist
        'labels_3d': np.array([2, 2, 0, 0, 1])  # 0=Pedestrian, 1=Cyclist, 2=Car
    }


def create_sample_ground_truth() -> Dict[str, Any]:
    """Create sample ground truth for demonstration"""
    return {
        'bboxes_3d': np.array([
            [10.2, 4.8, 0, 2.1, 2.0, 5.2, 0.12],  # Car
            [14.8, -2.9, 0, 1.9, 1.7, 4.6, 0.18], # Car
            [7.8, 2.1, 0, 0.65, 0.65, 1.75, 0.02], # Pedestrian
            [11.9, -0.8, 0, 0.75, 0.75, 1.82, 0.08], # Pedestrian
            [19.8, 2.9, 0, 1.25, 1.15, 1.85, 0.05], # Cyclist
            [25, 8, 0, 0.7, 0.7, 1.8, 0.0],       # Missed Pedestrian
            [30, -5, 0, 1.3, 1.2, 1.9, 0.1],      # Missed Cyclist
            [5, 1, 0, 2.2, 2.1, 5.0, 0.0],        # Missed Car
        ]),
        'labels_3d': np.array([2, 2, 0, 0, 1, 0, 1, 2])  # More GT than predictions
    }


def create_sample_batch_data() -> Dict[str, Any]:
    """Create sample batch data for demonstration"""
    return {
        'augmentation_info': [
            {'weather_type': 'rain', 'intensity': 0.4},
            {'weather_type': 'snow', 'intensity': 0.3},
            {'weather_type': 'fog', 'intensity': 0.5},
            {'weather_type': 'clear', 'intensity': 0.0}
        ],
        'gt_labels_3d': [np.array([2, 2, 0, 0, 1, 0, 1, 2])],
        'gt_bboxes_3d': [create_sample_ground_truth()['bboxes_3d']]
    }


if __name__ == "__main__":
    print("🌹 ROSE Enhanced Training Analytics Demo")
    print("=" * 50)
    
    # Run simulation
    components, training_report, detection_analysis = simulate_training_with_enhanced_analytics()
    
    print("\n📊 Summary of Key Issues Identified:")
    
    # Show problematic class analysis
    prob_analysis = detection_analysis.get('problematic_class_analysis', {})
    for class_name, analysis in prob_analysis.items():
        print(f"\n{class_name} Issues:")
        challenges = analysis.get('detection_challenges', [])
        patterns = analysis.get('failure_patterns', [])
        suggestions = analysis.get('improvement_suggestions', [])
        
        if challenges:
            print("  Challenges:", ", ".join(challenges))
        if patterns:
            print("  Failure Patterns:", ", ".join(patterns))
        if suggestions:
            print("  Suggestions:", "; ".join(suggestions[:3]))
    
    print("\n✨ Enhanced ROSE Training Analytics Demo Completed!")
    print("🔍 Check the generated reports and visualizations for detailed insights.")