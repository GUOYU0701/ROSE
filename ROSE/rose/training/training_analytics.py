"""
Training Analytics and Statistics Collection for ROSE Framework
Comprehensive analysis of data augmentation, SSL training, and model performance
"""
import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import torch
from collections import defaultdict, Counter


class ROSETrainingAnalytics:
    """
    Comprehensive training analytics for ROSE framework
    Tracks data augmentation, SSL metrics, detection performance
    """
    
    def __init__(self, work_dir: str, enabled: bool = True):
        self.work_dir = Path(work_dir)
        self.enabled = enabled
        
        if self.enabled:
            self.analytics_dir = self.work_dir / "analytics"
            self.analytics_dir.mkdir(exist_ok=True)
            
            # Initialize tracking dictionaries
            self.reset_tracking()
    
    def reset_tracking(self):
        """Reset all tracking statistics"""
        self.augmentation_stats = {
            'total_samples': 0,
            'weather_distribution': defaultdict(int),
            'intensity_distribution': defaultdict(list),
            'effectiveness_by_weather': defaultdict(list),
            'adaptation_history': [],
            'augmentation_timeline': []
        }
        
        self.ssl_stats = {
            'contrastive_losses': {
                'cross_modal': [],
                'spatial': [],
                'weather_aware': []
            },
            'consistency_losses': [],
            'teacher_student_divergence': [],
            'feature_alignment_scores': [],
            'ssl_convergence_metrics': []
        }
        
        self.detection_stats = {
            'class_performance': {
                'Car': {'tp': 0, 'fp': 0, 'fn': 0, 'precision': [], 'recall': [], 'ap': []},
                'Pedestrian': {'tp': 0, 'fp': 0, 'fn': 0, 'precision': [], 'recall': [], 'ap': []},
                'Cyclist': {'tp': 0, 'fp': 0, 'fn': 0, 'precision': [], 'recall': [], 'ap': []}
            },
            'overall_map': [],
            'loss_components': defaultdict(list),
            'epoch_metrics': []
        }
        
        self.training_timeline = []
        
    def log_augmentation_batch(self, batch_info: Dict[str, Any]):
        """Log augmentation statistics for a batch"""
        if not self.enabled:
            return
            
        batch_size = batch_info.get('batch_size', 1)
        self.augmentation_stats['total_samples'] += batch_size
        
        # Weather distribution
        for sample in batch_info.get('augmentation_info', []):
            weather_type = sample.get('weather_type', 'clear')
            intensity = sample.get('intensity', 0.0)
            effectiveness = sample.get('effectiveness_score', 0.0)
            
            self.augmentation_stats['weather_distribution'][weather_type] += 1
            self.augmentation_stats['intensity_distribution'][weather_type].append(intensity)
            self.augmentation_stats['effectiveness_by_weather'][weather_type].append(effectiveness)
        
        # Timeline tracking
        self.augmentation_stats['augmentation_timeline'].append({
            'timestamp': datetime.now().isoformat(),
            'batch_size': batch_size,
            'weather_types': [info.get('weather_type') for info in batch_info.get('augmentation_info', [])],
            'avg_intensity': np.mean([info.get('intensity', 0) for info in batch_info.get('augmentation_info', [])])
        })
    
    def log_ssl_metrics(self, ssl_metrics: Dict[str, float]):
        """Log SSL training metrics"""
        if not self.enabled:
            return
            
        # Contrastive losses
        if 'cross_modal_loss' in ssl_metrics:
            self.ssl_stats['contrastive_losses']['cross_modal'].append(ssl_metrics['cross_modal_loss'])
        if 'spatial_loss' in ssl_metrics:
            self.ssl_stats['contrastive_losses']['spatial'].append(ssl_metrics['spatial_loss'])
        if 'weather_aware_loss' in ssl_metrics:
            self.ssl_stats['contrastive_losses']['weather_aware'].append(ssl_metrics['weather_aware_loss'])
            
        # Consistency metrics
        if 'consistency_loss' in ssl_metrics:
            self.ssl_stats['consistency_losses'].append(ssl_metrics['consistency_loss'])
        if 'teacher_student_divergence' in ssl_metrics:
            self.ssl_stats['teacher_student_divergence'].append(ssl_metrics['teacher_student_divergence'])
        if 'feature_alignment' in ssl_metrics:
            self.ssl_stats['feature_alignment_scores'].append(ssl_metrics['feature_alignment'])
    
    def log_detection_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Log detection performance metrics"""
        if not self.enabled:
            return
            
        # Overall mAP
        if 'mAP' in metrics:
            self.detection_stats['overall_map'].append({
                'epoch': epoch,
                'mAP': metrics['mAP'],
                'mAP_easy': metrics.get('mAP_easy', 0),
                'mAP_moderate': metrics.get('mAP_moderate', 0),
                'mAP_hard': metrics.get('mAP_hard', 0)
            })
        
        # Class-specific performance
        for class_name in self.detection_stats['class_performance'].keys():
            if f'{class_name}_AP' in metrics:
                self.detection_stats['class_performance'][class_name]['ap'].append({
                    'epoch': epoch,
                    'AP': metrics[f'{class_name}_AP'],
                    'AP_easy': metrics.get(f'{class_name}_AP_easy', 0),
                    'AP_moderate': metrics.get(f'{class_name}_AP_moderate', 0),
                    'AP_hard': metrics.get(f'{class_name}_AP_hard', 0)
                })
        
        # Loss components
        loss_keys = ['bbox_loss', 'cls_loss', 'dir_cls_loss', 'ssl_loss', 'consistency_loss']
        for key in loss_keys:
            if key in metrics:
                self.detection_stats['loss_components'][key].append({
                    'epoch': epoch,
                    'value': metrics[key]
                })
    
    def log_adaptation_event(self, adaptation_info: Dict[str, Any]):
        """Log adaptive augmentation events"""
        if not self.enabled:
            return
            
        self.augmentation_stats['adaptation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'trigger_metric': adaptation_info.get('trigger_metric'),
            'previous_config': adaptation_info.get('previous_config'),
            'new_config': adaptation_info.get('new_config'),
            'adaptation_reason': adaptation_info.get('reason')
        })
    
    def compute_augmentation_analysis(self) -> Dict[str, Any]:
        """Compute comprehensive augmentation analysis"""
        if not self.enabled or self.augmentation_stats['total_samples'] == 0:
            return {}
        
        analysis = {}
        
        # Weather distribution analysis
        weather_dist = dict(self.augmentation_stats['weather_distribution'])
        total_samples = sum(weather_dist.values())
        weather_percentages = {k: (v / total_samples) * 100 for k, v in weather_dist.items()}
        
        analysis['weather_distribution'] = {
            'counts': weather_dist,
            'percentages': weather_percentages,
            'most_common': max(weather_dist, key=weather_dist.get),
            'diversity_index': self._compute_diversity_index(weather_dist)
        }
        
        # Intensity analysis by weather type
        intensity_stats = {}
        for weather_type, intensities in self.augmentation_stats['intensity_distribution'].items():
            if intensities:
                intensity_stats[weather_type] = {
                    'mean': np.mean(intensities),
                    'std': np.std(intensities),
                    'min': np.min(intensities),
                    'max': np.max(intensities),
                    'median': np.median(intensities)
                }
        
        analysis['intensity_analysis'] = intensity_stats
        
        # Effectiveness analysis
        effectiveness_stats = {}
        for weather_type, scores in self.augmentation_stats['effectiveness_by_weather'].items():
            if scores:
                effectiveness_stats[weather_type] = {
                    'mean_effectiveness': np.mean(scores),
                    'effectiveness_std': np.std(scores),
                    'effectiveness_trend': self._compute_trend(scores)
                }
        
        analysis['effectiveness_analysis'] = effectiveness_stats
        
        # Adaptation analysis
        analysis['adaptation_summary'] = {
            'total_adaptations': len(self.augmentation_stats['adaptation_history']),
            'adaptation_frequency': self._compute_adaptation_frequency(),
            'adaptation_triggers': self._analyze_adaptation_triggers()
        }
        
        return analysis
    
    def compute_ssl_analysis(self) -> Dict[str, Any]:
        """Compute SSL training analysis"""
        if not self.enabled:
            return {}
        
        analysis = {}
        
        # Contrastive loss analysis
        for loss_type, losses in self.ssl_stats['contrastive_losses'].items():
            if losses:
                analysis[f'{loss_type}_analysis'] = {
                    'mean_loss': np.mean(losses),
                    'loss_std': np.std(losses),
                    'loss_trend': self._compute_trend(losses),
                    'convergence_rate': self._compute_convergence_rate(losses),
                    'stability_score': self._compute_stability_score(losses)
                }
        
        # Consistency analysis
        if self.ssl_stats['consistency_losses']:
            consistency_losses = self.ssl_stats['consistency_losses']
            analysis['consistency_analysis'] = {
                'mean_consistency': np.mean(consistency_losses),
                'consistency_trend': self._compute_trend(consistency_losses),
                'consistency_stability': self._compute_stability_score(consistency_losses)
            }
        
        # Teacher-student analysis
        if self.ssl_stats['teacher_student_divergence']:
            divergences = self.ssl_stats['teacher_student_divergence']
            analysis['teacher_student_analysis'] = {
                'mean_divergence': np.mean(divergences),
                'divergence_trend': self._compute_trend(divergences),
                'convergence_quality': 1.0 / (1.0 + np.mean(divergences)) if divergences else 0
            }
        
        # Feature alignment analysis
        if self.ssl_stats['feature_alignment_scores']:
            alignments = self.ssl_stats['feature_alignment_scores']
            analysis['feature_alignment_analysis'] = {
                'mean_alignment': np.mean(alignments),
                'alignment_trend': self._compute_trend(alignments),
                'alignment_quality': np.mean(alignments) if alignments else 0
            }
        
        return analysis
    
    def compute_detection_analysis(self) -> Dict[str, Any]:
        """Compute detection performance analysis"""
        if not self.enabled:
            return {}
        
        analysis = {}
        
        # Overall performance trend
        if self.detection_stats['overall_map']:
            maps = [entry['mAP'] for entry in self.detection_stats['overall_map']]
            analysis['overall_performance'] = {
                'final_map': maps[-1] if maps else 0,
                'best_map': max(maps) if maps else 0,
                'performance_trend': self._compute_trend(maps),
                'improvement_rate': self._compute_improvement_rate(maps)
            }
        
        # Class-specific analysis
        class_analysis = {}
        for class_name, class_data in self.detection_stats['class_performance'].items():
            if class_data['ap']:
                aps = [entry['AP'] for entry in class_data['ap']]
                class_analysis[class_name] = {
                    'final_ap': aps[-1] if aps else 0,
                    'best_ap': max(aps) if aps else 0,
                    'performance_trend': self._compute_trend(aps),
                    'improvement_rate': self._compute_improvement_rate(aps),
                    'performance_stability': self._compute_stability_score(aps)
                }
        
        analysis['class_performance'] = class_analysis
        
        # Identify problematic classes
        problematic_classes = []
        for class_name, metrics in class_analysis.items():
            if metrics['final_ap'] < 0.3 or metrics['performance_trend'] < -0.1:
                problematic_classes.append({
                    'class': class_name,
                    'final_ap': metrics['final_ap'],
                    'trend': metrics['performance_trend'],
                    'issue': 'low_performance' if metrics['final_ap'] < 0.3 else 'declining_trend'
                })
        
        analysis['problematic_classes'] = problematic_classes
        
        # Loss component analysis
        loss_analysis = {}
        for loss_type, loss_history in self.detection_stats['loss_components'].items():
            if loss_history:
                losses = [entry['value'] for entry in loss_history]
                loss_analysis[loss_type] = {
                    'final_loss': losses[-1] if losses else 0,
                    'loss_trend': self._compute_trend(losses),
                    'convergence_rate': self._compute_convergence_rate(losses)
                }
        
        analysis['loss_analysis'] = loss_analysis
        
        return analysis
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        if not self.enabled:
            return {}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples_processed': self.augmentation_stats['total_samples'],
            'augmentation_analysis': self.compute_augmentation_analysis(),
            'ssl_analysis': self.compute_ssl_analysis(),
            'detection_analysis': self.compute_detection_analysis()
        }
        
        # Generate insights and recommendations
        report['insights'] = self._generate_insights(report)
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def save_analytics_report(self, epoch: Optional[int] = None):
        """Save comprehensive analytics report"""
        if not self.enabled:
            return
        
        report = self.generate_training_report()
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if epoch is not None:
            filename = f"training_analytics_epoch_{epoch}_{timestamp}.json"
        else:
            filename = f"training_analytics_{timestamp}.json"
        
        report_path = self.analytics_dir / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'timestamp': report['timestamp'],
            'total_samples': report['total_samples_processed'],
            'final_map': report['detection_analysis'].get('overall_performance', {}).get('final_map', 0),
            'problematic_classes': len(report['detection_analysis'].get('problematic_classes', [])),
            'key_insights': report.get('insights', {}).get('key_findings', [])
        }
        
        summary_path = self.analytics_dir / f"summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def create_visualizations(self):
        """Create comprehensive visualization plots"""
        if not self.enabled:
            return
        
        vis_dir = self.analytics_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Weather distribution plot
        self._plot_weather_distribution(vis_dir)
        
        # SSL metrics plots
        self._plot_ssl_metrics(vis_dir)
        
        # Detection performance plots
        self._plot_detection_performance(vis_dir)
        
        # Loss convergence plots
        self._plot_loss_convergence(vis_dir)
    
    def _compute_diversity_index(self, distribution: Dict) -> float:
        """Compute Shannon diversity index for weather distribution"""
        total = sum(distribution.values())
        if total == 0:
            return 0.0
        
        proportions = [count / total for count in distribution.values()]
        return -sum(p * np.log2(p) for p in proportions if p > 0)
    
    def _compute_trend(self, values: List[float], window: int = 10) -> float:
        """Compute trend using linear regression slope"""
        if len(values) < 2:
            return 0.0
        
        # Use last 'window' values for trend computation
        recent_values = values[-window:] if len(values) > window else values
        x = np.arange(len(recent_values))
        
        if len(recent_values) < 2:
            return 0.0
        
        # Linear regression
        slope = np.polyfit(x, recent_values, 1)[0]
        return slope
    
    def _compute_convergence_rate(self, losses: List[float]) -> float:
        """Compute convergence rate of loss values"""
        if len(losses) < 10:
            return 0.0
        
        # Compare first and last quarter
        quarter_size = len(losses) // 4
        first_quarter = np.mean(losses[:quarter_size])
        last_quarter = np.mean(losses[-quarter_size:])
        
        if first_quarter == 0:
            return 0.0
        
        return (first_quarter - last_quarter) / first_quarter
    
    def _compute_stability_score(self, values: List[float]) -> float:
        """Compute stability score (inverse of coefficient of variation)"""
        if len(values) < 2:
            return 0.0
        
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
        
        cv = np.std(values) / mean_val
        return 1.0 / (1.0 + cv)  # Normalized stability score
    
    def _compute_improvement_rate(self, values: List[float]) -> float:
        """Compute improvement rate over time"""
        if len(values) < 2:
            return 0.0
        
        initial = values[0]
        final = values[-1]
        
        if initial == 0:
            return 0.0
        
        return (final - initial) / initial
    
    def _compute_adaptation_frequency(self) -> float:
        """Compute frequency of adaptations"""
        if not self.augmentation_stats['adaptation_history']:
            return 0.0
        
        total_samples = self.augmentation_stats['total_samples']
        total_adaptations = len(self.augmentation_stats['adaptation_history'])
        
        return total_adaptations / max(total_samples / 1000, 1)  # Per 1000 samples
    
    def _analyze_adaptation_triggers(self) -> Dict[str, int]:
        """Analyze what triggers adaptations"""
        triggers = Counter()
        for adaptation in self.augmentation_stats['adaptation_history']:
            trigger = adaptation.get('adaptation_reason', 'unknown')
            triggers[trigger] += 1
        
        return dict(triggers)
    
    def _generate_insights(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training insights"""
        insights = {
            'key_findings': [],
            'performance_issues': [],
            'ssl_effectiveness': [],
            'augmentation_effectiveness': []
        }
        
        # Detection performance insights
        detection_analysis = report.get('detection_analysis', {})
        if 'problematic_classes' in detection_analysis:
            for problem in detection_analysis['problematic_classes']:
                if problem['issue'] == 'low_performance':
                    insights['performance_issues'].append(
                        f"{problem['class']} has low AP: {problem['final_ap']:.3f}"
                    )
                elif problem['issue'] == 'declining_trend':
                    insights['performance_issues'].append(
                        f"{problem['class']} shows declining performance trend"
                    )
        
        # SSL insights
        ssl_analysis = report.get('ssl_analysis', {})
        for analysis_type, metrics in ssl_analysis.items():
            if 'convergence_rate' in metrics and metrics['convergence_rate'] < 0.1:
                insights['ssl_effectiveness'].append(
                    f"Low convergence rate in {analysis_type}: {metrics['convergence_rate']:.3f}"
                )
        
        # Augmentation insights
        aug_analysis = report.get('augmentation_analysis', {})
        weather_dist = aug_analysis.get('weather_distribution', {})
        if weather_dist and 'diversity_index' in weather_dist:
            if weather_dist['diversity_index'] < 1.0:
                insights['augmentation_effectiveness'].append(
                    f"Low weather diversity: {weather_dist['diversity_index']:.3f}"
                )
        
        return insights
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate training recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        detection_analysis = report.get('detection_analysis', {})
        if 'problematic_classes' in detection_analysis:
            problematic = detection_analysis['problematic_classes']
            if any(p['class'] in ['Pedestrian', 'Cyclist'] for p in problematic):
                recommendations.append(
                    "Consider increasing augmentation intensity for small objects (Pedestrian/Cyclist)"
                )
                recommendations.append(
                    "Review anchor configurations for better small object detection"
                )
        
        # SSL recommendations
        ssl_analysis = report.get('ssl_analysis', {})
        if ssl_analysis:
            avg_convergence = np.mean([
                metrics.get('convergence_rate', 0) 
                for metrics in ssl_analysis.values() 
                if isinstance(metrics, dict) and 'convergence_rate' in metrics
            ])
            
            if avg_convergence < 0.2:
                recommendations.append(
                    "Consider adjusting SSL loss weights for better convergence"
                )
        
        # Augmentation recommendations
        aug_analysis = report.get('augmentation_analysis', {})
        weather_dist = aug_analysis.get('weather_distribution', {})
        if weather_dist and 'diversity_index' in weather_dist:
            if weather_dist['diversity_index'] < 1.5:
                recommendations.append(
                    "Increase weather augmentation diversity for better robustness"
                )
        
        return recommendations
    
    def _plot_weather_distribution(self, vis_dir: Path):
        """Plot weather distribution"""
        weather_dist = dict(self.augmentation_stats['weather_distribution'])
        if not weather_dist:
            return
        
        plt.figure(figsize=(10, 6))
        plt.bar(weather_dist.keys(), weather_dist.values())
        plt.title('Weather Augmentation Distribution')
        plt.xlabel('Weather Type')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(vis_dir / 'weather_distribution.png')
        plt.close()
    
    def _plot_ssl_metrics(self, vis_dir: Path):
        """Plot SSL training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Contrastive losses
        ax = axes[0, 0]
        for loss_type, losses in self.ssl_stats['contrastive_losses'].items():
            if losses:
                ax.plot(losses, label=loss_type)
        ax.set_title('Contrastive Losses')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Consistency losses
        ax = axes[0, 1]
        if self.ssl_stats['consistency_losses']:
            ax.plot(self.ssl_stats['consistency_losses'])
        ax.set_title('Consistency Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        
        # Teacher-student divergence
        ax = axes[1, 0]
        if self.ssl_stats['teacher_student_divergence']:
            ax.plot(self.ssl_stats['teacher_student_divergence'])
        ax.set_title('Teacher-Student Divergence')
        ax.set_xlabel('Step')
        ax.set_ylabel('Divergence')
        
        # Feature alignment
        ax = axes[1, 1]
        if self.ssl_stats['feature_alignment_scores']:
            ax.plot(self.ssl_stats['feature_alignment_scores'])
        ax.set_title('Feature Alignment Score')
        ax.set_xlabel('Step')
        ax.set_ylabel('Alignment Score')
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'ssl_metrics.png')
        plt.close()
    
    def _plot_detection_performance(self, vis_dir: Path):
        """Plot detection performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall mAP
        ax = axes[0, 0]
        if self.detection_stats['overall_map']:
            epochs = [entry['epoch'] for entry in self.detection_stats['overall_map']]
            maps = [entry['mAP'] for entry in self.detection_stats['overall_map']]
            ax.plot(epochs, maps)
        ax.set_title('Overall mAP')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        
        # Class-specific AP
        ax = axes[0, 1]
        for class_name, class_data in self.detection_stats['class_performance'].items():
            if class_data['ap']:
                epochs = [entry['epoch'] for entry in class_data['ap']]
                aps = [entry['AP'] for entry in class_data['ap']]
                ax.plot(epochs, aps, label=class_name)
        ax.set_title('Class-specific AP')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AP')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'detection_performance.png')
        plt.close()
    
    def _plot_loss_convergence(self, vis_dir: Path):
        """Plot loss convergence"""
        plt.figure(figsize=(12, 8))
        
        for loss_type, loss_history in self.detection_stats['loss_components'].items():
            if loss_history:
                epochs = [entry['epoch'] for entry in loss_history]
                losses = [entry['value'] for entry in loss_history]
                plt.plot(epochs, losses, label=loss_type)
        
        plt.title('Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(vis_dir / 'loss_convergence.png')
        plt.close()


class DetectionMetricsAnalyzer:
    """
    Specialized analyzer for detection performance issues
    Focus on Pedestrian and Cyclist detection problems
    """
    
    def __init__(self):
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.difficulty_levels = ['easy', 'moderate', 'hard']
    
    def analyze_class_performance(self, predictions: Dict, ground_truth: Dict,
                                class_name: str) -> Dict[str, Any]:
        """Analyze performance for specific class"""
        analysis = {
            'detection_rate': 0.0,
            'false_positive_rate': 0.0,
            'avg_confidence': 0.0,
            'size_distribution': {},
            'distance_distribution': {},
            'failure_cases': []
        }
        
        # Extract class-specific predictions and ground truth
        if class_name in predictions and class_name in ground_truth:
            pred_boxes = predictions[class_name]
            gt_boxes = ground_truth[class_name]
            
            # Compute detection rate
            matched_gt = self._match_predictions_to_gt(pred_boxes, gt_boxes)
            analysis['detection_rate'] = len(matched_gt) / max(len(gt_boxes), 1)
            
            # Compute false positive rate
            unmatched_pred = self._find_unmatched_predictions(pred_boxes, gt_boxes)
            analysis['false_positive_rate'] = len(unmatched_pred) / max(len(pred_boxes), 1)
            
            # Average confidence
            if pred_boxes:
                confidences = [box.get('confidence', 0) for box in pred_boxes]
                analysis['avg_confidence'] = np.mean(confidences)
            
            # Size and distance analysis
            analysis['size_distribution'] = self._analyze_object_sizes(gt_boxes)
            analysis['distance_distribution'] = self._analyze_object_distances(gt_boxes)
            
            # Identify failure cases
            analysis['failure_cases'] = self._identify_failure_patterns(pred_boxes, gt_boxes)
        
        return analysis
    
    def generate_improvement_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        for class_name, class_analysis in analysis.items():
            if class_analysis['detection_rate'] < 0.5:
                suggestions.append(f"Low detection rate for {class_name}: {class_analysis['detection_rate']:.3f}")
                
                # Size-based suggestions
                size_dist = class_analysis.get('size_distribution', {})
                if 'small' in size_dist and size_dist['small'] > 0.5:
                    suggestions.append(f"Consider multi-scale training for small {class_name} objects")
                
                # Distance-based suggestions
                dist_dist = class_analysis.get('distance_distribution', {})
                if 'far' in dist_dist and dist_dist['far'] > 0.3:
                    suggestions.append(f"Improve long-range detection for {class_name}")
            
            if class_analysis['false_positive_rate'] > 0.3:
                suggestions.append(f"High false positive rate for {class_name}: consider stricter NMS")
        
        return suggestions
    
    def _match_predictions_to_gt(self, predictions: List, ground_truth: List,
                               iou_threshold: float = 0.5) -> List:
        """Match predictions to ground truth boxes"""
        matched = []
        # Simplified matching - in real implementation, use proper IoU calculation
        min_count = min(len(predictions), len(ground_truth))
        matched = predictions[:min_count]  # Simplified
        return matched
    
    def _find_unmatched_predictions(self, predictions: List, ground_truth: List) -> List:
        """Find unmatched predictions (false positives)"""
        # Simplified implementation
        matched = self._match_predictions_to_gt(predictions, ground_truth)
        unmatched = [p for p in predictions if p not in matched]
        return unmatched
    
    def _analyze_object_sizes(self, boxes: List) -> Dict[str, float]:
        """Analyze object size distribution"""
        if not boxes:
            return {}
        
        sizes = []
        for box in boxes:
            # Assume box has dimensions
            volume = box.get('width', 1) * box.get('height', 1) * box.get('length', 1)
            sizes.append(volume)
        
        # Categorize sizes
        small_threshold = np.percentile(sizes, 33)
        large_threshold = np.percentile(sizes, 67)
        
        small_count = sum(1 for s in sizes if s <= small_threshold)
        medium_count = sum(1 for s in sizes if small_threshold < s <= large_threshold)
        large_count = sum(1 for s in sizes if s > large_threshold)
        
        total = len(sizes)
        return {
            'small': small_count / total,
            'medium': medium_count / total,
            'large': large_count / total
        }
    
    def _analyze_object_distances(self, boxes: List) -> Dict[str, float]:
        """Analyze object distance distribution"""
        if not boxes:
            return {}
        
        distances = []
        for box in boxes:
            # Assume box has centroid coordinates
            x, y, z = box.get('x', 0), box.get('y', 0), box.get('z', 0)
            distance = np.sqrt(x**2 + y**2 + z**2)
            distances.append(distance)
        
        # Categorize distances
        near_threshold = np.percentile(distances, 33)
        far_threshold = np.percentile(distances, 67)
        
        near_count = sum(1 for d in distances if d <= near_threshold)
        medium_count = sum(1 for d in distances if near_threshold < d <= far_threshold)
        far_count = sum(1 for d in distances if d > far_threshold)
        
        total = len(distances)
        return {
            'near': near_count / total,
            'medium': medium_count / total,
            'far': far_count / total
        }
    
    def _identify_failure_patterns(self, predictions: List, ground_truth: List) -> List[str]:
        """Identify common failure patterns"""
        patterns = []
        
        if len(predictions) < len(ground_truth) * 0.5:
            patterns.append("Low recall - many objects missed")
        
        if len(predictions) > len(ground_truth) * 1.5:
            patterns.append("High false positive rate")
        
        # Add more sophisticated pattern detection
        return patterns