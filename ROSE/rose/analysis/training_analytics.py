"""
ROSE训练分析系统
结合数据增强策略和SSL过程的统计分析
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class ROSETrainingAnalytics:
    """ROSE训练分析器"""
    
    def __init__(self, work_dir: str, enabled: bool = True):
        self.work_dir = Path(work_dir)
        self.enabled = enabled
        
        if self.enabled:
            self.analytics_dir = self.work_dir / 'training_analytics'
            self.analytics_dir.mkdir(parents=True, exist_ok=True)
            
            # 子目录
            self.dirs = {
                'plots': self.analytics_dir / 'performance_plots',
                'reports': self.analytics_dir / 'analysis_reports',
                'data': self.analytics_dir / 'collected_data',
                'comparisons': self.analytics_dir / 'strategy_comparisons'
            }
            
            for dir_path in self.dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # 数据收集
        self.training_data = {
            'epochs': [],
            'augmentation_strategies': [],
            'ssl_metrics': [],
            'detection_performance': [],
            'class_specific_performance': defaultdict(list),
            'weather_impact': defaultdict(list)
        }
        
    def record_epoch_data(self, epoch: int, epoch_data: Dict):
        """记录每个epoch的数据"""
        if not self.enabled:
            return
            
        epoch_record = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **epoch_data
        }
        
        self.training_data['epochs'].append(epoch_record)
        
        # 分类记录不同类型的数据
        if 'augmentation_strategy' in epoch_data:
            self.training_data['augmentation_strategies'].append({
                'epoch': epoch,
                'strategy': epoch_data['augmentation_strategy']
            })
            
        if 'ssl_metrics' in epoch_data:
            self.training_data['ssl_metrics'].append({
                'epoch': epoch,
                'metrics': epoch_data['ssl_metrics']
            })
            
        if 'detection_results' in epoch_data:
            self.training_data['detection_performance'].append({
                'epoch': epoch,
                'results': epoch_data['detection_results']
            })
            
            # 类别特定性能
            for class_name, class_metrics in epoch_data['detection_results'].items():
                if isinstance(class_metrics, dict):
                    self.training_data['class_specific_performance'][class_name].append({
                        'epoch': epoch,
                        'metrics': class_metrics
                    })
    
    def analyze_augmentation_impact(self) -> Dict:
        """分析数据增强策略的影响"""
        if not self.enabled or not self.training_data['epochs']:
            return {}
            
        analysis = {
            'weather_effectiveness': {},
            'augmentation_trends': {},
            'class_specific_impact': {},
            'optimal_strategies': {}
        }
        
        try:
            # 按天气类型分析性能
            weather_performance = defaultdict(list)
            
            for epoch_data in self.training_data['epochs']:
                if 'weather_distribution' in epoch_data and 'detection_results' in epoch_data:
                    weather_dist = epoch_data['weather_distribution']
                    detection_results = epoch_data['detection_results']
                    
                    for weather_type, ratio in weather_dist.items():
                        if weather_type in detection_results:
                            weather_performance[weather_type].append({
                                'epoch': epoch_data['epoch'],
                                'ratio': ratio,
                                'performance': detection_results[weather_type]
                            })
            
            # 计算每种天气的平均性能
            for weather_type, records in weather_performance.items():
                if records:
                    avg_performance = {}
                    for metric in ['recall', 'precision', 'f1']:
                        values = [r['performance'].get(metric, 0) for r in records if metric in r['performance']]
                        avg_performance[metric] = np.mean(values) if values else 0
                    
                    analysis['weather_effectiveness'][weather_type] = {
                        'average_performance': avg_performance,
                        'sample_count': len(records),
                        'trend': self._calculate_trend([r['performance'].get('f1', 0) for r in records])
                    }
            
            # 分析最优策略
            best_strategies = self._find_optimal_strategies()
            analysis['optimal_strategies'] = best_strategies
            
        except Exception as e:
            print(f"增强策略分析错误: {e}")
            
        return analysis
    
    def analyze_ssl_effectiveness(self) -> Dict:
        """分析SSL训练的有效性"""
        if not self.enabled or not self.training_data['ssl_metrics']:
            return {}
            
        analysis = {
            'ssl_convergence': {},
            'cross_modal_alignment': {},
            'consistency_metrics': {},
            'ssl_vs_detection_correlation': {}
        }
        
        try:
            ssl_data = []
            for record in self.training_data['ssl_metrics']:
                epoch = record['epoch']
                metrics = record['metrics']
                
                ssl_data.append({
                    'epoch': epoch,
                    'contrastive_loss': metrics.get('contrastive_loss', 0),
                    'consistency_loss': metrics.get('consistency_loss', 0),
                    'cross_modal_loss': metrics.get('cross_modal_loss', 0),
                    'total_ssl_loss': metrics.get('total_ssl_loss', 0)
                })
            
            if ssl_data:
                # SSL收敛分析
                epochs = [d['epoch'] for d in ssl_data]
                contrastive_losses = [d['contrastive_loss'] for d in ssl_data]
                consistency_losses = [d['consistency_loss'] for d in ssl_data]
                
                analysis['ssl_convergence'] = {
                    'contrastive_trend': self._calculate_trend(contrastive_losses),
                    'consistency_trend': self._calculate_trend(consistency_losses),
                    'convergence_rate': self._calculate_convergence_rate(contrastive_losses + consistency_losses)
                }
                
                # 跨模态对齐分析
                cross_modal_losses = [d['cross_modal_loss'] for d in ssl_data]
                analysis['cross_modal_alignment'] = {
                    'alignment_improvement': self._calculate_improvement_rate(cross_modal_losses),
                    'final_alignment_quality': cross_modal_losses[-1] if cross_modal_losses else 0
                }
        
        except Exception as e:
            print(f"SSL分析错误: {e}")
            
        return analysis
    
    def analyze_class_specific_performance(self) -> Dict:
        """分析类别特定性能"""
        analysis = {
            'class_trends': {},
            'problematic_classes': [],
            'improvement_patterns': {},
            'augmentation_class_correlation': {}
        }
        
        try:
            for class_name, class_records in self.training_data['class_specific_performance'].items():
                if not class_records:
                    continue
                    
                # 性能趋势
                epochs = [r['epoch'] for r in class_records]
                recall_values = [r['metrics'].get('recall', 0) for r in class_records]
                precision_values = [r['metrics'].get('precision', 0) for r in class_records]
                f1_values = [r['metrics'].get('f1', 0) for r in class_records]
                
                analysis['class_trends'][class_name] = {
                    'recall_trend': self._calculate_trend(recall_values),
                    'precision_trend': self._calculate_trend(precision_values),
                    'f1_trend': self._calculate_trend(f1_values),
                    'final_performance': {
                        'recall': recall_values[-1] if recall_values else 0,
                        'precision': precision_values[-1] if precision_values else 0,
                        'f1': f1_values[-1] if f1_values else 0
                    }
                }
                
                # 识别问题类别
                final_f1 = f1_values[-1] if f1_values else 0
                if final_f1 < 0.3:  # F1分数低于30%认为是问题类别
                    analysis['problematic_classes'].append({
                        'class': class_name,
                        'final_f1': final_f1,
                        'improvement_rate': self._calculate_improvement_rate(f1_values)
                    })
        
        except Exception as e:
            print(f"类别分析错误: {e}")
            
        return analysis
    
    def create_performance_visualizations(self):
        """创建性能可视化图表"""
        if not self.enabled:
            return
            
        try:
            # 1. 整体训练曲线
            self._plot_training_curves()
            
            # 2. 增强策略影响
            self._plot_augmentation_impact()
            
            # 3. SSL效果分析
            self._plot_ssl_effectiveness()
            
            # 4. 类别性能对比
            self._plot_class_performance()
            
            # 5. 天气影响分析
            self._plot_weather_impact()
            
        except Exception as e:
            print(f"可视化创建错误: {e}")
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        if not self.training_data['epochs']:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = []
        detection_losses = []
        ssl_losses = []
        total_losses = []
        map_scores = []
        
        for epoch_data in self.training_data['epochs']:
            epochs.append(epoch_data['epoch'])
            detection_losses.append(epoch_data.get('detection_loss', 0))
            ssl_losses.append(epoch_data.get('ssl_loss', 0))
            total_losses.append(epoch_data.get('total_loss', 0))
            map_scores.append(epoch_data.get('map_score', 0))
        
        # 损失曲线
        axes[0,0].plot(epochs, detection_losses, 'b-', label='Detection Loss', linewidth=2)
        axes[0,0].plot(epochs, ssl_losses, 'r-', label='SSL Loss', linewidth=2)
        axes[0,0].plot(epochs, total_losses, 'g-', label='Total Loss', linewidth=2)
        axes[0,0].set_title('Training Loss Curves')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # mAP曲线
        axes[0,1].plot(epochs, map_scores, 'purple', linewidth=3, marker='o', markersize=6)
        axes[0,1].set_title('mAP Performance')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('mAP Score')
        axes[0,1].grid(True, alpha=0.3)
        
        # 类别性能
        class_colors = ['red', 'blue', 'green', 'orange']
        for i, (class_name, records) in enumerate(self.training_data['class_specific_performance'].items()):
            if records:
                class_epochs = [r['epoch'] for r in records]
                class_f1 = [r['metrics'].get('f1', 0) for r in records]
                axes[1,0].plot(class_epochs, class_f1, color=class_colors[i % len(class_colors)], 
                             label=class_name, linewidth=2, marker='s', markersize=4)
        
        axes[1,0].set_title('Class-specific F1 Scores')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('F1 Score')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 增强策略分布
        if self.training_data['augmentation_strategies']:
            weather_counts = defaultdict(int)
            for strategy in self.training_data['augmentation_strategies']:
                for weather, count in strategy['strategy'].get('weather_distribution', {}).items():
                    weather_counts[weather] += count
            
            if weather_counts:
                axes[1,1].pie(weather_counts.values(), labels=weather_counts.keys(), autopct='%1.1f%%')
                axes[1,1].set_title('Overall Weather Distribution')
        
        plt.tight_layout()
        save_path = self.dirs['plots'] / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 训练曲线保存: {save_path}")
    
    def _plot_augmentation_impact(self):
        """绘制增强策略影响"""
        # 增强策略对不同类别的影响分析
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 天气类型vs性能
        weather_data = defaultdict(list)
        for epoch_data in self.training_data['epochs']:
            weather_dist = epoch_data.get('weather_distribution', {})
            detection_results = epoch_data.get('detection_results', {})
            
            for weather, ratio in weather_dist.items():
                if 'overall_f1' in detection_results:
                    weather_data[weather].append(detection_results['overall_f1'])
        
        if weather_data:
            weather_names = list(weather_data.keys())
            weather_means = [np.mean(weather_data[w]) for w in weather_names]
            weather_stds = [np.std(weather_data[w]) for w in weather_names]
            
            axes[0,0].bar(weather_names, weather_means, yerr=weather_stds, capsize=5, alpha=0.7)
            axes[0,0].set_title('Weather Type vs Performance')
            axes[0,0].set_ylabel('Average F1 Score')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self.dirs['plots'] / 'augmentation_impact.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 增强影响分析保存: {save_path}")
    
    def _plot_ssl_effectiveness(self):
        """绘制SSL效果"""
        if not self.training_data['ssl_metrics']:
            return
            
        ssl_data = []
        for record in self.training_data['ssl_metrics']:
            ssl_data.append({
                'epoch': record['epoch'],
                **record['metrics']
            })
        
        df = pd.DataFrame(ssl_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # SSL损失趋势
        if 'contrastive_loss' in df.columns:
            axes[0,0].plot(df['epoch'], df['contrastive_loss'], 'r-', label='Contrastive', linewidth=2)
        if 'consistency_loss' in df.columns:
            axes[0,0].plot(df['epoch'], df['consistency_loss'], 'b-', label='Consistency', linewidth=2)
        if 'cross_modal_loss' in df.columns:
            axes[0,0].plot(df['epoch'], df['cross_modal_loss'], 'g-', label='Cross-Modal', linewidth=2)
        
        axes[0,0].set_title('SSL Loss Components')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.dirs['plots'] / 'ssl_effectiveness.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ SSL效果分析保存: {save_path}")
    
    def _plot_class_performance(self):
        """绘制类别性能"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        class_names = list(self.training_data['class_specific_performance'].keys())
        metrics = ['recall', 'precision', 'f1']
        
        for i, metric in enumerate(metrics):
            class_values = []
            for class_name in class_names:
                records = self.training_data['class_specific_performance'][class_name]
                if records:
                    final_value = records[-1]['metrics'].get(metric, 0)
                    class_values.append(final_value)
                else:
                    class_values.append(0)
            
            bars = axes[i].bar(class_names, class_values, alpha=0.7)
            axes[i].set_title(f'Final {metric.title()} by Class')
            axes[i].set_ylabel(metric.title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, class_values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = self.dirs['plots'] / 'class_performance.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 类别性能分析保存: {save_path}")
    
    def _plot_weather_impact(self):
        """绘制天气影响"""
        # 天气对不同类别的具体影响
        pass
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return "insufficient_data"
        
        # 简单线性趋势
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """计算改进率"""
        if len(values) < 2:
            return 0
        return (values[-1] - values[0]) / max(abs(values[0]), 1e-6)
    
    def _calculate_convergence_rate(self, values: List[float]) -> float:
        """计算收敛率"""
        if len(values) < 3:
            return 0
        
        # 计算最后几个值的方差
        recent_values = values[-3:]
        return float(np.std(recent_values))
    
    def _find_optimal_strategies(self) -> Dict:
        """寻找最优策略"""
        # 基于性能数据找到最优的增强策略组合
        optimal = {
            'best_weather_combo': {},
            'best_intensity_levels': {},
            'recommendations': []
        }
        
        # 简化版实现
        weather_performance = defaultdict(list)
        
        for epoch_data in self.training_data['epochs']:
            weather_dist = epoch_data.get('weather_distribution', {})
            overall_performance = epoch_data.get('detection_results', {}).get('overall_f1', 0)
            
            dominant_weather = max(weather_dist.items(), key=lambda x: x[1])[0] if weather_dist else 'clear'
            weather_performance[dominant_weather].append(overall_performance)
        
        # 找出表现最好的天气类型
        best_weather = None
        best_performance = 0
        
        for weather, performances in weather_performance.items():
            avg_performance = np.mean(performances)
            if avg_performance > best_performance:
                best_performance = avg_performance
                best_weather = weather
        
        optimal['best_weather_combo']['weather_type'] = best_weather
        optimal['best_weather_combo']['average_performance'] = best_performance
        
        return optimal
    
    def generate_comprehensive_report(self) -> str:
        """生成综合分析报告"""
        if not self.enabled:
            return "Analytics disabled"
        
        # 执行所有分析
        aug_analysis = self.analyze_augmentation_impact()
        ssl_analysis = self.analyze_ssl_effectiveness()
        class_analysis = self.analyze_class_specific_performance()
        
        # 生成报告
        report = f"""
# ROSE训练综合分析报告

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据概览
- 总训练轮次: {len(self.training_data['epochs'])}
- 增强策略记录: {len(self.training_data['augmentation_strategies'])}
- SSL指标记录: {len(self.training_data['ssl_metrics'])}

## 数据增强策略分析
"""
        
        if aug_analysis.get('weather_effectiveness'):
            report += "\n### 天气增强效果:\n"
            for weather, data in aug_analysis['weather_effectiveness'].items():
                avg_f1 = data['average_performance'].get('f1', 0)
                trend = data['trend']
                report += f"- {weather}: F1={avg_f1:.3f} (趋势: {trend})\n"
        
        if aug_analysis.get('optimal_strategies'):
            optimal = aug_analysis['optimal_strategies']
            if 'best_weather_combo' in optimal:
                best_combo = optimal['best_weather_combo']
                report += f"\n### 最优策略:\n"
                report += f"- 最佳天气类型: {best_combo.get('weather_type', 'N/A')}\n"
                report += f"- 平均性能: {best_combo.get('average_performance', 0):.3f}\n"
        
        report += "\n## SSL训练分析\n"
        if ssl_analysis.get('ssl_convergence'):
            convergence = ssl_analysis['ssl_convergence']
            report += f"- 对比损失趋势: {convergence.get('contrastive_trend', 'N/A')}\n"
            report += f"- 一致性损失趋势: {convergence.get('consistency_trend', 'N/A')}\n"
            report += f"- 收敛率: {convergence.get('convergence_rate', 0):.4f}\n"
        
        report += "\n## 类别性能分析\n"
        if class_analysis.get('class_trends'):
            for class_name, trends in class_analysis['class_trends'].items():
                final_perf = trends['final_performance']
                report += f"\n### {class_name}:\n"
                report += f"- 最终F1: {final_perf.get('f1', 0):.3f}\n"
                report += f"- 召回率趋势: {trends.get('recall_trend', 'N/A')}\n"
                report += f"- 精确率趋势: {trends.get('precision_trend', 'N/A')}\n"
        
        if class_analysis.get('problematic_classes'):
            report += "\n### 问题类别:\n"
            for prob_class in class_analysis['problematic_classes']:
                report += f"- {prob_class['class']}: F1={prob_class['final_f1']:.3f}\n"
        
        report += f"\n## 文件位置\n"
        report += f"- 性能图表: {self.dirs['plots']}\n"
        report += f"- 分析报告: {self.dirs['reports']}\n"
        report += f"- 收集数据: {self.dirs['data']}\n"
        
        # 保存报告
        report_path = self.dirs['reports'] / 'comprehensive_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 综合分析报告保存: {report_path}")
        
        return report
    
    def save_training_data(self):
        """保存训练数据"""
        if not self.enabled:
            return
            
        try:
            # 保存原始数据
            data_path = self.dirs['data'] / 'training_data.json'
            with open(data_path, 'w') as f:
                json.dump(self.training_data, f, indent=2, default=str)
            
            # 保存分析结果
            analyses = {
                'augmentation_analysis': self.analyze_augmentation_impact(),
                'ssl_analysis': self.analyze_ssl_effectiveness(),
                'class_analysis': self.analyze_class_specific_performance()
            }
            
            analysis_path = self.dirs['data'] / 'analysis_results.json'
            with open(analysis_path, 'w') as f:
                json.dump(analyses, f, indent=2, default=str)
            
            print(f"✅ 训练数据保存完成")
            
        except Exception as e:
            print(f"❌ 训练数据保存失败: {e}")