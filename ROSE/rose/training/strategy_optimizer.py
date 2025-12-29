"""
Strategy Optimizer for ROSE Training
Analyzes performance and optimizes augmentation strategies
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import mmcv
from copy import deepcopy


class ROSEStrategyOptimizer:
    """Optimize augmentation strategies based on performance analysis"""
    
    def __init__(self, 
                 work_dir: str,
                 optimization_config: Optional[Dict] = None):
        """
        Initialize strategy optimizer
        
        Args:
            work_dir: Working directory containing evaluation results
            optimization_config: Configuration for optimization behavior
        """
        self.work_dir = work_dir
        self.eval_dir = os.path.join(work_dir, 'evaluations')
        self.strategy_dir = os.path.join(work_dir, 'augmentation_strategies')
        
        # Default optimization configuration
        self.optimization_config = optimization_config or {
            'performance_threshold': 0.65,  # Target overall mAP
            'improvement_threshold': 0.01,  # Minimum improvement to consider significant
            'problematic_class_threshold': 0.5,  # Threshold for problematic classes
            'intensity_adjustment_step': 0.05,  # Step size for intensity adjustments
            'probability_adjustment_step': 0.05,  # Step size for probability adjustments
            'max_intensity': 1.0,  # Maximum augmentation intensity
            'min_intensity': 0.1,  # Minimum augmentation intensity
            'focus_on_weak_classes': True,  # Whether to focus on weak performing classes
        }
        
        # Strategy evolution tracking
        self.optimization_history = []
        
    def load_evaluation_history(self) -> List[Dict]:
        """Load evaluation history from previous rounds"""
        history_path = os.path.join(self.eval_dir, 'evaluation_history.json')
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        return []
    
    def load_strategy_history(self) -> List[Dict]:
        """Load strategy history from previous rounds"""
        history_path = os.path.join(self.strategy_dir, 'strategy_history.json')
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        return []
    
    def analyze_performance_trends(self, evaluation_history: List[Dict]) -> Dict[str, Any]:
        """
        Analyze performance trends across training rounds
        
        Args:
            evaluation_history: List of evaluation results from previous rounds
            
        Returns:
            Dictionary containing trend analysis
        """
        if len(evaluation_history) < 1:
            return {
                'trend': 'insufficient_data', 
                'rounds': 0,
                'current_performance': {
                    'overall': 0.0,
                    'car': 0.0,
                    'pedestrian': 0.0,
                    'cyclist': 0.0,
                },
                'problematic_classes': ['car', 'pedestrian', 'cyclist']
            }
        
        # Extract performance metrics
        rounds = []
        overall_map = []
        car_map = []
        pedestrian_map = []
        cyclist_map = []
        
        for eval_result in evaluation_history:
            test_metrics = eval_result.get('test', {})
            rounds.append(eval_result.get('round', 0))
            overall_map.append(test_metrics.get('overall_3d_moderate', 0.0))
            car_map.append(test_metrics.get('car_3d_moderate', 0.0))
            pedestrian_map.append(test_metrics.get('pedestrian_3d_moderate', 0.0))
            cyclist_map.append(test_metrics.get('cyclist_3d_moderate', 0.0))
        
        # Calculate trends
        def calculate_trend(values):
            if len(values) < 2:
                return 0.0
            return (values[-1] - values[0]) / len(values)
        
        analysis = {
            'rounds_analyzed': len(rounds),
            'current_performance': {
                'overall': overall_map[-1] if overall_map else 0.0,
                'car': car_map[-1] if car_map else 0.0,
                'pedestrian': pedestrian_map[-1] if pedestrian_map else 0.0,
                'cyclist': cyclist_map[-1] if cyclist_map else 0.0,
            },
            'trends': {
                'overall': calculate_trend(overall_map),
                'car': calculate_trend(car_map),
                'pedestrian': calculate_trend(pedestrian_map),
                'cyclist': calculate_trend(cyclist_map),
            },
            'best_performance': {
                'overall': max(overall_map) if overall_map else 0.0,
                'car': max(car_map) if car_map else 0.0,
                'pedestrian': max(pedestrian_map) if pedestrian_map else 0.0,
                'cyclist': max(cyclist_map) if cyclist_map else 0.0,
            },
            'performance_stability': {
                'overall': np.std(overall_map) if len(overall_map) > 1 else 0.0,
                'car': np.std(car_map) if len(car_map) > 1 else 0.0,
                'pedestrian': np.std(pedestrian_map) if len(pedestrian_map) > 1 else 0.0,
                'cyclist': np.std(cyclist_map) if len(cyclist_map) > 1 else 0.0,
            }
        }
        
        # Identify problematic classes
        threshold = self.optimization_config['problematic_class_threshold']
        analysis['problematic_classes'] = []
        
        for class_name in ['car', 'pedestrian', 'cyclist']:
            if analysis['current_performance'][class_name] < threshold:
                analysis['problematic_classes'].append(class_name)
        
        return analysis
    
    def identify_effective_strategies(self, 
                                    evaluation_history: List[Dict],
                                    strategy_history: List[Dict]) -> Dict[str, Any]:
        """
        Identify most effective augmentation strategies
        
        Args:
            evaluation_history: Performance results
            strategy_history: Strategy configurations used
            
        Returns:
            Analysis of strategy effectiveness
        """
        if len(evaluation_history) != len(strategy_history):
            print("Warning: Evaluation and strategy histories have different lengths")
            min_len = min(len(evaluation_history), len(strategy_history))
            evaluation_history = evaluation_history[:min_len]
            strategy_history = strategy_history[:min_len]
        
        strategy_effectiveness = []
        
        for eval_result, strategy_result in zip(evaluation_history, strategy_history):
            test_metrics = eval_result.get('test', {})
            strategy_config = strategy_result.get('strategy', {})
            
            effectiveness = {
                'round': eval_result['round'],
                'performance': test_metrics.get('overall_3d_moderate', 0.0),
                'car_performance': test_metrics.get('car_3d_moderate', 0.0),
                'pedestrian_performance': test_metrics.get('pedestrian_3d_moderate', 0.0),
                'cyclist_performance': test_metrics.get('cyclist_3d_moderate', 0.0),
                'weather_configs': strategy_config.get('weather_configs', []),
                'weather_probabilities': strategy_config.get('weather_probabilities', []),
                'adaptation_enabled': strategy_config.get('adaptation_enabled', False)
            }
            strategy_effectiveness.append(effectiveness)
        
        # Find best performing strategies
        best_overall = max(strategy_effectiveness, key=lambda x: x['performance'])
        best_car = max(strategy_effectiveness, key=lambda x: x['car_performance'])
        best_pedestrian = max(strategy_effectiveness, key=lambda x: x['pedestrian_performance'])
        best_cyclist = max(strategy_effectiveness, key=lambda x: x['cyclist_performance'])
        
        return {
            'strategy_effectiveness': strategy_effectiveness,
            'best_strategies': {
                'overall': best_overall,
                'car': best_car,
                'pedestrian': best_pedestrian,
                'cyclist': best_cyclist
            }
        }
    
    def generate_optimization_recommendations(self, 
                                            performance_analysis: Dict,
                                            strategy_effectiveness: Dict) -> Dict[str, Any]:
        """
        Generate specific optimization recommendations
        
        Args:
            performance_analysis: Results from performance trend analysis
            strategy_effectiveness: Results from strategy effectiveness analysis
            
        Returns:
            Dictionary of specific optimization recommendations
        """
        recommendations = {
            'overall_strategy': 'maintain',  # maintain, increase, decrease, diversify
            'specific_adjustments': [],
            'focus_areas': [],
            'reasoning': []
        }
        
        current_perf = performance_analysis['current_performance']
        trends = performance_analysis['trends']
        problematic_classes = performance_analysis['problematic_classes']
        
        # Overall strategy recommendation
        if current_perf['overall'] < self.optimization_config['performance_threshold']:
            if trends['overall'] > 0:
                recommendations['overall_strategy'] = 'increase'
                recommendations['reasoning'].append("Performance below threshold but improving - increase augmentation")
            else:
                recommendations['overall_strategy'] = 'diversify'
                recommendations['reasoning'].append("Performance below threshold and declining - diversify augmentation")
        else:
            if trends['overall'] < -self.optimization_config['improvement_threshold']:
                recommendations['overall_strategy'] = 'decrease'
                recommendations['reasoning'].append("Good performance but declining - reduce augmentation intensity")
            else:
                recommendations['overall_strategy'] = 'maintain'
                recommendations['reasoning'].append("Good performance and stable - maintain current strategy")
        
        # Class-specific recommendations
        for class_name in problematic_classes:
            recommendations['focus_areas'].append(class_name)
            
            if class_name == 'pedestrian':
                recommendations['specific_adjustments'].append({
                    'target': 'pedestrian_detection',
                    'action': 'increase_small_object_augmentation',
                    'details': {
                        'increase_rain_intensity': 0.1,
                        'reduce_fog_intensity': 0.05,  # Fog heavily impacts small objects
                        'increase_augmentation_probability': 0.1
                    }
                })
            elif class_name == 'cyclist':
                recommendations['specific_adjustments'].append({
                    'target': 'cyclist_detection',
                    'action': 'increase_motion_blur_resistance',
                    'details': {
                        'increase_snow_augmentation': 0.1,
                        'add_motion_artifacts': True,
                        'increase_augmentation_probability': 0.1
                    }
                })
            elif class_name == 'car':
                recommendations['specific_adjustments'].append({
                    'target': 'car_detection',
                    'action': 'increase_occlusion_handling',
                    'details': {
                        'increase_fog_intensity': 0.1,
                        'add_partial_occlusion': True
                    }
                })
        
        # Weather-specific recommendations based on best strategies
        best_overall_strategy = strategy_effectiveness['best_strategies']['overall']
        best_weather_configs = best_overall_strategy.get('weather_configs', [])
        
        if best_weather_configs:
            recommendations['specific_adjustments'].append({
                'target': 'weather_distribution',
                'action': 'adopt_best_weather_mix',
                'details': {
                    'recommended_configs': best_weather_configs[:3],  # Top 3 configurations
                    'recommended_probabilities': best_overall_strategy.get('weather_probabilities', [])
                }
            })
        
        return recommendations
    
    def apply_optimization_strategy(self, 
                                  current_strategy: Dict,
                                  recommendations: Dict) -> Dict:
        """
        Apply optimization recommendations to generate new strategy
        
        Args:
            current_strategy: Current augmentation strategy
            recommendations: Optimization recommendations
            
        Returns:
            Optimized strategy configuration
        """
        optimized_strategy = deepcopy(current_strategy)
        
        # Apply overall strategy changes
        overall_strategy = recommendations['overall_strategy']
        intensity_step = self.optimization_config['intensity_adjustment_step']
        prob_step = self.optimization_config['probability_adjustment_step']
        
        weather_configs = optimized_strategy.get('weather_configs', [])
        weather_probs = optimized_strategy.get('weather_probabilities', [])
        
        if overall_strategy == 'increase':
            # Increase augmentation intensity and probability
            for config in weather_configs:
                if config['weather_type'] != 'clear':
                    config['intensity'] = min(
                        config['intensity'] + intensity_step,
                        self.optimization_config['max_intensity']
                    )
            
            # Increase non-clear weather probabilities
            if weather_probs:
                clear_prob = weather_probs[0]  # Assuming first is clear
                remaining_prob = sum(weather_probs[1:])
                
                # Redistribute probability from clear to augmented
                new_clear_prob = max(clear_prob - prob_step, 0.2)  # Minimum 20% clear
                prob_increase = (clear_prob - new_clear_prob) / len(weather_probs[1:])
                
                weather_probs[0] = new_clear_prob
                for i in range(1, len(weather_probs)):
                    weather_probs[i] += prob_increase
                    
        elif overall_strategy == 'decrease':
            # Decrease augmentation intensity
            for config in weather_configs:
                if config['weather_type'] != 'clear':
                    config['intensity'] = max(
                        config['intensity'] - intensity_step,
                        self.optimization_config['min_intensity']
                    )
                    
        elif overall_strategy == 'diversify':
            # Add new weather types or adjust distribution
            if len(weather_configs) < 6:  # Add more weather variations
                new_weather_config = {
                    'weather_type': 'mixed',
                    'intensity': 0.3,
                    'brightness_factor': 0.8,
                    'contrast_factor': 0.9,
                    'noise_level': 0.02,
                    'blur_kernel_size': 1
                }
                weather_configs.append(new_weather_config)
                weather_probs.append(0.1)
                
                # Renormalize probabilities
                total_prob = sum(weather_probs)
                weather_probs = [p / total_prob for p in weather_probs]
        
        # Apply specific adjustments
        for adjustment in recommendations['specific_adjustments']:
            target = adjustment['target']
            details = adjustment['details']
            
            if 'increase_rain_intensity' in details:
                for config in weather_configs:
                    if config['weather_type'] == 'rain':
                        config['intensity'] = min(
                            config['intensity'] + details['increase_rain_intensity'],
                            self.optimization_config['max_intensity']
                        )
            
            if 'reduce_fog_intensity' in details:
                for config in weather_configs:
                    if config['weather_type'] == 'fog':
                        config['intensity'] = max(
                            config['intensity'] - details['reduce_fog_intensity'],
                            self.optimization_config['min_intensity']
                        )
            
            if 'increase_augmentation_probability' in details:
                if weather_probs and weather_probs[0] > 0.2:  # Ensure minimum clear probability
                    prob_decrease = details['increase_augmentation_probability']
                    weather_probs[0] = max(weather_probs[0] - prob_decrease, 0.2)
                    
                    # Distribute to other weather types
                    prob_increase = prob_decrease / (len(weather_probs) - 1)
                    for i in range(1, len(weather_probs)):
                        weather_probs[i] += prob_increase
        
        # Update strategy
        optimized_strategy['weather_configs'] = weather_configs
        optimized_strategy['weather_probabilities'] = weather_probs
        
        return optimized_strategy
    
    def optimize_strategy(self, current_round: int) -> Dict[str, Any]:
        """
        Complete strategy optimization for next round
        
        Args:
            current_round: Current training round number
            
        Returns:
            Optimization results and new strategy
        """
        print(f"\n{'='*60}")
        print(f"Strategy Optimization for Round {current_round + 1}")
        print(f"{'='*60}")
        
        # Load historical data
        evaluation_history = self.load_evaluation_history()
        strategy_history = self.load_strategy_history()
        
        if len(evaluation_history) < 1:
            print("No evaluation history available - using default strategy")
            return self._get_default_strategy()
        
        # Analyze performance trends
        performance_analysis = self.analyze_performance_trends(evaluation_history)
        
        # Identify effective strategies
        strategy_effectiveness = self.identify_effective_strategies(
            evaluation_history, strategy_history)
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations(
            performance_analysis, strategy_effectiveness)
        
        # Get current strategy (latest from history)
        current_strategy = strategy_history[-1]['strategy'] if strategy_history else self._get_default_strategy()
        
        # Apply optimizations
        optimized_strategy = self.apply_optimization_strategy(current_strategy, recommendations)
        
        # Create optimization summary
        optimization_result = {
            'current_round': current_round,
            'next_round': current_round + 1,
            'performance_analysis': performance_analysis,
            'recommendations': recommendations,
            'current_strategy': current_strategy,
            'optimized_strategy': optimized_strategy,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save optimization result
        optimization_path = os.path.join(
            self.work_dir, 'optimization_results', f'optimization_round_{current_round + 1}.json')
        os.makedirs(os.path.dirname(optimization_path), exist_ok=True)
        
        with open(optimization_path, 'w') as f:
            json.dump(optimization_result, f, indent=2)
        
        # Update optimization history
        self.optimization_history.append(optimization_result)
        
        # Print summary
        self._print_optimization_summary(optimization_result)
        
        return optimization_result
    
    def _get_default_strategy(self) -> Dict:
        """Get default augmentation strategy"""
        return {
            'epoch': 0,
            'total_epochs': 10,
            'weather_configs': [
                {'weather_type': 'clear', 'intensity': 0.0},
                {'weather_type': 'rain', 'intensity': 0.3, 'rain_rate': 5.0},
                {'weather_type': 'fog', 'intensity': 0.2, 'visibility_range': 50.0},
                {'weather_type': 'snow', 'intensity': 0.2, 'snow_rate': 3.0}
            ],
            'weather_probabilities': [0.4, 0.3, 0.2, 0.1],
            'adaptation_enabled': True
        }
    
    def _print_optimization_summary(self, optimization_result: Dict):
        """Print optimization summary"""
        perf_analysis = optimization_result['performance_analysis']
        recommendations = optimization_result['recommendations']
        
        print("\nPerformance Analysis:")
        current_perf = perf_analysis['current_performance']
        print(f"  Overall mAP: {current_perf['overall']:.4f}")
        print(f"  Car mAP: {current_perf['car']:.4f}")
        print(f"  Pedestrian mAP: {current_perf['pedestrian']:.4f}")
        print(f"  Cyclist mAP: {current_perf['cyclist']:.4f}")
        
        if perf_analysis['problematic_classes']:
            print(f"  Problematic classes: {', '.join(perf_analysis['problematic_classes'])}")
        
        print(f"\nOptimization Strategy: {recommendations['overall_strategy']}")
        print("Reasoning:")
        for reason in recommendations['reasoning']:
            print(f"  - {reason}")
        
        if recommendations['focus_areas']:
            print(f"Focus areas: {', '.join(recommendations['focus_areas'])}")
        
        print(f"Strategy optimization completed for round {optimization_result['next_round']}")