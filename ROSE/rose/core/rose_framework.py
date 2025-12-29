"""
ROSE (Roadside Oversight-guided Scenario Enhancement) 核心框架
路侧多模态感知的场景增强系统
"""

import os
import sys
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime

# MMDet3D imports
from mmdet3d.registry import MODELS
from mmengine import Config
from mmengine.registry import MODELS as MMENGINE_MODELS

class ROSEFramework:
    """ROSE框架主类"""
    
    def __init__(self, 
                 base_config_path: str,
                 work_dir: str,
                 data_root: str):
        """
        初始化ROSE框架
        
        Args:
            base_config_path: 基础mvxnet配置文件路径
            work_dir: 工作目录
            data_root: 数据集根目录
        """
        self.base_config_path = base_config_path
        self.work_dir = Path(work_dir)
        self.data_root = Path(data_root)
        
        # 创建工作目录结构
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.dirs = {
            'configs': self.work_dir / 'configs',
            'augmented_data': self.work_dir / 'augmented_data',
            'checkpoints': self.work_dir / 'checkpoints',
            'logs': self.work_dir / 'logs',
            'visualizations': self.work_dir / 'visualizations',
            'reports': self.work_dir / 'reports',
            'strategies': self.work_dir / 'augmentation_strategies'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.augmentation_engine = None
        self.ssl_trainer = None
        self.detection_model = None
        self.current_round = 0
        
        print(f"✅ ROSE框架初始化完成")
        print(f"   工作目录: {self.work_dir}")
        print(f"   数据根目录: {self.data_root}")
    
    def initialize_components(self):
        """初始化所有组件"""
        print("初始化ROSE框架组件...")
        
        # 1. 初始化数据增强引擎
        self._initialize_augmentation_engine()
        
        # 2. 初始化SSL训练器
        self._initialize_ssl_trainer()
        
        # 3. 初始化检测模型
        self._initialize_detection_model()
        
        print("✅ 所有组件初始化完成")
    
    def _initialize_augmentation_engine(self):
        """初始化数据增强引擎"""
        from rose.augmentation.multimodal_augmentor import MultiModalAugmentor
        
        self.augmentation_engine = MultiModalAugmentor(
            work_dir=str(self.work_dir),
            data_root=str(self.data_root)
        )
        print("✅ 数据增强引擎初始化完成")
    
    def _initialize_ssl_trainer(self):
        """初始化SSL训练器"""
        from rose.ssl.ssl_trainer import ROSESSLTrainer
        
        self.ssl_trainer = ROSESSLTrainer(
            work_dir=str(self.work_dir)
        )
        print("✅ SSL训练器初始化完成")
    
    def _initialize_detection_model(self):
        """初始化检测模型"""
        # 加载基础配置
        base_cfg = Config.fromfile(self.base_config_path)
        
        # 创建模型
        from mmdet3d.utils import register_all_modules
        register_all_modules()
        
        self.detection_model = MODELS.build(base_cfg.model)
        print("✅ 检测模型初始化完成")
    
    def run_training_round(self, round_num: int, epochs: int = 10):
        """
        运行一轮完整的训练
        
        Args:
            round_num: 轮次编号
            epochs: 训练epoch数
        """
        print(f"\n{'='*60}")
        print(f"开始第 {round_num} 轮ROSE训练")
        print(f"{'='*60}")
        
        self.current_round = round_num
        
        # 1. 生成增强策略
        strategy = self._generate_augmentation_strategy(round_num)
        
        # 2. 执行数据增强
        augmented_dataset = self._execute_data_augmentation(strategy)
        
        # 3. 创建训练配置
        train_config = self._create_training_config(strategy, epochs)
        
        # 4. 执行训练
        training_results = self._execute_training(train_config)
        
        # 5. 验证和可视化
        validation_results = self._execute_validation(training_results)
        
        # 6. 生成报告
        self._generate_round_report(round_num, strategy, training_results, validation_results)
        
        return {
            'round': round_num,
            'strategy': strategy,
            'training_results': training_results,
            'validation_results': validation_results
        }
    
    def _generate_augmentation_strategy(self, round_num: int) -> Dict:
        """生成增强策略"""
        print(f"生成第{round_num}轮增强策略...")
        
        if round_num == 1:
            # 第一轮使用默认策略
            strategy = {
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'weather_distribution': {
                    'clear': 0.4,
                    'rain_light': 0.2, 
                    'rain_heavy': 0.15,
                    'fog_light': 0.15,
                    'fog_heavy': 0.1
                },
                'augmentation_parameters': {
                    'rain_light': {'intensity': 0.3, 'rate': 3.0},
                    'rain_heavy': {'intensity': 0.6, 'rate': 8.0},
                    'fog_light': {'intensity': 0.2, 'visibility': 80.0},
                    'fog_heavy': {'intensity': 0.5, 'visibility': 30.0}
                },
                'ssl_parameters': {
                    'lambda_det': 1.0,
                    'lambda_cm': 0.5,
                    'lambda_cons': 0.3
                }
            }
        else:
            # 基于上一轮结果调整策略
            strategy = self._adapt_strategy_from_previous_round(round_num)
        
        # 保存策略文件
        strategy_file = self.dirs['strategies'] / f'round_{round_num}_strategy.yaml'
        with open(strategy_file, 'w') as f:
            yaml.dump(strategy, f, default_flow_style=False)
        
        print(f"✅ 策略已保存: {strategy_file}")
        return strategy
    
    def _adapt_strategy_from_previous_round(self, round_num: int) -> Dict:
        """基于上一轮结果调整增强策略"""
        # 读取上一轮验证结果
        prev_round = round_num - 1
        prev_results_file = self.dirs['reports'] / f'round_{prev_round}_results.json'
        
        if prev_results_file.exists():
            with open(prev_results_file, 'r') as f:
                prev_results = json.load(f)
            
            # 分析性能并调整策略
            validation_results = prev_results.get('validation_results', {})
            class_performance = validation_results.get('class_performance', {})
            
            # 基于类别性能调整增强强度
            strategy = {
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'adapted_from_round': prev_round,
                'weather_distribution': {},
                'augmentation_parameters': {},
                'ssl_parameters': {}
            }
            
            # 如果小目标性能较差，增加对应天气条件的训练
            pedestrian_map = class_performance.get('Pedestrian', {}).get('mAP', 0)
            cyclist_map = class_performance.get('Cyclist', {}).get('mAP', 0)
            
            if pedestrian_map < 0.3 or cyclist_map < 0.3:
                # 增加雾天和雨天训练（对小目标更有挑战性）
                strategy['weather_distribution'] = {
                    'clear': 0.3,
                    'rain_light': 0.25,
                    'rain_heavy': 0.2, 
                    'fog_light': 0.15,
                    'fog_heavy': 0.1
                }
                # 增加SSL权重
                strategy['ssl_parameters'] = {
                    'lambda_det': 1.0,
                    'lambda_cm': 0.7,  # 增加跨模态对比
                    'lambda_cons': 0.5  # 增加一致性约束
                }
            else:
                # 标准分布
                strategy['weather_distribution'] = {
                    'clear': 0.4,
                    'rain_light': 0.2,
                    'rain_heavy': 0.15,
                    'fog_light': 0.15,
                    'fog_heavy': 0.1
                }
                strategy['ssl_parameters'] = {
                    'lambda_det': 1.0,
                    'lambda_cm': 0.5,
                    'lambda_cons': 0.3
                }
            
            strategy['augmentation_parameters'] = {
                'rain_light': {'intensity': 0.3, 'rate': 3.0},
                'rain_heavy': {'intensity': 0.6, 'rate': 8.0}, 
                'fog_light': {'intensity': 0.2, 'visibility': 80.0},
                'fog_heavy': {'intensity': 0.5, 'visibility': 30.0}
            }
            
        else:
            # 如果没有上一轮结果，使用默认策略
            strategy = self._generate_augmentation_strategy(1)
            strategy['round'] = round_num
        
        return strategy
    
    def _execute_data_augmentation(self, strategy: Dict) -> str:
        """执行数据增强"""
        print("执行多模态数据增强...")
        
        augmented_dataset_dir = self.dirs['augmented_data'] / f'round_{self.current_round}'
        augmented_dataset_dir.mkdir(exist_ok=True)
        
        # 使用增强引擎处理数据
        self.augmentation_engine.process_dataset(
            strategy=strategy,
            output_dir=str(augmented_dataset_dir)
        )
        
        print(f"✅ 数据增强完成: {augmented_dataset_dir}")
        return str(augmented_dataset_dir)
    
    def _create_training_config(self, strategy: Dict, epochs: int) -> str:
        """创建训练配置文件"""
        print("创建训练配置...")
        
        # 加载基础配置
        base_cfg = Config.fromfile(self.base_config_path)
        
        # 修改配置以适应ROSE训练
        rose_config = self._adapt_config_for_rose(base_cfg, strategy, epochs)
        
        # 保存配置
        config_file = self.dirs['configs'] / f'rose_round_{self.current_round}_config.py'
        rose_config.dump(str(config_file))
        
        print(f"✅ 训练配置已生成: {config_file}")
        return str(config_file)
    
    def _adapt_config_for_rose(self, base_cfg: Config, strategy: Dict, epochs: int) -> Config:
        """调整配置以支持ROSE训练"""
        # 克隆基础配置
        cfg = base_cfg.copy()
        
        # 设置工作目录
        cfg.work_dir = str(self.dirs['checkpoints'] / f'round_{self.current_round}')
        
        # 设置训练轮数
        cfg.train_cfg.max_epochs = epochs
        
        # 修改数据集配置
        cfg.train_dataloader.dataset = dict(
            type='ROSEDataset',
            data_root=str(self.data_root),
            augmented_data_root=str(self.dirs['augmented_data'] / f'round_{self.current_round}'),
            ann_file='kitti_infos_train.pkl',
            pipeline=cfg.train_dataloader.dataset.pipeline,
            metainfo=cfg.train_dataloader.dataset.metainfo,
            box_type_3d='LiDAR',
            augmentation_strategy=strategy
        )
        
        # 添加SSL配置
        cfg.model.update({
            'ssl_config': strategy['ssl_parameters'],
            'enable_ssl': True
        })
        
        # 添加自定义钩子
        cfg.custom_hooks = [
            dict(
                type='ROSETrainingHook',
                work_dir=cfg.work_dir,
                augmentation_strategy=strategy,
                save_interval=100,
                visualization_interval=200
            )
        ]
        
        # 设置可视化
        cfg.default_hooks.visualization.update({
            'draw': True,
            'interval': 200,
            'test_out_dir': str(self.dirs['visualizations'] / f'round_{self.current_round}')
        })
        
        return cfg
    
    def _execute_training(self, train_config: str) -> Dict:
        """执行训练"""
        print("开始训练...")
        
        import subprocess
        
        # 构建训练命令
        cmd = [
            'python', 'tools/train.py',
            train_config,
            '--work-dir', str(self.dirs['checkpoints'] / f'round_{self.current_round}')
        ]
        
        # 执行训练
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/guoyu/mmdetection3d-1.2.0')
        
        if result.returncode == 0:
            print("✅ 训练完成")
            return {
                'status': 'success',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ 训练失败: {result.stderr}")
            return {
                'status': 'failed', 
                'stdout': result.stdout,
                'stderr': result.stderr
            }
    
    def _execute_validation(self, training_results: Dict) -> Dict:
        """执行验证"""
        print("执行模型验证...")
        
        checkpoint_dir = self.dirs['checkpoints'] / f'round_{self.current_round}'
        latest_checkpoint = checkpoint_dir / 'latest.pth'
        
        if not latest_checkpoint.exists():
            print("❌ 未找到训练权重文件")
            return {'status': 'failed', 'error': 'No checkpoint found'}
        
        # 执行验证
        import subprocess
        
        config_file = self.dirs['configs'] / f'rose_round_{self.current_round}_config.py'
        
        cmd = [
            'python', 'tools/test.py',
            str(config_file),
            str(latest_checkpoint),
            '--out', str(self.dirs['reports'] / f'round_{self.current_round}_predictions.pkl'),
            '--eval', 'mAP'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/guoyu/mmdetection3d-1.2.0')
        
        if result.returncode == 0:
            print("✅ 验证完成")
            
            # 解析验证结果
            validation_results = self._parse_validation_results(result.stdout)
            
            return {
                'status': 'success',
                'results': validation_results,
                'stdout': result.stdout
            }
        else:
            print(f"❌ 验证失败: {result.stderr}")
            return {
                'status': 'failed',
                'error': result.stderr
            }
    
    def _parse_validation_results(self, stdout: str) -> Dict:
        """解析验证结果"""
        # 从stdout中提取mAP结果
        results = {
            'class_performance': {},
            'overall_performance': {}
        }
        
        lines = stdout.split('\n')
        for line in lines:
            if 'Pedestrian' in line and 'mAP' in line:
                # 解析行人mAP
                parts = line.split()
                if len(parts) >= 2:
                    results['class_performance']['Pedestrian'] = {
                        'mAP': float(parts[-1]) if parts[-1].replace('.', '').isdigit() else 0.0
                    }
            elif 'Cyclist' in line and 'mAP' in line:
                # 解析骑行者mAP
                parts = line.split()
                if len(parts) >= 2:
                    results['class_performance']['Cyclist'] = {
                        'mAP': float(parts[-1]) if parts[-1].replace('.', '').isdigit() else 0.0
                    }
            elif 'Car' in line and 'mAP' in line:
                # 解析汽车mAP
                parts = line.split()
                if len(parts) >= 2:
                    results['class_performance']['Car'] = {
                        'mAP': float(parts[-1]) if parts[-1].replace('.', '').isdigit() else 0.0
                    }
        
        return results
    
    def _generate_round_report(self, round_num: int, strategy: Dict, 
                             training_results: Dict, validation_results: Dict):
        """生成轮次报告"""
        print("生成轮次报告...")
        
        report = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'training_results': training_results,
            'validation_results': validation_results,
            'summary': self._generate_summary(validation_results)
        }
        
        # 保存JSON报告
        report_file = self.dirs['reports'] / f'round_{round_num}_results.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 生成Markdown报告
        md_report = self._generate_markdown_report(report)
        md_file = self.dirs['reports'] / f'round_{round_num}_report.md'
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        print(f"✅ 报告已生成: {report_file}")
    
    def _generate_summary(self, validation_results: Dict) -> Dict:
        """生成性能总结"""
        if validation_results.get('status') != 'success':
            return {'status': 'failed'}
        
        results = validation_results.get('results', {})
        class_perf = results.get('class_performance', {})
        
        # 计算平均mAP
        maps = []
        for class_name, perf in class_perf.items():
            if 'mAP' in perf:
                maps.append(perf['mAP'])
        
        avg_map = np.mean(maps) if maps else 0.0
        
        return {
            'status': 'success',
            'average_mAP': avg_map,
            'class_count': len(class_perf),
            'best_class': max(class_perf.items(), key=lambda x: x[1].get('mAP', 0))[0] if class_perf else 'None',
            'worst_class': min(class_perf.items(), key=lambda x: x[1].get('mAP', 0))[0] if class_perf else 'None'
        }
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """生成Markdown报告"""
        round_num = report['round']
        timestamp = report['timestamp']
        strategy = report['strategy']
        validation = report['validation_results']
        summary = report['summary']
        
        md = f"""# ROSE训练第{round_num}轮报告

## 基本信息
- **轮次**: {round_num}
- **时间**: {timestamp}
- **状态**: {'成功' if summary.get('status') == 'success' else '失败'}

## 增强策略
### 天气分布
"""
        
        for weather, prob in strategy.get('weather_distribution', {}).items():
            md += f"- {weather}: {prob:.1%}\n"
        
        md += "\n### SSL参数\n"
        ssl_params = strategy.get('ssl_parameters', {})
        for param, value in ssl_params.items():
            md += f"- {param}: {value}\n"
        
        md += "\n## 验证结果\n"
        
        if validation.get('status') == 'success':
            results = validation.get('results', {})
            class_perf = results.get('class_performance', {})
            
            md += f"- **平均mAP**: {summary.get('average_mAP', 0):.3f}\n\n"
            md += "### 类别性能\n"
            
            for class_name, perf in class_perf.items():
                map_score = perf.get('mAP', 0)
                md += f"- **{class_name}**: {map_score:.3f}\n"
        else:
            md += "验证失败\n"
        
        md += f"""

## 文件位置
- **配置文件**: `{self.dirs['configs']}/rose_round_{round_num}_config.py`
- **权重文件**: `{self.dirs['checkpoints']}/round_{round_num}/`
- **可视化结果**: `{self.dirs['visualizations']}/round_{round_num}/`
- **增强数据**: `{self.dirs['augmented_data']}/round_{round_num}/`

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return md
    
    def run_complete_training(self, total_rounds: int = 3, epochs_per_round: int = 10):
        """运行完整的多轮训练"""
        print(f"\n{'='*80}")
        print(f"ROSE框架 - 完整训练流程")
        print(f"总轮数: {total_rounds}, 每轮epoch: {epochs_per_round}")
        print(f"{'='*80}")
        
        all_results = []
        
        for round_num in range(1, total_rounds + 1):
            try:
                round_results = self.run_training_round(round_num, epochs_per_round)
                all_results.append(round_results)
                
                print(f"✅ 第{round_num}轮训练完成")
                
                # 简短总结
                summary = round_results.get('validation_results', {}).get('results', {})
                if summary:
                    class_perf = summary.get('class_performance', {})
                    for class_name, perf in class_perf.items():
                        map_score = perf.get('mAP', 0)
                        print(f"   {class_name}: {map_score:.3f}")
                
            except Exception as e:
                print(f"❌ 第{round_num}轮训练失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 生成最终总结报告
        self._generate_final_report(all_results)
        
        return all_results
    
    def _generate_final_report(self, all_results: List[Dict]):
        """生成最终总结报告"""
        print("生成最终总结报告...")
        
        final_report = {
            'framework': 'ROSE',
            'timestamp': datetime.now().isoformat(),
            'total_rounds': len(all_results),
            'results': all_results,
            'performance_trend': self._analyze_performance_trend(all_results),
            'best_round': self._find_best_round(all_results)
        }
        
        # 保存最终报告
        final_report_file = self.dirs['reports'] / 'ROSE_FINAL_REPORT.json'
        with open(final_report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # 生成最终Markdown报告
        final_md = self._generate_final_markdown_report(final_report)
        final_md_file = self.dirs['reports'] / 'ROSE_FINAL_REPORT.md'
        with open(final_md_file, 'w', encoding='utf-8') as f:
            f.write(final_md)
        
        print(f"✅ 最终报告已生成: {final_report_file}")
        
        return final_report
    
    def _analyze_performance_trend(self, all_results: List[Dict]) -> Dict:
        """分析性能趋势"""
        trend = {'rounds': [], 'avg_mAP': [], 'class_performance': {}}
        
        for result in all_results:
            round_num = result['round']
            validation = result.get('validation_results', {})
            
            if validation.get('status') == 'success':
                results = validation.get('results', {})
                class_perf = results.get('class_performance', {})
                
                # 计算平均mAP
                maps = [perf.get('mAP', 0) for perf in class_perf.values()]
                avg_map = np.mean(maps) if maps else 0
                
                trend['rounds'].append(round_num)
                trend['avg_mAP'].append(avg_map)
                
                # 记录各类性能
                for class_name, perf in class_perf.items():
                    if class_name not in trend['class_performance']:
                        trend['class_performance'][class_name] = []
                    trend['class_performance'][class_name].append(perf.get('mAP', 0))
        
        return trend
    
    def _find_best_round(self, all_results: List[Dict]) -> Dict:
        """找到最佳轮次"""
        best_round = None
        best_map = 0
        
        for result in all_results:
            validation = result.get('validation_results', {})
            if validation.get('status') == 'success':
                results = validation.get('results', {})
                class_perf = results.get('class_performance', {})
                
                maps = [perf.get('mAP', 0) for perf in class_perf.values()]
                avg_map = np.mean(maps) if maps else 0
                
                if avg_map > best_map:
                    best_map = avg_map
                    best_round = result
        
        return best_round if best_round else {}
    
    def _generate_final_markdown_report(self, final_report: Dict) -> str:
        """生成最终Markdown报告"""
        total_rounds = final_report['total_rounds']
        timestamp = final_report['timestamp']
        trend = final_report.get('performance_trend', {})
        best_round = final_report.get('best_round', {})
        
        md = f"""# ROSE框架完整训练报告

## 训练概况
- **框架**: ROSE (Roadside Oversight-guided Scenario Enhancement)
- **完成时间**: {timestamp}
- **总轮数**: {total_rounds}
- **数据集**: DAIR-V2X-I

## 性能趋势
"""
        
        if trend.get('avg_mAP'):
            md += "### 平均mAP趋势\n"
            for round_num, avg_map in zip(trend['rounds'], trend['avg_mAP']):
                md += f"- 第{round_num}轮: {avg_map:.3f}\n"
            
            md += "\n### 各类别性能趋势\n"
            for class_name, class_maps in trend.get('class_performance', {}).items():
                md += f"\n**{class_name}**:\n"
                for round_num, class_map in zip(trend['rounds'], class_maps):
                    md += f"- 第{round_num}轮: {class_map:.3f}\n"
        
        md += "\n## 最佳轮次\n"
        if best_round:
            best_round_num = best_round.get('round', 0)
            best_validation = best_round.get('validation_results', {})
            best_results = best_validation.get('results', {})
            best_class_perf = best_results.get('class_performance', {})
            
            md += f"- **最佳轮次**: 第{best_round_num}轮\n"
            md += f"- **性能表现**:\n"
            
            for class_name, perf in best_class_perf.items():
                map_score = perf.get('mAP', 0)
                md += f"  - {class_name}: {map_score:.3f}\n"
        
        md += f"""

## 输出文件说明

### 每轮训练输出
- **配置文件**: `configs/rose_round_X_config.py`
- **训练权重**: `checkpoints/round_X/`
- **增强数据**: `augmented_data/round_X/`
- **可视化结果**: `visualizations/round_X/`
- **轮次报告**: `reports/round_X_report.md`

### 增强策略文件
- **策略配置**: `augmentation_strategies/round_X_strategy.yaml`

### 检测结果
- **mAP性能**: 每轮报告中包含三类目标(Pedestrian, Cyclist, Car)的mAP
- **3D检测框**: 可视化目录中包含绘制3D检测框的图像和点云
- **增强数据可视化**: 增强前后的图像和点云对比

## 技术特性

### 数据增强
- ✅ 物理一致性的图像-点云同步增强
- ✅ 多种天气类型(雨、雾、雪)模拟
- ✅ 基于LISA的点云物理建模
- ✅ 增强策略的自适应调整

### SSL训练
- ✅ 跨模态对比学习
- ✅ 师生一致性约束
- ✅ 动态损失权重调整

### 3D检测
- ✅ 基于MVXNet的多模态融合
- ✅ 针对DAIR-V2X数据集优化
- ✅ 完整的训练验证流程

---
*ROSE框架 v1.0 - 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return md