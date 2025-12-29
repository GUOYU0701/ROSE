# ROSE: Roadside Oversight-guided Scenario Enhancement

ROSE is an advanced **roadside multi-modal 3D object detection framework** that integrates **physically consistent data augmentation**, **self-supervised learning (SSL)**, and an **intelligent training analytics system**, with a dedicated focus on improving detection performance for **Pedestrians** and **Cyclists**.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org)
[![MMDetection3D](https://img.shields.io/badge/MMDetection3D-1.2.0-green.svg)](https://github.com/open-mmlab/mmdetection3d)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Highlights

### 🎯 Enhanced 3D Object Detection

* **Task-specialized optimization**: tailored for **Pedestrian** and **Cyclist** detection.
* **Multi-modal fusion**: deep fusion architecture for **images + point clouds**.
* **Intelligent diagnostics**: real-time performance monitoring and failure analysis.

### 🌦️ Physically Consistent Data Augmentation

* **LISA integration**: LiDAR weather simulation based on **Mie scattering theory**.
* **Image augmentation**: physics-inspired weather effects (**rain / snow / fog**).
* **Adaptive scheduling**: automatically adjusts augmentation strength according to model performance.

### 🔗 Advanced Self-Supervised Learning (SSL)

* **Cross-modal learning**: image–point-cloud feature alignment and contrastive learning.
* **Class-aware objectives**: enhanced SSL losses designed for small/rare classes.
* **Teacher–student training**: an EMA teacher provides stable supervision.

### 📊 Intelligent Training Analytics

* **Real-time monitoring**: comprehensive statistics, logging, and visualization.
* **Failure diagnosis**: automatic discovery of common failure patterns.
* **Actionable suggestions**: AI-driven recommendations for performance improvement.

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/GUOYU0701/ROSE.git
cd ROSE

# 2. Create a conda environment
conda create -n rose python=3.8
conda activate rose

# 3. Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mmengine==0.8.4 mmcv==2.0.1 mmdet==3.1.0 mmdet3d==1.2.0
pip install numpy matplotlib scipy opencv-python pillow pyyaml pandas seaborn

# 4. Build the LISA library
cd LISA-main/pylisa && ./build.sh && cd ../..

# 5. Verify installation
python scripts/verify_installation.py
```

For detailed installation notes, please refer to [INSTALL.md](INSTALL.md).

---

## 📦 Dataset Preparation

```bash
# DAIR-V2X directory structure
/path/to/data/DAIR-V2X/
├── training/
│   ├── image_2/           # RGB images
│   ├── velodyne_reduced/  # point clouds
│   ├── label_2/           # 3D annotations
│   └── calib/             # calibration files
├── ImageSets/             # dataset splits
└── kitti_infos_*.pkl      # preprocessed metadata
```

---

## 💡 Usage

### 🎯 Recommended Training Pipelines (All Known Issues Fixed)

```bash
# 1. Pedestrian & Cyclist focused training (recommended)
python scripts/train_pedestrian_cyclist_focused.py \
    configs/rose_pedestrian_cyclist_optimized.py \
    --work-dir work_dirs/pedestrian_cyclist_focused \
    --validate

# 2. Training with full visualization enabled
python scripts/train_rose.py configs/rose_with_visualization.py \
    --work-dir work_dirs/rose_with_viz

# 3. Standard training pipeline (SSL fixed)
python scripts/train_rose.py configs/rose_mvxnet_dair_v2x.py \
    --work-dir work_dirs/rose_standard
```

### 📊 Analytics-Enhanced Training

```bash
# Run the enhanced training + analytics demo
python scripts/enhanced_training_example.py

# Generate a detailed training analytics report
python -c "
from rose.training.training_analytics import ROSETrainingAnalytics
analytics = ROSETrainingAnalytics('work_dirs/rose_enhanced')
report = analytics.generate_training_report()
analytics.save_analytics_report()
analytics.create_visualizations()
print('📊 Analytics report saved under analytics/')
"
```

### 🧪 Testing & Evaluation

```bash
# Basic evaluation
python scripts/test_rose.py configs/rose_mvxnet_dair_v2x.py \
    work_dirs/rose_enhanced/best_model.pth \
    --work-dir test_results

# Weather robustness evaluation
python scripts/test_rose.py configs/rose_mvxnet_dair_v2x.py \
    work_dirs/rose_enhanced/best_model.pth \
    --test-augmentation --weather-type rain \
    --weather-intensity 0.5 --visualize

# Detailed evaluation and analysis
python scripts/evaluate_rose.py \
    --config configs/rose_mvxnet_dair_v2x.py \
    --checkpoint work_dirs/rose_enhanced/best_model.pth \
    --detailed-analysis
```

---

## ⚙️ Advanced Configuration

### Training Configuration (Pedestrian/Cyclist Optimized)

```python
enhanced_ssl_config = dict(
    lambda_det=1.0,                 # base detection loss weight
    lambda_cm=0.6,                  # ⬆️ stronger cross-modal learning
    lambda_cons=0.4,                # ⬆️ higher small-object consistency
    lambda_spatial=0.3,             # ⬆️ stronger spatial relation learning
    lambda_weather=0.5,             # ⬆️ weather adaptation
    ema_decay=0.999,                # teacher model stability
    consistency_warmup_epochs=3,    # start consistency earlier
    enable_pseudo_labeling=True     # pseudo-label assisted training
)

adaptive_weather_config = dict(
    weather_configs=[
        dict(weather_type='rain', intensity=0.4, rain_rate=6.0),
        dict(weather_type='snow', intensity=0.3, rain_rate=4.0),
        dict(weather_type='fog', intensity=0.5, fog_type='moderate_advection_fog'),
        dict(weather_type='clear', intensity=0.0)
    ],
    weather_probabilities=[0.3, 0.25, 0.25, 0.2],
    adaptation_enabled=True,
    performance_threshold=0.6,
    total_epochs=80
)
```

### Analytics System Configuration

```python
analytics_config = dict(
    enabled=True,
    real_time_monitoring=True,
    class_focus=['Pedestrian', 'Cyclist'],
    failure_pattern_detection=True,
    improvement_suggestions=True,
    visualization_frequency=10,
    report_generation=True
)
```

---

## 🏗️ Project Structure

```text
ROSE/
├── rose/                                  # Core ROSE framework
│   ├── augmentation/                      # Intelligent augmentation
│   │   ├── weather_augmentor.py           # Multi-modal weather augmentor
│   │   ├── image_augment.py               # Physics-inspired image augmentation
│   │   ├── point_cloud_augment.py         # LISA-integrated point cloud augmentation
│   │   └── config.py                      # Adaptive augmentation configs
│   ├── ssl_training/                      # Advanced SSL training
│   │   ├── ssl_trainer.py                 # Enhanced SSL coordinator ⭐
│   │   ├── contrastive_loss.py            # Cross-modal contrastive learning
│   │   ├── consistency_loss.py            # Teacher–student consistency learning
│   │   └── ema_teacher.py                 # EMA teacher module
│   ├── training/                          # Training pipeline
│   │   ├── rose_trainer.py                # Main training orchestrator
│   │   ├── rose_detector.py               # Enhanced 3D detector
│   │   ├── rose_dataset.py                # Multi-modal dataset wrapper
│   │   ├── training_hooks.py              # Monitoring hooks
│   │   └── training_analytics.py          # Training analytics system ⭐
│   └── visualization/                     # Visualization
│       ├── visualizer.py                  # Unified visualization interface
│       ├── detection_visualizer.py        # Failure diagnosis visualization ⭐
│       └── augmentation_visualizer.py     # Augmentation effect analysis
├── configs/                               # Training configurations
│   ├── rose_mvxnet_dair_v2x.py
│   ├── rose_pedestrian_cyclist_optimized.py
│   ├── rose_with_visualization.py
│   ├── rose_full_training.py
│   └── rose_enhanced_adaptive.py
├── LISA-main/                             # Physically based scattering augmentation
│   ├── pylisa/                            # Python interface
│   │   ├── lisa.py
│   │   ├── mie_wrapper.py
│   │   └── atmos_models.py
│   └── tests/
├── scripts/                               # Entry scripts
│   ├── train_rose.py
│   ├── train_pedestrian_cyclist_focused.py   # Focus training ⭐
│   ├── test_rose.py
│   ├── evaluate_rose.py
│   ├── demo_augmentation_visualization.py    # Augmentation demo ⭐
│   ├── enhanced_training_example.py          # End-to-end demo ⭐
│   ├── verify_installation.py
│   └── analyze_dataset.py
├── README.md
├── INSTALL.md
├── LICENSE
└── requirements.txt
```

---

## 🤝 Contributing

We welcome contributions in all forms!

### Development Setup

```bash
pip install pre-commit black flake8 isort pytest
pre-commit install
```

---

## 📚 Resources

* [INSTALL.md](INSTALL.md) — Detailed installation guide
* [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) — Execution guide
* [Training tutorial](scripts/enhanced_training_example.py) — Full-feature demo
* [LISA theory](LISA-main/README.md) — Mie scattering and atmospheric models

---

## 📄 Citation

If ROSE is helpful for your research, please consider citing:

```bibtex
@misc{rose,
  title={ROSE: Roadside Oversight-guided Scenario Enhancement for Robust Multi-modal 3D Object Detection},
  author={ROSE Team},
  year={2025},
  note={Enhanced SSL training and intelligent analytics for small object detection},
  url={https://github.com/GUOYU0701/ROSE}
}
```

---

## ⚖️ License

ROSE is released under the MIT License. See [LICENSE](LICENSE) for details.
