# ROSE Installation Guide

## 系统要求

- **操作系统**: Linux (推荐 Ubuntu 18.04/20.04/22.04)
- **Python**: 3.8 / 3.9 / 3.10
- **CUDA**: 11.3 或更高版本
- **GPU**: NVIDIA GPU with >= 8GB memory (推荐 16GB+)

## 安装步骤

### 1. 创建 Conda 环境

```bash
conda create -n rose python=3.8
conda activate rose
```

### 2. 安装 PyTorch

根据您的 CUDA 版本安装对应的 PyTorch。以 CUDA 11.3 为例：

```bash
pip install torch==1.13.0+cu113 torchvision==0.14.0+cu113 torchaudio==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

其他 CUDA 版本请参考 [PyTorch官网](https://pytorch.org/)。

### 3. 安装 MMDetection3D 及其依赖

```bash
# 安装 mmengine
pip install mmengine==0.8.4

# 安装 mmcv (需要与 PyTorch 版本匹配)
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.13/index.html

# 安装 mmdet
pip install mmdet==3.1.0

# 安装 mmdet3d
pip install mmdet3d==1.2.0
```

**注意**: `mmcv` 的安装URL需要根据您的 CUDA 和 PyTorch 版本调整。详见 [MMCV官方文档](https://mmcv.readthedocs.io/)。

### 4. 安装 ROSE 依赖

```bash
cd /path/to/ROSE-NEW
pip install -r requirements.txt
```

### 5. 安装 LISA 库

LISA 库需要编译 C 扩展：

```bash
cd LISA-main
python setup.py build_ext --inplace
cd ..
```

如果遇到编译错误，请确保已安装必要的编译工具：

```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum install gcc gcc-c++ python3-devel
```

### 6. 安装 ROSE 包

```bash
# 开发模式安装 (推荐)
pip install -e .

# 或者正常安装
pip install .
```

### 7. 验证安装

运行安装验证脚本：

```bash
python scripts/verify_installation.py
```

如果所有检查通过，说明安装成功！

## 数据集准备

### DAIR-V2X 数据集

1. 从官方源下载 DAIR-V2X 数据集：
   - [DAIR-V2X 官方网站](https://thudair.baai.ac.cn/index)

2. 组织数据集结构：

```
/path/to/data/DAIR-V2X/
├── training/
│   ├── image_2/           # RGB 图像
│   ├── velodyne_reduced/  # LiDAR 点云数据
│   ├── label_2/           # 3D 标注文件
│   └── calib/             # 标定文件
├── testing/
│   ├── image_2/
│   ├── velodyne_reduced/
│   └── calib/
└── ImageSets/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

3. 在配置文件中设置数据集路径：

修改 `configs/rose_mvxnet_dair_v2x.py` 中的 `data_root` 参数：

```python
data_root = '/path/to/data/DAIR-V2X/'
```

### 示例数据

项目包含示例增强数据 (`augmented_data/`)，可用于快速测试：

```bash
# 分析示例数据
python scripts/analyze_dataset.py

# 可视化增强效果
python scripts/demo_augmentation_visualization.py
```

## 常见问题

### Q1: MMCV 安装失败

**A**: 确保 CUDA、PyTorch 和 MMCV 版本兼容。使用预编译的 wheel 文件而非源码安装。

### Q2: LISA 编译错误

**A**:
- 确保已安装 `gcc` 和 `python3-dev`
- 检查 Python 版本是否为 3.8-3.10
- 尝试更新 setuptools: `pip install --upgrade setuptools`

### Q3: CUDA 版本不匹配

**A**: 使用 `nvcc --version` 查看 CUDA 版本，确保与 PyTorch 安装时指定的版本一致。

### Q4: 内存不足

**A**:
- 减小批大小 (batch size)
- 使用梯度累积
- 使用混合精度训练 (AMP)

## 下一步

安装完成后，请参考：
- [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) - 训练和评估指南
- [README.md](README.md) - 项目功能介绍
- [configs/](configs/) - 配置文件说明

## 技术支持

如遇到问题，请：
1. 检查本指南的常见问题部分
2. 查看 MMDetection3D 官方文档
3. 提交 GitHub Issue
