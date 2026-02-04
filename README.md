# Smart Crowd Monitoring System with P2PNet

A production-ready, modular crowd counting and monitoring system using **P2PNet (Point-to-Point Network)** with real-time ROI-based alerting capabilities.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Key Features

- **Point-Based Localization**: Precise head location detection using P2PNet framework
- **VGG-16 Backbone**: Feature extraction with batch normalization for stable training
- **Feature Pyramid Network**: Multi-scale feature fusion for robust detection
- **Hungarian Matching**: Optimal one-to-one matching between predictions and ground truth
- **Real-Time ROI Monitoring**: Custom polygon-based regions with automatic alerting
- **Multiple Input Sources**: Support for images, videos, and webcam streams
- **Production-Ready**: Modular code with proper error handling and documentation

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Input Image                        │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│         VGG-16 Backbone (with BN)                   │
│    ┌──────────┬──────────┬──────────┐              │
│    │   C3     │    C4    │    C5    │              │
│    │ (256ch)  │ (512ch)  │ (512ch)  │              │
└────┴──────────┴──────────┴──────────┴──────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│       Feature Pyramid Network (FPN)                 │
│    ┌──────────┬──────────┬──────────┐              │
│    │   P3     │    P4    │    P5    │              │
│    │ (256ch)  │ (256ch)  │ (256ch)  │              │
└────┴──────────┴──────────┴──────────┴──────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│          Parallel Prediction Heads                  │
│    ┌────────────────┬────────────────┐             │
│    │  Regression    │ Classification │             │
│    │   (x, y)       │ (FG vs BG)     │             │
└────┴────────────────┴────────────────┴─────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│        Hungarian Algorithm Matching                 │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│   Composite Loss (Euclidean + Cross-Entropy)        │
└─────────────────────────────────────────────────────┘
```

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 16GB RAM minimum
- 8GB GPU memory recommended

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/smart-crowd-monitoring.git
cd smart-crowd-monitoring
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 📊 Dataset Preparation

### ShanghaiTech Dataset

Download the ShanghaiTech dataset from the [official source](https://github.com/desenzhou/ShanghaiTechDataset).

Expected directory structure:
```
data/
└── ShanghaiTech/
    ├── part_A/
    │   ├── train_data/
    │   │   ├── images/
    │   │   │   ├── IMG_1.jpg
    │   │   │   ├── IMG_2.jpg
    │   │   │   └── ...
    │   │   └── ground_truth/
    │   │       ├── GT_IMG_1.mat
    │   │       ├── GT_IMG_2.mat
    │   │       └── ...
    │   └── test_data/
    │       ├── images/
    │       └── ground_truth/
    └── part_B/
        ├── train_data/
        └── test_data/
```

### Annotation Format

The `.mat` files contain head point coordinates:
```matlab
image_info:
  location: [N x 2] array of (x, y) coordinates
```

## 🎓 Training

### Basic Training

Train on ShanghaiTech Part A:
```bash
python train.py --part A --epochs 1500 --batch_size 8
```

Train on ShanghaiTech Part B:
```bash
python train.py --part B --epochs 1500 --batch_size 8
```

### Advanced Options

```bash
python train.py \
    --part A \
    --epochs 1500 \
    --batch_size 8 \
    --resume checkpoints/checkpoint_epoch_500.pth \
    --pretrained
```

### Training Features

- ✅ **Automatic Mixed Precision (AMP)** for faster training
- ✅ **Cosine Annealing LR Scheduler** for better convergence
- ✅ **Hungarian Algorithm** for optimal matching
- ✅ **TensorBoard Logging** for monitoring
- ✅ **Checkpoint Management** with best model saving

### Monitor Training

```bash
tensorboard --logdir logs/
```

Open browser at `http://localhost:6006`

## 🔮 Inference

### Single Image

```bash
python inference.py \
    --input path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output results/output.jpg \
    --threshold 0.5
```

### Video

```bash
python inference.py \
    --input path/to/video.mp4 \
    --checkpoint checkpoints/best_model.pth \
    --output results/output.mp4
```

### With ROI

```bash
python inference.py \
    --input path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    --roi "100,100,500,100,500,400,100,400"  # x1,y1,x2,y2,...
```

## 🖥️ Streamlit Dashboard

### Launch the Dashboard

```bash
streamlit run app.py
```

### Features

1. **Video Upload**: Process pre-recorded videos
2. **Webcam Stream**: Real-time monitoring from webcam
3. **ROI Definition**: Define custom polygon regions
4. **Real-Time Alerts**: Get notifications when thresholds exceeded
5. **Interactive Settings**: Adjust parameters on-the-fly

### Dashboard Interface

- **Detection Threshold**: Confidence score filtering (0.0 - 1.0)
- **ROI Alert Threshold**: Number of people to trigger alert
- **Point Visualization**: Customize detection marker size
- **ROI Polygon**: Define monitoring zones with corner points

## 📁 Project Structure

```
smart-crowd-monitoring/
├── config.py              # Centralized configuration
├── dataset.py             # ShanghaiTech dataset loader
├── models.py              # P2PNet architecture
├── utils.py               # Hungarian matcher, losses, ROI detection
├── train.py               # Training script
├── inference.py           # Standalone inference
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
│
├── data/                  # Dataset directory
│   └── ShanghaiTech/
│
├── checkpoints/           # Model checkpoints
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
│
├── logs/                  # TensorBoard logs
│
└── results/               # Inference outputs
```

## ⚙️ Configuration

Edit `config.py` to customize:

### Model Parameters
```python
BACKBONE = "vgg16_bn"           # VGG-16 with Batch Norm
FPN_CHANNELS = 256              # FPN feature channels
NUM_CLASSES = 2                 # Foreground vs Background
MAX_POINTS = 1000               # Max proposals
```

### Training Parameters
```python
BATCH_SIZE = 8
NUM_EPOCHS = 1500
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
USE_AMP = True                  # Automatic Mixed Precision
```

### Inference Parameters
```python
SCORE_THRESHOLD = 0.5           # Detection confidence
ROI_COUNT_THRESHOLD = 50        # Alert threshold
```

## 📈 Performance

### Expected Results on ShanghaiTech

| Dataset | MAE | RMSE |
|---------|-----|------|
| Part A (Dense) | ~60-70 | ~90-100 |
| Part B (Sparse) | ~8-10 | ~13-15 |

*Note: Performance depends on training duration and hyperparameters*

### Inference Speed

- **GPU (RTX 3090)**: ~30 FPS on 1024×768 images
- **CPU (Intel i9)**: ~3 FPS on 1024×768 images

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size in config.py
BATCH_SIZE = 4  # or lower
```

**2. Dataset Not Found**
```bash
# Verify dataset structure
python -c "from dataset import create_dataloaders; create_dataloaders('A')"
```

**3. Checkpoint Loading Error**
```python
# Ensure checkpoint matches model architecture
# Check Config.MAX_POINTS matches checkpoint
```

**4. Slow Training**
```python
# Enable AMP in config.py
USE_AMP = True

# Reduce NUM_WORKERS if CPU bottleneck
NUM_WORKERS = 2
```

### Performance Optimization Tips

1. **Use Mixed Precision**: Enable `USE_AMP = True`
2. **Optimal Batch Size**: Find largest batch that fits in GPU
3. **Data Loading**: Increase `NUM_WORKERS` for faster I/O
4. **Pin Memory**: Keep `PIN_MEMORY = True` for GPU training
5. **Learning Rate**: Use learning rate finder for optimal value

## 📚 Key Improvements Over Previous Approaches

### Why This System is Better

1. **Point-Based vs Density Maps**
   - ✅ Precise localization (exact coordinates)
   - ✅ Better for sparse crowds
   - ✅ Easier to interpret results

2. **Hungarian Matching**
   - ✅ Optimal assignment problem solution
   - ✅ Better gradient flow
   - ✅ Handles variable crowd sizes

3. **Feature Pyramid Network**
   - ✅ Multi-scale detection
   - ✅ Robust to scale variations
   - ✅ Better feature representation

4. **Production-Ready Code**
   - ✅ Modular architecture
   - ✅ Comprehensive error handling
   - ✅ Extensive documentation
   - ✅ Easy to extend and customize

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **P2PNet Paper**: "Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework"
- **ShanghaiTech Dataset**: Provided by Fudan University
- **PyTorch Team**: For the excellent deep learning framework

## 📧 Contact

For questions or issues, please:
- Open a GitHub issue
- Email: your.email@example.com

## 🔗 References

1. Song et al., "Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework", ICCV 2021
2. Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", ICLR 2015
3. Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017
4. Zhang et al., "Single-Image Crowd Counting via Multi-Column Convolutional Neural Network", CVPR 2016

---

**Note**: This is a research/educational project. For production deployment in critical applications, additional testing and validation is recommended.
