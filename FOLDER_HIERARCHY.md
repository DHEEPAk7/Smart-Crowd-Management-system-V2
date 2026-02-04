# Smart Crowd Monitoring System - Folder Hierarchy

```
smart-crowd-monitoring/
│
├── 📄 config.py                      # Centralized configuration
├── 📄 models.py                      # P2PNet architecture
├── 📄 dataset.py                     # Dataset loader
├── 📄 utils.py                       # Utilities (Hungarian, ROI, losses)
├── 📄 train.py                       # Training script
├── 📄 inference.py                   # Standalone inference
├── 📄 app.py                         # Streamlit dashboard
├── 📄 demo.py                        # Demo with synthetic data
├── 📄 setup.py                       # Automated setup script
│
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # Main documentation
├── 📄 QUICKSTART.md                  # Quick start guide
├── 📄 PROJECT_OVERVIEW.md            # Project overview
│
├── 📁 data/                          # Dataset directory
│   └── 📁 ShanghaiTech/
│       ├── 📁 part_A/
│       │   ├── 📁 train_data/
│       │   │   ├── 📁 images/
│       │   │   │   ├── IMG_1.jpg
│       │   │   │   ├── IMG_2.jpg
│       │   │   │   ├── IMG_3.jpg
│       │   │   │   └── ... (300 images)
│       │   │   └── 📁 ground_truth/
│       │   │       ├── GT_IMG_1.mat
│       │   │       ├── GT_IMG_2.mat
│       │   │       ├── GT_IMG_3.mat
│       │   │       └── ... (300 .mat files)
│       │   │
│       │   └── 📁 test_data/
│       │       ├── 📁 images/
│       │       │   ├── IMG_1.jpg
│       │       │   ├── IMG_2.jpg
│       │       │   └── ... (182 images)
│       │       └── 📁 ground_truth/
│       │           ├── GT_IMG_1.mat
│       │           ├── GT_IMG_2.mat
│       │           └── ... (182 .mat files)
│       │
│       └── 📁 part_B/
│           ├── 📁 train_data/
│           │   ├── 📁 images/
│           │   │   └── ... (400 images)
│           │   └── 📁 ground_truth/
│           │       └── ... (400 .mat files)
│           │
│           └── 📁 test_data/
│               ├── 📁 images/
│               │   └── ... (316 images)
│               └── 📁 ground_truth/
│                   └── ... (316 .mat files)
│
├── 📁 checkpoints/                   # Model checkpoints
│   ├── best_model.pth                # Best performing model
│   ├── checkpoint_epoch_50.pth       # Checkpoint at epoch 50
│   ├── checkpoint_epoch_100.pth      # Checkpoint at epoch 100
│   ├── checkpoint_epoch_150.pth      # Checkpoint at epoch 150
│   └── ... (saved every 50 epochs)
│
├── 📁 logs/                          # TensorBoard logs
│   ├── events.out.tfevents.*         # Training events
│   └── ... (experiment logs)
│
├── 📁 results/                       # Inference outputs
│   ├── demo_roi_detection.jpg        # Demo output
│   ├── demo_inference.jpg            # Demo output
│   ├── test_IMG_1.jpg                # Test results
│   ├── monitored_video.mp4           # Processed videos
│   └── ... (your inference results)
│
└── 📁 venv/                          # Virtual environment (optional)
    ├── bin/
    ├── lib/
    └── ...
```

## 📊 Directory Sizes (Approximate)

```
Directory           Size        Purpose
─────────────────────────────────────────────────────────────
data/               ~2-3 GB     ShanghaiTech dataset
checkpoints/        ~500 MB     Trained model weights
logs/               ~50 MB      TensorBoard training logs
results/            Varies      Your inference outputs
venv/               ~2 GB       Python virtual environment
```

## 🗂️ Detailed Structure Explanation

### Root Directory Files

| File | Size | Purpose |
|------|------|---------|
| `config.py` | 4 KB | All hyperparameters and settings |
| `models.py` | 13 KB | P2PNet model architecture |
| `dataset.py` | 11 KB | ShanghaiTech data loader |
| `utils.py` | 15 KB | Hungarian matching, losses, ROI |
| `train.py` | 13 KB | Complete training pipeline |
| `inference.py` | 8.5 KB | Standalone inference script |
| `app.py` | 14 KB | Streamlit web dashboard |
| `demo.py` | 11 KB | Test without dataset |
| `setup.py` | 7.5 KB | Automated setup |

### Data Directory (`data/`)

**Purpose**: Stores the ShanghaiTech crowd counting dataset

**Structure**:
- `part_A/`: Dense crowd scenes (1024×768 avg)
  - `train_data/`: 300 training images
  - `test_data/`: 182 test images
- `part_B/`: Sparse crowd scenes (768×1024 avg)
  - `train_data/`: 400 training images
  - `test_data/`: 316 test images

**Note**: You need to download this separately from:
https://github.com/desenzhou/ShanghaiTechDataset

### Checkpoints Directory (`checkpoints/`)

**Purpose**: Stores trained model weights

**Auto-generated files**:
- `best_model.pth`: Best model based on validation MAE
- `checkpoint_epoch_N.pth`: Saved every 50 epochs
- Each checkpoint ~170 MB (30M parameters × 4 bytes)

### Logs Directory (`logs/`)

**Purpose**: TensorBoard training logs

**Auto-generated files**:
- Event files for training metrics
- Scalars: loss, MAE, RMSE, learning rate
- View with: `tensorboard --logdir logs/`

### Results Directory (`results/`)

**Purpose**: Store inference outputs

**Your outputs**:
- Annotated images with detected points
- Processed videos with crowd counts
- ROI monitoring visualizations

## 📝 Setup Instructions

### Option 1: Automatic Setup

```bash
# Run the setup script
python setup.py

# This will create all necessary directories
```

### Option 2: Manual Setup

```bash
# Create directories
mkdir -p data/ShanghaiTech/part_A/{train_data,test_data}/{images,ground_truth}
mkdir -p data/ShanghaiTech/part_B/{train_data,test_data}/{images,ground_truth}
mkdir -p checkpoints logs results

# Download dataset
# Extract ShanghaiTech to data/ShanghaiTech/
```

### Option 3: Using Python

```python
from pathlib import Path

# Create directory structure
dirs = [
    "data/ShanghaiTech/part_A/train_data/images",
    "data/ShanghaiTech/part_A/train_data/ground_truth",
    "data/ShanghaiTech/part_A/test_data/images",
    "data/ShanghaiTech/part_A/test_data/ground_truth",
    "data/ShanghaiTech/part_B/train_data/images",
    "data/ShanghaiTech/part_B/train_data/ground_truth",
    "data/ShanghaiTech/part_B/test_data/images",
    "data/ShanghaiTech/part_B/test_data/ground_truth",
    "checkpoints",
    "logs",
    "results"
]

for dir_path in dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
```

## 🎯 File Relationships

```
Workflow Flow:
─────────────

1. CONFIGURATION
   config.py ──────────┐
                       ├──> All other files import config
                       │
2. DATA LOADING        │
   dataset.py <────────┤
   data/ <─────────────┤
                       │
3. MODEL ARCHITECTURE  │
   models.py <─────────┤
                       │
4. TRAINING            │
   utils.py <──────────┤
   train.py <──────────┤
   ├─> checkpoints/    │
   └─> logs/           │
                       │
5. INFERENCE           │
   inference.py <──────┤
   app.py <────────────┤
   └─> results/        │
                       │
6. TESTING             │
   demo.py <───────────┘
   └─> results/
```

## 🔍 Quick Verification Commands

```bash
# Check directory structure
tree -L 3 -d

# Or using ls
ls -R

# Verify data directory
ls -lh data/ShanghaiTech/part_A/train_data/images/ | head

# Check checkpoints
ls -lh checkpoints/

# View logs
ls -lh logs/

# Count files
find data/ShanghaiTech/part_A/train_data/images -name "*.jpg" | wc -l
# Should output: 300

find data/ShanghaiTech/part_A/test_data/images -name "*.jpg" | wc -l
# Should output: 182
```

## 📦 What Gets Generated During Usage

### During Training (`python train.py`)
```
checkpoints/
├── checkpoint_epoch_50.pth    (created at epoch 50)
├── checkpoint_epoch_100.pth   (created at epoch 100)
├── checkpoint_epoch_150.pth   (created at epoch 150)
└── best_model.pth             (updated when validation improves)

logs/
└── events.out.tfevents.*      (TensorBoard logs)
```

### During Inference (`python inference.py`)
```
results/
├── annotated_image.jpg        (your outputs)
├── processed_video.mp4        (your outputs)
└── ...
```

### During Demo (`python demo.py`)
```
results/
├── demo_roi_detection.jpg     (ROI visualization)
└── demo_inference.jpg         (inference example)
```

## 💾 Storage Requirements

**Minimum**:
- Code: ~100 MB
- Dataset: ~2-3 GB
- Models: ~500 MB
- Logs: ~50 MB
- **Total**: ~3-4 GB

**Recommended**:
- SSD for faster data loading
- 10+ GB free space for checkpoints and experiments
- GPU with 8+ GB VRAM for training

## 🗂️ Alternative Layouts

### For Production Deployment

```
production/
├── src/
│   ├── config.py
│   ├── models.py
│   ├── utils.py
│   └── inference.py
├── models/
│   └── best_model.pth
├── app.py
├── Dockerfile
└── requirements.txt
```

### For Research/Experimentation

```
research/
├── experiments/
│   ├── exp1_baseline/
│   ├── exp2_higher_lr/
│   └── exp3_more_augmentation/
├── notebooks/
│   ├── data_exploration.ipynb
│   └── error_analysis.ipynb
├── src/
│   └── ... (all .py files)
└── data/
    └── ShanghaiTech/
```

---

**The provided hierarchy is optimized for development, training, and testing.**

Use `python setup.py` to automatically create all directories!
