"""
Configuration file for P2PNet Crowd Monitoring System
"""

import torch
from pathlib import Path


class Config:
    """Centralized configuration for the entire system"""
    
    # ==================== PATHS ====================
    DATASET_ROOT = Path("./data/ShanghaiTech")
    PART_A_TRAIN = DATASET_ROOT / "part_A" / "train_data"
    PART_A_TEST = DATASET_ROOT / "part_A" / "test_data"
    PART_B_TRAIN = DATASET_ROOT / "part_B" / "train_data"
    PART_B_TEST = DATASET_ROOT / "part_B" / "test_data"
    
    CHECKPOINT_DIR = Path("./checkpoints")
    LOG_DIR = Path("./logs")
    
    # ==================== MODEL ====================
    BACKBONE = "vgg16_bn"  # VGG-16 with Batch Normalization
    PRETRAINED = True  # Use ImageNet pretrained weights
    NUM_CLASSES = 2  # Foreground vs Background
    
    # Feature Pyramid Network settings
    FPN_CHANNELS = 256
    NUM_REGRESSION_PARAMS = 2  # (x, y) coordinates
    
    # ==================== TRAINING ====================
    BATCH_SIZE = 8
    NUM_EPOCHS = 1500
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    
    # Learning rate scheduler (Cosine Annealing)
    LR_SCHEDULER = "cosine"
    T_MAX = 1500  # Maximum number of iterations
    ETA_MIN = 1e-7  # Minimum learning rate
    
    # Automatic Mixed Precision
    USE_AMP = True
    
    # Loss weights
    LAMBDA_COORD = 1.0  # Weight for coordinate loss
    LAMBDA_CLASS = 1.0  # Weight for classification loss
    
    # ==================== DATA AUGMENTATION ====================
    IMG_SIZE = (768, 1024)  # (height, width)
    RANDOM_FLIP = True
    COLOR_JITTER = True
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # ==================== INFERENCE ====================
    SCORE_THRESHOLD = 0.5  # Confidence threshold for detections
    NMS_THRESHOLD = 0.3  # Non-maximum suppression threshold
    
    # ==================== ROI ALERTING ====================
    ROI_COUNT_THRESHOLD = 50  # Alert when count exceeds this
    ALERT_COLOR = (0, 0, 255)  # Red in BGR
    NORMAL_COLOR = (0, 255, 0)  # Green in BGR
    
    # ==================== VISUALIZATION ====================
    POINT_COLOR = (255, 0, 0)  # Red dots for detected heads
    POINT_RADIUS = 3
    POINT_THICKNESS = -1  # Filled circle
    
    ROI_BORDER_COLOR = (255, 255, 0)  # Yellow for ROI boundary
    ROI_BORDER_THICKNESS = 2
    
    # ==================== SYSTEM ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # DataLoader workers
    PIN_MEMORY = True
    
    # Logging
    LOG_INTERVAL = 10  # Print every N batches
    SAVE_INTERVAL = 50  # Save checkpoint every N epochs
    
    # ==================== HUNGARIAN MATCHING ====================
    MAX_POINTS = 1000  # Maximum number of points to match
    MATCHING_COST_WEIGHT = 1.0
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 50)
        print("P2PNet Configuration")
        print("=" * 50)
        print(f"Device: {cls.DEVICE}")
        print(f"Backbone: {cls.BACKBONE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Use AMP: {cls.USE_AMP}")
        print(f"Score Threshold: {cls.SCORE_THRESHOLD}")
        print(f"ROI Alert Threshold: {cls.ROI_COUNT_THRESHOLD}")
        print("=" * 50)
