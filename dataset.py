"""
Dataset loader for ShanghaiTech crowd counting dataset
Handles .mat annotation files and converts them to coordinate tensors
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import scipy.io as sio
from typing import Tuple, Dict, List
import torchvision.transforms as T
import cv2

from config import Config


class ShanghaiTechDataset(Dataset):
    """
    ShanghaiTech Dataset loader for crowd counting
    
    Parses .mat annotation files containing head point coordinates
    and returns images with corresponding point tensors.
    """
    
    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        transform: bool = True,
        max_points: int = 1000
    ):
        """
        Args:
            data_root: Root directory containing images and ground_truth folders
            split: 'train' or 'test'
            transform: Whether to apply data augmentation
            max_points: Maximum number of points to consider (for padding)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.max_points = max_points
        
        # Paths
        self.img_dir = self.data_root / "images"
        self.gt_dir = self.data_root / "ground_truth"
        
        # Validate directories
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {self.gt_dir}")
        
        # Get image paths
        self.image_paths = sorted(list(self.img_dir.glob("*.jpg")))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
        print(f"Loaded {len(self.image_paths)} images from {self.data_root}")
        
        # Setup transforms
        self.transform = transform
        if self.transform:
            self.img_transform = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                if Config.COLOR_JITTER else T.Lambda(lambda x: x),
                T.ToTensor(),
                T.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
            ])
        else:
            self.img_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - 'image': Normalized image tensor [3, H, W]
                - 'points': Point coordinates [N, 2]
                - 'count': Number of people (scalar)
                - 'image_path': Path to original image
        """
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to load image {img_path}: {e}")
        
        # Get corresponding annotation file
        # Convert IMG_1.jpg -> GT_IMG_1.mat
        img_name = img_path.stem  # e.g., "IMG_1"
        gt_name = f"GT_{img_name}.mat"
        gt_path = self.gt_dir / gt_name
        
        # Load ground truth points
        try:
            points = self._load_gt_points(gt_path)
        except Exception as e:
            raise IOError(f"Failed to load ground truth {gt_path}: {e}")
        
        # Store original dimensions
        orig_width, orig_height = image.size
        
        # Resize image if needed
        if image.size != (Config.IMG_SIZE[1], Config.IMG_SIZE[0]):
            # Calculate scaling factors
            scale_x = Config.IMG_SIZE[1] / orig_width
            scale_y = Config.IMG_SIZE[0] / orig_height
            
            # Resize image
            image = image.resize(
                (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
                Image.BILINEAR
            )
            
            # Scale points accordingly
            if len(points) > 0:
                points[:, 0] *= scale_x  # x coordinates
                points[:, 1] *= scale_y  # y coordinates
        
        # Apply random horizontal flip
        if self.transform and Config.RANDOM_FLIP and np.random.rand() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(points) > 0:
                points[:, 0] = Config.IMG_SIZE[1] - points[:, 0]
        
        # Convert image to tensor
        image_tensor = self.img_transform(image)
        
        # Convert points to tensor and pad/truncate
        if len(points) == 0:
            # No people in image
            points_tensor = torch.zeros((self.max_points, 2), dtype=torch.float32)
            valid_count = 0
        else:
            # Limit to max_points
            if len(points) > self.max_points:
                indices = np.random.choice(len(points), self.max_points, replace=False)
                points = points[indices]
            
            valid_count = len(points)
            
            # Create padded tensor
            points_tensor = torch.zeros((self.max_points, 2), dtype=torch.float32)
            points_tensor[:valid_count] = torch.from_numpy(points).float()
        
        return {
            'image': image_tensor,
            'points': points_tensor,
            'count': torch.tensor(valid_count, dtype=torch.long),
            'image_path': str(img_path),
            'orig_size': torch.tensor([orig_height, orig_width], dtype=torch.long)
        }
    
    def _load_gt_points(self, gt_path: Path) -> np.ndarray:
        """
        Load ground truth points from .mat file
        
        Args:
            gt_path: Path to .mat annotation file
            
        Returns:
            numpy array of shape [N, 2] containing (x, y) coordinates
        """
        if not gt_path.exists():
            # Return empty array if annotation doesn't exist
            print(f"Warning: Annotation file not found: {gt_path}")
            return np.zeros((0, 2), dtype=np.float32)
        
        try:
            mat_data = sio.loadmat(str(gt_path))
            
            # ShanghaiTech annotations are typically stored as 'image_info'
            # with structure: [{'location': array([[x1, y1], [x2, y2], ...])}]
            if 'image_info' in mat_data:
                points = mat_data['image_info'][0][0][0][0][0]  # Navigate nested structure
            elif 'annPoints' in mat_data:
                points = mat_data['annPoints']
            else:
                # Try to find any field with 2D array
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        data = mat_data[key]
                        if isinstance(data, np.ndarray) and len(data.shape) >= 2:
                            if data.shape[-1] == 2 or data.shape[0] == 2:
                                points = data
                                break
                else:
                    raise KeyError(f"Could not find point coordinates in {gt_path}")
            
            # Ensure shape is [N, 2]
            points = np.array(points, dtype=np.float32)
            if points.shape[1] != 2 and points.shape[0] == 2:
                points = points.T
            
            return points
            
        except Exception as e:
            raise RuntimeError(f"Error parsing .mat file {gt_path}: {e}")


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched dictionary
    """
    images = torch.stack([item['image'] for item in batch])
    points = torch.stack([item['points'] for item in batch])
    counts = torch.stack([item['count'] for item in batch])
    orig_sizes = torch.stack([item['orig_size'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'points': points,
        'counts': counts,
        'orig_sizes': orig_sizes,
        'image_paths': image_paths
    }


def create_dataloaders(
    part: str = "A",
    batch_size: int = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test dataloaders for ShanghaiTech dataset
    
    Args:
        part: 'A' or 'B'
        batch_size: Batch size (defaults to Config.BATCH_SIZE)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    
    # Select correct paths
    if part.upper() == "A":
        train_root = Config.PART_A_TRAIN
        test_root = Config.PART_A_TEST
    elif part.upper() == "B":
        train_root = Config.PART_B_TRAIN
        test_root = Config.PART_B_TEST
    else:
        raise ValueError(f"Invalid part: {part}. Must be 'A' or 'B'")
    
    # Create datasets
    train_dataset = ShanghaiTechDataset(
        train_root,
        split='train',
        transform=True,
        max_points=Config.MAX_POINTS
    )
    
    test_dataset = ShanghaiTechDataset(
        test_root,
        split='test',
        transform=False,
        max_points=Config.MAX_POINTS
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing ShanghaiTech Dataset Loader...")
    
    try:
        train_loader, test_loader = create_dataloaders(part="A", batch_size=2)
        
        # Test one batch
        batch = next(iter(train_loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Image shape: {batch['images'].shape}")
        print(f"Points shape: {batch['points'].shape}")
        print(f"Counts: {batch['counts']}")
        print(f"\nDataset test passed successfully!")
        
    except Exception as e:
        print(f"\nDataset test failed: {e}")
        import traceback
        traceback.print_exc()
