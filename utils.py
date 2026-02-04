"""
Utility functions for P2PNet training and inference
Includes Hungarian matching, loss functions, and ROI detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional

from config import Config


class HungarianMatcher:
    """
    Hungarian Algorithm for optimal one-to-one matching
    between predicted points and ground truth points
    """
    
    def __init__(self, cost_weight: float = 1.0):
        """
        Args:
            cost_weight: Weight for coordinate matching cost
        """
        self.cost_weight = cost_weight
    
    @torch.no_grad()
    def forward(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_points: torch.Tensor,
        gt_counts: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching
        
        Args:
            pred_points: Predicted coordinates [B, N, 2]
            pred_logits: Classification logits [B, N, num_classes]
            gt_points: Ground truth points [B, M, 2]
            gt_counts: Number of valid GT points per sample [B]
            
        Returns:
            List of (pred_indices, gt_indices) tuples for each batch element
        """
        batch_size = pred_points.shape[0]
        matches = []
        
        for i in range(batch_size):
            # Get valid ground truth points
            num_gt = gt_counts[i].item()
            if num_gt == 0:
                # No ground truth points - all predictions are background
                matches.append((torch.tensor([]), torch.tensor([])))
                continue
            
            gt_pts = gt_points[i, :num_gt]  # [num_gt, 2]
            
            # Compute cost matrix
            # L2 distance between all prediction-GT pairs
            pred_pts = pred_points[i]  # [N, 2]
            
            # Expand dimensions for broadcasting
            pred_expanded = pred_pts.unsqueeze(1)  # [N, 1, 2]
            gt_expanded = gt_pts.unsqueeze(0)  # [1, num_gt, 2]
            
            # Euclidean distance cost
            coord_cost = torch.cdist(pred_pts, gt_pts, p=2)  # [N, num_gt]
            
            # Classification cost (prefer foreground class for matched points)
            pred_probs = F.softmax(pred_logits[i], dim=-1)  # [N, num_classes]
            # Cost is negative log probability of foreground class
            cls_cost = -torch.log(pred_probs[:, 1:2] + 1e-8)  # [N, 1]
            cls_cost = cls_cost.expand(-1, num_gt)  # [N, num_gt]
            
            # Total cost
            cost = self.cost_weight * coord_cost + cls_cost
            
            # Convert to numpy for scipy
            cost_matrix = cost.cpu().numpy()
            
            # Hungarian algorithm
            pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
            
            # Convert back to torch tensors
            pred_indices = torch.as_tensor(pred_indices, dtype=torch.long)
            gt_indices = torch.as_tensor(gt_indices, dtype=torch.long)
            
            matches.append((pred_indices, gt_indices))
        
        return matches


class P2PNetLoss(nn.Module):
    """
    Composite loss function for P2PNet
    Combines coordinate regression loss and classification loss
    """
    
    def __init__(
        self,
        lambda_coord: float = 1.0,
        lambda_class: float = 1.0
    ):
        """
        Args:
            lambda_coord: Weight for coordinate loss
            lambda_class: Weight for classification loss
        """
        super().__init__()
        
        self.lambda_coord = lambda_coord
        self.lambda_class = lambda_class
        
        self.matcher = HungarianMatcher(cost_weight=Config.MATCHING_COST_WEIGHT)
        
        # Classification loss (Cross Entropy)
        self.cls_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            outputs: Model predictions
                - 'pred_points': [B, N, 2]
                - 'pred_logits': [B, N, num_classes]
            targets: Ground truth
                - 'points': [B, M, 2]
                - 'counts': [B]
                
        Returns:
            Dictionary of losses
        """
        pred_points = outputs['pred_points']
        pred_logits = outputs['pred_logits']
        gt_points = targets['points']
        gt_counts = targets['counts']
        
        batch_size = pred_points.shape[0]
        num_proposals = pred_points.shape[1]
        
        # Perform Hungarian matching
        matches = self.matcher.forward(pred_points, pred_logits, gt_points, gt_counts)
        
        # Initialize loss accumulators
        total_coord_loss = 0.0
        total_cls_loss = 0.0
        num_matched = 0
        
        for i in range(batch_size):
            pred_indices, gt_indices = matches[i]
            
            # Classification labels
            # All proposals are background (class 0) by default
            cls_labels = torch.zeros(num_proposals, dtype=torch.long, device=pred_logits.device)
            
            if len(pred_indices) > 0:
                # Matched proposals are foreground (class 1)
                cls_labels[pred_indices] = 1
                
                # Coordinate loss (only for matched pairs)
                matched_pred = pred_points[i, pred_indices]  # [num_matched, 2]
                matched_gt = gt_points[i, gt_indices]  # [num_matched, 2]
                
                # Euclidean distance loss
                coord_loss = F.mse_loss(matched_pred, matched_gt, reduction='sum')
                total_coord_loss += coord_loss
                num_matched += len(pred_indices)
            
            # Classification loss (for all proposals)
            cls_loss = self.cls_criterion(pred_logits[i], cls_labels)
            total_cls_loss += cls_loss
        
        # Average losses
        if num_matched > 0:
            coord_loss = total_coord_loss / num_matched
        else:
            coord_loss = torch.tensor(0.0, device=pred_points.device)
        
        cls_loss = total_cls_loss / batch_size
        
        # Weighted total loss
        total_loss = self.lambda_coord * coord_loss + self.lambda_class * cls_loss
        
        return {
            'loss': total_loss,
            'coord_loss': coord_loss,
            'cls_loss': cls_loss,
            'num_matched': torch.tensor(num_matched, dtype=torch.float32)
        }


class ROIDetector:
    """
    Region of Interest detector for crowd counting and alerting
    """
    
    def __init__(self, roi_polygon: Optional[np.ndarray] = None):
        """
        Args:
            roi_polygon: Polygon vertices as numpy array [[x1,y1], [x2,y2], ...]
        """
        self.roi_polygon = roi_polygon
        self.roi_count = 0
        self.is_alert = False
    
    def set_roi(self, polygon: np.ndarray):
        """Set ROI polygon"""
        self.roi_polygon = polygon
    
    def count_in_roi(self, points: np.ndarray) -> int:
        """
        Count points inside ROI polygon
        
        Args:
            points: Point coordinates as numpy array [N, 2]
            
        Returns:
            Number of points inside ROI
        """
        if self.roi_polygon is None or len(points) == 0:
            return 0
        
        count = 0
        for point in points:
            # Use OpenCV's pointPolygonTest
            # Returns positive if inside, negative if outside, 0 if on edge
            result = cv2.pointPolygonTest(
                self.roi_polygon.astype(np.float32),
                tuple(point.astype(float)),
                measureDist=False
            )
            if result >= 0:
                count += 1
        
        self.roi_count = count
        self.is_alert = count > Config.ROI_COUNT_THRESHOLD
        
        return count
    
    def draw_roi(
        self,
        image: np.ndarray,
        fill: bool = False,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Draw ROI polygon on image
        
        Args:
            image: Input image (BGR)
            fill: Whether to fill the polygon
            alpha: Transparency for filled polygon
            
        Returns:
            Image with ROI drawn
        """
        if self.roi_polygon is None:
            return image
        
        output = image.copy()
        
        # Choose color based on alert status
        color = Config.ALERT_COLOR if self.is_alert else Config.NORMAL_COLOR
        
        if fill:
            # Create overlay for transparency
            overlay = output.copy()
            cv2.fillPoly(overlay, [self.roi_polygon.astype(np.int32)], color)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        # Draw border
        cv2.polylines(
            output,
            [self.roi_polygon.astype(np.int32)],
            isClosed=True,
            color=color,
            thickness=Config.ROI_BORDER_THICKNESS
        )
        
        return output
    
    def get_alert_message(self) -> str:
        """Get alert message if threshold exceeded"""
        if self.is_alert:
            return f"⚠️ ALERT: {self.roi_count} people in ROI (Threshold: {Config.ROI_COUNT_THRESHOLD})"
        else:
            return f"✓ Normal: {self.roi_count} people in ROI"


def visualize_predictions(
    image: np.ndarray,
    points: np.ndarray,
    scores: Optional[np.ndarray] = None,
    roi_detector: Optional[ROIDetector] = None
) -> np.ndarray:
    """
    Visualize crowd counting predictions on image
    
    Args:
        image: Input image (BGR or RGB)
        points: Detected point coordinates [N, 2]
        scores: Confidence scores [N] (optional)
        roi_detector: ROI detector instance (optional)
        
    Returns:
        Annotated image
    """
    # Convert to BGR if needed
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        output = image.copy()
    else:
        output = image[:, :, :3].copy()
    
    # Draw ROI if available
    if roi_detector is not None and roi_detector.roi_polygon is not None:
        output = roi_detector.draw_roi(output, fill=True, alpha=0.2)
    
    # Draw points
    for i, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        
        # Skip if out of bounds
        if x < 0 or y < 0 or x >= output.shape[1] or y >= output.shape[0]:
            continue
        
        # Draw circle
        cv2.circle(
            output,
            (x, y),
            Config.POINT_RADIUS,
            Config.POINT_COLOR,
            Config.POINT_THICKNESS
        )
    
    # Add count text
    count_text = f"Count: {len(points)}"
    cv2.putText(
        output,
        count_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    # Add ROI count if available
    if roi_detector is not None and roi_detector.roi_polygon is not None:
        roi_text = f"ROI: {roi_detector.roi_count}"
        color = Config.ALERT_COLOR if roi_detector.is_alert else Config.NORMAL_COLOR
        cv2.putText(
            output,
            roi_text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA
        )
    
    return output


def denormalize_image(
    tensor: torch.Tensor,
    mean: List[float] = None,
    std: List[float] = None
) -> np.ndarray:
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized image as numpy array [H, W, C] in BGR
    """
    if mean is None:
        mean = Config.NORMALIZE_MEAN
    if std is None:
        std = Config.NORMALIZE_STD
    
    # Clone to avoid modifying original
    img = tensor.clone()
    
    # Denormalize
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    
    # Clip to valid range
    img = torch.clamp(img, 0, 1)
    
    # Convert to numpy and transpose to HWC
    img = img.permute(1, 2, 0).cpu().numpy()
    
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return img


class MetricTracker:
    """Track and compute metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.losses = []
        self.coord_losses = []
        self.cls_losses = []
        self.mae_errors = []
        self.mse_errors = []
    
    def update(
        self,
        loss: float,
        coord_loss: float,
        cls_loss: float,
        pred_count: int,
        gt_count: int
    ):
        """Update metrics with batch results"""
        self.losses.append(loss)
        self.coord_losses.append(coord_loss)
        self.cls_losses.append(cls_loss)
        
        # MAE and MSE
        error = abs(pred_count - gt_count)
        self.mae_errors.append(error)
        self.mse_errors.append(error ** 2)
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics"""
        return {
            'loss': np.mean(self.losses) if self.losses else 0.0,
            'coord_loss': np.mean(self.coord_losses) if self.coord_losses else 0.0,
            'cls_loss': np.mean(self.cls_losses) if self.cls_losses else 0.0,
            'mae': np.mean(self.mae_errors) if self.mae_errors else 0.0,
            'mse': np.mean(self.mse_errors) if self.mse_errors else 0.0,
            'rmse': np.sqrt(np.mean(self.mse_errors)) if self.mse_errors else 0.0
        }


if __name__ == "__main__":
    # Test Hungarian matcher
    print("Testing Hungarian Matcher...")
    
    matcher = HungarianMatcher()
    
    # Create dummy data
    pred_points = torch.randn(2, 100, 2)
    pred_logits = torch.randn(2, 100, 2)
    gt_points = torch.randn(2, 50, 2)
    gt_counts = torch.tensor([30, 45])
    
    matches = matcher.forward(pred_points, pred_logits, gt_points, gt_counts)
    
    print(f"Batch 0: Matched {len(matches[0][0])} pairs")
    print(f"Batch 1: Matched {len(matches[1][0])} pairs")
    
    # Test loss
    print("\nTesting P2PNet Loss...")
    criterion = P2PNetLoss()
    
    outputs = {'pred_points': pred_points, 'pred_logits': pred_logits}
    targets = {'points': gt_points, 'counts': gt_counts}
    
    losses = criterion(outputs, targets)
    print(f"Total Loss: {losses['loss']:.4f}")
    print(f"Coord Loss: {losses['coord_loss']:.4f}")
    print(f"Class Loss: {losses['cls_loss']:.4f}")
    
    print("\nAll tests passed successfully!")
