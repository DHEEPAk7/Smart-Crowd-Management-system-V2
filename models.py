"""
P2PNet: Point-to-Point Network for Crowd Counting
Architecture with VGG-16 backbone and Feature Pyramid Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple, List

from config import Config


class VGG16Backbone(nn.Module):
    """
    VGG-16 with Batch Normalization backbone
    Extracts multi-scale features for FPN
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained VGG16 with batch normalization
        vgg16_bn = models.vgg16_bn(pretrained=pretrained)
        
        # Extract feature layers
        self.features = vgg16_bn.features
        
        # Define feature extraction points for FPN
        # Conv3_3, Conv4_3, Conv5_3 outputs
        self.conv3_idx = 23  # After 3rd pooling
        self.conv4_idx = 33  # After 4th pooling
        self.conv5_idx = 43  # After 5th pooling
        
        # Channel dimensions at each level
        self.out_channels = [256, 512, 512]  # C3, C4, C5
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass extracting multi-scale features
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary of feature maps at different scales
        """
        features = {}
        
        # Extract features at different depths
        for idx, layer in enumerate(self.features):
            x = layer(x)
            
            if idx == self.conv3_idx:
                features['C3'] = x
            elif idx == self.conv4_idx:
                features['C4'] = x
            elif idx == self.conv5_idx:
                features['C5'] = x
        
        return features


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion
    """
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        
        self.out_channels = out_channels
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_conv3 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.lateral_conv5 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)
        
        # Output convolutions (3x3 conv to reduce aliasing)
        self.output_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for conv layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Top-down pathway with lateral connections
        
        Args:
            features: Multi-scale features from backbone
            
        Returns:
            Fused pyramid features
        """
        # Lateral connections
        P5 = self.lateral_conv5(features['C5'])
        P4 = self.lateral_conv4(features['C4'])
        P3 = self.lateral_conv3(features['C3'])
        
        # Top-down fusion
        # P5 -> P4
        P4 = P4 + F.interpolate(P5, size=P4.shape[-2:], mode='bilinear', align_corners=False)
        
        # P4 -> P3
        P3 = P3 + F.interpolate(P4, size=P3.shape[-2:], mode='bilinear', align_corners=False)
        
        # Output convolutions
        P5 = self.output_conv5(P5)
        P4 = self.output_conv4(P4)
        P3 = self.output_conv3(P3)
        
        return {'P3': P3, 'P4': P4, 'P5': P5}


class PointRegressionHead(nn.Module):
    """
    Regression head for predicting point coordinates
    """
    
    def __init__(self, in_channels: int, num_points: int = 1000):
        super().__init__()
        
        self.num_points = num_points
        
        # Regression network
        self.reg_conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.reg_bn1 = nn.BatchNorm2d(256)
        self.reg_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.reg_bn2 = nn.BatchNorm2d(256)
        
        # Output: 2 coordinates (x, y) per point
        self.reg_output = nn.Conv2d(256, num_points * 2, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, C, H, W]
            
        Returns:
            Point coordinates [B, num_points, 2]
        """
        x = F.relu(self.reg_bn1(self.reg_conv1(x)))
        x = F.relu(self.reg_bn2(self.reg_conv2(x)))
        x = self.reg_output(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)  # [B, num_points*2, 1, 1]
        x = x.view(x.size(0), self.num_points, 2)  # [B, num_points, 2]
        
        return x


class ClassificationHead(nn.Module):
    """
    Classification head for foreground/background classification
    """
    
    def __init__(self, in_channels: int, num_points: int = 1000, num_classes: int = 2):
        super().__init__()
        
        self.num_points = num_points
        self.num_classes = num_classes
        
        # Classification network
        self.cls_conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.cls_bn1 = nn.BatchNorm2d(256)
        self.cls_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.cls_bn2 = nn.BatchNorm2d(256)
        
        # Output: num_classes per point
        self.cls_output = nn.Conv2d(256, num_points * num_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper bias for class imbalance"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Prior probability for foreground class (assume 5% of proposals are positive)
        prior_prob = 0.05
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_output.bias, bias_value.item())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, C, H, W]
            
        Returns:
            Classification logits [B, num_points, num_classes]
        """
        x = F.relu(self.cls_bn1(self.cls_conv1(x)))
        x = F.relu(self.cls_bn2(self.cls_conv2(x)))
        x = self.cls_output(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)  # [B, num_points*num_classes, 1, 1]
        x = x.view(x.size(0), self.num_points, self.num_classes)  # [B, num_points, num_classes]
        
        return x


class P2PNet(nn.Module):
    """
    Complete P2PNet architecture for crowd counting
    
    Components:
        - VGG-16 backbone with batch normalization
        - Feature Pyramid Network
        - Parallel regression and classification heads
    """
    
    def __init__(
        self,
        num_points: int = 1000,
        num_classes: int = 2,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.num_points = num_points
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = VGG16Backbone(pretrained=pretrained)
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels=self.backbone.out_channels,
            out_channels=Config.FPN_CHANNELS
        )
        
        # Prediction heads (using P3 features for high resolution)
        self.regression_head = PointRegressionHead(
            in_channels=Config.FPN_CHANNELS,
            num_points=num_points
        )
        
        self.classification_head = ClassificationHead(
            in_channels=Config.FPN_CHANNELS,
            num_points=num_points,
            num_classes=num_classes
        )
        
        print(f"P2PNet initialized with {num_points} point proposals")
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Dictionary containing:
                - 'pred_points': Predicted coordinates [B, num_points, 2]
                - 'pred_logits': Classification logits [B, num_points, num_classes]
        """
        # Extract multi-scale features
        backbone_features = self.backbone(x)
        
        # Feature pyramid
        fpn_features = self.fpn(backbone_features)
        
        # Use P3 (highest resolution) for predictions
        features = fpn_features['P3']
        
        # Parallel branches
        pred_points = self.regression_head(features)
        pred_logits = self.classification_head(features)
        
        return {
            'pred_points': pred_points,
            'pred_logits': pred_logits
        }
    
    def predict(
        self,
        x: torch.Tensor,
        score_threshold: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference with thresholding
        
        Args:
            x: Input images [B, 3, H, W]
            score_threshold: Confidence threshold (default from Config)
            
        Returns:
            Tuple of (filtered_points, scores) as lists per batch
        """
        if score_threshold is None:
            score_threshold = Config.SCORE_THRESHOLD
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            pred_points = outputs['pred_points']  # [B, N, 2]
            pred_logits = outputs['pred_logits']  # [B, N, 2]
            
            # Convert logits to probabilities
            pred_probs = F.softmax(pred_logits, dim=-1)
            foreground_probs = pred_probs[:, :, 1]  # Probability of foreground class
            
            # Filter by threshold
            batch_points = []
            batch_scores = []
            
            for i in range(x.size(0)):
                mask = foreground_probs[i] > score_threshold
                filtered_points = pred_points[i][mask]
                filtered_scores = foreground_probs[i][mask]
                
                batch_points.append(filtered_points)
                batch_scores.append(filtered_scores)
            
            return batch_points, batch_scores


def build_model(pretrained: bool = True) -> P2PNet:
    """
    Build P2PNet model with configuration
    
    Args:
        pretrained: Whether to use ImageNet pretrained backbone
        
    Returns:
        P2PNet model
    """
    model = P2PNet(
        num_points=Config.MAX_POINTS,
        num_classes=Config.NUM_CLASSES,
        pretrained=pretrained
    )
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing P2PNet model...")
    
    model = build_model(pretrained=False)
    model.eval()
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 768, 1024)
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"\nModel output shapes:")
    print(f"  Predicted points: {outputs['pred_points'].shape}")
    print(f"  Predicted logits: {outputs['pred_logits'].shape}")
    
    # Test prediction
    points, scores = model.predict(dummy_input, score_threshold=0.5)
    print(f"\nAfter thresholding:")
    print(f"  Batch 0: {len(points[0])} detected points")
    print(f"  Batch 1: {len(points[1])} detected points")
    
    print("\nModel test passed successfully!")
