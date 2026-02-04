"""
Demo script for P2PNet Crowd Monitoring System
Tests the system with synthetic data (no dataset required)
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from config import Config
from models import build_model
from utils import P2PNetLoss, ROIDetector, visualize_predictions


def create_synthetic_image(width=1024, height=768, num_people=50):
    """
    Create a synthetic crowd image with random points
    
    Args:
        width: Image width
        height: Image height
        num_people: Number of simulated people
        
    Returns:
        Tuple of (image, points)
    """
    # Create blank image
    image = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Add some random rectangles to simulate background
    for _ in range(20):
        x1 = np.random.randint(0, width - 100)
        y1 = np.random.randint(0, height - 100)
        x2 = x1 + np.random.randint(50, 150)
        y2 = y1 + np.random.randint(50, 150)
        color = tuple(np.random.randint(150, 255, 3).tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    
    # Generate random points
    points = []
    for _ in range(num_people):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        points.append([x, y])
        
        # Draw a circle to represent a person's head
        cv2.circle(image, (x, y), 5, (50, 50, 50), -1)
    
    points = np.array(points, dtype=np.float32)
    
    return image, points


def demo_model_forward_pass():
    """Demo: Model architecture and forward pass"""
    print("\n" + "=" * 70)
    print("DEMO 1: Model Forward Pass")
    print("=" * 70)
    
    # Build model
    print("\nBuilding P2PNet model...")
    model = build_model(pretrained=False)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 768, 1024)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"Output shapes:")
    print(f"  Predicted points: {outputs['pred_points'].shape}")
    print(f"  Predicted logits: {outputs['pred_logits'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n✓ Model forward pass successful!")


def demo_loss_computation():
    """Demo: Loss computation with Hungarian matching"""
    print("\n" + "=" * 70)
    print("DEMO 2: Loss Computation with Hungarian Matching")
    print("=" * 70)
    
    # Create synthetic data
    batch_size = 2
    num_predictions = Config.MAX_POINTS
    
    pred_points = torch.randn(batch_size, num_predictions, 2) * 100 + 400
    pred_logits = torch.randn(batch_size, num_predictions, Config.NUM_CLASSES)
    
    # Ground truth
    gt_points = torch.randn(batch_size, 100, 2) * 100 + 400
    gt_counts = torch.tensor([50, 75])
    
    print(f"Predictions: {num_predictions} point proposals per image")
    print(f"Ground truth: {gt_counts.tolist()} people per image")
    
    # Compute loss
    criterion = P2PNetLoss()
    outputs = {'pred_points': pred_points, 'pred_logits': pred_logits}
    targets = {'points': gt_points, 'counts': gt_counts}
    
    print("\nComputing loss...")
    losses = criterion(outputs, targets)
    
    print(f"\nLoss breakdown:")
    print(f"  Total loss: {losses['loss']:.4f}")
    print(f"  Coordinate loss: {losses['coord_loss']:.4f}")
    print(f"  Classification loss: {losses['cls_loss']:.4f}")
    print(f"  Matched pairs: {losses['num_matched']:.0f}")
    
    print("\n✓ Loss computation successful!")


def demo_roi_detection():
    """Demo: ROI detection and alerting"""
    print("\n" + "=" * 70)
    print("DEMO 3: ROI Detection and Alerting")
    print("=" * 70)
    
    # Create synthetic image
    image, gt_points = create_synthetic_image(num_people=60)
    
    print(f"Synthetic image created: {image.shape}")
    print(f"Ground truth: {len(gt_points)} people")
    
    # Define ROI polygon (center region)
    roi_polygon = np.array([
        [300, 200],
        [700, 200],
        [700, 550],
        [300, 550]
    ])
    
    # Create ROI detector
    roi_detector = ROIDetector(roi_polygon)
    roi_count = roi_detector.count_in_roi(gt_points)
    
    print(f"\nROI polygon defined with {len(roi_polygon)} corners")
    print(f"People in ROI: {roi_count}")
    print(f"Alert status: {roi_detector.is_alert}")
    print(f"Message: {roi_detector.get_alert_message()}")
    
    # Visualize
    annotated = visualize_predictions(image, gt_points, roi_detector=roi_detector)
    
    # Save result
    output_path = Path("results/demo_roi_detection.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    
    print(f"\n✓ Visualization saved to: {output_path}")


def demo_inference():
    """Demo: End-to-end inference"""
    print("\n" + "=" * 70)
    print("DEMO 4: End-to-End Inference")
    print("=" * 70)
    
    # Create synthetic image
    image, gt_points = create_synthetic_image(num_people=45)
    
    print("Creating synthetic crowd scene...")
    print(f"Ground truth: {len(gt_points)} people")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(pretrained=False)
    model.eval()
    
    # Preprocess image
    import torchvision.transforms as T
    from PIL import Image as PILImage
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = PILImage.fromarray(image_rgb)
    
    transform = T.Compose([
        T.Resize((Config.IMG_SIZE[0], Config.IMG_SIZE[1])),
        T.ToTensor(),
        T.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ])
    
    input_tensor = transform(image_pil).unsqueeze(0)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Inference
    print("\nRunning inference...")
    with torch.no_grad():
        points_batch, scores_batch = model.predict(input_tensor, score_threshold=0.3)
    
    points = points_batch[0].cpu().numpy()
    scores = scores_batch[0].cpu().numpy()
    
    # Scale back to original size
    points[:, 0] *= (image.shape[1] / Config.IMG_SIZE[1])
    points[:, 1] *= (image.shape[0] / Config.IMG_SIZE[0])
    
    print(f"\nResults:")
    print(f"  Detected: {len(points)} people")
    print(f"  Ground truth: {len(gt_points)} people")
    print(f"  Error: {abs(len(points) - len(gt_points))}")
    
    # Visualize
    annotated = visualize_predictions(image, points, scores)
    
    # Save
    output_path = Path("results/demo_inference.jpg")
    cv2.imwrite(str(output_path), annotated)
    
    print(f"\n✓ Result saved to: {output_path}")


def demo_training_step():
    """Demo: Single training iteration"""
    print("\n" + "=" * 70)
    print("DEMO 5: Single Training Step")
    print("=" * 70)
    
    # Create model and optimizer
    model = build_model(pretrained=False)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = P2PNetLoss()
    
    # Create synthetic batch
    batch_size = 2
    images = torch.randn(batch_size, 3, 768, 1024)
    points = torch.randn(batch_size, 100, 2) * 100 + 400
    counts = torch.tensor([45, 60])
    
    print(f"Batch size: {batch_size}")
    print(f"Ground truth counts: {counts.tolist()}")
    
    # Forward pass
    print("\nForward pass...")
    outputs = model(images)
    
    # Compute loss
    print("Computing loss...")
    targets = {'points': points, 'counts': counts}
    losses = criterion(outputs, targets)
    
    # Backward pass
    print("Backward pass...")
    optimizer.zero_grad()
    losses['loss'].backward()
    optimizer.step()
    
    print(f"\nTraining step completed:")
    print(f"  Loss: {losses['loss'].item():.4f}")
    print(f"  Coordinate loss: {losses['coord_loss'].item():.4f}")
    print(f"  Classification loss: {losses['cls_loss'].item():.4f}")
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"  Gradients computed: {has_grad}")
    
    print("\n✓ Training step successful!")


def run_all_demos():
    """Run all demos"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           P2PNet Crowd Monitoring System - Demo Suite            ║
║                                                                   ║
║  This demo suite tests all components with synthetic data        ║
║  No dataset required!                                            ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    Config.create_dirs()
    
    try:
        # Run demos
        demo_model_forward_pass()
        demo_loss_computation()
        demo_roi_detection()
        demo_inference()
        demo_training_step()
        
        # Summary
        print("\n" + "=" * 70)
        print("DEMO SUITE COMPLETE!")
        print("=" * 70)
        print("\nAll components tested successfully! ✓")
        print("\nNext steps:")
        print("1. Download the ShanghaiTech dataset")
        print("2. Run: python train.py --part A")
        print("3. Run: streamlit run app.py")
        print("\nCheck the 'results/' folder for demo outputs.")
        print("\n" + "=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_demos()
