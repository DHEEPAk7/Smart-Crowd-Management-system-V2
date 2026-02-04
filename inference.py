"""
Standalone inference script for P2PNet
Test the model on images or videos without Streamlit
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from config import Config
from models import build_model
from utils import ROIDetector, visualize_predictions
import torchvision.transforms as T
from PIL import Image


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess image for model inference
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Preprocessed tensor
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(image_rgb, (Config.IMG_SIZE[1], Config.IMG_SIZE[0]))
    
    # Convert to PIL
    image_pil = Image.fromarray(image_resized)
    
    # Transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ])
    
    tensor = transform(image_pil).unsqueeze(0)
    
    return tensor


def inference_image(
    model: torch.nn.Module,
    image_path: str,
    output_path: str = None,
    roi_polygon: np.ndarray = None
):
    """
    Run inference on a single image
    
    Args:
        model: P2PNet model
        image_path: Path to input image
        output_path: Path to save annotated image (optional)
        roi_polygon: ROI polygon coordinates (optional)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_h, original_w = image.shape[:2]
    
    # Preprocess
    input_tensor = preprocess_image(image).to(Config.DEVICE)
    
    # Inference
    model.eval()
    with torch.no_grad():
        points_batch, scores_batch = model.predict(input_tensor, Config.SCORE_THRESHOLD)
    
    # Get results
    points = points_batch[0].cpu().numpy()
    scores = scores_batch[0].cpu().numpy()
    
    # Scale points back to original size
    points[:, 0] *= (original_w / Config.IMG_SIZE[1])
    points[:, 1] *= (original_h / Config.IMG_SIZE[0])
    
    # Setup ROI detector if provided
    roi_detector = None
    roi_count = 0
    if roi_polygon is not None:
        roi_detector = ROIDetector(roi_polygon)
        roi_count = roi_detector.count_in_roi(points)
    
    # Visualize
    annotated = visualize_predictions(image, points, scores, roi_detector)
    
    # Display results
    print(f"\nResults for {Path(image_path).name}:")
    print(f"  Total count: {len(points)}")
    if roi_detector is not None:
        print(f"  ROI count: {roi_count}")
        print(f"  Alert: {roi_detector.is_alert}")
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"  Saved to: {output_path}")
    
    # Display (optional)
    cv2.imshow('P2PNet Inference', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return len(points), roi_count


def inference_video(
    model: torch.nn.Module,
    video_path: str,
    output_path: str = None,
    roi_polygon: np.ndarray = None
):
    """
    Run inference on a video
    
    Args:
        model: P2PNet model
        video_path: Path to input video
        output_path: Path to save annotated video (optional)
        roi_polygon: ROI polygon coordinates (optional)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Setup ROI detector
    roi_detector = None
    if roi_polygon is not None:
        roi_detector = ROIDetector(roi_polygon)
    
    # Process frames
    model.eval()
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    total_count_sum = 0
    roi_count_sum = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        input_tensor = preprocess_image(frame).to(Config.DEVICE)
        
        # Inference
        with torch.no_grad():
            points_batch, scores_batch = model.predict(input_tensor, Config.SCORE_THRESHOLD)
        
        # Get results
        points = points_batch[0].cpu().numpy()
        scores = scores_batch[0].cpu().numpy()
        
        # Scale points back
        points[:, 0] *= (width / Config.IMG_SIZE[1])
        points[:, 1] *= (height / Config.IMG_SIZE[0])
        
        # ROI count
        roi_count = 0
        if roi_detector is not None:
            roi_count = roi_detector.count_in_roi(points)
            roi_count_sum += roi_count
        
        # Visualize
        annotated = visualize_predictions(frame, points, scores, roi_detector)
        
        # Write frame
        if writer is not None:
            writer.write(annotated)
        
        # Display
        cv2.imshow('P2PNet Video Inference', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Update stats
        total_count_sum += len(points)
        frame_count += 1
        pbar.update(1)
    
    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    pbar.close()
    
    # Print statistics
    print(f"\nVideo processing complete!")
    print(f"  Total frames: {frame_count}")
    print(f"  Average count: {total_count_sum / frame_count:.2f}")
    if roi_detector is not None:
        print(f"  Average ROI count: {roi_count_sum / frame_count:.2f}")
    if output_path:
        print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='P2PNet Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output (optional)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Detection confidence threshold')
    parser.add_argument('--roi', type=str, default=None,
                        help='ROI polygon as comma-separated coordinates: x1,y1,x2,y2,...')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Update config
    if args.threshold is not None:
        Config.SCORE_THRESHOLD = args.threshold
    if args.device is not None:
        Config.DEVICE = torch.device(args.device)
    
    # Parse ROI
    roi_polygon = None
    if args.roi:
        coords = [float(x) for x in args.roi.split(',')]
        if len(coords) % 2 != 0:
            raise ValueError("ROI coordinates must be in pairs (x,y)")
        roi_polygon = np.array(coords).reshape(-1, 2)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = build_model(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    model.eval()
    print("Model loaded successfully!")
    
    # Determine input type
    input_path = Path(args.input)
    if not input_path.exists():
        raise ValueError(f"Input file not found: {args.input}")
    
    # Run inference
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Image inference
        inference_image(model, str(input_path), args.output, roi_polygon)
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video inference
        inference_video(model, str(input_path), args.output, roi_polygon)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


if __name__ == "__main__":
    main()
