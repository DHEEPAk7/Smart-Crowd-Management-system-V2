"""
Streamlit Dashboard for Smart Crowd Monitoring System
Supports video upload, webcam stream, and ROI-based alerting
"""

import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import time

from config import Config
from models import build_model
from utils import ROIDetector, visualize_predictions, denormalize_image
import torchvision.transforms as T


class CrowdMonitoringApp:
    """
    Streamlit application for real-time crowd monitoring
    """
    
    def __init__(self):
        """Initialize the application"""
        st.set_page_config(
            page_title="Smart Crowd Monitoring System",
            page_icon="👥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.setup_session_state()
        self.load_model()
        
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'roi_points' not in st.session_state:
            st.session_state.roi_points = []
        if 'roi_detector' not in st.session_state:
            st.session_state.roi_detector = ROIDetector()
        if 'processing' not in st.session_state:
            st.session_state.processing = False
    
    @st.cache_resource
    def load_model(_self, checkpoint_path: str = None):
        """
        Load P2PNet model
        
        Args:
            checkpoint_path: Path to model checkpoint (optional)
        """
        with st.spinner('Loading P2PNet model...'):
            model = build_model(pretrained=True)
            model.to(Config.DEVICE)
            model.eval()
            
            # Load checkpoint if provided
            if checkpoint_path and Path(checkpoint_path).exists():
                checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                st.success(f'Loaded checkpoint from {checkpoint_path}')
            
            return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (Config.IMG_SIZE[1], Config.IMG_SIZE[0]))
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        # Apply transforms
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def process_frame(self, frame: np.ndarray, model: torch.nn.Module) -> tuple:
        """
        Process a single frame
        
        Args:
            frame: Input frame (BGR)
            model: P2PNet model
            
        Returns:
            Tuple of (annotated_frame, count, roi_count)
        """
        # Preprocess
        input_tensor = self.preprocess_image(frame).to(Config.DEVICE)
        
        # Inference
        with torch.no_grad():
            points_batch, scores_batch = model.predict(input_tensor, Config.SCORE_THRESHOLD)
        
        # Get points for first (and only) image in batch
        points = points_batch[0].cpu().numpy()
        scores = scores_batch[0].cpu().numpy()
        
        # Scale points back to original frame size
        h_ratio = frame.shape[0] / Config.IMG_SIZE[0]
        w_ratio = frame.shape[1] / Config.IMG_SIZE[1]
        points[:, 0] *= w_ratio
        points[:, 1] *= h_ratio
        
        # Count in ROI
        roi_count = 0
        if len(st.session_state.roi_points) > 0:
            roi_polygon = np.array(st.session_state.roi_points)
            roi_count = st.session_state.roi_detector.count_in_roi(points)
        
        # Visualize
        annotated_frame = visualize_predictions(
            frame,
            points,
            scores,
            st.session_state.roi_detector if len(st.session_state.roi_points) > 0 else None
        )
        
        return annotated_frame, len(points), roi_count
    
    def process_video(self, video_path: str, model: torch.nn.Module):
        """
        Process video file
        
        Args:
            video_path: Path to video file
            model: P2PNet model
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error: Could not open video file")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create placeholders
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame
            annotated_frame, count, roi_count = self.process_frame(frame, model)
            
            # Display
            video_placeholder.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )
            
            # Update stats
            stats_placeholder.metric(
                label="Total Count",
                value=count,
                delta=f"ROI: {roi_count}" if len(st.session_state.roi_points) > 0 else None
            )
            
            # Alert if threshold exceeded
            if st.session_state.roi_detector.is_alert:
                st.warning(st.session_state.roi_detector.get_alert_message())
            
            # Update progress
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
            
            # Control playback speed
            time.sleep(1 / fps)
        
        cap.release()
        st.success("Video processing completed!")
    
    def process_webcam(self, model: torch.nn.Module):
        """
        Process webcam stream
        
        Args:
            model: P2PNet model
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not access webcam")
            return
        
        # Create placeholders
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        stop_button = st.button("Stop Webcam")
        
        while not stop_button:
            ret, frame = cap.read()
            
            if not ret:
                st.error("Error: Could not read frame from webcam")
                break
            
            # Process frame
            annotated_frame, count, roi_count = self.process_frame(frame, model)
            
            # Display
            video_placeholder.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )
            
            # Update stats
            stats_placeholder.metric(
                label="Total Count",
                value=count,
                delta=f"ROI: {roi_count}" if len(st.session_state.roi_points) > 0 else None
            )
            
            # Alert if threshold exceeded
            if st.session_state.roi_detector.is_alert:
                st.warning(st.session_state.roi_detector.get_alert_message())
        
        cap.release()
    
    def roi_selector_ui(self):
        """UI for ROI polygon selection"""
        st.sidebar.header("🎯 Region of Interest (ROI)")
        
        st.sidebar.write("Define a polygon ROI by entering corner points:")
        
        # Number of points
        num_points = st.sidebar.number_input(
            "Number of corners",
            min_value=3,
            max_value=10,
            value=4
        )
        
        # Input fields for each point
        points = []
        cols = st.sidebar.columns(2)
        
        for i in range(num_points):
            with cols[0]:
                x = st.number_input(f"Point {i+1} - X", value=0, key=f"x_{i}")
            with cols[1]:
                y = st.number_input(f"Point {i+1} - Y", value=0, key=f"y_{i}")
            points.append([x, y])
        
        # Set ROI button
        if st.sidebar.button("Set ROI"):
            st.session_state.roi_points = points
            roi_polygon = np.array(points)
            st.session_state.roi_detector.set_roi(roi_polygon)
            st.sidebar.success("ROI set successfully!")
        
        # Clear ROI button
        if st.sidebar.button("Clear ROI"):
            st.session_state.roi_points = []
            st.session_state.roi_detector.set_roi(None)
            st.sidebar.success("ROI cleared!")
        
        # Display current ROI
        if len(st.session_state.roi_points) > 0:
            st.sidebar.write("Current ROI points:")
            st.sidebar.write(st.session_state.roi_points)
    
    def settings_ui(self):
        """Settings panel"""
        st.sidebar.header("⚙️ Settings")
        
        # Score threshold
        score_threshold = st.sidebar.slider(
            "Detection Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=Config.SCORE_THRESHOLD,
            step=0.05
        )
        Config.SCORE_THRESHOLD = score_threshold
        
        # ROI alert threshold
        roi_threshold = st.sidebar.number_input(
            "ROI Alert Threshold",
            min_value=1,
            max_value=500,
            value=Config.ROI_COUNT_THRESHOLD
        )
        Config.ROI_COUNT_THRESHOLD = roi_threshold
        
        # Point visualization
        point_radius = st.sidebar.slider(
            "Point Radius",
            min_value=1,
            max_value=10,
            value=Config.POINT_RADIUS
        )
        Config.POINT_RADIUS = point_radius
    
    def run(self):
        """Main application loop"""
        # Header
        st.title("👥 Smart Crowd Monitoring System")
        st.markdown("### Real-time crowd counting with P2PNet")
        
        # Sidebar
        self.settings_ui()
        self.roi_selector_ui()
        
        # Load model
        checkpoint_path = st.sidebar.text_input(
            "Checkpoint Path (optional)",
            value=""
        )
        
        model = self.load_model(checkpoint_path if checkpoint_path else None)
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["📹 Video Upload", "📷 Webcam", "ℹ️ Info"])
        
        with tab1:
            st.header("Upload Video")
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv']
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                if st.button("Process Video"):
                    self.process_video(tmp_path, model)
        
        with tab2:
            st.header("Webcam Stream")
            if st.button("Start Webcam"):
                self.process_webcam(model)
        
        with tab3:
            st.header("System Information")
            
            st.markdown("""
            ## About P2PNet
            
            P2PNet (Point-to-Point Network) is a state-of-the-art crowd counting method that:
            - Uses point-based localization instead of density maps
            - Employs VGG-16 backbone with Feature Pyramid Network
            - Achieves optimal matching via Hungarian Algorithm
            - Provides precise head location coordinates
            
            ## Features
            
            - **Real-time Processing**: Fast inference with AMP support
            - **ROI Monitoring**: Define custom regions and get alerts
            - **Multiple Sources**: Support for video files and webcam
            - **Accurate Counting**: Point-based approach for better precision
            
            ## Usage Instructions
            
            1. **Set Detection Threshold**: Adjust confidence threshold in settings
            2. **Define ROI** (optional): Enter polygon corners for specific area monitoring
            3. **Upload Video or Start Webcam**: Choose your input source
            4. **Monitor Results**: View real-time counts and alerts
            
            ## Configuration
            
            - **Model**: P2PNet with VGG-16 backbone
            - **Input Size**: 768 × 1024
            - **Score Threshold**: {:.2f}
            - **ROI Alert Threshold**: {}
            - **Device**: {}
            """.format(
                Config.SCORE_THRESHOLD,
                Config.ROI_COUNT_THRESHOLD,
                Config.DEVICE
            ))
            
            # Display model info
            if model is not None:
                st.subheader("Model Statistics")
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Parameters", f"{total_params:,}")
                col2.metric("Trainable Parameters", f"{trainable_params:,}")
                col3.metric("Model Size", f"{total_params * 4 / 1024 / 1024:.2f} MB")


def main():
    """Entry point"""
    app = CrowdMonitoringApp()
    app.run()


if __name__ == "__main__":
    main()
