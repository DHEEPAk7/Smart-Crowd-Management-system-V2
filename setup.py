#!/usr/bin/env python3
"""
Setup script for Smart Crowd Monitoring System
Automates environment setup, dependency installation, and directory creation
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header(message):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70 + "\n")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")


def create_directories():
    """Create necessary project directories"""
    print_header("Creating Project Directories")
    
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
        print(f"✓ Created: {dir_path}")


def install_dependencies():
    """Install required Python packages"""
    print_header("Installing Dependencies")
    
    try:
        # Upgrade pip
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("\nInstalling required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("\n✓ All dependencies installed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        print("   Please install manually using: pip install -r requirements.txt")
        sys.exit(1)


def verify_installation():
    """Verify that key packages are installed correctly"""
    print_header("Verifying Installation")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'streamlit': 'Streamlit',
        'scipy': 'SciPy',
        'numpy': 'NumPy'
    }
    
    all_good = True
    for package, name in packages.items():
        try:
            if package == 'cv2':
                import cv2
                version = cv2.__version__
            else:
                module = __import__(package)
                version = module.__version__
            
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: NOT INSTALLED")
            all_good = False
    
    if not all_good:
        print("\n⚠️  Some packages are missing. Please install them manually.")
        sys.exit(1)


def check_cuda():
    """Check CUDA availability"""
    print_header("Checking CUDA Support")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA is not available - will use CPU")
            print("   For GPU acceleration, install CUDA toolkit and PyTorch with CUDA support")
            print("   Visit: https://pytorch.org/get-started/locally/")
    
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check CUDA")


def create_sample_config():
    """Create a sample configuration file"""
    print_header("Creating Sample Configuration")
    
    sample_config = """# Sample configuration for quick testing
# Copy this to config_custom.py and modify as needed

# Quick Test Settings
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# For CPU testing (if no GPU available)
# import torch
# DEVICE = torch.device("cpu")
# USE_AMP = False

print("Sample configuration ready for testing!")
"""
    
    with open('config_sample.py', 'w') as f:
        f.write(sample_config)
    
    print("✓ Created config_sample.py")


def print_next_steps():
    """Print instructions for next steps"""
    print_header("Setup Complete!")
    
    print("""
Next Steps:

1. Download the ShanghaiTech dataset:
   - Visit: https://github.com/desenzhou/ShanghaiTechDataset
   - Extract to: ./data/ShanghaiTech/

2. Verify dataset structure:
   python -c "from dataset import create_dataloaders; create_dataloaders('A')"

3. Train the model:
   python train.py --part A --epochs 100

4. Run inference:
   python inference.py --input path/to/image.jpg --checkpoint checkpoints/best_model.pth

5. Launch Streamlit dashboard:
   streamlit run app.py

For detailed instructions, see README.md

Happy crowd counting! 👥
""")


def main():
    """Main setup function"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║        Smart Crowd Monitoring System - Setup Script              ║
║                    P2PNet Implementation                          ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Step 1: Check Python version
        check_python_version()
        
        # Step 2: Create directories
        create_directories()
        
        # Step 3: Install dependencies
        install_choice = input("\nInstall dependencies from requirements.txt? (y/n): ")
        if install_choice.lower() == 'y':
            install_dependencies()
            verify_installation()
            check_cuda()
        else:
            print("⚠️  Skipping dependency installation")
            print("   Please install manually: pip install -r requirements.txt")
        
        # Step 4: Create sample config
        create_sample_config()
        
        # Step 5: Print next steps
        print_next_steps()
        
        print("\n✓ Setup completed successfully!\n")
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
