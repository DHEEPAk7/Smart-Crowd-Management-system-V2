# Quick Start Guide

Get up and running with the Smart Crowd Monitoring System in 5 minutes!

## 🚀 Fastest Path to Running

### Option 1: Test with Demo (No Dataset Needed)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo suite
python demo.py
```

This will test all components with synthetic data and create sample visualizations in `results/`.

### Option 2: Train on ShanghaiTech

```bash
# 1. Setup environment
python setup.py

# 2. Download dataset
# Visit: https://github.com/desenzhou/ShanghaiTechDataset
# Extract to: ./data/ShanghaiTech/

# 3. Train
python train.py --part A --epochs 100

# 4. Monitor training
tensorboard --logdir logs/
```

### Option 3: Launch Dashboard

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Streamlit
streamlit run app.py

# 3. Upload video or use webcam
# Dashboard will open at http://localhost:8501
```

## 📦 What's Included

```
├── config.py          - All settings in one place
├── dataset.py         - ShanghaiTech data loader
├── models.py          - P2PNet architecture
├── utils.py           - Hungarian matcher, losses, ROI
├── train.py           - Full training pipeline
├── inference.py       - Standalone inference
├── app.py            - Streamlit dashboard
├── demo.py           - Test without dataset
└── setup.py          - Automated setup
```

## ⚡ Common Commands

### Training
```bash
# Part A (dense crowds)
python train.py --part A --epochs 1500

# Part B (sparse crowds)
python train.py --part B --epochs 1500

# Resume from checkpoint
python train.py --part A --resume checkpoints/checkpoint_epoch_500.pth
```

### Inference
```bash
# Single image
python inference.py --input image.jpg --checkpoint checkpoints/best_model.pth --output result.jpg

# Video with ROI
python inference.py --input video.mp4 --checkpoint checkpoints/best_model.pth --roi "100,100,500,100,500,400,100,400"
```

### Streamlit Dashboard
```bash
streamlit run app.py
```

## 🎯 Key Features Quick Reference

| Feature | File | Description |
|---------|------|-------------|
| Hungarian Matching | `utils.py` | Optimal point assignment |
| FPN | `models.py` | Multi-scale features |
| AMP Training | `train.py` | Faster training |
| ROI Alerting | `utils.py` | Zone monitoring |
| Real-time UI | `app.py` | Interactive dashboard |

## 🔧 Quick Configuration

Edit `config.py`:

```python
# For faster testing
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

# For CPU-only
import torch
DEVICE = torch.device("cpu")
USE_AMP = False

# For more detections
SCORE_THRESHOLD = 0.3

# For ROI alerts
ROI_COUNT_THRESHOLD = 30
```

## 🐛 Troubleshooting One-Liners

```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test dataset loading
python -c "from dataset import create_dataloaders; create_dataloaders('A')"

# Verify model
python -c "from models import build_model; m = build_model(); print('OK')"

# Check dependencies
python -c "import torch, cv2, streamlit; print('All OK')"
```

## 📊 Expected Performance

| Metric | Part A | Part B |
|--------|--------|--------|
| MAE | ~60-70 | ~8-10 |
| Training | ~8-12 hours | ~4-6 hours |
| Inference | ~30 FPS (GPU) | ~30 FPS (GPU) |

## 🎓 Learning Resources

1. **P2PNet Paper**: Understanding the architecture
2. **Hungarian Algorithm**: Why optimal matching matters
3. **Feature Pyramid Networks**: Multi-scale detection
4. **Crowd Counting**: Overview and challenges

## 💡 Pro Tips

1. **Start Small**: Test with `demo.py` first
2. **Use Checkpoints**: Save frequently during training
3. **Monitor Logs**: TensorBoard is your friend
4. **Tune Threshold**: Adjust based on your use case
5. **ROI First**: Define zones before processing video

## 🚨 Common Gotchas

❌ **CUDA OOM**: Reduce batch size  
❌ **Dataset not found**: Check path in `config.py`  
❌ **Slow training**: Enable AMP, increase workers  
❌ **Poor accuracy**: Train longer, tune learning rate  
❌ **No detections**: Lower score threshold  

## 📞 Need Help?

1. Run demo: `python demo.py`
2. Check README: Full documentation
3. Open issue: GitHub issues
4. Review code: Extensive comments

---

**Ready to go!** Start with `python demo.py` to verify everything works.
