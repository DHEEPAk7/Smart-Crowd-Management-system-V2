"""
Training script for P2PNet with Automatic Mixed Precision
Includes cosine annealing scheduler and checkpoint management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from tqdm import tqdm
import argparse

from config import Config
from dataset import create_dataloaders
from models import build_model
from utils import P2PNetLoss, MetricTracker


class Trainer:
    """
    Training manager for P2PNet
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device = None
    ):
        """
        Args:
            model: P2PNet model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else Config.DEVICE
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function
        self.criterion = P2PNetLoss(
            lambda_coord=Config.LAMBDA_COORD,
            lambda_class=Config.LAMBDA_CLASS
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=Config.T_MAX,
            eta_min=Config.ETA_MIN
        )
        
        # AMP scaler for mixed precision training
        self.scaler = GradScaler() if Config.USE_AMP else None
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=Config.LOG_DIR)
        
        # Tracking
        self.current_epoch = 0
        self.best_mae = float('inf')
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        
        print(f"Trainer initialized on {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{Config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)
            points = batch['points'].to(self.device)
            counts = batch['counts'].to(self.device)
            
            # Forward pass with AMP
            if Config.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    
                    targets = {'points': points, 'counts': counts}
                    losses = self.criterion(outputs, targets)
                    loss = losses['loss']
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                
                targets = {'points': points, 'counts': counts}
                losses = self.criterion(outputs, targets)
                loss = losses['loss']
                
                # Standard backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Compute predicted counts (number of points with high confidence)
            with torch.no_grad():
                pred_logits = outputs['pred_logits']
                pred_probs = torch.softmax(pred_logits, dim=-1)
                foreground_probs = pred_probs[:, :, 1]
                pred_counts = (foreground_probs > Config.SCORE_THRESHOLD).sum(dim=1)
            
            # Update metrics
            for i in range(images.size(0)):
                self.train_metrics.update(
                    loss=loss.item(),
                    coord_loss=losses['coord_loss'].item(),
                    cls_loss=losses['cls_loss'].item(),
                    pred_count=pred_counts[i].item(),
                    gt_count=counts[i].item()
                )
            
            # Update progress bar
            if batch_idx % Config.LOG_INTERVAL == 0:
                metrics = self.train_metrics.compute()
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'mae': f"{metrics['mae']:.2f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        
        # Log to tensorboard
        self.writer.add_scalar('Train/Loss', metrics['loss'], self.current_epoch)
        self.writer.add_scalar('Train/CoordLoss', metrics['coord_loss'], self.current_epoch)
        self.writer.add_scalar('Train/ClsLoss', metrics['cls_loss'], self.current_epoch)
        self.writer.add_scalar('Train/MAE', metrics['mae'], self.current_epoch)
        self.writer.add_scalar('Train/RMSE', metrics['rmse'], self.current_epoch)
        self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on validation set"""
        self.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(self.val_loader, desc="Validating")
        
        for batch in pbar:
            # Move data to device
            images = batch['images'].to(self.device)
            points = batch['points'].to(self.device)
            counts = batch['counts'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            targets = {'points': points, 'counts': counts}
            losses = self.criterion(outputs, targets)
            
            # Compute predicted counts
            pred_logits = outputs['pred_logits']
            pred_probs = torch.softmax(pred_logits, dim=-1)
            foreground_probs = pred_probs[:, :, 1]
            pred_counts = (foreground_probs > Config.SCORE_THRESHOLD).sum(dim=1)
            
            # Update metrics
            for i in range(images.size(0)):
                self.val_metrics.update(
                    loss=losses['loss'].item(),
                    coord_loss=losses['coord_loss'].item(),
                    cls_loss=losses['cls_loss'].item(),
                    pred_count=pred_counts[i].item(),
                    gt_count=counts[i].item()
                )
        
        # Compute epoch metrics
        metrics = self.val_metrics.compute()
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', metrics['loss'], self.current_epoch)
        self.writer.add_scalar('Val/MAE', metrics['mae'], self.current_epoch)
        self.writer.add_scalar('Val/RMSE', metrics['rmse'], self.current_epoch)
        
        return metrics
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth', is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_mae': self.best_mae,
            'config': {
                'num_points': Config.MAX_POINTS,
                'num_classes': Config.NUM_CLASSES,
                'img_size': Config.IMG_SIZE
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = Config.CHECKPOINT_DIR / filename
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = Config.CHECKPOINT_DIR / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_mae = checkpoint['best_mae']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int = None):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs (defaults to Config.NUM_EPOCHS)
        """
        if num_epochs is None:
            num_epochs = Config.NUM_EPOCHS
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Step scheduler
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.2f} | RMSE: {train_metrics['rmse']:.2f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.2f} | RMSE: {val_metrics['rmse']:.2f}")
            
            # Save checkpoint
            if (epoch + 1) % Config.SAVE_INTERVAL == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Save best model
            if val_metrics['mae'] < self.best_mae:
                self.best_mae = val_metrics['mae']
                self.save_checkpoint(is_best=True)
                print(f"  ✓ New best MAE: {self.best_mae:.2f}")
            
            print("-" * 70)
        
        # Training complete
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/3600:.2f} hours")
        print(f"Best validation MAE: {self.best_mae:.2f}")
        
        self.writer.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train P2PNet for crowd counting')
    parser.add_argument('--part', type=str, default='A', choices=['A', 'B'],
                        help='ShanghaiTech dataset part')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default from config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pretrained backbone')
    
    args = parser.parse_args()
    
    # Create directories
    Config.create_dirs()
    Config.print_config()
    
    # Create dataloaders
    print(f"\nLoading ShanghaiTech Part {args.part} dataset...")
    train_loader, val_loader = create_dataloaders(
        part=args.part,
        batch_size=args.batch_size
    )
    
    # Build model
    print("\nBuilding P2PNet model...")
    model = build_model(pretrained=args.pretrained)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
