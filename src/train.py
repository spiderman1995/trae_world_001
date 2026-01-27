import os
import argparse
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from src.data.dataset import StockDataset
from src.models.feature_extractor import FeatureExtractor
from src.models.transformer import StockViT
from src.models.loss import MultiTaskLoss, QuantileLoss, SmoothQuantileLoss, PairwiseRankingLoss, physics_constraint_loss

# Setup Logging
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def train(args):
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting training with args: {args}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Dataset
    logger.info("Initializing dataset...")
    train_dataset = StockDataset(args.data_dir, seq_len=args.seq_len, pred_len=args.pred_len, mode='train')
    # Use smaller batch size for laptop
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    logger.info(f"Dataset size: {len(train_dataset)} samples")
    logger.info(f"Batch size: {args.batch_size}, Batches per epoch: {len(train_loader)}")
    
    # 2. Models
    logger.info("Building models...")
    # 1DCNN Feature Extractor
    # Reduce dimension from 10000 to 1024 for laptop 4050 GPU (6GB VRAM)
    embed_dim = 1024 # Was 10000
    feature_extractor = FeatureExtractor(input_channels=18, output_dim=embed_dim).to(device)
    # ViT
    vit_model = StockViT(seq_len=args.seq_len, embed_dim=embed_dim, depth=args.depth, num_heads=args.num_heads).to(device)
    
    logger.info(f"Models moved to {device}. Embed Dim: {embed_dim}")
    
    # 3. Losses
    mtl_loss_wrapper = MultiTaskLoss(num_tasks=4).to(device)
    
    criterion_high = SmoothQuantileLoss(quantile=0.9, delta=0.1)
    criterion_low = QuantileLoss(quantile=0.1) # Or SmoothQuantileLoss
    criterion_sharpe = PairwiseRankingLoss(margin=0.1)
    criterion_dir = nn.BCEWithLogitsLoss()
    
    # 4. Optimizer
    # Optimize all parameters
    params = list(feature_extractor.parameters()) + list(vit_model.parameters()) + list(mtl_loss_wrapper.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # Logging
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Training Loop
    global_step = 0
    feature_extractor.train()
    vit_model.train()
    
    logger.info("Starting training loop...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{args.epochs} started")
        epoch_loss = 0
        
        # Enhanced TQDM
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=True)
        
        for batch_idx, (seq_data, targets) in enumerate(pbar):
            # seq_data: [B, 180, 18, 1442]
            # Reshape for CNN: [B*180, 18, 1442]
            B, Seq, C, L = seq_data.shape
            
            # Move to device
            seq_data_flat = seq_data.view(B * Seq, C, L).to(device)
            
            target_high = targets['high'].to(device)
            target_low = targets['low'].to(device)
            target_sharpe = targets['sharpe'].to(device)
            target_dir = targets['direction'].to(device)
            current_price = targets['current_price'].to(device)
            
            optimizer.zero_grad()
            
            # Use torch.amp.autocast for newer PyTorch versions
            with torch.amp.autocast('cuda'):
                # 1. Feature Extraction
                # [B*180, 10000]
                features_flat = feature_extractor(seq_data_flat)
                
                # Reshape for ViT: [B, 180, 10000]
                features_seq = features_flat.view(B, Seq, -1)
                
                # 2. ViT
                outputs = vit_model(features_seq)
                
                pred_high = outputs['high']
                pred_low = outputs['low']
                pred_sharpe = outputs['sharpe']
                pred_dir = outputs['direction']
                
                # 3. Calculate Losses
                # L1: High Price
                # Ensure squeeze() doesn't remove batch dim if B=1
                loss_high = criterion_high(pred_high.view(-1), target_high.view(-1))
                
                # L2: Low Price
                loss_low = criterion_low(pred_low.view(-1), target_low.view(-1))
                
                # L3: Sharpe Ranking
                loss_sharpe = criterion_sharpe(pred_sharpe, target_sharpe)
                
                # L4: Direction BCE
                # Target must be float for BCEWithLogitsLoss
                loss_dir = criterion_dir(pred_dir.view(-1), target_dir.view(-1).float())
                
                # Physics Constraint
                # Extract BigBuy feature from last time step
                # Assume BigBuy is feature index 9
                # seq_data_flat: [B*Seq, 18, 1442]
                # We need the last day (index Seq-1 for each batch), last tick? or sum?
                # User: "BigBuy > Threshold". Usually sum of BigBuy volume ratio.
                # Feature index 9 is "buy_big_order_amt_ratio".
                # We can take the mean or max over the day.
                # "Big Buy" is a sparse signal, MAX or SUM is better than MEAN to capture the signal strength.
                last_day_indices = torch.arange(Seq - 1, B * Seq, Seq, device=device)
                last_day_data = seq_data_flat[last_day_indices] # [B, 18, 1442]
                big_buy_feat = last_day_data[:, 9, :].max(dim=1)[0] # [B] Take Max over ticks
                
                loss_physics = physics_constraint_loss(pred_high, None, big_buy_feat, threshold=0.3)
                
                # Combine Losses
                # Only combine 4 main losses with dynamic weights
                losses_list = torch.stack([loss_high, loss_low, loss_sharpe, loss_dir])
                loss_mtl = mtl_loss_wrapper(losses_list)
                
                total_loss = loss_mtl + 0.1 * loss_physics
            
            # Backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            global_step += 1
            
            # Logging
            if batch_idx % 10 == 0:
                writer.add_scalar("Loss/Total", total_loss.item(), global_step)
                writer.add_scalar("Loss/High", loss_high.item(), global_step)
                writer.add_scalar("Loss/Low", loss_low.item(), global_step)
                writer.add_scalar("Loss/Sharpe", loss_sharpe.item(), global_step)
                writer.add_scalar("Loss/Dir", loss_dir.item(), global_step)
                writer.add_scalar("Loss/Physics", loss_physics.item(), global_step)
                
                # Log weights
                sigmas = torch.exp(mtl_loss_wrapper.log_vars)
                writer.add_scalar("Weight/High", 1/(2*sigmas[0].item()**2), global_step)
                writer.add_scalar("Weight/Low", 1/(2*sigmas[1].item()**2), global_step)
                writer.add_scalar("Weight/Sharpe", 1/(2*sigmas[2].item()**2), global_step)
                writer.add_scalar("Weight/Dir", 1/(2*sigmas[3].item()**2), global_step)
                
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}", 
                "H": f"{loss_high.item():.2f}",
                "L": f"{loss_low.item():.2f}",
                "S": f"{loss_sharpe.item():.2f}",
                "D": f"{loss_dir.item():.2f}"
            })
            
        epoch_duration = time.time() - epoch_start_time
        mean_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Completed. Mean Loss: {mean_epoch_loss:.4f}. Duration: {epoch_duration:.2f}s")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'feature_extractor': feature_extractor.state_dict(),
            'vit': vit_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    total_duration = time.time() - start_time
    logger.info(f"Training completed in {total_duration/60:.2f} minutes.")
    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"D:\temp\0_tempdata7")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2) # Small batch for 4050 + Large Model
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=180)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--depth", type=int, default=4) # Smaller ViT for test
    parser.add_argument("--num_heads", type=int, default=4)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main()
