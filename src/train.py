import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from src.models.loss import MultiTaskLoss, PeakDayLoss

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
    train_dataset = StockDataset(
        args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=args.start_date,
        end_date=args.end_date
    )
    train_size = len(train_dataset)
    if train_size <= 0:
        raise ValueError(
            f"Empty train dataset. data_dir={args.data_dir}, date_range={args.start_date}..{args.end_date}, "
            f"seq_len={args.seq_len}, pred_len={args.pred_len}."
        )
    # Use smaller batch size for laptop
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    logger.info(f"Dataset size: {train_size} samples")
    logger.info(f"Batch size: {args.batch_size}, Batches per epoch: {len(train_loader)}")
    
    # 2. Models
    logger.info("Building models...")
    # 1DCNN Feature Extractor
    # Reduce dimension from 10000 to 1024 for laptop 4050 GPU (6GB VRAM)
    embed_dim = 1024 # Was 10000
    feature_extractor = FeatureExtractor(input_channels=18, output_dim=embed_dim).to(device)
    # ViT
    vit_model = StockViT(seq_len=args.seq_len, pred_len=args.pred_len, embed_dim=embed_dim, depth=args.depth, num_heads=args.num_heads).to(device)
    
    logger.info(f"Models moved to {device}. Embed Dim: {embed_dim}")
    
    # 3. Losses
    mtl_loss_wrapper = MultiTaskLoss(num_tasks=4).to(device)
    
    criterion_day = PeakDayLoss(sigma=args.day_sigma)
    criterion_value = nn.SmoothL1Loss()
    
    # 4. Optimizer
    # Optimize all parameters
    params = list(feature_extractor.parameters()) + list(vit_model.parameters()) + list(mtl_loss_wrapper.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
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
            
            target_max_value = targets['max_value'].to(device)
            target_min_value = targets['min_value'].to(device)
            target_max_day = targets['max_day'].to(device)
            target_min_day = targets['min_day'].to(device)
            
            optimizer.zero_grad()
            
            # Use torch.amp.autocast for newer PyTorch versions
            with torch.cuda.amp.autocast():
                # 1. Feature Extraction
                # [B*180, 10000]
                features_flat = feature_extractor(seq_data_flat)
                
                # Reshape for ViT: [B, 180, 10000]
                features_seq = features_flat.view(B, Seq, -1)
                
                # 2. ViT
                outputs = vit_model(features_seq)
                
                pred_max_value = outputs['max_value']
                pred_min_value = outputs['min_value']
                pred_max_day = outputs['max_day']
                pred_min_day = outputs['min_day']
                
                loss_max_value = criterion_value(pred_max_value.view(-1), target_max_value.view(-1))
                loss_min_value = criterion_value(pred_min_value.view(-1), target_min_value.view(-1))
                loss_max_day = criterion_day(pred_max_day, target_max_day)
                loss_min_day = criterion_day(pred_min_day, target_min_day)
                
                losses_list = torch.stack([loss_max_value, loss_min_value, loss_max_day, loss_min_day])
                loss_mtl = mtl_loss_wrapper(losses_list)
                
                total_loss = loss_mtl
            
            # Backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            global_step += 1
            
            # Logging
            if batch_idx % 10 == 0:
                writer.add_scalar("Loss/Total", total_loss.item(), global_step)
                writer.add_scalar("Loss/MaxValue", loss_max_value.item(), global_step)
                writer.add_scalar("Loss/MinValue", loss_min_value.item(), global_step)
                writer.add_scalar("Loss/MaxDay", loss_max_day.item(), global_step)
                writer.add_scalar("Loss/MinDay", loss_min_day.item(), global_step)
                
                sigmas = torch.exp(mtl_loss_wrapper.log_vars)
                writer.add_scalar("Weight/MaxValue", 1/(2*sigmas[0].item()**2), global_step)
                writer.add_scalar("Weight/MinValue", 1/(2*sigmas[1].item()**2), global_step)
                writer.add_scalar("Weight/MaxDay", 1/(2*sigmas[2].item()**2), global_step)
                writer.add_scalar("Weight/MinDay", 1/(2*sigmas[3].item()**2), global_step)
                
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}", 
                "MaxV": f"{loss_max_value.item():.2f}",
                "MinV": f"{loss_min_value.item():.2f}",
                "MaxD": f"{loss_max_day.item():.2f}",
                "MinD": f"{loss_min_day.item():.2f}"
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
            'mean': train_dataset.mean,
            'std': train_dataset.std,
            'config': {
                'seq_len': args.seq_len,
                'pred_len': args.pred_len,
                'embed_dim': embed_dim,
                'depth': args.depth,
                'num_heads': args.num_heads,
                'input_channels': 18,
            },
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    total_duration = time.time() - start_time
    logger.info(f"Training completed in {total_duration/60:.2f} minutes.")
    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"D:\temp\0_tempdata8")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2) # Small batch for 4050 + Large Model
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=180)
    parser.add_argument("--pred_len", type=int, default=60)
    parser.add_argument("--depth", type=int, default=4) # Smaller ViT for test
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--day_sigma", type=float, default=2.0)
    parser.add_argument("--start_date", type=str, default="2024-01-01")
    parser.add_argument("--end_date", type=str, default="2024-12-31")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main()
