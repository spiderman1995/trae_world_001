import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import glob

from src.data.dataset import StockDataset
from src.models.feature_extractor import FeatureExtractor
from src.models.transformer import StockViT
from src.models.loss import MultiTaskLoss, PeakDayLoss

# Setup Logging
def setup_logging(output_dir, fold_idx=0):
    log_file = os.path.join(output_dir, f"train_fold_{fold_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
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

def get_sorted_dates(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "daily_summary_*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    dates = []
    for f in files:
        basename = os.path.basename(f)
        date_str = basename.replace("daily_summary_", "").replace(".csv", "")
        try:
            dates.append(pd.to_datetime(date_str))
        except Exception:
            continue
    dates = sorted(set(dates))
    if not dates:
        raise FileNotFoundError(f"No valid daily_summary_*.csv dates in {data_dir}")
    return dates

def build_day_range_folds(dates, train_days, test_days, step_days):
    folds = []
    max_day = len(dates)
    
    start_day = 1
    while True:
        train_end_day = start_day + train_days - 1
        test_start_day = train_end_day + 1
        test_end_day = test_start_day + test_days - 1
        
        if test_end_day > max_day:
            break
            
        train_range = (start_day, train_end_day)
        test_range = (test_start_day, test_end_day)
        
        train_period = (
            dates[train_range[0] - 1].strftime("%Y-%m-%d"),
            dates[train_range[1] - 1].strftime("%Y-%m-%d"),
        )
        test_period = (
            dates[test_range[0] - 1].strftime("%Y-%m-%d"),
            dates[test_range[1] - 1].strftime("%Y-%m-%d"),
        )
        
        folds.append({
            "train_range": train_range,
            "test_range": test_range,
            "train_period": train_period,
            "test_period": test_period,
        })
        
        start_day += step_days
        
    if not folds:
        raise ValueError(
            f"Could not create any folds. Needed {train_days + test_days} days, but only {max_day} are available."
        )
        
    return folds

def get_period_by_day_range(dates, day_range):
    start_day, end_day = day_range
    if start_day < 1 or end_day < 1 or start_day > end_day:
        raise ValueError(f"Invalid day_range={day_range}. Expect 1-based inclusive indices.")
    if end_day > len(dates):
        raise ValueError(f"day_range={day_range} exceeds available days={len(dates)}.")
    return (
        dates[start_day - 1].strftime("%Y-%m-%d"),
        dates[end_day - 1].strftime("%Y-%m-%d"),
    )

def train_one_fold(args, fold_idx, dates, train_period, test_period, train_range, test_range):
    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    
    logger = setup_logging(fold_dir, fold_idx)
    logger.info(f"=== Starting Fold {fold_idx} ===")
    logger.info(f"Train Days: {train_range[0]} to {train_range[1]}")
    logger.info(f"Test Days: {test_range[0]} to {test_range[1]}")
    logger.info(f"Train Period: {train_period[0]} to {train_period[1]}")
    logger.info(f"Test Period: {test_period[0]} to {test_period[1]}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Building models...")
    embed_dim = 1024 
    feature_extractor = FeatureExtractor(input_channels=18, output_dim=embed_dim).to(device)
    vit_model = StockViT(
        seq_len=args.seq_len, 
        pred_len=args.pred_len, 
        embed_dim=embed_dim, 
        depth=args.depth, 
        num_heads=args.num_heads,
        drop_ratio=args.drop_ratio,
        attn_drop_ratio=args.attn_drop_ratio
    ).to(device)
    
    mtl_loss_wrapper = MultiTaskLoss(num_tasks=4).to(device)
    
    criterion_day = PeakDayLoss(sigma=args.day_sigma)
    criterion_value = nn.SmoothL1Loss()
    
    params = list(feature_extractor.parameters()) + list(vit_model.parameters()) + list(mtl_loss_wrapper.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    
    writer = SummaryWriter(log_dir=os.path.join(fold_dir, "logs"))
    
    train_dataset = StockDataset(
        args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=train_period[0],
        end_date=train_period[1]
    )
    train_size = len(train_dataset)
    num_train_stocks = len(set(idx[0] for idx in train_dataset.indices))
    logger.info(f"Found {num_train_stocks} unique stocks for training period.")
    logger.info(f"Train dataset size: {train_size}")
    if train_size <= 0:
        raise ValueError(
            f"Empty train dataset. data_dir={args.data_dir}, train_days={train_range}, "
            f"train_period={train_period[0]}..{train_period[1]}, seq_len={args.seq_len}, pred_len={args.pred_len}. "
            f"Check logs for ChiNext50 hit and StockID normalization diagnostics."
        )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    global_step = 0
    feature_extractor.train()
    vit_model.train()
    
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx} Train Epoch {epoch+1}", unit="batch", leave=True)
        for batch_idx, (seq_data, targets) in enumerate(pbar):
            if epoch == 0 and batch_idx == 0:
                logger.info(f"Data shape per batch: [Batch, Seq_Len, Channels, Tick_Len] = {list(seq_data.shape)}")
            B, Seq, C, L = seq_data.shape
            seq_data_flat = seq_data.view(B * Seq, C, L).to(device)
            
            target_max_value = targets['max_value'].to(device)
            target_min_value = targets['min_value'].to(device)
            target_max_day = targets['max_day'].to(device)
            target_min_day = targets['min_day'].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                features_flat = feature_extractor(seq_data_flat)
                features_seq = features_flat.view(B, Seq, -1)
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
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            global_step += 1
            if batch_idx % 10 == 0:
                writer.add_scalar("Train/Loss", total_loss.item(), global_step)
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})
    
    torch.save({
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
    }, os.path.join(fold_dir, f"model_final.pth"))

    logger.info("Starting Validation...")
    eval_range = (max(1, test_range[0] - args.seq_len), test_range[1])
    eval_period = get_period_by_day_range(dates, eval_range)
    logger.info(f"Eval Days: {eval_range[0]} to {eval_range[1]}")
    logger.info(f"Eval Period: {eval_period[0]} to {eval_period[1]}")
    test_dataset = StockDataset(
        args.data_dir, 
        seq_len=args.seq_len, 
        pred_len=args.pred_len, 
        start_date=eval_period[0], 
        end_date=eval_period[1],
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    test_size = len(test_dataset)
    logger.info(f"Test dataset size: {test_size}")
    if test_size <= 0:
        raise ValueError(
            f"Empty test dataset. data_dir={args.data_dir}, test_days={test_range}, "
            f"test_period={test_period[0]}..{test_period[1]}, eval_days={eval_range}, eval_period={eval_period[0]}..{eval_period[1]}, "
            f"seq_len={args.seq_len}, pred_len={args.pred_len}. "
            f"Check logs for ChiNext50 hit and StockID normalization diagnostics."
        )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    val_loss, topk_max, topk_min = validate(feature_extractor, vit_model, test_loader, device, criterion_day, criterion_value, args.topk, fold_idx)
    logger.info(f"Fold {fold_idx} Final Val Loss: {val_loss:.4f}")
    logger.info(f"Fold {fold_idx} Top{args.topk} Max-Day Acc: {topk_max:.4f}")
    logger.info(f"Fold {fold_idx} Top{args.topk} Min-Day Acc: {topk_min:.4f}")
    writer.add_scalar("Val/Loss", val_loss, global_step)
    writer.add_scalar(f"Val/Top{args.topk}_MaxDay", topk_max, global_step)
    writer.add_scalar(f"Val/Top{args.topk}_MinDay", topk_min, global_step)
    writer.close()

    return train_size, test_size

def validate(feature_extractor, vit_model, loader, device, criterion_day, criterion_value, topk, fold_idx):
    feature_extractor.eval()
    vit_model.eval()
    total_loss = 0
    count = 0
    hit_max = 0
    hit_min = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Fold {fold_idx} Val", unit="batch", leave=True)
        for seq_data, targets in pbar:
            B, Seq, C, L = seq_data.shape
            seq_data_flat = seq_data.view(B * Seq, C, L).to(device)
            
            target_max_value = targets['max_value'].to(device)
            target_min_value = targets['min_value'].to(device)
            target_max_day = targets['max_day'].to(device)
            target_min_day = targets['min_day'].to(device)
            
            features_flat = feature_extractor(seq_data_flat)
            features_seq = features_flat.view(B, Seq, -1)
            outputs = vit_model(features_seq)
            
            loss_max_value = criterion_value(outputs['max_value'].view(-1), target_max_value.view(-1))
            loss_min_value = criterion_value(outputs['min_value'].view(-1), target_min_value.view(-1))
            loss_max_day = criterion_day(outputs['max_day'], target_max_day)
            loss_min_day = criterion_day(outputs['min_day'], target_min_day)
            
            total_loss += (loss_max_value + loss_min_value + loss_max_day + loss_min_day).item()
            count += 1
            
            pred_max_day = torch.argmax(outputs['max_day'], dim=1)
            pred_min_day = torch.argmax(outputs['min_day'], dim=1)
            hit_max += (torch.abs(pred_max_day - target_max_day) <= topk).float().sum().item()
            hit_min += (torch.abs(pred_min_day - target_min_day) <= topk).float().sum().item()
            total_samples += B
            
    avg_loss = total_loss / max(count, 1)
    total_samples = max(total_samples, 1)
    topk_max = hit_max / total_samples
    topk_min = hit_min / total_samples
    return avg_loss, topk_max, topk_min

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"D:\temp\0_tempdata8")
    parser.add_argument("--output_dir", type=str, default="runs_rolling")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=60) 
    parser.add_argument("--pred_len", type=int, default=60)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--day_sigma", type=float, default=2.0)
    parser.add_argument("--topk", type=int, default=3)
    
    # For tuning
    parser.add_argument("--train_days", type=int, default=120)
    parser.add_argument("--test_days", type=int, default=60)
    parser.add_argument("--step_days", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--drop_ratio", type=float, default=0.0)
    parser.add_argument("--attn_drop_ratio", type=float, default=0.0)
    parser.add_argument("--start_date", type=str, default=None, help="Start date for training data (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, help="End date for training data (YYYY-MM-DD)")

    args = parser.parse_args()
    
    dates = get_sorted_dates(args.data_dir)
    # 根据参数过滤日期
    if args.start_date:
        dates = [d for d in dates if d >= pd.to_datetime(args.start_date)]
    if args.end_date:
        dates = [d for d in dates if d <= pd.to_datetime(args.end_date)]

    if not dates:
        raise ValueError("No data available in the specified date range.")

    folds = build_day_range_folds(dates, args.train_days, args.test_days, args.step_days)
    
    fold_summaries = []
    for i, fold in enumerate(folds):
        print(f"\n{'='*20} Running Fold {i+1} {'='*20}")
        train_size, test_size = train_one_fold(
            args,
            i+1,
            dates,
            fold["train_period"],
            fold["test_period"],
            fold["train_range"],
            fold["test_range"]
        )
        fold_summaries.append({
            "Fold": i + 1,
            "Train Period": f"{fold['train_period'][0]} to {fold['train_period'][1]}",
            "Test Period": f"{fold['test_period'][0]} to {fold['test_period'][1]}",
            "Num Train Samples": train_size,
            "Num Test Samples": test_size
        })

    # 打印最终的总结报告
    print(f"\n\n{'='*25} Training Summary Report {'='*25}")
    summary_df = pd.DataFrame(fold_summaries)
    print(summary_df.to_string())
    print(f"{'='*75}")

if __name__ == "__main__":
    main()
