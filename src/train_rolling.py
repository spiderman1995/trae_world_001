import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import numpy as np
import glob

import signal

from src.data.dataset import StockDataset, GlobalDataCache
from src.data.stock_pool import StockPool
from src.data.chinext50 import get_chinext50_constituents
from src.models.feature_extractor import FeatureExtractor
from src.models.transformer import StockViT
from src.models.loss import MultiTaskLoss, PeakDayLoss
from src.config import DATA_DIR
import platform

# ---- 优雅停止：Ctrl+C 完成当前epoch后退出 ----
_stop_requested = False

def _handle_stop(signum, frame):
    global _stop_requested
    if _stop_requested:
        print("\nForce exit!")
        sys.exit(1)
    _stop_requested = True
    print("\n" + "=" * 50)
    print(" STOP REQUESTED — finishing current epoch...")
    print("=" * 50)

signal.signal(signal.SIGINT, _handle_stop)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, _handle_stop)

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

def set_seed(seed, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

def train_one_fold(args, fold_idx, dates, train_period, test_period, train_range, test_range,
                   checkpoint_path=None, stock_pool=None, sampled_stocks=None, global_cache=None):
    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    logger = setup_logging(fold_dir, fold_idx)
    logger.info(f"=== Starting Fold {fold_idx} ===")
    logger.info(f"Train Days: {train_range[0]} to {train_range[1]}")
    logger.info(f"Test Days: {test_range[0]} to {test_range[1]}")
    logger.info(f"Train Period: {train_period[0]} to {train_period[1]}")
    logger.info(f"Test Period: {test_period[0]} to {test_period[1]}")

    # 股票采样（如果未由 main 预计算）
    if sampled_stocks is None:
        if args.stock_pool == "random" and stock_pool is not None:
            # 注意：end_date 只用到训练期末，避免用测试期的退市/停牌信息做筛选（未来信息泄漏）
            sampled_stocks = stock_pool.sample_stocks(
                n=args.num_stocks,
                start_date=train_period[0],
                end_date=train_period[1],
                min_trading_days=args.seq_len + args.pred_len,
                stock_prefix=("30", "31"),
                min_list_days=args.min_list_days,
                exclude_delisted=True,
                seed=args.seed + fold_idx,
            )
        else:
            sampled_stocks = list(get_chinext50_constituents())
    logger.info(f"Using {len(sampled_stocks)} stocks for this fold.")
    
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

    # Load weights from previous fold if a checkpoint is provided
    # warm_start_mode:
    #   full      - 加载 CNN + ViT + heads（v7/v8 行为，跨 fold 累积）
    #   cnn_only  - 只加载 CNN，ViT + heads 每 fold 重新随机初始化（v9，切断 ViT 累积污染）
    if checkpoint_path and os.path.exists(checkpoint_path):
        mode = getattr(args, 'warm_start_mode', 'full')
        logger.info(f"Loading weights from checkpoint: {checkpoint_path} (warm_start_mode={mode})")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            feature_extractor.load_state_dict(checkpoint['feature_extractor'], strict=False)
            if mode == 'full':
                vit_model.load_state_dict(checkpoint['vit'], strict=False)
                logger.info("Loaded CNN + ViT weights.")
            elif mode == 'cnn_only':
                logger.info("Loaded CNN only. ViT + heads use fresh random init (v9 strategy).")
            else:
                raise ValueError(f"Unknown warm_start_mode: {mode}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting with random weights.")
    else:
        logger.info("No checkpoint found or provided, starting with random weights.")
    
    mtl_loss_wrapper = MultiTaskLoss(num_tasks=4).to(device)
    
    criterion_day = PeakDayLoss(sigma=args.day_sigma)
    criterion_value = nn.SmoothL1Loss(beta=args.smooth_l1_beta)
    
    params = list(feature_extractor.parameters()) + list(vit_model.parameters()) + list(mtl_loss_wrapper.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # Learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    else:
        scheduler = None
    logger.info(f"LR Scheduler: {args.scheduler}, Grad Clip: {args.max_grad_norm}")

    writer = SummaryWriter(log_dir=os.path.join(fold_dir, "logs"))
    
    train_dataset = StockDataset(
        args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=train_period[0],
        end_date=train_period[1],
        stock_ids=sampled_stocks,
        sample_stride=args.sample_stride,
        global_cache=global_cache,
    )
    train_size = len(train_dataset)
    num_train_stocks = len(set(idx[0] for idx in train_dataset.indices))
    logger.info(f"Found {num_train_stocks} unique stocks for training period.")
    logger.info(f"Train dataset size: {train_size}")
    if train_size <= 0:
        raise ValueError(
            f"Empty train dataset. data_dir={args.data_dir}, train_days={train_range}, "
            f"train_period={train_period[0]}..{train_period[1]}, seq_len={args.seq_len}, pred_len={args.pred_len}. "
            f"Check logs for stock pool hit and StockID normalization diagnostics."
        )
    # Windows 下 num_workers>0 会因 Dataset 过大导致 pickle 失败，强制为 0
    dl_workers = 0 if platform.system() == "Windows" else args.num_workers
    dl_kwargs = dict(num_workers=dl_workers, pin_memory=True)
    if dl_workers > 0:
        dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dl_kwargs,
    )

    logger.info("Preparing validation dataset...")
    eval_range = (max(1, test_range[0] - args.seq_len), test_range[1])
    eval_period = get_period_by_day_range(dates, eval_range)
    logger.info(f"Val Days: {eval_range[0]} to {eval_range[1]}")
    logger.info(f"Val Period: {eval_period[0]} to {eval_period[1]}")
    val_dataset = StockDataset(
        args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=eval_period[0],
        end_date=eval_period[1],
        mean=train_dataset.mean,
        std=train_dataset.std,
        stock_ids=sampled_stocks,
        sample_stride=args.sample_stride,
        global_cache=global_cache,
    )
    test_size = len(val_dataset)
    logger.info(f"Val dataset size: {test_size}")
    if test_size <= 0:
        raise ValueError(
            f"Empty val dataset. data_dir={args.data_dir}, test_days={test_range}, "
            f"test_period={test_period[0]}..{test_period[1]}, eval_days={eval_range}, eval_period={eval_period[0]}..{eval_period[1]}, "
            f"seq_len={args.seq_len}, pred_len={args.pred_len}."
        )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **dl_kwargs,
    )

    global_step = 0
    best_val_loss = float("inf")
    best_topk_max = 0.0
    best_topk_min = 0.0
    best_epoch = 0
    best_rank_ic_max = 0.0
    best_rank_ic_min = 0.0
    best_train_loss = float("inf")
    log_var_max_day_peak = -float("inf")
    log_var_min_day_peak = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(fold_dir, "model_best.pth")

    for epoch in range(args.epochs):
        feature_extractor.train()
        vit_model.train()
        epoch_train_loss = 0.0
        epoch_train_steps = 0

        pbar = tqdm(train_loader, desc=f"Fold {fold_idx} Train Epoch {epoch+1}", unit="batch", leave=True)
        for batch_idx, (seq_data, targets) in enumerate(pbar):
            if epoch == 0 and batch_idx == 0:
                logger.info(f"Data shape per batch: [Batch, Seq_Len, Channels, Tick_Len] = {list(seq_data.shape)}")
            B, Seq, C, L = seq_data.shape
            seq_data_flat = seq_data.view(B * Seq, C, L).to(device, non_blocking=True)

            target_max_value = targets['max_value'].to(device, non_blocking=True)
            target_min_value = targets['min_value'].to(device, non_blocking=True)
            target_max_day = targets['max_day'].to(device, non_blocking=True)
            target_min_day = targets['min_day'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
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
                total_loss = mtl_loss_wrapper(losses_list)

            # NaN/Inf 检测：避免训练数小时后才发现 loss 已崩
            if not torch.isfinite(total_loss):
                logger.error(f"Fold {fold_idx} epoch {epoch+1} batch {batch_idx}: "
                             f"non-finite loss detected ({total_loss.item()}). "
                             f"Skipping batch. Check grad explosion, bad data, or mixed-precision issues.")
                optimizer.zero_grad()
                continue

            scaler.scale(total_loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            # 首 batch 报告 GPU 显存实际占用（用户可据此调 batch_size）
            if epoch == 0 and batch_idx == 0 and device.type == 'cuda':
                alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
                logger.info(f"GPU memory after first batch: allocated={alloc_gb:.2f}GB, "
                            f"reserved={reserved_gb:.2f}GB / total={total_gb:.0f}GB "
                            f"({reserved_gb/total_gb*100:.0f}% utilized)")

            global_step += 1
            epoch_train_loss += total_loss.item()
            epoch_train_steps += 1

            if batch_idx % 10 == 0:
                writer.add_scalar("Train/Loss_step", total_loss.item(), global_step)
                writer.add_scalar("Train/loss_max_value", loss_max_value.item(), global_step)
                writer.add_scalar("Train/loss_min_value", loss_min_value.item(), global_step)
                writer.add_scalar("Train/loss_max_day", loss_max_day.item(), global_step)
                writer.add_scalar("Train/loss_min_day", loss_min_day.item(), global_step)
                for i, name in enumerate(["max_val", "min_val", "max_day", "min_day"]):
                    writer.add_scalar(f"Train/log_var_{name}", mtl_loss_wrapper.log_vars[i].item(), global_step)
            postfix = {"Loss": f"{total_loss.item():.4f}"}
            if _stop_requested:
                postfix["STATUS"] = "STOPPING"
            pbar.set_postfix(postfix)

        mean_train_loss = epoch_train_loss / max(epoch_train_steps, 1)
        writer.add_scalar("Train/Loss_epoch", mean_train_loss, epoch + 1)

        (val_loss, topk_max, topk_min,
         val_loss_max_value, val_loss_min_value, val_loss_max_day, val_loss_min_day,
         rank_ic_max, rank_ic_min) = validate(
            feature_extractor, vit_model, mtl_loss_wrapper, val_loader, device, criterion_day, criterion_value, args.topk, fold_idx
        )
        writer.add_scalar("Val/Loss_epoch", val_loss, epoch + 1)
        writer.add_scalar(f"Val/Top{args.topk}_MaxDay_epoch", topk_max, epoch + 1)
        writer.add_scalar(f"Val/Top{args.topk}_MinDay_epoch", topk_min, epoch + 1)
        writer.add_scalar("Val/loss_max_value", val_loss_max_value, epoch + 1)
        writer.add_scalar("Val/loss_min_value", val_loss_min_value, epoch + 1)
        writer.add_scalar("Val/loss_max_day", val_loss_max_day, epoch + 1)
        writer.add_scalar("Val/loss_min_day", val_loss_min_day, epoch + 1)
        writer.add_scalar("Val/RankIC_max_value", rank_ic_max, epoch + 1)
        writer.add_scalar("Val/RankIC_min_value", rank_ic_min, epoch + 1)

        # Step scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        writer.add_scalar("Train/LR", current_lr, epoch + 1)

        logger.info(
            f"Fold {fold_idx} Epoch {epoch+1}: TrainLoss={mean_train_loss:.4f}, "
            f"ValLoss={val_loss:.4f}, Top{args.topk}Max={topk_max:.4f}, Top{args.topk}Min={topk_min:.4f}, "
            f"RankIC_max={rank_ic_max:.4f}, RankIC_min={rank_ic_min:.4f}, LR={current_lr:.2e}"
        )

        # 追踪 log_var 峰值（判断模型是否在"放弃"某任务）
        log_var_max_day_peak = max(log_var_max_day_peak, mtl_loss_wrapper.log_vars[2].item())
        log_var_min_day_peak = max(log_var_min_day_peak, mtl_loss_wrapper.log_vars[3].item())
        if mean_train_loss < best_train_loss:
            best_train_loss = mean_train_loss

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_topk_max = topk_max
            best_topk_min = topk_min
            best_epoch = epoch + 1
            best_rank_ic_max = rank_ic_max
            best_rank_ic_min = rank_ic_min
            patience_counter = 0
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
                'best_val_loss': best_val_loss,
                'best_topk_max': best_topk_max,
                'best_topk_min': best_topk_min,
                'best_epoch': epoch + 1,
            }, best_model_path)
            logger.info(f"Fold {fold_idx} improved ValLoss to {best_val_loss:.4f}, saved best model.")
        else:
            patience_counter += 1
            logger.info(f"Fold {fold_idx} early-stop counter: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.info(f"Fold {fold_idx} early stopped at epoch {epoch+1}.")
                break

        if _stop_requested:
            logger.info(f"Fold {fold_idx} graceful stop at epoch {epoch+1}, saving best model.")
            break

    model_save_path = os.path.join(fold_dir, "model_final.pth")
    if os.path.exists(best_model_path):
        best_ckpt = torch.load(best_model_path, map_location="cpu")
        torch.save(best_ckpt, model_save_path)
    else:
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
            'best_val_loss': best_val_loss,
            'best_topk_max': best_topk_max,
            'best_topk_min': best_topk_min,
        }, model_save_path)

    logger.info(f"Fold {fold_idx} Best Val Loss: {best_val_loss:.4f} (epoch {best_epoch})")
    logger.info(f"Fold {fold_idx} Best Top{args.topk} Max-Day Acc: {best_topk_max:.4f}")
    logger.info(f"Fold {fold_idx} Best Top{args.topk} Min-Day Acc: {best_topk_min:.4f}")

    # ---- Fold Doctor：基于已有指标做方向性诊断，给出调整建议 ----
    logger.info(f"--- Fold {fold_idx} Doctor ---")
    gap = best_val_loss - best_train_loss

    # 1. best_epoch 位置 — 指示 lr / patience / epochs 是否合适
    if best_epoch == 1:
        logger.warning(f"  [HINT] best_epoch=1 → lr may be too large (overshooting), "
                       f"or data too easy. Try --lr {args.lr * 0.3:.1e}")
    elif best_epoch >= args.epochs - 1:
        logger.warning(f"  [HINT] best_epoch={best_epoch}/{args.epochs} → model may still be learning, "
                       f"consider --epochs {args.epochs + 10}")
    elif best_epoch <= 3:
        logger.info(f"  [INFO] best_epoch={best_epoch} is early, confirm patience={args.patience} "
                    f"isn't cutting off useful learning")

    # 2. train-val gap — 过拟合程度
    if gap > best_train_loss * 0.5:
        logger.warning(f"  [HINT] train-val gap large ({gap:.3f} = {gap/best_train_loss*100:.0f}% of train). "
                       f"Consider: ↑weight_decay ({args.weight_decay} → {args.weight_decay*2:.0e}), "
                       f"↑drop_ratio ({args.drop_ratio} → {min(args.drop_ratio+0.1, 0.5):.1f})")

    # 3. RankIC 健康
    if best_rank_ic_max < 0 or best_rank_ic_min < 0:
        logger.warning(f"  [HINT] RankIC negative (max={best_rank_ic_max:.3f}, min={best_rank_ic_min:.3f}) → "
                       f"model may be predicting inverse direction, check label construction")
    elif best_rank_ic_max < 0.01 and best_rank_ic_min < 0.01:
        logger.warning(f"  [HINT] RankIC near zero (max={best_rank_ic_max:.3f}, min={best_rank_ic_min:.3f}) → "
                       f"value heads found no signal, check pred_len or feature set")

    # 4. log_var 放弃警告（MTL 把某任务权重压得很低）
    if log_var_max_day_peak > 0.5:
        logger.warning(f"  [HINT] log_var_max_day peaked at {log_var_max_day_peak:.2f} "
                       f"(weight={2.71828**-log_var_max_day_peak:.2f}) → max_day task is being abandoned by MTL. "
                       f"Consider: ↑day_sigma (smoother target), freeze log_var, or add day-head regularization")
    if log_var_min_day_peak > 0.5:
        logger.warning(f"  [HINT] log_var_min_day peaked at {log_var_min_day_peak:.2f} → min_day task being abandoned")

    # 5. Top1_MinDay 假高分
    if best_topk_min > 0.95:
        logger.info(f"  [INFO] Top1_MinDay={best_topk_min:.3f} suspiciously high — "
                    f"likely predicting dominant mode day (skewed distribution), not real predictive skill")

    logger.info(f"--- end Fold {fold_idx} Doctor ---")
    writer.close()

    return {
        'train_size': train_size,
        'test_size': test_size,
        'model_save_path': model_save_path,
        'best_val_loss': best_val_loss,
        'best_topk_max': best_topk_max,
        'best_topk_min': best_topk_min,
        'best_epoch': best_epoch,
        'best_rank_ic_max': best_rank_ic_max,
        'best_rank_ic_min': best_rank_ic_min,
        'best_train_loss': best_train_loss,
        'log_var_max_day_peak': log_var_max_day_peak,
    }

def validate(feature_extractor, vit_model, mtl_loss_wrapper, loader, device, criterion_day, criterion_value, topk, fold_idx):
    feature_extractor.eval()
    vit_model.eval()
    total_loss = 0
    total_loss_max_value = 0
    total_loss_min_value = 0
    total_loss_max_day = 0
    total_loss_min_day = 0
    count = 0
    hit_max = 0
    hit_min = 0
    total_samples = 0
    all_pred_max_value = []
    all_true_max_value = []
    all_pred_min_value = []
    all_true_min_value = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Fold {fold_idx} Val", unit="batch", leave=True)
        for seq_data, targets in pbar:
            B, Seq, C, L = seq_data.shape
            seq_data_flat = seq_data.view(B * Seq, C, L).to(device, non_blocking=True)

            target_max_value = targets['max_value'].to(device, non_blocking=True)
            target_min_value = targets['min_value'].to(device, non_blocking=True)
            target_max_day = targets['max_day'].to(device, non_blocking=True)
            target_min_day = targets['min_day'].to(device, non_blocking=True)

            features_flat = feature_extractor(seq_data_flat)
            features_seq = features_flat.view(B, Seq, -1)
            outputs = vit_model(features_seq)

            loss_max_value = criterion_value(outputs['max_value'].view(-1), target_max_value.view(-1))
            loss_min_value = criterion_value(outputs['min_value'].view(-1), target_min_value.view(-1))
            loss_max_day = criterion_day(outputs['max_day'], target_max_day)
            loss_min_day = criterion_day(outputs['min_day'], target_min_day)

            losses_list = torch.stack([loss_max_value, loss_min_value, loss_max_day, loss_min_day])
            total_loss += mtl_loss_wrapper(losses_list).item()
            total_loss_max_value += loss_max_value.item()
            total_loss_min_value += loss_min_value.item()
            total_loss_max_day += loss_max_day.item()
            total_loss_min_day += loss_min_day.item()
            count += 1

            pred_max_day = torch.argmax(outputs['max_day'], dim=1)
            pred_min_day = torch.argmax(outputs['min_day'], dim=1)
            hit_max += (torch.abs(pred_max_day - target_max_day) <= topk).float().sum().item()
            hit_min += (torch.abs(pred_min_day - target_min_day) <= topk).float().sum().item()
            total_samples += B

            all_pred_max_value.append(outputs['max_value'].view(-1).cpu())
            all_true_max_value.append(target_max_value.view(-1).cpu())
            all_pred_min_value.append(outputs['min_value'].view(-1).cpu())
            all_true_min_value.append(target_min_value.view(-1).cpu())

    avg_loss = total_loss / max(count, 1)
    avg_loss_max_value = total_loss_max_value / max(count, 1)
    avg_loss_min_value = total_loss_min_value / max(count, 1)
    avg_loss_max_day = total_loss_max_day / max(count, 1)
    avg_loss_min_day = total_loss_min_day / max(count, 1)
    total_samples = max(total_samples, 1)
    topk_max = hit_max / total_samples
    topk_min = hit_min / total_samples

    # Rank IC: Spearman correlation between predicted and true values
    from scipy.stats import spearmanr
    pred_mv = torch.cat(all_pred_max_value).numpy()
    true_mv = torch.cat(all_true_max_value).numpy()
    pred_minv = torch.cat(all_pred_min_value).numpy()
    true_minv = torch.cat(all_true_min_value).numpy()
    rank_ic_max = spearmanr(pred_mv, true_mv).correlation if len(pred_mv) > 2 else 0.0
    rank_ic_min = spearmanr(pred_minv, true_minv).correlation if len(pred_minv) > 2 else 0.0
    if not np.isfinite(rank_ic_max):
        rank_ic_max = 0.0
    if not np.isfinite(rank_ic_min):
        rank_ic_min = 0.0

    return (avg_loss, topk_max, topk_min,
            avg_loss_max_value, avg_loss_min_value, avg_loss_max_day, avg_loss_min_day,
            rank_ic_max, rank_ic_min)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output_dir", type=str, default="runs_rolling")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=180)
    parser.add_argument("--pred_len", type=int, default=15)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--day_sigma", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=1)
    
    # For tuning
    parser.add_argument("--train_days", type=int, default=480)
    parser.add_argument("--test_days", type=int, default=60)
    parser.add_argument("--step_days", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--drop_ratio", type=float, default=0.2)
    parser.add_argument("--attn_drop_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "plateau"],
                        help="LR scheduler: none, cosine (CosineAnnealingLR), plateau (ReduceLROnPlateau)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (0=disabled)")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a model checkpoint to start training from (for the first fold).")
    parser.add_argument("--warm_start_mode", type=str, default="full", choices=["full", "cnn_only"],
                        help="Cross-fold warm-start strategy: full (load CNN+ViT, default), "
                             "cnn_only (load CNN only, ViT+heads reset each fold). "
                             "v9 hypothesis: cnn_only avoids ViT cumulative pollution.")
    parser.add_argument("--start_date", type=str, default=None, help="Start date for training data (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, help="End date for training data (YYYY-MM-DD)")
    # 股票采样
    parser.add_argument("--stock_pool", type=str, default="random", choices=["random", "chinext50"],
                        help="Stock sampling strategy: random (sample from all), chinext50 (fixed 50)")
    parser.add_argument("--num_stocks", type=int, default=500, help="Number of stocks to sample per fold (when stock_pool=random)")
    parser.add_argument("--min_list_days", type=int, default=180, help="Exclude stocks listed less than N calendar days (IPO filter)")
    parser.add_argument("--sample_stride", type=int, default=10, help="Sample-level sliding window stride (days). Larger = less overlap, fewer samples")
    # 数据加载
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers (0=main thread)")
    # Loss
    parser.add_argument("--smooth_l1_beta", type=float, default=0.1, help="SmoothL1Loss beta (transition point from L2 to L1)")
    # 性能
    parser.add_argument("--cudnn_benchmark", action="store_true", help="Enable cudnn.benchmark (faster conv, non-deterministic)")
    parser.add_argument("--preload", type=str, default="auto", choices=["auto", "on", "off"],
                        help="Pre-load all data into memory to avoid repeated CSV reads across folds (auto/on/off)")

    args = parser.parse_args()
    set_seed(args.seed, benchmark=args.cudnn_benchmark)

    # ---- 保存启动信息到 output_dir/run_config.txt（便于复盘）----
    import socket
    from datetime import datetime
    os.makedirs(args.output_dir, exist_ok=True)
    _config_path = os.path.join(args.output_dir, "run_config.txt")
    _config_file = open(_config_path, "w", encoding="utf-8")

    def tee(msg=""):
        """同时输出到终端和 run_config.txt"""
        print(msg)
        _config_file.write(msg + "\n")
        _config_file.flush()

    tee(f"=== Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    tee(f"Host: {socket.gethostname()}")
    tee(f"Working dir: {os.getcwd()}")
    tee(f"Command: {' '.join(sys.argv)}")
    tee()

    # ---- 启动自检（新环境容易漏掉的优化项 + sanity check）----
    tee("=" * 60)
    tee(" Startup self-check")
    tee("=" * 60)

    # 1. 性能优化项
    from src.data.dataset import _CSV_ENGINE
    if _CSV_ENGINE == "pyarrow":
        tee(" [OK]   pyarrow detected → CSV parsing 2-3x faster")
    else:
        tee(" [HINT] pyarrow NOT installed → `pip install pyarrow` for 2-3x faster CSV reads")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        tee(f" [OK]   CUDA: {gpu_name} ({gpu_mem_gb:.0f} GB)")
        if args.batch_size >= 64 and gpu_mem_gb < 24:
            tee(f" [WARN] batch_size={args.batch_size} on {gpu_mem_gb:.0f}GB GPU may OOM → try 16-32")
        if args.cudnn_benchmark:
            tee(" [OK]   cudnn.benchmark=True → Conv auto-selects fastest algorithm")
        else:
            tee(" [HINT] cudnn.benchmark=False → add --cudnn_benchmark for 5-15% Conv speedup")
    else:
        tee(" [WARN] CUDA NOT available, training on CPU will be very slow")

    if args.num_workers == 0:
        tee(" [HINT] DataLoader num_workers=0 → add --num_workers 4 for parallel data prep (ignored on Windows)")
    else:
        tee(f" [OK]   DataLoader num_workers={args.num_workers}")

    if torch.__version__.startswith("1."):
        tee(f" [INFO] PyTorch {torch.__version__} (legacy). Flash Attention & torch.compile require 2.0+")
    else:
        tee(f" [OK]   PyTorch {torch.__version__}")

    # 2. 数据与输出 sanity check
    tee()
    csv_files = glob.glob(os.path.join(args.data_dir, "daily_summary_*.csv"))
    if not csv_files:
        tee(f" [ERROR] No CSV files found in {args.data_dir}")
    else:
        tee(f" [OK]   Data dir: {len(csv_files)} CSV files at {args.data_dir}")
        cache_files = glob.glob(os.path.join(args.data_dir, ".stock_pool_cache_*.pkl"))
        if cache_files:
            tee(f" [OK]   StockPool cache exists ({len(cache_files)} file(s)) → fast load")
        else:
            tee(" [INFO] StockPool cache NOT found → first scan takes ~minutes")

    existing_folds = glob.glob(os.path.join(args.output_dir, "fold_*"))
    if existing_folds:
        tee(f" [WARN] Output dir {args.output_dir} has {len(existing_folds)} existing fold_*/ dirs → "
            f"will OVERWRITE. Ctrl+C to abort, rename if you want to keep them.")
    else:
        tee(f" [OK]   Output dir {args.output_dir} is empty")

    # 3. warm-start / resume 状态
    if args.resume_from:
        if os.path.exists(args.resume_from):
            tee(f" [OK]   Warm-start from {args.resume_from}")
        else:
            tee(f" [ERROR] --resume_from path does not exist: {args.resume_from}")
    else:
        tee(" [INFO] No --resume_from, fold 1 trains from random init. "
            "Fold N+1 warm-starts from fold N automatically.")

    if args.warm_start_mode == "cnn_only":
        tee(" [INFO] warm_start_mode=cnn_only: each fold loads CNN only, ViT+heads reset (v9).")
    else:
        tee(" [INFO] warm_start_mode=full: each fold loads full CNN+ViT (default).")

    # 4. 磁盘空间（checkpoint 写盘）
    import shutil
    out_parent = os.path.dirname(os.path.abspath(args.output_dir)) or "."
    free_gb = shutil.disk_usage(out_parent).free / (1024 ** 3)
    if free_gb < 10:
        tee(f" [WARN] Only {free_gb:.1f}GB free on {out_parent} → checkpoints (~100MB each) may fail")
    else:
        tee(f" [OK]   Disk space: {free_gb:.0f}GB free on {out_parent}")

    # 5. Git commit（可复现性）
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL, cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
        suffix = " (+uncommitted changes)" if dirty else ""
        tee(f" [OK]   Git commit: {git_hash}{suffix}")
    except Exception:
        tee(" [INFO] Not a git repo (or git unavailable)")

    tee("=" * 60)
    tee()

    # 最终生效的 args（方便复现和排错）
    tee("=" * 60)
    tee(" Final args")
    tee("=" * 60)
    for k, v in sorted(vars(args).items()):
        tee(f"   {k:20s} = {v}")
    tee("=" * 60)
    tee()

    # 参数校验
    if args.train_days < args.seq_len + args.pred_len:
        raise ValueError(
            f"train_days({args.train_days}) must >= seq_len+pred_len({args.seq_len + args.pred_len}). "
            f"Each sample needs {args.seq_len + args.pred_len} consecutive days."
        )

    dates = get_sorted_dates(args.data_dir)
    # 根据参数过滤日期
    if args.start_date:
        dates = [d for d in dates if d >= pd.to_datetime(args.start_date)]
    if args.end_date:
        dates = [d for d in dates if d <= pd.to_datetime(args.end_date)]

    if not dates:
        raise ValueError("No data available in the specified date range.")

    folds = build_day_range_folds(dates, args.train_days, args.test_days, args.step_days)

    # 4. 训练规模预估
    samples_per_stock_per_fold = max(0, (args.train_days - args.seq_len - args.pred_len) // args.sample_stride + 1)
    est_samples_per_fold = samples_per_stock_per_fold * args.num_stocks
    est_batches_per_epoch = est_samples_per_fold // args.batch_size
    tee("=" * 60)
    tee(" Training scale estimate")
    tee("=" * 60)
    tee(f" Total folds:                {len(folds)}")
    tee(f" Samples/stock/fold:         {samples_per_stock_per_fold}")
    tee(f" Samples/fold (est):         ~{est_samples_per_fold:,} (× {args.num_stocks} stocks)")
    tee(f" Batches/epoch (est):        ~{est_batches_per_epoch:,}")
    tee(f" Max epochs/fold:            {args.epochs} (early stop patience={args.patience})")
    tee(f" Train period first fold:    {folds[0]['train_period'][0]} .. {folds[0]['train_period'][1]}")
    tee(f" Test  period last  fold:    {folds[-1]['test_period'][0]} .. {folds[-1]['test_period'][1]}")
    tee("=" * 60)
    tee()

    # run_config.txt 头部记录完成，后续训练运行时信息仍正常打印到终端
    # （配置文件保持打开，程序退出时 OS 自动 flush/close）
    _config_file.write(f"\n=== Config header done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    _config_file.flush()

    # ---- 预计算股票采样 + 数据预加载 ----
    pool = None
    global_cache = None
    fold_stocks = {}

    if args.stock_pool == "random":
        print("Scanning stock pool (one-time)...")
        pool = StockPool(args.data_dir, start_date=args.start_date, end_date=args.end_date)

        # 预计算所有fold的股票采样
        # 注意：end_date 只用到训练期末，避免用测试期的退市/停牌信息做筛选（未来信息泄漏）
        all_needed_stocks = set()
        for i, fold in enumerate(folds):
            fold_idx = i + 1
            sampled = pool.sample_stocks(
                n=args.num_stocks,
                start_date=fold["train_period"][0],
                end_date=fold["train_period"][1],
                min_trading_days=args.seq_len + args.pred_len,
                stock_prefix=("30", "31"),
                min_list_days=args.min_list_days,
                exclude_delisted=True,
                seed=args.seed + fold_idx,
            )
            fold_stocks[fold_idx] = sampled
            all_needed_stocks.update(sampled)

        print(f"Unique stocks across all folds: {len(all_needed_stocks)}")

        # 预加载判断
        est_mem_gb = len(all_needed_stocks) * len(dates) * 18 * 1424 * 4 * 2 / (1024 ** 3)
        do_preload = (args.preload == "on" or
                      (args.preload == "auto" and est_mem_gb < 128))

        if do_preload:
            print(f"Pre-loading data (estimated {est_mem_gb:.1f} GB)...")
            global_cache = GlobalDataCache(
                args.data_dir, all_needed_stocks,
                start_date=dates[0].strftime("%Y-%m-%d"),
                end_date=dates[-1].strftime("%Y-%m-%d"),
            )
        else:
            print(f"Skipping preload (estimated {est_mem_gb:.1f} GB > 128 GB limit). Per-fold CSV loading.")

    import time
    overall_start = time.time()

    previous_model_path = args.resume_from
    fold_summaries = []
    for i, fold in enumerate(folds):
        if _stop_requested:
            print(f"\nGraceful stop before fold {i+1}. Completed {i} folds.")
            break

        fold_start = time.time()
        print(f"\n{'='*20} Running Fold {i+1} {'='*20}")
        result = train_one_fold(
            args,
            i+1,
            dates,
            fold["train_period"],
            fold["test_period"],
            fold["train_range"],
            fold["test_range"],
            checkpoint_path=previous_model_path,
            stock_pool=pool,
            sampled_stocks=fold_stocks.get(i+1),
            global_cache=global_cache,
        )
        fold_elapsed = time.time() - fold_start
        previous_model_path = result['model_save_path']
        fold_summaries.append({
            "Fold": i + 1,
            "Test Period": f"{fold['test_period'][0][:7]}..{fold['test_period'][1][:7]}",
            "Samples": result['train_size'],
            "BestEp": result['best_epoch'],
            "ValLoss": round(result['best_val_loss'], 4),
            "Top1_Max": round(result['best_topk_max'], 3),
            "IC_max": round(result['best_rank_ic_max'], 3),
            "IC_min": round(result['best_rank_ic_min'], 3),
            "Min": round(fold_elapsed / 60, 1),
        })

        # ---- Trend Watch：每 3 fold 做一次跨 fold 漂移检查 ----
        if len(fold_summaries) >= 6 and (i + 1) % 3 == 0:
            recent_vals = [s['ValLoss'] for s in fold_summaries[-3:]]
            early_vals = [s['ValLoss'] for s in fold_summaries[:3]]
            recent_mean = sum(recent_vals) / 3
            early_mean = sum(early_vals) / 3
            recent_ic = [s['IC_max'] for s in fold_summaries[-3:]]

            if recent_mean > early_mean * 1.3:
                print(f"\n[TREND WARN] Recent 3 folds ValLoss avg={recent_mean:.3f} is "
                      f"{recent_mean/early_mean:.1f}x early avg ({early_mean:.3f}). "
                      f"Likely overfitting accumulation. Consider stopping or switching strategy.")

            if all(ic < 0.01 for ic in recent_ic):
                print(f"\n[TREND WARN] Recent 3 folds all RankIC_max < 0.01 ({recent_ic}). "
                      f"Value head predictive signal degraded. Check pred_len / features / regime shift.")

    overall_elapsed = time.time() - overall_start

    # ---- 事后总结报告（同时写入 run_config.txt，便于复盘）----
    tee()
    tee(f"\n{'='*30} Training Summary Report {'='*30}")
    summary_df = pd.DataFrame(fold_summaries)
    summary_str = summary_df.to_string(index=False)
    tee(summary_str)
    tee("=" * 85)

    if fold_summaries:
        # 最佳 fold 推荐（用于回测）
        best_row = min(fold_summaries, key=lambda r: r["ValLoss"])
        mean_val = sum(r["ValLoss"] for r in fold_summaries) / len(fold_summaries)
        tee(f"\n Total folds completed:     {len(fold_summaries)} / {len(folds)}")
        tee(f" Total training time:       {overall_elapsed/60:.1f} min ({overall_elapsed/3600:.2f} h)")
        tee(f" Mean Best ValLoss:         {mean_val:.4f}")
        best_fold_num = best_row['Fold']
        best_ckpt_path = os.path.join(args.output_dir, f'fold_{best_fold_num}', 'model_final.pth')
        tee(f" Best fold (lowest ValLoss): Fold {best_fold_num} → ValLoss={best_row['ValLoss']:.4f}")
        tee(f" Recommended checkpoint:    {best_ckpt_path}")

        # 诊断性提示
        val_losses = [r["ValLoss"] for r in fold_summaries]
        if len(val_losses) >= 3:
            trend_recent = sum(val_losses[-3:]) / 3
            trend_early = sum(val_losses[:3]) / 3
            if trend_recent > trend_early * 1.3:
                tee(f" [WARN] Val loss in recent folds ({trend_recent:.3f}) is {trend_recent/trend_early:.1f}x "
                    f"higher than early folds ({trend_early:.3f}) → possible overfitting / drift, "
                    f"use early fold checkpoint for backtest.")
    tee("=" * 85)
    tee(f"\n=== Run finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    _config_file.close()

if __name__ == "__main__":
    main()
