import argparse
import os
import random
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import StockDataset
from src.models.feature_extractor import FeatureExtractor
from src.models.transformer import StockViT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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


def build_dataset_period(data_dir, seq_len, test_start_date, test_end_date):
    dates = get_sorted_dates(data_dir)
    start_date = pd.to_datetime(test_start_date)
    end_date = pd.to_datetime(test_end_date)
    if start_date > end_date:
        raise ValueError("test_start_date must be <= test_end_date")
    filtered = [d for d in dates if start_date <= d <= end_date]
    if not filtered:
        raise ValueError("No test dates found in the specified range.")
    first_idx = dates.index(filtered[0])
    ext_start_idx = max(0, first_idx - seq_len)
    ext_start_date = dates[ext_start_idx].strftime("%Y-%m-%d")
    ext_end_date = filtered[-1].strftime("%Y-%m-%d")
    return ext_start_date, ext_end_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--test_start_date", type=str, required=True)
    parser.add_argument("--test_end_date", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--rebalance_interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="weekly_test_report.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    seq_len = int(cfg.get("seq_len", 60))
    pred_len = int(cfg.get("pred_len", 60))
    embed_dim = int(cfg.get("embed_dim", 1024))
    depth = int(cfg.get("depth", 4))
    num_heads = int(cfg.get("num_heads", 4))
    input_channels = int(cfg.get("input_channels", 18))

    feature_extractor = FeatureExtractor(input_channels=input_channels, output_dim=embed_dim).to(device)
    vit_model = StockViT(
        seq_len=seq_len,
        pred_len=pred_len,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads
    ).to(device)
    feature_extractor.load_state_dict(checkpoint["feature_extractor"])
    vit_model.load_state_dict(checkpoint["vit"])
    feature_extractor.eval()
    vit_model.eval()

    ext_start_date, ext_end_date = build_dataset_period(
        args.data_dir, seq_len, args.test_start_date, args.test_end_date
    )
    dataset = StockDataset(
        args.data_dir,
        seq_len=seq_len,
        pred_len=pred_len,
        start_date=ext_start_date,
        end_date=ext_end_date,
        mean=checkpoint.get("mean"),
        std=checkpoint.get("std")
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    criterion_value = nn.SmoothL1Loss(reduction="none")
    target_start_date = pd.to_datetime(args.test_start_date)
    target_end_date = pd.to_datetime(args.test_end_date)

    by_week = defaultdict(list)
    pointer = 0

    with torch.no_grad():
        for seq_data, targets in tqdm(loader, desc="Weekly Test"):
            B, Seq, C, L = seq_data.shape
            seq_data_flat = seq_data.view(B * Seq, C, L).to(device)

            outputs = vit_model(feature_extractor(seq_data_flat).view(B, Seq, -1))
            pred_max_value = outputs["max_value"].view(-1).detach().cpu()
            pred_min_value = outputs["min_value"].view(-1).detach().cpu()
            pred_max_day = torch.argmax(outputs["max_day"], dim=1).detach().cpu()
            pred_min_day = torch.argmax(outputs["min_day"], dim=1).detach().cpu()

            tgt_max_value = targets["max_value"].view(-1).cpu()
            tgt_min_value = targets["min_value"].view(-1).cpu()
            tgt_max_day = targets["max_day"].view(-1).cpu()
            tgt_min_day = targets["min_day"].view(-1).cpu()

            val_loss_max = criterion_value(pred_max_value, tgt_max_value)
            val_loss_min = criterion_value(pred_min_value, tgt_min_value)
            day_err = (
                (pred_max_day - tgt_max_day).abs().float() / max(pred_len, 1)
                + (pred_min_day - tgt_min_day).abs().float() / max(pred_len, 1)
            )
            sample_loss = val_loss_max + val_loss_min + day_err

            for i in range(B):
                stock_id, start_idx = dataset.indices[pointer + i]
                data_dict = dataset.stock_data[stock_id]
                rebalance_idx = start_idx + seq_len
                if rebalance_idx >= len(data_dict["dates"]):
                    continue
                rebalance_date = pd.to_datetime(data_dict["dates"][rebalance_idx])
                if rebalance_date < target_start_date or rebalance_date > target_end_date:
                    continue
                score = float(pred_max_value[i].item() - pred_min_value[i].item())
                realized = float(tgt_max_value[i].item() - tgt_min_value[i].item())
                by_week[rebalance_date].append({
                    "stock_id": stock_id,
                    "score": score,
                    "realized": realized,
                    "loss": float(sample_loss[i].item())
                })
            pointer += B

    # 按日历日期每 N 个交易日选一次调仓日（与 backtest.py 逻辑一致）
    all_dates_sorted = sorted(by_week.keys())
    weekly_dates = [d for i, d in enumerate(all_dates_sorted)
                    if i % args.rebalance_interval == 0]
    if not weekly_dates:
        raise ValueError("No weekly rebalance samples found. Check test date range and rebalance_interval.")

    report_rows = []
    best_loss = float("inf")
    bad_rounds = 0

    for d in weekly_dates:
        rows = by_week[d]
        week_df = pd.DataFrame(rows)
        week_loss = float(week_df["loss"].mean())
        if len(week_df) >= 2:
            rank_ic = float(week_df["score"].corr(week_df["realized"], method="spearman"))
        else:
            rank_ic = float("nan")

        improved = week_loss < (best_loss - args.min_delta)
        if improved:
            best_loss = week_loss
            bad_rounds = 0
        else:
            bad_rounds += 1

        report_rows.append({
            "rebalance_date": d.strftime("%Y-%m-%d"),
            "num_stocks": int(len(week_df)),
            "weekly_loss": week_loss,
            "rank_ic": rank_ic,
            "best_loss_so_far": best_loss,
            "early_stop_counter": bad_rounds
        })

        if bad_rounds >= args.patience:
            break

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(args.output_csv, index=False)

    print(report_df.to_string(index=False))
    print(f"\nSaved weekly test report to: {args.output_csv}")
    if len(report_df) > 0:
        print(f"Final best weekly loss: {report_df['best_loss_so_far'].iloc[-1]:.6f}")
        valid_rank_ic = report_df["rank_ic"].dropna()
        if len(valid_rank_ic) > 0:
            print(f"Average RankIC: {valid_rank_ic.mean():.6f}")


if __name__ == "__main__":
    main()
