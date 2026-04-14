"""
diagnose_labels.py — 标签分布诊断脚本

统计 min_day / max_day / max_value / min_value 的分布情况，
用于判断标签是否退化（集中在某几天）。

用法：
    python src/diagnose_labels.py \
        --data_dir "D:/temp/0_tempdata8" \
        --seq_len 60 --pred_len 60 \
        --start_date 2020-01-01 --end_date 2023-12-31
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from collections import Counter
from src.data.dataset import StockDataset


def main():
    parser = argparse.ArgumentParser(description="Diagnose label distribution")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--pred_len", type=int, default=60)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to scan (0=all)")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = StockDataset(
        args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    total = len(dataset)
    if total == 0:
        print("Empty dataset, nothing to diagnose.")
        return

    n = total if args.max_samples <= 0 else min(args.max_samples, total)
    print(f"Scanning {n} / {total} samples...\n")

    max_days = []
    min_days = []
    max_values = []
    min_values = []

    for i in range(n):
        _, targets = dataset[i]
        max_days.append(int(targets["max_day"]))
        min_days.append(int(targets["min_day"]))
        max_values.append(float(targets["max_value"]))
        min_values.append(float(targets["min_value"]))

        if (i + 1) % 5000 == 0:
            print(f"  scanned {i+1}/{n}")

    max_days = np.array(max_days)
    min_days = np.array(min_days)
    max_values = np.array(max_values)
    min_values = np.array(min_values)

    # ---- max_day 分布 ----
    print("=" * 60)
    print("MAX_DAY DISTRIBUTION (pred window中最高价出现在第几天)")
    print("=" * 60)
    print_day_distribution(max_days, args.pred_len)

    # ---- min_day 分布 ----
    print("\n" + "=" * 60)
    print("MIN_DAY DISTRIBUTION (pred window中最低价出现在第几天)")
    print("=" * 60)
    print_day_distribution(min_days, args.pred_len)

    # ---- value 分布 ----
    print("\n" + "=" * 60)
    print("VALUE DISTRIBUTION (价格比率)")
    print("=" * 60)
    print(f"  max_value (最高价/当前价):")
    print(f"    mean={max_values.mean():.4f}  std={max_values.std():.4f}  "
          f"min={max_values.min():.4f}  max={max_values.max():.4f}  "
          f"median={np.median(max_values):.4f}")
    print(f"  min_value (最低价/当前价):")
    print(f"    mean={min_values.mean():.4f}  std={min_values.std():.4f}  "
          f"min={min_values.min():.4f}  max={min_values.max():.4f}  "
          f"median={np.median(min_values):.4f}")

    # ---- 关键指标 ----
    print("\n" + "=" * 60)
    print("KEY DIAGNOSTICS")
    print("=" * 60)

    # min_day 集中度
    min_counter = Counter(min_days)
    top1_day, top1_count = min_counter.most_common(1)[0]
    top3_days = min_counter.most_common(3)
    top3_pct = sum(c for _, c in top3_days) / n * 100

    print(f"  min_day 最常见的天: day={top1_day}, 占比={top1_count/n*100:.1f}%")
    print(f"  min_day Top3 天: {[(d, f'{c/n*100:.1f}%') for d, c in top3_days]}")
    print(f"  min_day Top3 合计占比: {top3_pct:.1f}%")

    max_counter = Counter(max_days)
    top1_day_max, top1_count_max = max_counter.most_common(1)[0]
    top3_days_max = max_counter.most_common(3)
    top3_pct_max = sum(c for _, c in top3_days_max) / n * 100

    print(f"\n  max_day 最常见的天: day={top1_day_max}, 占比={top1_count_max/n*100:.1f}%")
    print(f"  max_day Top3 天: {[(d, f'{c/n*100:.1f}%') for d, c in top3_days_max]}")
    print(f"  max_day Top3 合计占比: {top3_pct_max:.1f}%")

    # 退化判定
    print("\n" + "-" * 60)
    if top3_pct > 50:
        print(f"  !! MIN_DAY 退化: Top3天占 {top3_pct:.0f}% > 50%, 模型可以靠记住位置得高分")
    else:
        print(f"  OK MIN_DAY 分布相对均匀 (Top3占 {top3_pct:.0f}%)")

    if top3_pct_max > 50:
        print(f"  !! MAX_DAY 退化: Top3天占 {top3_pct_max:.0f}% > 50%")
    else:
        print(f"  OK MAX_DAY 分布相对均匀 (Top3占 {top3_pct_max:.0f}%)")

    # 头尾集中度（第0天和最后一天）
    edge_days = {0, args.pred_len - 1}
    min_edge_pct = sum(1 for d in min_days if d in edge_days) / n * 100
    max_edge_pct = sum(1 for d in max_days if d in edge_days) / n * 100
    print(f"\n  min_day 在头尾两天(day 0 或 day {args.pred_len-1})的占比: {min_edge_pct:.1f}%")
    print(f"  max_day 在头尾两天的占比: {max_edge_pct:.1f}%")

    # 前5天和后5天
    head5 = set(range(5))
    tail5 = set(range(args.pred_len - 5, args.pred_len))
    min_head5 = sum(1 for d in min_days if d in head5) / n * 100
    min_tail5 = sum(1 for d in min_days if d in tail5) / n * 100
    max_head5 = sum(1 for d in max_days if d in head5) / n * 100
    max_tail5 = sum(1 for d in max_days if d in tail5) / n * 100
    print(f"\n  min_day 前5天占比: {min_head5:.1f}%,  后5天占比: {min_tail5:.1f}%")
    print(f"  max_day 前5天占比: {max_head5:.1f}%,  后5天占比: {max_tail5:.1f}%")

    print("=" * 60)


def print_day_distribution(days, pred_len):
    counter = Counter(days)
    n = len(days)

    # 按天数分桶显示
    bucket_size = max(1, pred_len // 10)
    buckets = []
    for start in range(0, pred_len, bucket_size):
        end = min(start + bucket_size, pred_len)
        count = sum(counter.get(d, 0) for d in range(start, end))
        pct = count / n * 100
        bar = "#" * int(pct / 2)
        buckets.append((start, end - 1, count, pct, bar))

    print(f"  Total samples: {n}")
    print(f"  Day range: 0 ~ {pred_len - 1}")
    print(f"  Mean: {days.mean():.1f},  Std: {days.std():.1f},  Median: {np.median(days):.1f}")
    print()
    print(f"  {'Bucket':>12}  {'Count':>7}  {'Pct':>6}  Histogram")
    print(f"  {'-'*12}  {'-'*7}  {'-'*6}  {'-'*30}")
    for start, end, count, pct, bar in buckets:
        print(f"  day {start:>2}-{end:>2}     {count:>6}  {pct:>5.1f}%  {bar}")

    # Top 5 具体天
    print(f"\n  Top 5 most frequent days:")
    for day, count in counter.most_common(5):
        print(f"    day {day:>2}: {count:>6} ({count/n*100:.1f}%)")


if __name__ == "__main__":
    main()
