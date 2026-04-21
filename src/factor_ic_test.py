"""
单因子IC测试 + 信号衰减曲线

原理：
  IC (Information Coefficient) = 同一天所有股票的"因子值"和"未来N天收益"之间的 Spearman 排序相关系数。
  每个交易日算一次IC，最终取所有交易日的均值(Mean IC)和IC均值/IC标准差(ICIR)。

  Mean IC > 0.03 → 因子有弱预测力
  Mean IC > 0.05 → 因子有中等预测力
  ICIR > 0.5    → 因子预测力稳定

  对不同预测天数(1,3,5,10,15,30,60)分别计算IC，画出"信号衰减曲线"，
  找到IC衰减到~0的拐点 → 该因子的有效预测跨度。

用法：
  python -m src.factor_ic_test --data_dir E:/pre_process/version1 --start_date 2020-01-01 --end_date 2022-12-31 --num_stocks 200
"""

import argparse
import os
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from src.data.stock_pool import StockPool
from src.data.dataset import RAW_TICKS, CLEAN_TICKS, trim_auction_zeros

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "trade_count", "total_trade_amt", "avg_trade_price",
    "total_buy_order_id_count", "total_sell_order_id_count", "active_buy_amt",
    "ratio_6", "ratio_7", "ratio_8", "ratio_9", "ratio_10", "ratio_11",
    "ratio_12", "ratio_13", "ratio_14", "ratio_15", "ratio_16", "ratio_17",
]

HORIZONS = [1, 3, 5, 10, 15, 30, 60]


def load_daily_features(data_dir, start_date, end_date, stock_ids):
    """加载日级聚合特征和收盘价。返回 {stock_id: DataFrame(date, 18个日均值, close_price)}"""
    import glob

    files = sorted(glob.glob(os.path.join(data_dir, "daily_summary_*.csv")))
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)

    filtered = []
    for f in files:
        basename = os.path.basename(f)
        date_str = basename.replace("daily_summary_", "").replace(".csv", "")
        try:
            file_date = pd.to_datetime(date_str)
        except Exception:
            continue
        if sd <= file_date <= ed:
            filtered.append((f, file_date))

    stock_set = set(str(s) for s in stock_ids)
    records = []

    for fpath, fdate in tqdm(filtered, desc="Loading daily data"):
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue

        feature_cols = [c for c in df.columns if c not in ("StockID", "Time")]
        if len(feature_cols) != 18:
            continue

        for stock_id_raw, group in df.groupby("StockID"):
            if isinstance(stock_id_raw, (int, np.integer)):
                stock_id = f"{int(stock_id_raw):06d}"
            else:
                stock_id = str(stock_id_raw).strip()

            if stock_id not in stock_set:
                continue
            if len(group) != RAW_TICKS:
                continue

            feats = group[feature_cols].values  # [1442, 18]
            feats = np.where(np.isfinite(feats), feats, 0)

            close_price = feats[-1, 2]  # 最后一个tick的avg_trade_price
            daily_means = feats.mean(axis=0)  # 18个特征的日均值

            record = {"date": fdate, "stock_id": stock_id, "close_price": close_price}
            for i, name in enumerate(FEATURE_NAMES):
                record[name] = daily_means[i]
            records.append(record)

    df_all = pd.DataFrame(records)
    logger.info(f"Loaded {len(df_all)} stock-day records, {df_all['stock_id'].nunique()} stocks, {df_all['date'].nunique()} days.")
    return df_all


def compute_future_returns(df_all, horizon):
    """计算每只股票未来N天的最大收益率。"""
    results = []
    for stock_id, sdf in df_all.groupby("stock_id"):
        sdf = sdf.sort_values("date").reset_index(drop=True)
        prices = sdf["close_price"].values
        dates = sdf["date"].values

        for i in range(len(sdf) - horizon):
            future_prices = prices[i + 1: i + 1 + horizon]
            current_price = prices[i]
            if current_price <= 0:
                continue
            max_return = future_prices.max() / current_price - 1.0
            results.append({
                "date": dates[i],
                "stock_id": stock_id,
                "future_max_return": max_return,
            })

    return pd.DataFrame(results)


def compute_ic(df_all, horizon):
    """计算单因子IC：每天跨股票的 Spearman 相关。"""
    df_returns = compute_future_returns(df_all, horizon)
    df_merged = pd.merge(df_all, df_returns, on=["date", "stock_id"])

    ic_results = {name: [] for name in FEATURE_NAMES}

    for date, group in df_merged.groupby("date"):
        if len(group) < 10:
            continue
        for name in FEATURE_NAMES:
            factor_values = group[name].values
            returns = group["future_max_return"].values
            if np.std(factor_values) < 1e-10 or np.std(returns) < 1e-10:
                continue
            ic, _ = spearmanr(factor_values, returns)
            if np.isfinite(ic):
                ic_results[name].append(ic)

    summary = {}
    for name in FEATURE_NAMES:
        ics = ic_results[name]
        if len(ics) > 0:
            mean_ic = np.mean(ics)
            std_ic = np.std(ics)
            icir = mean_ic / std_ic if std_ic > 1e-10 else 0
            summary[name] = {"mean_ic": mean_ic, "std_ic": std_ic, "icir": icir, "n_days": len(ics)}
        else:
            summary[name] = {"mean_ic": 0, "std_ic": 0, "icir": 0, "n_days": 0}

    return summary


def main():
    parser = argparse.ArgumentParser(description="单因子IC测试 + 信号衰减曲线")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--start_date", type=str, default="2020-01-01")
    parser.add_argument("--end_date", type=str, default="2022-12-31")
    parser.add_argument("--num_stocks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="ic_test_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 采样股票
    logger.info("Sampling stocks...")
    pool = StockPool(args.data_dir, args.start_date, args.end_date)
    stock_ids = pool.sample_stocks(
        n=args.num_stocks,
        start_date=args.start_date,
        end_date=args.end_date,
        stock_prefix=("30", "31"),
        seed=args.seed,
    )
    logger.info(f"Sampled {len(stock_ids)} stocks.")

    # 2. 加载日级数据
    df_all = load_daily_features(args.data_dir, args.start_date, args.end_date, stock_ids)

    # 3. 对每个预测跨度计算IC
    all_results = []
    for horizon in HORIZONS:
        logger.info(f"Computing IC for horizon={horizon} days...")
        summary = compute_ic(df_all, horizon)
        for name, metrics in summary.items():
            all_results.append({
                "feature": name,
                "horizon": horizon,
                **metrics,
            })

    df_results = pd.DataFrame(all_results)

    # 4. 输出结果
    # 信号衰减表
    pivot_ic = df_results.pivot(index="feature", columns="horizon", values="mean_ic")
    pivot_icir = df_results.pivot(index="feature", columns="horizon", values="icir")

    print("\n" + "=" * 80)
    print("Mean IC (信号强度，>0.03有效)")
    print("=" * 80)
    print(pivot_ic.to_string(float_format="%.4f"))

    print("\n" + "=" * 80)
    print("ICIR (信号稳定性，>0.5稳定)")
    print("=" * 80)
    print(pivot_icir.to_string(float_format="%.4f"))

    # 保存
    csv_path = os.path.join(args.output_dir, "factor_ic_results.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # 5. 找最佳预测窗口
    print("\n" + "=" * 80)
    print("各特征最佳预测窗口")
    print("=" * 80)
    for name in FEATURE_NAMES:
        feat_data = df_results[df_results["feature"] == name]
        best_row = feat_data.loc[feat_data["mean_ic"].abs().idxmax()]
        print(f"  {name:30s} → best horizon={int(best_row['horizon']):2d}天, IC={best_row['mean_ic']:.4f}, ICIR={best_row['icir']:.4f}")


if __name__ == "__main__":
    main()
