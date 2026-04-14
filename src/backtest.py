"""
backtest.py — 基于真实 daily_summary_YYYY-MM-DD.csv 数据的滚动回测

功能：
  - 模型推理 + 方向过滤选股 + 等权调仓
  - 支持交易成本（佣金 + 印花税）
  - 记录每日净值（非仅调仓日）
  - 等权基准线对比
  - 可选自动出图（--plot）

用法示例：
    python src/backtest.py \
        --checkpoint_path runs_rolling_v1/fold_7/model_final.pth \
        --data_dir "D:/temp/0_tempdata8" \
        --start_date 2024-07-01 \
        --end_date   2024-09-30 \
        --top_k 5 \
        --rebalance_interval 5 \
        --commission 0.0000875 \
        --stamp_tax 0.0 \
        --plot
"""

import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.feature_extractor import FeatureExtractor
from src.models.transformer import StockViT
from src.data.chinext50 import get_chinext50_constituents
from src.data.dataset import RAW_TICKS, CLEAN_TICKS, trim_auction_zeros


# ---------------------------------------------------------------------------
# 价格取值说明
# ---------------------------------------------------------------------------
# 使用 avg_trade_price（特征索引 2）的最后一个 tick（-1）作为日终参考价。
# 在 A 股市场，收盘集合竞价（14:57-15:00）产出的成交价即为官方收盘价，
# 也是清算结算的参考价。最后一个 tick 对应此价格，是日频回测的标准选择。
PRICE_FEATURE_IDX = 2   # avg_trade_price
PRICE_TICK_IDX = -1      # 最后一个 tick（≈收盘价）


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def get_sorted_dates(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "daily_summary_*.csv")))
    dates = []
    for f in files:
        basename = os.path.basename(f)
        date_str = basename.replace("daily_summary_", "").replace(".csv", "")
        try:
            dates.append(pd.to_datetime(date_str))
        except Exception:
            continue
    return sorted(set(dates))


def load_stock_data(data_dir, load_dates, valid_stocks):
    """
    读取指定日期列表的 CSV 文件，返回：
        {stock_id: {"dates": [pd.Timestamp, ...], "data": np.ndarray [Days, 18, 1424]}}
    """
    stock_buf = {}

    for d in tqdm(load_dates, desc="Loading data"):
        fname = os.path.join(data_dir, f"daily_summary_{d.strftime('%Y-%m-%d')}.csv")
        if not os.path.exists(fname):
            continue
        try:
            df = pd.read_csv(fname)
            feature_cols = [c for c in df.columns if c not in ("StockID", "Time")]
            if len(feature_cols) != 18:
                print(f"Warning: {fname} has {len(feature_cols)} feature cols, expected 18, skipping.")
                continue

            for raw_id, grp in df.groupby("StockID"):
                if isinstance(raw_id, float):
                    raw_id = int(raw_id)
                sid = f"{int(raw_id):06d}" if isinstance(raw_id, int) else str(raw_id).strip()
                if sid not in valid_stocks:
                    continue

                feats = grp[feature_cols].to_numpy(dtype=np.float32)
                if feats.shape[0] not in (RAW_TICKS, CLEAN_TICKS):
                    continue
                feats = feats.T  # [18, 1442] or [18, 1424]
                if feats.shape[1] == RAW_TICKS:
                    feats = trim_auction_zeros(feats)  # [18, 1424]

                if sid not in stock_buf:
                    stock_buf[sid] = {"dates": [], "data": []}
                stock_buf[sid]["dates"].append(d)
                stock_buf[sid]["data"].append(feats)

        except Exception as e:
            print(f"Warning: skipping {fname}: {e}")
            continue

    stock_data = {}
    for sid, content in stock_buf.items():
        pairs = sorted(zip(content["dates"], content["data"]), key=lambda x: x[0])
        stock_data[sid] = {
            "dates": [p[0] for p in pairs],
            "data": np.stack([p[1] for p in pairs]),
        }
    return stock_data


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def normalize(arr, mean, std):
    """arr: [seq_len, 18, 1424]"""
    arr = arr.copy()
    arr[:, :6, :] = np.log1p(arr[:, :6, :])
    if mean is not None and std is not None:
        m = mean.numpy().reshape(1, 18, 1)
        s = std.numpy().reshape(1, 18, 1)
        arr = (arr - m) / (s + 1e-6)
    return arr


def get_price(stock_data, stock_id, date):
    """取 date 当天的收盘参考价。"""
    content = stock_data.get(stock_id)
    if content is None:
        return None
    try:
        idx = content["dates"].index(date)
    except ValueError:
        return None
    price = float(content["data"][idx, PRICE_FEATURE_IDX, PRICE_TICK_IDX])
    return price if price > 0 else None


def get_all_prices(stock_data, date):
    """获取某日所有股票的价格 {stock_id: price}。"""
    prices = {}
    for sid, content in stock_data.items():
        try:
            idx = content["dates"].index(date)
            p = float(content["data"][idx, PRICE_FEATURE_IDX, PRICE_TICK_IDX])
            if p > 0:
                prices[sid] = p
        except ValueError:
            continue
    return prices


# ---------------------------------------------------------------------------
# 模型推理
# ---------------------------------------------------------------------------

def predict_scores(stock_data, rebalance_date, seq_len, mean, std,
                   feature_extractor, vit_model, batch_size, device):
    """
    返回：{stock_id: {"score": float, "max_day": int, "min_day": int}}
    """
    inputs = []
    stock_ids = []

    for sid, content in stock_data.items():
        dates = content["dates"]
        try:
            idx = dates.index(rebalance_date)
        except ValueError:
            continue
        if idx < seq_len:
            continue
        window = content["data"][idx - seq_len: idx]
        inputs.append(normalize(window, mean, std))
        stock_ids.append(sid)

    if not inputs:
        return {}

    inputs_tensor = torch.FloatTensor(np.stack(inputs))
    results = {}

    with torch.no_grad():
        for start in range(0, len(inputs_tensor), batch_size):
            batch = inputs_tensor[start: start + batch_size].to(device)
            B, Seq, C, L = batch.shape
            flat = batch.view(B * Seq, C, L)
            feats = feature_extractor(flat).view(B, Seq, -1)
            outputs = vit_model(feats)
            pred_max = outputs["max_value"].view(-1).cpu().numpy()
            pred_min = outputs["min_value"].view(-1).cpu().numpy()
            pred_max_day = torch.argmax(outputs["max_day"], dim=1).cpu().numpy()
            pred_min_day = torch.argmax(outputs["min_day"], dim=1).cpu().numpy()
            for i, sid in enumerate(stock_ids[start: start + batch_size]):
                results[sid] = {
                    "score": float(pred_max[i] - pred_min[i]),
                    "max_day": int(pred_max_day[i]),
                    "min_day": int(pred_min_day[i]),
                }

    return results


# ---------------------------------------------------------------------------
# 等权基准线
# ---------------------------------------------------------------------------

def compute_benchmark(stock_data, test_dates, initial_capital):
    """
    计算等权基准：每日所有可用股票的平均收益率。
    模拟一个 "每天都持有全部股票、等权" 的被动组合。
    """
    benchmark = [initial_capital]

    for i in range(1, len(test_dates)):
        today = test_dates[i]
        yesterday = test_dates[i - 1]
        returns = []

        for sid, content in stock_data.items():
            dates = content["dates"]
            if today in dates and yesterday in dates:
                idx_t = dates.index(today)
                idx_y = dates.index(yesterday)
                pt = float(content["data"][idx_t, PRICE_FEATURE_IDX, PRICE_TICK_IDX])
                py = float(content["data"][idx_y, PRICE_FEATURE_IDX, PRICE_TICK_IDX])
                if py > 0 and pt > 0:
                    returns.append(pt / py - 1)

        avg_ret = np.mean(returns) if returns else 0.0
        benchmark.append(benchmark[-1] * (1 + avg_ret))

    return benchmark


# ---------------------------------------------------------------------------
# 回测引擎
# ---------------------------------------------------------------------------

class BacktestEngine:
    def __init__(self, data_dir, checkpoint_path, start_date, end_date,
                 initial_capital=1_000_000, top_k=5, rebalance_interval=5,
                 commission=0.0000875, stamp_tax=0.0,
                 batch_size=8, device=None):
        self.data_dir = data_dir
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.top_k = top_k
        self.rebalance_interval = rebalance_interval
        self.commission = commission      # 佣金（双向），默认万0.875
        self.stamp_tax = stamp_tax        # 印花税（仅卖出），默认0
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载 checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        cfg = checkpoint.get("config", {})
        self.seq_len = int(cfg.get("seq_len", 60))
        self.pred_len = int(cfg.get("pred_len", 60))
        self.embed_dim = int(cfg.get("embed_dim", 1024))
        self.depth = int(cfg.get("depth", 4))
        self.num_heads = int(cfg.get("num_heads", 4))
        self.input_channels = int(cfg.get("input_channels", 18))
        self.mean = checkpoint.get("mean")
        self.std = checkpoint.get("std")

        self.feature_extractor = FeatureExtractor(
            input_channels=self.input_channels, output_dim=self.embed_dim
        ).to(self.device)
        self.vit_model = StockViT(
            seq_len=self.seq_len, pred_len=self.pred_len,
            embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads
        ).to(self.device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.vit_model.load_state_dict(checkpoint["vit"])
        self.feature_extractor.eval()
        self.vit_model.eval()

    def run(self):
        all_dates = get_sorted_dates(self.data_dir)
        test_dates = [d for d in all_dates if self.start_date <= d <= self.end_date]
        if not test_dates:
            raise ValueError("No dates found in the backtest period.")

        # 向前多加载 seq_len 天历史
        first_test_idx = all_dates.index(test_dates[0])
        hist_start_idx = max(0, first_test_idx - self.seq_len)
        load_dates = all_dates[hist_start_idx: all_dates.index(test_dates[-1]) + 1]

        print(f"Data range: {load_dates[0].date()} → {load_dates[-1].date()}")
        valid_stocks = set(get_chinext50_constituents())
        stock_data = load_stock_data(self.data_dir, load_dates, valid_stocks)
        print(f"Loaded {len(stock_data)} stocks.")

        # 调仓日集合
        rebalance_set = set(
            d for i, d in enumerate(test_dates) if i % self.rebalance_interval == 0
        )

        # ------ 基准线 ------
        print("Computing benchmark...")
        benchmark_values = compute_benchmark(stock_data, test_dates, self.initial_capital)

        # ------ 逐日循环 ------
        cash = float(self.initial_capital)
        positions = {}          # stock_id -> shares
        last_known_prices = {}  # stock_id -> price（用于无数据日的估值）
        daily_history = []
        total_commission_paid = 0.0
        total_stamp_tax_paid = 0.0

        for day_idx, current_date in enumerate(tqdm(test_dates, desc="Backtest")):
            today_prices = get_all_prices(stock_data, current_date)
            last_known_prices.update(today_prices)

            is_rebalance = current_date in rebalance_set
            num_bullish = 0
            selected = []

            if is_rebalance:
                # ---- 推理 & 选股 ----
                scores = predict_scores(
                    stock_data, current_date, self.seq_len, self.mean, self.std,
                    self.feature_extractor, self.vit_model, self.batch_size, self.device
                )

                # 方向过滤：min_day < max_day（先跌后涨）
                bullish = {sid: info for sid, info in scores.items()
                           if info["min_day"] < info["max_day"]}
                num_bullish = len(bullish)

                sorted_stocks = sorted(bullish.items(), key=lambda x: x[1]["score"], reverse=True)
                selected = [sid for sid, _ in sorted_stocks[: self.top_k]]

                # ---- 卖出全部持仓（扣佣金 + 印花税）----
                for sid, shares in list(positions.items()):
                    price = today_prices.get(sid) or last_known_prices.get(sid)
                    if price:
                        gross = shares * price
                        comm = gross * self.commission
                        stamp = gross * self.stamp_tax
                        cash += gross - comm - stamp
                        total_commission_paid += comm
                        total_stamp_tax_paid += stamp
                positions.clear()

                # ---- 买入（扣佣金）----
                buyable = [sid for sid in selected if sid in today_prices]
                if buyable:
                    capital_each = cash / len(buyable)
                    for sid in buyable:
                        effective_price = today_prices[sid] * (1 + self.commission)
                        shares = capital_each / effective_price
                        positions[sid] = shares
                        comm = shares * today_prices[sid] * self.commission
                        cash -= capital_each
                        total_commission_paid += comm

            # ---- 记录每日净值 ----
            holdings_value = 0.0
            for sid, shares in positions.items():
                price = today_prices.get(sid) or last_known_prices.get(sid)
                if price:
                    holdings_value += shares * price
            total_value = cash + holdings_value

            daily_history.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "portfolio_value": total_value,
                "benchmark_value": benchmark_values[day_idx],
                "cash": cash,
                "num_holdings": len(positions),
                "is_rebalance": is_rebalance,
                "num_bullish": num_bullish if is_rebalance else np.nan,
                "top5_stocks": ",".join(selected[:5]) if is_rebalance else "",
            })

        if not daily_history:
            print("No data in backtest period.")
            return pd.DataFrame()

        df = pd.DataFrame(daily_history).set_index("date")
        df.index = pd.to_datetime(df.index)

        self._print_report(df, total_commission_paid, total_stamp_tax_paid)
        return df

    def _print_report(self, df, total_comm, total_stamp):
        initial = self.initial_capital
        final = df["portfolio_value"].iloc[-1]
        bench_final = df["benchmark_value"].iloc[-1]
        total_return = (final / initial) - 1
        bench_return = (bench_final / initial) - 1

        df_tmp = df.copy()
        df_tmp["returns"] = df_tmp["portfolio_value"].pct_change()
        df_tmp["bench_returns"] = df_tmp["benchmark_value"].pct_change()

        n_days = len(df_tmp)
        ann_factor = 252 / max(n_days, 1)
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        ann_vol = df_tmp["returns"].std() * (252 ** 0.5)
        sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0.0

        running_max = df["portfolio_value"].cummax()
        max_dd = ((df["portfolio_value"] - running_max) / running_max).min()

        bench_ann = (1 + bench_return) ** (252 / max(n_days, 1)) - 1
        excess_return = ann_return - bench_ann

        n_rebalance = int(df["is_rebalance"].sum())

        print("\n" + "=" * 45)
        print("         BACKTEST PERFORMANCE REPORT")
        print("=" * 45)
        print(f"  Period:            {df.index[0].date()} → {df.index[-1].date()} ({n_days} trading days)")
        print(f"  Rebalance Count:   {n_rebalance} times (every {self.rebalance_interval} days)")
        print(f"  Initial Capital:   {initial:>15,.0f}")
        print(f"  Final Value:       {final:>15,.0f}")
        print(f"  Benchmark Final:   {bench_final:>15,.0f}")
        print("-" * 45)
        print(f"  Total Return:      {total_return:>14.2%}")
        print(f"  Benchmark Return:  {bench_return:>14.2%}")
        print(f"  Excess Return(ann):{excess_return:>14.2%}")
        print(f"  Ann. Return:       {ann_return:>14.2%}")
        print(f"  Ann. Volatility:   {ann_vol:>14.2%}")
        print(f"  Sharpe Ratio:      {sharpe:>14.2f}")
        print(f"  Max Drawdown:      {max_dd:>14.2%}")
        print("-" * 45)
        print(f"  Commission(万{self.commission*10000:.3f}): {total_comm:>12,.2f}")
        print(f"  Stamp Tax({self.stamp_tax:.1%}):     {total_stamp:>12,.2f}")
        print(f"  Total Cost:        {total_comm + total_stamp:>12,.2f}")
        print("=" * 45 + "\n")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest with real market data")
    parser.add_argument("--checkpoint_path",    type=str, required=True,
                        help="Path to model_final.pth")
    parser.add_argument("--data_dir",           type=str, required=True,
                        help="Dir containing daily_summary_YYYY-MM-DD.csv")
    parser.add_argument("--start_date",         type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date",           type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--initial_capital",    type=float, default=1_000_000)
    parser.add_argument("--top_k",              type=int,   default=5,
                        help="Number of stocks to hold each period")
    parser.add_argument("--rebalance_interval", type=int,   default=5,
                        help="Rebalance every N trading days")
    parser.add_argument("--commission",         type=float, default=0.0000875,
                        help="Commission rate per trade (default: 0.0000875 = 万0.875)")
    parser.add_argument("--stamp_tax",          type=float, default=0.0,
                        help="Stamp tax rate, sell-only (default: 0.0)")
    parser.add_argument("--batch_size",         type=int,   default=8)
    parser.add_argument("--output_csv",         type=str,   default="backtest_history.csv")
    parser.add_argument("--plot",               action="store_true",
                        help="Auto-generate performance chart after backtest")
    parser.add_argument("--plot_output",        type=str,   default="backtest_analysis.png",
                        help="Path for the output chart image")
    args = parser.parse_args()

    engine = BacktestEngine(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        top_k=args.top_k,
        rebalance_interval=args.rebalance_interval,
        commission=args.commission,
        stamp_tax=args.stamp_tax,
        batch_size=args.batch_size,
    )
    df = engine.run()
    if not df.empty:
        df.to_csv(args.output_csv)
        print(f"Saved daily history to {args.output_csv}")

        if args.plot:
            from src.visualization import plot_backtest_results
            plot_backtest_results(df=df, output_path=args.plot_output)


if __name__ == "__main__":
    main()
