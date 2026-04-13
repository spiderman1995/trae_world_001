"""
visualization.py — 回测结果可视化（4 子图）

子图：
  1. 组合净值 vs 等权基准
  2. 回撤曲线
  3. 每期收益率柱状图（调仓周期）
  4. 持仓数 & 可选看涨股票数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates


def plot_backtest_results(df=None, csv_path=None, output_path="backtest_analysis.png"):
    """
    生成回测分析图表。

    Parameters
    ----------
    df : pd.DataFrame, optional
        backtest.py 返回的 DataFrame（index=date，含 portfolio_value / benchmark_value /
        num_holdings / num_bullish / is_rebalance 等列）。
    csv_path : str, optional
        若 df 为 None，则从 CSV 读取（兼容独立运行）。
    output_path : str
        输出图片路径。
    """
    if df is None:
        if csv_path is None:
            csv_path = "backtest_history.csv"
        try:
            df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        except FileNotFoundError:
            print(f"Error: {csv_path} not found. Run backtest first.")
            return

    if df.empty:
        print("Empty DataFrame, nothing to plot.")
        return

    has_benchmark = "benchmark_value" in df.columns
    has_holdings = "num_holdings" in df.columns
    has_bullish = "num_bullish" in df.columns
    has_rebalance = "is_rebalance" in df.columns

    # ---- 准备数据 ----
    dates = df.index
    nav = df["portfolio_value"]
    bench = df["benchmark_value"] if has_benchmark else None

    # 回撤
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max

    # 每调仓周期收益率
    period_returns = []
    period_bench_returns = []
    period_labels = []
    if has_rebalance:
        rebalance_mask = df["is_rebalance"].fillna(False).astype(bool)
        rebalance_idx = df.index[rebalance_mask].tolist()
        # 添加最后一天（如果不是调仓日）
        if len(rebalance_idx) > 0 and rebalance_idx[-1] != dates[-1]:
            rebalance_idx.append(dates[-1])
        for i in range(len(rebalance_idx) - 1):
            d_start = rebalance_idx[i]
            d_end = rebalance_idx[i + 1]
            v_start = nav.loc[d_start]
            v_end = nav.loc[d_end]
            ret = (v_end / v_start) - 1 if v_start > 0 else 0.0
            period_returns.append(ret)
            if has_benchmark:
                b_start = bench.loc[d_start]
                b_end = bench.loc[d_end]
                b_ret = (b_end / b_start) - 1 if b_start > 0 else 0.0
                period_bench_returns.append(b_ret)
            period_labels.append(d_start.strftime("%m/%d"))

    # ---- 绘图 ----
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=False,
                             gridspec_kw={"height_ratios": [3, 1.2, 1.5, 1]})
    fig.suptitle("Backtest Performance Analysis", fontsize=16, fontweight="bold")

    # ===== 子图 1：净值曲线 =====
    ax1 = axes[0]
    ax1.plot(dates, nav, label="Portfolio", color="#1f77b4", linewidth=1.5)
    if has_benchmark and bench is not None:
        ax1.plot(dates, bench, label="Benchmark (Equal-Weight)", color="#ff7f0e",
                 linewidth=1.2, linestyle="--", alpha=0.8)
    # 标记调仓日
    if has_rebalance:
        reb_dates = dates[df["is_rebalance"].fillna(False).astype(bool)]
        reb_values = nav.loc[reb_dates]
        ax1.scatter(reb_dates, reb_values, color="#2ca02c", marker="^", s=30,
                    zorder=5, label="Rebalance", alpha=0.7)

    ax1.set_ylabel("Portfolio Value")
    ax1.set_title("Net Asset Value")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.tick_params(axis="x", rotation=30)

    # ===== 子图 2：回撤 =====
    ax2 = axes[1]
    ax2.fill_between(dates, drawdown, 0, color="#d62728", alpha=0.3)
    ax2.plot(dates, drawdown, color="#d62728", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Portfolio Drawdown")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.tick_params(axis="x", rotation=30)

    # ===== 子图 3：每期收益率柱状图 =====
    ax3 = axes[2]
    if period_returns:
        x = np.arange(len(period_returns))
        width = 0.35
        colors = ["#2ca02c" if r >= 0 else "#d62728" for r in period_returns]

        if period_bench_returns:
            bars1 = ax3.bar(x - width / 2, period_returns, width, color=colors,
                            alpha=0.8, label="Portfolio")
            ax3.bar(x + width / 2, period_bench_returns, width, color="#ff7f0e",
                    alpha=0.5, label="Benchmark")
            ax3.legend(fontsize=8)
        else:
            ax3.bar(x, period_returns, width * 2, color=colors, alpha=0.8)

        ax3.set_xticks(x)
        ax3.set_xticklabels(period_labels, rotation=45, fontsize=7)
        ax3.axhline(0, color="black", linewidth=0.5)
    else:
        ax3.text(0.5, 0.5, "No rebalance periods", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=10, color="gray")
    ax3.set_ylabel("Period Return")
    ax3.set_title("Per-Rebalance-Period Returns")
    ax3.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # ===== 子图 4：持仓数 & 看涨股票数 =====
    ax4 = axes[3]
    if has_holdings:
        ax4.step(dates, df["num_holdings"], where="post", color="#1f77b4",
                 linewidth=1.2, label="Holdings")
    if has_bullish:
        bullish_vals = df["num_bullish"].copy()
        # 只在调仓日有值，前向填充以便可视化
        bullish_vals = bullish_vals.ffill()
        ax4.step(dates, bullish_vals, where="post", color="#9467bd",
                 linewidth=1.0, linestyle="--", alpha=0.7, label="Bullish Candidates")
    ax4.set_ylabel("Count")
    ax4.set_title("Holdings & Bullish Candidates")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(fontsize=8, loc="upper left")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax4.tick_params(axis="x", rotation=30)
    # 整数 Y 轴
    ax4.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nAnalysis plot saved to {output_path}")


if __name__ == "__main__":
    plot_backtest_results(csv_path="backtest_history.csv")
