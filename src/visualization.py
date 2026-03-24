
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_backtest_results(csv_path="backtest_history.csv", output_path="backtest_analysis.png"):
    """
    读取回测历史数据并生成分析图表。
    """
    try:
        # 读取数据
        history = pd.read_csv(csv_path, index_col='date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        print("Please run the backtest first to generate the history file.")
        return

    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Backtest Performance Analysis', fontsize=16)

    # --- 子图1: 净值曲线 ---
    ax1.plot(history.index, history['portfolio_value'], label='Portfolio Value', color='#1f77b4')
    ax1.set_ylabel('Portfolio Value')
    ax1.set_title('Portfolio Value Over Time')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    # 格式化Y轴为货币格式
    formatter = mticker.FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax1.yaxis.set_major_formatter(formatter)

    # --- 子图2: 回撤曲线 ---
    running_max = history['portfolio_value'].cummax()
    drawdown = (history['portfolio_value'] - running_max) / running_max
    
    ax2.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.3)
    ax2.plot(drawdown.index, drawdown, color='#d62728', linewidth=1)
    ax2.set_ylabel('Drawdown')
    ax2.set_title('Portfolio Drawdown')
    ax2.grid(True, linestyle='--', alpha=0.6)
    # 格式化Y轴为百分比格式
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # 优化X轴日期显示
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图表
    plt.savefig(output_path)
    print(f"\nAnalysis plot saved to {output_path}")

if __name__ == '__main__':
    plot_backtest_results()
