
import pandas as pd
import torch
import numpy as np
from models.transformer import StockViT

class BacktestEngine:
    """
    回测引擎，负责协调数据、策略和投资组合，驱动整个回测流程。
    """
    def __init__(self, start_date, end_date, initial_capital, data, strategy):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = data
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)

    def run(self):
        """
        执行回测
        """
        # 获取时间范围内的数据
        backtest_data = self.data.get_data_in_range(self.start_date, self.end_date)
        
        # 按周进行迭代
        for week_start_date, weekly_data in backtest_data.groupby(pd.Grouper(freq='W')):
            if weekly_data.empty:
                continue
            # 1. 生成信号
            signals, predictions = self.strategy.generate_signals(weekly_data)
            
            # 2. 投资组合根据信号进行调仓
            self.portfolio.rebalance(week_start_date, weekly_data, signals, predictions)
            
        # 3. 生成性能报告
        self.portfolio.generate_performance_report()

class Strategy:
    """
    策略基类
    """
    def __init__(self, model):
        self.model = model

    def generate_signals(self, data):
        raise NotImplementedError("Should implement generate_signals()")

class TopKStrategy(Strategy):
    """
    每周买入预测排名前K%的策略
    """
    def __init__(self, model, k=0.1):
        super().__init__(model)
        self.k = k

    def generate_signals(self, weekly_data):
        """
        根据模型预测，生成交易信号
        """
        # 1. 使用模型进行预测
        # 假设 weekly_data 需要转换为模型所需的 Tensor 格式
        # inputs = self._prepare_data_for_model(weekly_data)
        # with torch.no_grad():
        #    predictions = self.model(inputs)
        # predictions = self._format_predictions(predictions)
        predictions = self._mock_predict(weekly_data) # 暂时保留模拟预测

        # 2. 根据预测结果进行排名
        predictions.sort(key=lambda x: x[1], reverse=True) # 按预测收益率降序排列

        # 3. 选出前K%的资产
        top_k_assets = {pred[0] for pred in predictions[:int(len(predictions) * self.k)]}

        # 4. 生成信号
        signals = {}
        # 假设数据中有 asset_id 列
        unique_assets = weekly_data['asset_id'].unique()
        for asset_id in unique_assets:
            if asset_id in top_k_assets:
                signals[asset_id] = 'BUY'
            else:
                signals[asset_id] = 'SELL' # 如果不在前K%，则卖出
        
        return signals, predictions # 返回信号和排序后的预测

    def _mock_predict(self, weekly_data):
        """
        模拟模型预测，返回随机结果
        """
        asset_ids = weekly_data['asset_id'].unique()
        return [(asset_id, np.random.rand(), np.random.randint(1, 5)) for asset_id in asset_ids]

class Portfolio:
    """
    投资组合，负责管理资金、持仓和交易记录。
    """
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.history = []
        self.last_value = initial_capital # 追踪上期价值

    def rebalance(self, date, data, signals, predictions):
        """
        根据信号执行调仓，并打印详细日志
        """
        print(f"\n--- Rebalancing on {date.strftime('%Y-%m-%d')} ---")

        # 打印预测排名
        print("\nTop 5 Predicted Assets:")
        for asset_id, pred_return, _ in predictions[:5]:
            print(f"  - {asset_id}: Predicted Return = {pred_return:.4f}")

        # 记录交易
        trades = {'BUY': [], 'SELL': []}

        # 卖出
        for asset_id, signal in signals.items():
            if signal == 'SELL' and asset_id in self.positions:
                price = data[data['asset_id'] == asset_id]['price'].iloc[0]
                self.cash += self.positions.pop(asset_id) * price
                trades['SELL'].append(asset_id)

        # 买入
        buy_signals = {asset_id: signal for asset_id, signal in signals.items() if signal == 'BUY'}
        if buy_signals:
            capital_per_asset = self.cash / len(buy_signals)
            for asset_id in buy_signals:
                price = data[data['asset_id'] == asset_id]['price'].iloc[0]
                quantity = capital_per_asset / price
                self.positions[asset_id] = self.positions.get(asset_id, 0) + quantity
                self.cash -= capital_per_asset
                trades['BUY'].append(asset_id)
        
        # 打印交易总结
        print("\nTrades Executed:")
        print(f"  - Sold: {trades['SELL'] if trades['SELL'] else 'None'}")
        print(f"  - Bought: {trades['BUY'] if trades['BUY'] else 'None'}")

        # --- 新增日志 ---
        portfolio_value = self.cash + self._calculate_positions_value(date, data)
        period_return = (portfolio_value / self.last_value) - 1 if self.last_value != 0 else 0
        self.last_value = portfolio_value

        print("\nCurrent Portfolio:")
        if not self.positions:
            print("  - Empty")
        else:
            for asset_id, quantity in self.positions.items():
                print(f"  - {asset_id}: {quantity:.4f} shares")
        
        print("\nPortfolio Snapshot:")
        print(f"  - Cash: {self.cash:.2f}")
        print(f"  - Positions Value: {portfolio_value - self.cash:.2f}")
        print(f"  - Total Value: {portfolio_value:.2f}")
        print(f"  - Period Return: {period_return:.2%}")
        print("---------------------------------------")

        self._record_history(date, data)

    def _calculate_positions_value(self, date, data):
        """
        计算当前持仓总价值
        """
        value = 0
        for asset_id, quantity in self.positions.items():
            # It's possible the asset for the last day is not in the current weekly data
            if not data[data['asset_id'] == asset_id].empty:
                price = data[data['asset_id'] == asset_id]['price'].iloc[0]
                value += quantity * price
        return value

    def _record_history(self, date, data):
        """
        记录每个调仓日的投资组合状态
        """
        portfolio_value = self.cash + self._calculate_positions_value(date, data)
        self.history.append((date, portfolio_value))


    def generate_performance_report(self):
        """
        生成并打印性能报告
        """
        print("\n\n--- Final Backtest Performance Report ---")
        
        if not self.history:
            print("No trades were made.")
            return

        portfolio_df = pd.DataFrame(self.history, columns=['date', 'portfolio_value']).set_index('date')
        
        # --- 收益指标 ---
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        annualized_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else 0
        
        # --- 风险指标 ---
        # 1. 计算周期回报率
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # 2. 年化波动率 (假设每周调仓，一年52周)
        annualized_volatility = portfolio_df['returns'].std() * np.sqrt(52)
        
        # 3. 夏普比率 (假设无风险利率为0)
        sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0
        
        # 4. 最大回撤
        running_max = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()

        print("\n------ Performance Metrics ------")
        print(f"Initial Portfolio Value: {self.initial_capital:.2f}")
        print(f"Final Portfolio Value:   {portfolio_df['portfolio_value'].iloc[-1]:.2f}")
        print(f"Total Return:            {total_return:.2%}")
        print(f"Annualized Return:       {annualized_return:.2%}")
        
        print("\n------ Risk Metrics ------")
        print(f"Annualized Volatility:   {annualized_volatility:.2%}")
        print(f"Sharpe Ratio:            {sharpe_ratio:.2f}")
        print(f"Max Drawdown:            {max_drawdown:.2%}")
        print("---------------------------------------")

        # 保存历史记录到 CSV
        portfolio_df.to_csv("backtest_history.csv")
        print("\nBacktest history saved to backtest_history.csv")

class DataLoader:
    """
    数据加载器，负责提供回测所需的数据。
    """
    def __init__(self, start_date, end_date, asset_ids):
        self.start_date = start_date
        self.end_date = end_date
        self.asset_ids = asset_ids

    def get_data_in_range(self, start_date, end_date):
        """
        生成并返回指定时间范围内的模拟数据
        """
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        df_list = []
        for asset_id in self.asset_ids:
            # 生成随机价格数据作为模拟
            price = 100 + np.random.randn(len(dates)).cumsum()
            df = pd.DataFrame({'date': dates, 'asset_id': asset_id, 'price': price})
            df_list.append(df)
        
        full_df = pd.concat(df_list).set_index('date')
        return full_df

def load_model(model_path, device):
    """
    加载训练好的模型
    """
    # 使用从错误日志中推断出的参数来实例化模型
    model = StockViT(seq_len=60, embed_dim=1024)
    # 加载整个 checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    # 从 checkpoint 中提取模型的 state_dict
    # 假设模型权重保存在 'vit' 键下
    model_weights = checkpoint.get('vit', checkpoint)
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    # 1. 设置回测参数
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    initial_capital = 1000000
    asset_ids = [f'asset_{i}' for i in range(100)] # 模拟100个资产
    model_path = 'runs_rolling_v1/fold_7/model_final.pth' # 模型路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载数据和模型
    data_loader = DataLoader(start_date, end_date, asset_ids)
    try:
        model = load_model(model_path, device)
    except FileNotFoundError:
        print(f"Warning: Model file not found at {model_path}. Using a dummy model.")
        model = None # Or a dummy model object

    # 3. 初始化策略和回测引擎
    strategy = TopKStrategy(model, k=0.1)
    engine = BacktestEngine(start_date, end_date, initial_capital, data_loader, strategy)

    # 4. 运行回测
    engine.run()
