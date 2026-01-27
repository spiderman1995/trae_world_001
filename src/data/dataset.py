import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .chinext50 import get_chinext50_constituents

class StockDataset(Dataset):
    def __init__(self, data_dir, seq_len=180, pred_len=40, mode='train'):
        """
        Args:
            data_dir (str): Path to directory containing daily_summary_YYYY-MM-DD.csv files.
            seq_len (int): Input sequence length (days). Default 180.
            pred_len (int): Prediction horizon (days). Default 40 (approx 2 months).
            mode (str): 'train' or 'val'.
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        
        # 1. Load and organize data
        self.stock_data = self._load_data()
        
        # 2. Build index map (stock, start_index)
        self.indices = self._build_indices()
        
    def _load_data(self):
        print(f"Loading data from {self.data_dir}...")
        file_patterns = os.path.join(self.data_dir, "daily_summary_*.csv")
        files = sorted(glob.glob(file_patterns))
        
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            
        # Read all files
        all_dfs = []
        for f in tqdm(files, desc="Reading CSVs"):
            try:
                # Read only necessary columns if possible to save memory
                # But here we read all and filter
                df = pd.read_csv(f)
                
                # Extract date from filename: daily_summary_2024-01-02.csv
                basename = os.path.basename(f)
                date_str = basename.replace("daily_summary_", "").replace(".csv", "")
                df['date'] = pd.to_datetime(date_str)
                
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
        if not all_dfs:
            return {}
            
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # Filter for ChiNext 50
        # For simplicity, we use the static list. Ideally pass date to get_chinext50_constituents
        # valid_stocks = get_chinext50_constituents()
        # full_df['StockID'] = full_df['StockID'].astype(str)
        # full_df = full_df[full_df['StockID'].isin(valid_stocks)]
        
        # Pivot or organize by StockID
        # We need to ensure each day has 1442 rows.
        # Wait, the CSV format: One row per tick?
        # Let's check the CSV content again.
        # Row 1: 300562, 09:29:50, ...
        # Row 2: 300562, 09:30:00, ...
        # Yes, it is tick data (aggregated).
        # We need to group by StockID and Date.
        
        # Sort by Stock, Date, Time
        full_df.sort_values(['StockID', 'date', 'Time'], inplace=True)
        
        grouped = full_df.groupby('StockID')
        
        stock_data = {}
        for stock_id, group in tqdm(grouped, desc="Processing Stocks"):
            stock_id = str(stock_id)
            # Group by date to check integrity
            daily_groups = group.groupby('date')
            
            stock_tensors = []
            valid_dates = []
            
            for date, day_df in daily_groups:
                # Check shape
                # Drop non-feature columns
                feats = day_df.drop(['StockID', 'Time', 'date'], axis=1).values
                
                # Ensure fixed size 1442
                # If less, pad? If more, truncate?
                # User said "Daily data shape is 1442*18". We assume it's consistent.
                # If not, we might need padding.
                if feats.shape[0] != 1442:
                    # Simple padding or truncation for robustness
                    if feats.shape[0] < 1442:
                        pad = np.zeros((1442 - feats.shape[0], feats.shape[1]))
                        feats = np.vstack([feats, pad])
                    else:
                        feats = feats[:1442]
                        
                stock_tensors.append(feats)
                valid_dates.append(date)
                
            if not stock_tensors:
                continue
                
            # Stack: [Days, 1442, 18]
            data_tensor = np.stack(stock_tensors).astype(np.float32)
            
            # Normalize (Global or Per-Stock?)
            # Per-stock Z-score is safer for stationarity
            mean = np.mean(data_tensor, axis=(0, 1), keepdims=True)
            std = np.std(data_tensor, axis=(0, 1), keepdims=True) + 1e-5
            data_tensor = (data_tensor - mean) / std
            
            # Transpose to [Days, Channels, Length] for 1DCNN
            # Original: [Days, Length, Channels] -> [Days, Channels, Length]
            data_tensor = np.transpose(data_tensor, (0, 2, 1))
            
            stock_data[stock_id] = {
                'data': torch.from_numpy(data_tensor),
                'dates': valid_dates,
                # Store un-normalized closes for target calculation? 
                # We need Close price. 
                # Wait, "avg_trade_price" is in features. Is there "Close"?
                # The prompt said "18 features... excluding index and stock code".
                # The CSV header has "avg_trade_price".
                # I'll use "avg_trade_price" of the last tick as Close? Or is there a High/Low column?
                # CSV features: trade_count, total_trade_amt, avg_trade_price...
                # It doesn't seem to have OHLC explicitly.
                # I will use `avg_trade_price` as the price proxy.
                # High = max(avg_trade_price over day), Low = min...
                # Actually, the user asks to predict "Future 2 months High/Low".
                # This usually refers to Daily High/Low.
                # If I only have `avg_trade_price` of 10s ticks, I can approximate Daily High = Max(10s Avg Prices).
                'prices': torch.tensor([df['avg_trade_price'].mean() for _, df in daily_groups], dtype=torch.float32) 
                # Note: Using daily mean price as "Close" for simplicity, or we can use the last tick.
            }
            
        return stock_data

    def _build_indices(self):
        indices = []
        for stock_id, data_dict in self.stock_data.items():
            num_days = len(data_dict['dates'])
            # We need seq_len history + pred_len future
            # Valid start indices
            # If we are at index `i`, we take `data[i : i+seq_len]` as input.
            # Target is from `i+seq_len` to `i+seq_len+pred_len`.
            # So max index `i` is `num_days - seq_len - pred_len`.
            
            # If num_days is insufficient, skip
            if num_days < self.seq_len + self.pred_len:
                # print(f"Skipping {stock_id}: {num_days} days < {self.seq_len + self.pred_len}")
                continue
                
            # If num_days == seq_len + pred_len, we have 1 valid sample (index 0).
            # Indices range: 0 to num_days - seq_len - pred_len (exclusive? No, Range is exclusive)
            # range(1) -> [0]. So we need +1 if we want to include the last possible one?
            # Let's check math:
            # Data: [0, 1, 2, 3] (4 days)
            # Seq=2, Pred=1. Total=3.
            # i=0: Input [0,1], Target [2]. Valid.
            # i=1: Input [1,2], Target [3]. Valid.
            # num_days - seq - pred = 4 - 2 - 1 = 1. range(1) is [0]. Misses i=1.
            # So range should be range(num_days - self.seq_len - self.pred_len + 1)
            
            for i in range(num_days - self.seq_len - self.pred_len + 1):
                indices.append((stock_id, i))
        
        # If indices is empty, it means no stock has enough history.
        # This often happens in testing with small data samples.
        # For debugging, we can relax the constraint or warn.
        if not indices:
             print(f"WARNING: No valid sequences found! Max days: {max([len(d['dates']) for d in self.stock_data.values()]) if self.stock_data else 0}. Req: {self.seq_len + self.pred_len}")
        
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        stock_id, start_idx = self.indices[idx]
        data_dict = self.stock_data[stock_id]
        
        # Input Sequence
        # Shape: [180, 18, 1442]
        seq_data = data_dict['data'][start_idx : start_idx + self.seq_len]
        
        # Targets
        # Future window
        future_start = start_idx + self.seq_len
        future_end = future_start + self.pred_len
        future_prices = data_dict['prices'][future_start : future_end]
        current_price = data_dict['prices'][future_start - 1] # Last day of input
        
        # 1. Extreme Values (Relative to Current Price)
        # We predict price, but usually regression is easier on returns or log-prices.
        # User says "Predict High/Low".
        # Let's predict the value directly or ratio?
        # Ratio is better: Max / Current, Min / Current.
        target_high = torch.max(future_prices) / current_price
        target_low = torch.min(future_prices) / current_price
        
        # 2. Sharpe Ratio
        # Sharpe = Mean(Returns) / Std(Returns) * sqrt(252)
        # Daily returns in future window
        returns = torch.diff(future_prices) / future_prices[:-1]
        if torch.std(returns) == 0:
            target_sharpe = torch.tensor(0.0)
        else:
            target_sharpe = torch.mean(returns) / torch.std(returns) * np.sqrt(252)
            
        # 3. Direction
        # Rise if Return > 0 (Cumulative)
        # (End - Start) / Start
        total_return = (future_prices[-1] - current_price) / current_price
        target_direction = (total_return > 0).float()
        
        # 4. BigBuy (Physics Constraint)
        # Need to extract "BigBuy" feature from the last day of input?
        # User: "Extract BigBuy ... from raw input Batch"
        # We'll pass the input, and the model can extract it.
        # Or we pass it explicitly.
        # Let's extract it here for convenience?
        # "buy_big_order_amt_ratio" is column index 9 (0-based) in CSV?
        # CSV: StockID, Time, trade_count, total_trade_amt, avg_trade_price, total_buy...
        # 0: StockID, 1: Time, 2: trade_count...
        # Let's assume the feature tensor preserves order.
        # We'll let the model extract it from the tensor.
        
        return seq_data, {
            'high': target_high,
            'low': target_low,
            'sharpe': target_sharpe,
            'direction': target_direction,
            'current_price': current_price
        }

