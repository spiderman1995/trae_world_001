import os
import glob
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .chinext50 import get_chinext50_constituents

# ---------------------------------------------------------------------------
# 尾盘集合竞价零行处理
# 原始 CSV 每天 1442 tick (9:30~15:00, 每10秒)
# 其中 14:57:00~14:59:50 (index 1423~1440, 共18行) 为集合竞价期间零行
# 15:00:00 (index 1441) 为集合竞价结果，保留
# 加载后自动裁剪为 1424 tick，供网络使用
# ---------------------------------------------------------------------------
RAW_TICKS = 1442
AUCTION_ZERO_START = 1423  # 14:57:00
AUCTION_ZERO_END = 1441    # 14:59:50 (exclusive end for slice)
CLEAN_TICKS = RAW_TICKS - (AUCTION_ZERO_END - AUCTION_ZERO_START)  # 1424


def trim_auction_zeros(arr):
    """去除集合竞价零行。arr 可以是 [1442, 18] 或 [18, 1442]（按 axis=-1 裁剪）。"""
    return np.concatenate([arr[..., :AUCTION_ZERO_START], arr[..., AUCTION_ZERO_END:]], axis=-1)


class StockDataset(Dataset):
    def __init__(self, data_dir, seq_len=180, pred_len=60, start_date=None, end_date=None,
                 mean=None, std=None, stock_ids=None, sample_stride=10):
        """
        Args:
            data_dir (str): Path to directory containing daily_summary_YYYY-MM-DD.csv files.
            seq_len (int): Input sequence length (days). Default 180.
            pred_len (int): Prediction horizon (days). Default 60.
            start_date (str): Filter data starting from this date (YYYY-MM-DD).
            end_date (str): Filter data up to this date (YYYY-MM-DD).
            mean (Tensor): Pre-computed mean for normalization.
            std (Tensor): Pre-computed std for normalization.
            stock_ids (list): Stock IDs to include. None = fallback to ChiNext50.
            sample_stride (int): Sliding window stride for sample generation. Default 10.
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.sample_stride = sample_stride
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        # 股票池：外部传入或回退到 ChiNext50
        if stock_ids is not None:
            self.stock_ids = set(str(s) for s in stock_ids)
        else:
            self.stock_ids = set(get_chinext50_constituents())
        
        # 1. Get file list
        self.filtered_files = self._get_filtered_files()

        # 2. 一次读取：加载数据 + 计算统计量（如果需要）
        need_stats = (mean is None or std is None)
        if not need_stats:
            self.mean = mean
            self.std = std
        self.stock_data, computed_mean, computed_std = self._load_data_and_stats(compute_stats=need_stats)
        if need_stats:
            self.mean = computed_mean
            self.std = computed_std

        # 3. Build index map (stock, start_index)
        self.indices = self._build_indices()

    def _get_filtered_files(self):
        file_patterns = os.path.join(self.data_dir, "daily_summary_*.csv")
        files = sorted(glob.glob(file_patterns))
        
        filtered_files = []
        for f in files:
            basename = os.path.basename(f)
            date_str = basename.replace("daily_summary_", "").replace(".csv", "")
            try:
                file_date = pd.to_datetime(date_str)
                if self.start_date and file_date < self.start_date:
                    continue
                if self.end_date and file_date > self.end_date:
                    continue
                filtered_files.append(f)
            except:
                continue
        return filtered_files

    def _load_data_and_stats(self, compute_stats=True):
        """
        一次读取完成两件事：加载数据到内存 + 计算 mean/std（如需要）。
        避免对同一批 CSV 文件重复读取。

        Returns:
            (stock_data_dict, mean_tensor_or_None, std_tensor_or_None)
        """
        logger = logging.getLogger(__name__)
        if not self.filtered_files:
            return {}, None, None

        valid_stocks = self.stock_ids

        def _to_stock_id_str(value):
            if isinstance(value, (int, np.integer)):
                return f"{int(value):06d}"
            return str(value).strip()

        # ---- 第一步：8 线程并行读 CSV + 预处理 ----
        def _read_one_csv(f):
            df = pd.read_csv(f)
            if "StockID" not in df.columns or "Time" not in df.columns:
                raise ValueError(f"Invalid CSV schema (missing StockID/Time). file={f}")
            feature_cols = [c for c in df.columns if c not in ("StockID", "Time")]
            if len(feature_cols) != 18:
                raise ValueError(f"Invalid CSV schema (expected 18 feature cols). file={f}")
            df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            if df[feature_cols].isna().any().any():
                nan_count = df[feature_cols].isna().sum().sum()
                df[feature_cols] = df.groupby('StockID')[feature_cols].apply(
                    lambda g: g.interpolate(method='linear', limit_direction='both')
                )
                df[feature_cols] = df[feature_cols].fillna(0)
                logger.warning("Interpolated %d NaN/Inf values in file=%s", nan_count, f)
            basename = os.path.basename(f)
            date_str = basename.replace("daily_summary_", "").replace(".csv", "")
            df['date'] = pd.to_datetime(date_str)
            return df

        logger.info(f"Loading {len(self.filtered_files)} files (8 threads, single pass)...")
        all_dfs = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_read_one_csv, f): f for f in self.filtered_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Reading CSVs"):
                f = futures[future]
                try:
                    all_dfs.append(future.result())
                except Exception as e:
                    logger.exception(f"Error reading {f}: {e}")

        all_dfs.sort(key=lambda df: df['date'].iloc[0])

        if not all_dfs:
            return {}, None, None

        # ---- 第二步：单次遍历，同时存数据 + 累加统计量 ----
        stock_data = {}
        total_sum = np.zeros(18)
        total_sq_sum = np.zeros(18)
        total_count = 0

        for day_df in tqdm(all_dfs, desc="Processing & Stats"):
            current_date = day_df['date'].iloc[0]
            feature_cols = [c for c in day_df.columns if c not in ("StockID", "Time", "date")]

            for stock_id_raw, group in day_df.groupby('StockID'):
                stock_id = _to_stock_id_str(stock_id_raw)
                if stock_id not in valid_stocks:
                    continue

                feats = group[feature_cols].values
                if feats.shape[1] != 18:
                    continue

                # 缺失值插值兜底
                if not np.isfinite(feats).all():
                    feats = feats.astype(np.float64)
                    mask = ~np.isfinite(feats)
                    for col in range(feats.shape[1]):
                        col_mask = mask[:, col]
                        if col_mask.any():
                            valid_m = ~col_mask
                            if valid_m.sum() >= 2:
                                feats[col_mask, col] = np.interp(
                                    np.where(col_mask)[0], np.where(valid_m)[0], feats[valid_m, col]
                                )
                            else:
                                feats[col_mask, col] = 0

                # tick 数验证 + 裁剪
                if feats.shape[0] == RAW_TICKS:
                    feats = feats.T  # [18, 1442]
                    feats = trim_auction_zeros(feats)  # [18, 1424]
                elif feats.shape[0] == CLEAN_TICKS:
                    feats = feats.T  # [18, 1424]
                else:
                    continue  # 跳过异常行数

                # 累加统计量（在 log1p 变换后的空间）
                if compute_stats:
                    stats_arr = feats.copy()
                    stats_arr[:6, :] = np.log1p(stats_arr[:6, :])
                    total_sum += np.sum(stats_arr, axis=1)   # [18]
                    total_sq_sum += np.sum(stats_arr ** 2, axis=1)
                    total_count += stats_arr.shape[1]  # 1424 ticks

                # 存入数据字典
                if stock_id not in stock_data:
                    stock_data[stock_id] = {'data': [], 'dates': []}
                stock_data[stock_id]['data'].append(feats)
                stock_data[stock_id]['dates'].append(current_date)

        # ---- 第三步：整理输出 ----
        final_stock_data = {}
        for stock_id, content in stock_data.items():
            if len(content['data']) > 0:
                final_stock_data[stock_id] = {
                    'data': np.stack(content['data']),
                    'dates': content['dates']
                }

        logger.info(f"Loaded {len(final_stock_data)} stocks, {sum(len(v['dates']) for v in final_stock_data.values())} stock-days.")

        # 计算 mean/std
        computed_mean, computed_std = None, None
        if compute_stats and total_count > 0:
            mean = total_sum / total_count
            var = (total_sq_sum / total_count) - (mean ** 2)
            std = np.sqrt(np.maximum(var, 1e-10))
            computed_mean = torch.FloatTensor(mean)
            computed_std = torch.FloatTensor(std)
            logger.info(f"Stats computed over {total_count} ticks.")

        return final_stock_data, computed_mean, computed_std
        
    def _build_indices(self):
        indices = []
        for stock_id, data_dict in self.stock_data.items():
            num_days = len(data_dict['dates'])
            total_window = self.seq_len + self.pred_len
            if num_days < total_window:
                continue

            for i in range(0, num_days - total_window + 1, self.sample_stride):
                indices.append((stock_id, i))
                
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        stock_id, start_idx = self.indices[idx]
        data_dict = self.stock_data[stock_id]
        
        # Input Sequence
        # Shape: [Seq, C, L]
        input_seq = data_dict['data'][start_idx : start_idx + self.seq_len]
        
        # Target Window
        target_start = start_idx + self.seq_len
        target_end = target_start + self.pred_len
        target_seq = data_dict['data'][target_start : target_end]
        
        # Calculate Targets
        # 1. High/Low Ratio relative to CURRENT price (Last price of input sequence)
        # Current Price is usually the Close price of the last day of input.
        # We need raw price to calculate ratios.
        # But we only have features. Assuming one feature is Price or we can infer return?
        # If we don't have raw price, we can predict return.
        # Let's assume the features are normalized, so we can't get absolute price easily.
        # However, the task usually implies predicting return.
        # "High/Low" might mean Max(High)/Current - 1.
        
        # Let's assume input data is RAW for now (from code analysis above, it seems raw).
        # Feature 2 is avg_trade_price.
        # "300562, 09:29:50, ..."
        # Usually Tick Data: trade_count, total_trade_amt, avg_trade_price, ...
        # We use Feature 2 for price-related calculations.
        
        # Get Current Price (Last tick of last input day)
        # input_seq: [Seq, C, L] -> [seq_len, 18, 1424]
        # Last day: input_seq[-1] -> [18, 1424]
        # Last tick: input_seq[-1, 2, -1] (Using avg_trade_price)
        
        # 基准价：最后一个 tick（集合竞价结果 15:00:00）
        current_price = input_seq[-1, 2, -1] + 1e-6

        # 数据已在预处理阶段去除集合竞价零行，可直接取 max/min
        daily_max = target_seq[:, 2, :].max(axis=1)
        daily_min = target_seq[:, 2, :].min(axis=1)

        max_day = int(np.argmax(daily_max))
        min_day = int(np.argmin(daily_min))

        max_value = daily_max[max_day] / current_price - 1.0  # 收益率，中心在0附近
        min_value = daily_min[min_day] / current_price - 1.0
        
        # Standardize input_seq for model input
        input_data = input_seq.copy()
        
        # 1. Apply log1p to first 6 columns
        input_data[:, :6, :] = np.log1p(input_data[:, :6, :])
        
        # 2. Apply Z-Score using global stats
        if self.mean is not None and self.std is not None:
            # Reshape stats for broadcasting: [1, 18, 1]
            mean = self.mean.view(1, 18, 1).numpy()
            std = self.std.view(1, 18, 1).numpy()
            input_data = (input_data - mean) / (std + 1e-6)
        
        return torch.FloatTensor(input_data), {
            'max_value': torch.tensor(max_value, dtype=torch.float),
            'min_value': torch.tensor(min_value, dtype=torch.float),
            'max_day': torch.tensor(max_day, dtype=torch.long),
            'min_day': torch.tensor(min_day, dtype=torch.long),
            'current_price': torch.tensor(current_price, dtype=torch.float)
        }
