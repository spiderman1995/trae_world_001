import os
import glob
import logging
import re
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
                 mean=None, std=None, stock_ids=None):
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
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        # 股票池：外部传入或回退到 ChiNext50
        if stock_ids is not None:
            self.stock_ids = set(str(s) for s in stock_ids)
        else:
            self.stock_ids = set(get_chinext50_constituents())
        
        # 1. Get file list
        self.filtered_files = self._get_filtered_files()
        
        # 2. Compute global statistics using STREAMING approach (Memory Efficient)
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            self.mean, self.std = self._compute_streaming_stats()
        
        # 3. Load and organize data (Keep in RAM for speed, but could be lazy)
        self.stock_data = self._load_data()
        
        # 4. Build index map (stock, start_index)
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

    def _compute_streaming_stats(self):
        """
        Compute global mean and std using a streaming approach to save memory.
        Reads file -> updates running sums -> discards data.
        """
        logger = logging.getLogger(__name__)
        if not self.filtered_files:
            return None, None
            
        logger.info("Pre-checking files for corruption...")
        safe_files = []
        for f in tqdm(self.filtered_files, desc="Pre-checking files"):
            try:
                pd.read_csv(f, nrows=5)  # Try reading only a few rows
                safe_files.append(f)
            except Exception as e:
                logger.warning(f"Excluding corrupted file {f}: {e}")

        if not safe_files:
            logger.error("No valid files found after pre-check.")
            return None, None

        logger.info(f"Found {len(safe_files)} non-corrupted files out of {len(self.filtered_files)}.")


        logger.info("Computing global statistics using streaming approach...")
        total_sum = np.zeros(18)
        total_sq_sum = np.zeros(18)
        total_count = 0

        valid_stocks = self.stock_ids

        for f in tqdm(safe_files, desc="Streaming Stats"):
            try:
                df = pd.read_csv(f)
                # Filter for valid stocks first to save computation
                df = df[df['StockID'].astype(str).str.strip().isin(valid_stocks)]
                if df.empty:
                    continue

                feature_cols = [c for c in df.columns if c not in ("StockID", "Time")]

                # Per-stock validation: accept 1442 raw ticks, trim auction zeros to 1424
                for _, grp in df.groupby('StockID'):
                    if len(grp) != RAW_TICKS:
                        continue
                    stock_arr = grp[feature_cols].to_numpy()  # [1442, 18]
                    if not np.isfinite(stock_arr).all():
                        continue
                    # 裁掉集合竞价零行: [1442, 18] → [1424, 18]
                    stock_arr = np.concatenate([stock_arr[:AUCTION_ZERO_START], stock_arr[AUCTION_ZERO_END:]], axis=0)
                    stock_arr[:, :6] = np.log1p(stock_arr[:, :6])
                    total_sum += np.sum(stock_arr, axis=0)
                    total_sq_sum += np.sum(stock_arr**2, axis=0)
                    total_count += stock_arr.shape[0]
            except Exception as e:
                logger.warning(f"Skipping corrupted file {f}: {e}")
                continue
                
        if total_count == 0:
            return None, None
            
        mean = total_sum / total_count
        # Variance = E[X^2] - (E[X])^2
        var = (total_sq_sum / total_count) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 1e-10)) # Numerical stability
        
        logger.info(f"Stats computed over {total_count} ticks.")
        return torch.FloatTensor(mean), torch.FloatTensor(std)
        
    def _load_data(self):
        logger = logging.getLogger(__name__)
        if not self.filtered_files:
            return {}
            
        logger.info(f"Loading data from {len(self.filtered_files)} files...")
        def is_valid_stock_id(value):
            if value is None:
                return False
            if isinstance(value, (int, np.integer)):
                return True
            s = str(value).strip()
            return re.fullmatch(r"\d{6}", s) is not None

        def to_stock_id_str(value):
            if isinstance(value, (int, np.integer)):
                return f"{int(value):06d}"
            return str(value).strip()

        def stock_id_format_stats(raw_ids):
            raw_ids = [x for x in raw_ids if x is not None]
            as_str = [str(x).strip() for x in raw_ids]
            stats = {
                "total": len(as_str),
                "prefix_SZ_SH": sum(1 for s in as_str if s.startswith(("SZ", "SH"))),
                "suffix_DOT_SZ_SH": sum(1 for s in as_str if s.endswith((".SZ", ".SH"))),
                "contains_dot": sum(1 for s in as_str if "." in s),
                "float_like_dot0": sum(1 for s in as_str if s.endswith(".0")),
                "not_6_digits": sum(1 for s in as_str if re.fullmatch(r"\d{6}", s) is None),
            }
            return stats

        # Read all files
        all_dfs = []
        for f in tqdm(self.filtered_files, desc="Reading CSVs"):
            try:
                # Read only necessary columns if possible to save memory
                # But here we read all and filter
                df = pd.read_csv(f)

                if "StockID" not in df.columns or "Time" not in df.columns:
                    logger.error("Invalid CSV schema: missing StockID/Time. file=%s cols=%s", f, list(df.columns))
                    raise ValueError(f"Invalid CSV schema (missing StockID/Time). file={f}")

                feature_cols = [c for c in df.columns if c not in ("StockID", "Time")]
                if len(feature_cols) != 18:
                    logger.error(
                        "Invalid CSV schema: expected 18 feature cols (20 total including StockID/Time). file=%s col_count=%d feature_col_count=%d",
                        f,
                        len(df.columns),
                        len(feature_cols),
                    )
                    logger.error("Feature cols in file=%s: %s", f, feature_cols)
                    raise ValueError(f"Invalid CSV schema (expected 18 feature cols). file={f}")

                df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
                if df[feature_cols].isna().any().any():
                    nan_cols = [c for c in feature_cols if df[c].isna().any()]
                    logger.error(
                        "NaN detected in feature columns. file=%s nan_cols=%s",
                        f,
                        nan_cols,
                    )
                    raise ValueError(f"NaN detected in feature columns. file={f} nan_cols={nan_cols}")
                arr = df[feature_cols].to_numpy()
                if not np.isfinite(arr).all():
                    logger.error("Inf detected in feature values. file=%s", f)
                    raise ValueError(f"Inf detected in feature values. file={f}")
                
                # Extract date from filename: daily_summary_2024-01-02.csv
                basename = os.path.basename(f)
                date_str = basename.replace("daily_summary_", "").replace(".csv", "")
                df['date'] = pd.to_datetime(date_str)
                df['_source_file'] = basename
                
                all_dfs.append(df)
            except Exception as e:
                logger.exception(f"Error reading {f}: {e}")
                
        if not all_dfs:
            return {}
            
        stock_data = {}

        valid_stocks = self.stock_ids

        day_stats = []
        
        # Iterate over daily dataframes
        for day_df in tqdm(all_dfs, desc="Processing Daily Data"):
            # day_df has 'date' column added
            current_date = day_df['date'].iloc[0]
            source_file = day_df['_source_file'].iloc[0] if '_source_file' in day_df.columns else "unknown"
            
            raw_unique = list(pd.unique(day_df['StockID']))
            fmt = stock_id_format_stats(raw_unique)
            invalid_samples = [str(x).strip() for x in raw_unique if not is_valid_stock_id(x)]
            if invalid_samples:
                logger.error(
                    "Invalid StockID format detected. Expect 6-digit codes only. date=%s file=%s invalid_sample=%s format_stats=%s",
                    str(current_date)[:10],
                    source_file,
                    invalid_samples[:10],
                    fmt,
                )
                raise ValueError(f"Invalid StockID format in file={source_file} date={str(current_date)[:10]}. Expect 6-digit codes.")

            stock_ids = [to_stock_id_str(x) for x in raw_unique if x is not None]
            hits = sorted(set(stock_ids) & valid_stocks)
            day_stats.append({
                "date": current_date,
                "file_stocks": len(set(stock_ids)),
                "pool_hits": len(hits),
                "format": fmt,
                "sample_raw": [str(x) for x in raw_unique[:10]],
                "sample_norm": [str(x) for x in stock_ids[:10]],
            })
            if len(hits) == 0:
                logger.warning(
                    "No stocks from pool matched on date=%s file=%s. raw_sample=%s format_stats=%s",
                    str(current_date)[:10],
                    [str(x) for x in raw_unique[:10]],
                    source_file,
                    fmt,
                )

            daily_grouped = day_df.groupby('StockID')
            
            for stock_id, group in daily_grouped:
                if stock_id is None or (isinstance(stock_id, float) and np.isnan(stock_id)):
                    logger.error("Missing StockID value in file=%s date=%s", source_file, str(current_date)[:10])
                    raise ValueError(f"Missing StockID value in file={source_file} date={str(current_date)[:10]}")

                stock_id = to_stock_id_str(stock_id)
                if re.fullmatch(r"\d{6}", stock_id) is None:
                    logger.error(
                        "Invalid StockID format in grouped data. Expect 6-digit. file=%s date=%s stock_id=%s",
                        source_file,
                        str(current_date)[:10],
                        stock_id,
                    )
                    raise ValueError(f"Invalid StockID format in file={source_file} date={str(current_date)[:10]} stock_id={stock_id}")
                
                # Filter for ChiNext 50
                if stock_id not in valid_stocks:
                    continue
                
                # Check shape
                feats = group.drop(['StockID', 'Time', 'date', '_source_file'], axis=1).values
                if feats.shape[1] != 18:
                    logger.error(
                        "Invalid feature column count for stock/day. Expect 18. file=%s date=%s stock_id=%s feat_cols=%d",
                        source_file,
                        str(current_date)[:10],
                        stock_id,
                        feats.shape[1],
                    )
                    raise ValueError(f"Invalid feature column count in file={source_file} date={str(current_date)[:10]} stock_id={stock_id}")
                if not np.isfinite(feats).all():
                    logger.error(
                        "Non-finite feature values for stock/day. file=%s date=%s stock_id=%s",
                        source_file,
                        str(current_date)[:10],
                        stock_id,
                    )
                    raise ValueError(f"Non-finite feature values in file={source_file} date={str(current_date)[:10]} stock_id={stock_id}")
                
                # Ensure raw size 1442 (or already trimmed 1424)
                if feats.shape[0] == RAW_TICKS:
                    pass  # will trim below
                elif feats.shape[0] == CLEAN_TICKS:
                    pass  # already clean
                else:
                    logger.error(
                        "Invalid tick row count for stock/day. Expect %d or %d. file=%s date=%s stock_id=%s rows=%d",
                        RAW_TICKS, CLEAN_TICKS,
                        source_file,
                        str(current_date)[:10],
                        stock_id,
                        feats.shape[0],
                    )
                    raise ValueError(f"Invalid tick row count in file={source_file} date={str(current_date)[:10]} stock_id={stock_id} rows={feats.shape[0]}")

                # Transpose to [C, L]
                feats = feats.T  # [18, 1442] or [18, 1424]

                # 裁掉集合竞价零行（如果是原始1442）
                if feats.shape[1] == RAW_TICKS:
                    feats = trim_auction_zeros(feats)  # [18, 1424]
                
                if stock_id not in stock_data:
                    stock_data[stock_id] = {'data': [], 'dates': []}
                    
                stock_data[stock_id]['data'].append(feats)
                stock_data[stock_id]['dates'].append(current_date)

        # Convert lists to numpy arrays
        final_stock_data = {}
        for stock_id, content in stock_data.items():
            if len(content['data']) > 0:
                final_stock_data[stock_id] = {
                    'data': np.stack(content['data']),
                    'dates': content['dates']
                }

        if day_stats:
            zero_days = [str(d["date"])[:10] for d in day_stats if d["pool_hits"] == 0]
            logger.info(
                "Dataset summary: days=%d, zero_hit_days=%d, kept_stocks=%d, date_range=%s..%s",
                len(day_stats),
                len(zero_days),
                len(final_stock_data),
                str(day_stats[0]["date"])[:10],
                str(day_stats[-1]["date"])[:10],
            )
            if zero_days:
                logger.warning("Zero pool-hit dates (first 20): %s", zero_days[:20])

        return final_stock_data
        
    def _build_indices(self):
        indices = []
        for stock_id, data_dict in self.stock_data.items():
            num_days = len(data_dict['dates'])
            # We need seq_len + pred_len days for one sample
            # seq_len for input, pred_len for target
            
            # Example: Day 0 to 179 (Input), Day 180 to 219 (Target)
            # Total days needed: 180 + 40 = 220
            
            # Stride = 1 (Moving window)
            total_window = self.seq_len + self.pred_len
            if num_days < total_window:
                continue
                
            for i in range(num_days - total_window + 1):
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
