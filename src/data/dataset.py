import os
import glob
import logging
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


# pyarrow 可选加速（pandas read_csv 引擎，2-3x 更快）
try:
    import pyarrow  # noqa: F401
    _CSV_ENGINE = "pyarrow"
except ImportError:
    _CSV_ENGINE = None


def _read_csvs_to_per_stock_arrays(filtered_files, valid_stock_ids, desc="Reading CSVs"):
    """共享的 CSV 读取 + 按股票聚合（供 StockDataset 和 GlobalDataCache 复用）。

    - 8 线程并行读 CSV（用 pyarrow 引擎如果可用）
    - NaN/Inf 插值填充（只在 read 阶段做一次）
    - 裁剪集合竞价零行 → [18, 1424]

    Returns:
        {stock_id: {'data': ndarray[D, 18, 1424] float32, 'dates': [Timestamp]}}
    """
    logger = logging.getLogger(__name__)
    if not filtered_files:
        return {}

    def _to_sid(value):
        if isinstance(value, (int, np.integer)):
            return f"{int(value):06d}"
        return str(value).strip()

    def _read_one_csv(f):
        if _CSV_ENGINE:
            df = pd.read_csv(f, engine=_CSV_ENGINE)
        else:
            df = pd.read_csv(f)
        if "StockID" not in df.columns or "Time" not in df.columns:
            raise ValueError(f"Invalid CSV schema. file={f}")
        feature_cols = [c for c in df.columns if c not in ("StockID", "Time")]
        if len(feature_cols) != 18:
            raise ValueError(f"Expected 18 feature cols. file={f}")
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

    all_dfs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_read_one_csv, f): f for f in filtered_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            f = futures[future]
            try:
                all_dfs.append(future.result())
            except Exception as e:
                logger.exception(f"Error reading {f}: {e}")

    all_dfs.sort(key=lambda df: df['date'].iloc[0])
    if not all_dfs:
        return {}

    stock_lists = {}
    for day_df in all_dfs:
        current_date = day_df['date'].iloc[0]
        feature_cols = [c for c in day_df.columns if c not in ("StockID", "Time", "date")]

        for stock_id_raw, group in day_df.groupby('StockID'):
            stock_id = _to_sid(stock_id_raw)
            if stock_id not in valid_stock_ids:
                continue

            feats = group[feature_cols].values
            if feats.shape[1] != 18:
                continue

            if feats.shape[0] == RAW_TICKS:
                feats = feats.T
                feats = trim_auction_zeros(feats)
            elif feats.shape[0] == CLEAN_TICKS:
                feats = feats.T
            else:
                continue

            if stock_id not in stock_lists:
                stock_lists[stock_id] = {'data': [], 'dates': []}
            stock_lists[stock_id]['data'].append(feats.astype(np.float32))
            stock_lists[stock_id]['dates'].append(current_date)

    result = {}
    for stock_id, content in stock_lists.items():
        if content['data']:
            result[stock_id] = {
                'data': np.stack(content['data']),
                'dates': content['dates'],
            }
    return result


class GlobalDataCache:
    """一次性预加载所有需要的股票数据，供跨fold复用，避免重复读CSV。"""

    def __init__(self, data_dir, stock_ids, start_date=None, end_date=None):
        self.stock_ids = set(str(s) for s in stock_ids)
        self._raw = {}
        self._norm = {}
        self._load(data_dir, start_date, end_date)

    def _load(self, data_dir, start_date, end_date):
        logger = logging.getLogger(__name__)
        sd = pd.to_datetime(start_date) if start_date else None
        ed = pd.to_datetime(end_date) if end_date else None

        file_patterns = os.path.join(data_dir, "daily_summary_*.csv")
        files = sorted(glob.glob(file_patterns))
        filtered = []
        for f in files:
            basename = os.path.basename(f)
            date_str = basename.replace("daily_summary_", "").replace(".csv", "")
            try:
                file_date = pd.to_datetime(date_str)
            except Exception:
                continue
            if sd and file_date < sd:
                continue
            if ed and file_date > ed:
                continue
            filtered.append(f)

        if not filtered:
            logger.warning("GlobalDataCache: no CSV files found.")
            return

        logger.info(f"GlobalDataCache: loading {len(filtered)} files for {len(self.stock_ids)} stocks "
                    f"(engine={_CSV_ENGINE or 'default'})...")
        stock_arrays = _read_csvs_to_per_stock_arrays(
            filtered, self.stock_ids, desc="GlobalCache loading"
        )

        for stock_id, content in stock_arrays.items():
            raw = content['data']  # [D, 18, 1424] float32
            self._raw[stock_id] = {'data': raw, 'dates': content['dates']}
            normed = raw.copy()
            normed[:, :6, :] = np.log1p(normed[:, :6, :])
            self._norm[stock_id] = normed

        logger.info(f"GlobalDataCache: {len(self._raw)} stocks, "
                     f"{sum(v['data'].shape[0] for v in self._raw.values())} stock-days, "
                     f"{self.memory_usage_gb():.1f} GB")

    def slice(self, start_date, end_date, stock_ids):
        """返回指定日期范围和股票子集的数据视图（numpy view，不拷贝）。"""
        sd = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        ed = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
        stock_set = set(str(s) for s in stock_ids)

        raw_slice = {}
        norm_slice = {}
        for sid in stock_set:
            if sid not in self._raw:
                continue
            dates = self._raw[sid]['dates']
            s_idx = 0
            while s_idx < len(dates) and dates[s_idx] < sd:
                s_idx += 1
            e_idx = len(dates)
            while e_idx > s_idx and dates[e_idx - 1] > ed:
                e_idx -= 1
            if s_idx >= e_idx:
                continue
            raw_slice[sid] = {
                'data': self._raw[sid]['data'][s_idx:e_idx],
                'dates': dates[s_idx:e_idx],
            }
            norm_slice[sid] = self._norm[sid][s_idx:e_idx]

        return raw_slice, norm_slice

    def memory_usage_gb(self):
        total = sum(v['data'].nbytes for v in self._raw.values())
        total += sum(v.nbytes for v in self._norm.values())
        return total / (1024 ** 3)


class StockDataset(Dataset):
    def __init__(self, data_dir, seq_len=180, pred_len=60, start_date=None, end_date=None,
                 mean=None, std=None, stock_ids=None, sample_stride=10,
                 global_cache=None):
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
            global_cache (GlobalDataCache): Pre-loaded data cache. None = read CSVs from disk.
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

        need_stats = (mean is None or std is None)
        if not need_stats:
            self.mean = mean
            self.std = std

        if global_cache is not None:
            # 快速路径：从预加载缓存切片（numpy view，不重复读CSV）
            self.stock_data, self.normalized_data = global_cache.slice(
                start_date, end_date, self.stock_ids
            )
        else:
            # 慢速路径：从磁盘读CSV
            self.filtered_files = self._get_filtered_files()
            self.stock_data = self._load_data()
            self._prenormalize()

        # mean/std 已被 RevIN 替代，仅为 checkpoint 兼容保留占位
        if need_stats:
            self.mean = torch.zeros(18)
            self.std = torch.ones(18)

        # Build index map (stock, start_index)
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

    def _load_data(self):
        """读CSV并按股票聚合，返回 {stock_id: {'data': ndarray[D,18,1424], 'dates': list}}。"""
        logger = logging.getLogger(__name__)
        if not self.filtered_files:
            return {}
        logger.info(f"Loading {len(self.filtered_files)} files (engine={_CSV_ENGINE or 'default'})...")
        stock_data = _read_csvs_to_per_stock_arrays(
            self.filtered_files, self.stock_ids, desc="Reading CSVs"
        )
        logger.info(f"Loaded {len(stock_data)} stocks, "
                    f"{sum(len(v['dates']) for v in stock_data.values())} stock-days.")
        return stock_data
        
    def _prenormalize(self):
        """一次性对所有数据做 log1p（仅前6个特征）。z-score 已移除，由模型内 RevIN 替代。"""
        logger = logging.getLogger(__name__)

        self.normalized_data = {}
        for stock_id, content in self.stock_data.items():
            raw = content['data']  # [num_days, 18, 1424]
            normed = raw.copy()
            normed[:, :6, :] = np.log1p(normed[:, :6, :])
            self.normalized_data[stock_id] = normed.astype(np.float32)

        logger.info(f"Pre-normalized {len(self.normalized_data)} stocks in memory.")

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
        raw_data = data_dict['data']  # [num_days, 18, 1424] 原始数据

        # 标签计算（用原始价格）
        target_start = start_idx + self.seq_len
        target_end = target_start + self.pred_len
        target_seq = raw_data[target_start : target_end]

        current_price = raw_data[start_idx + self.seq_len - 1, 2, -1] + 1e-6

        daily_max = target_seq[:, 2, :].max(axis=1)
        daily_min = target_seq[:, 2, :].min(axis=1)

        max_day = int(np.argmax(daily_max))
        min_day = int(np.argmin(daily_min))

        max_value = daily_max[max_day] / current_price - 1.0
        min_value = daily_min[min_day] / current_price - 1.0

        # 模型输入（预归一化数据，直接切片，无需重复计算）
        # z-score 已移除，由模型内 RevIN 替代
        if self.normalized_data is not None:
            input_data = self.normalized_data[stock_id][start_idx : start_idx + self.seq_len]
        else:
            input_seq = raw_data[start_idx : start_idx + self.seq_len]
            input_data = input_seq.copy()
            input_data[:, :6, :] = np.log1p(input_data[:, :6, :])

        return torch.from_numpy(input_data), {
            'max_value': torch.tensor(max_value, dtype=torch.float),
            'min_value': torch.tensor(min_value, dtype=torch.float),
            'max_day': torch.tensor(max_day, dtype=torch.long),
            'min_day': torch.tensor(min_day, dtype=torch.long),
            'current_price': torch.tensor(current_price, dtype=torch.float)
        }
