"""
股票池发现与随机采样模块。

扫描 daily_summary CSV 文件，建立 {stock_id: [trading_dates]} 映射，
按条件过滤后随机采样 N 只股票用于训练。
"""

import os
import glob
import logging
import random
import re

import pandas as pd
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StockPool:
    """扫描数据目录，发现可用股票并支持随机采样。"""

    def __init__(self, data_dir, start_date=None, end_date=None):
        """
        轻量扫描：只读每个 CSV 的 StockID 列和行数，建立可用性映射。

        Args:
            data_dir: 包含 daily_summary_YYYY-MM-DD.csv 的目录
            start_date: 扫描起始日期 (YYYY-MM-DD)，None 表示不限
            end_date: 扫描截止日期 (YYYY-MM-DD)，None 表示不限
        """
        self.data_dir = data_dir
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

        # {stock_id: set of dates}
        self.availability = self._scan()

    def _scan(self):
        """扫描所有 CSV，只读 StockID 列，统计每只股票在哪些日期出现且 tick 数正确。"""
        file_patterns = os.path.join(self.data_dir, "daily_summary_*.csv")
        files = sorted(glob.glob(file_patterns))

        filtered = []
        for f in files:
            basename = os.path.basename(f)
            date_str = basename.replace("daily_summary_", "").replace(".csv", "")
            try:
                file_date = pd.to_datetime(date_str)
            except Exception:
                continue
            if self.start_date and file_date < self.start_date:
                continue
            if self.end_date and file_date > self.end_date:
                continue
            filtered.append((f, file_date))

        if not filtered:
            logger.warning("No CSV files found in date range.")
            return {}

        logger.info(f"StockPool: scanning {len(filtered)} files for stock availability...")

        availability = {}  # stock_id -> set of dates

        for fpath, fdate in tqdm(filtered, desc="Scanning stocks"):
            try:
                df = pd.read_csv(fpath, usecols=["StockID"])
            except Exception as e:
                logger.warning(f"Skipping {fpath}: {e}")
                continue

            # 统计每只股票的行数
            counts = df["StockID"].value_counts()
            for raw_id, cnt in counts.items():
                stock_id = self._normalize_id(raw_id)
                if stock_id is None:
                    continue
                # 只接受 1442 tick 的完整交易日
                if cnt != 1442:
                    continue
                if stock_id not in availability:
                    availability[stock_id] = set()
                availability[stock_id].add(fdate)

        logger.info(f"StockPool: found {len(availability)} unique stocks across {len(filtered)} days.")
        return availability

    @staticmethod
    def _normalize_id(raw_id):
        """将原始 StockID 转为 6 位字符串，无效返回 None。"""
        if raw_id is None:
            return None
        if isinstance(raw_id, (int, np.integer)):
            s = f"{int(raw_id):06d}"
        else:
            s = str(raw_id).strip()
        if re.fullmatch(r"\d{6}", s):
            return s
        return None

    def get_available_stocks(self, start_date, end_date, min_trading_days=None,
                             stock_prefix=None):
        """
        返回在 [start_date, end_date] 区间内满足条件的股票列表。

        Args:
            start_date: 区间起始日期
            end_date: 区间截止日期
            min_trading_days: 最少交易天数。None 则使用区间内所有交易日数的 90%。
            stock_prefix: 股票代码前缀过滤，如 ("30", "31")。None 不过滤。

        Returns:
            list of stock_id strings
        """
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)

        # 区间内的所有交易日
        all_trading_dates = set()
        for stock_dates in self.availability.values():
            all_trading_dates |= {d for d in stock_dates if sd <= d <= ed}
        num_trading_days = len(all_trading_dates)

        if min_trading_days is None:
            min_trading_days = int(num_trading_days * 0.90)

        valid = []
        for stock_id, dates in self.availability.items():
            # 前缀过滤
            if stock_prefix and not any(stock_id.startswith(p) for p in stock_prefix):
                continue
            # 区间内的交易天数
            days_in_range = sum(1 for d in dates if sd <= d <= ed)
            if days_in_range >= min_trading_days:
                valid.append(stock_id)

        logger.info(
            f"StockPool: {len(valid)} stocks have >= {min_trading_days} trading days "
            f"in [{start_date}, {end_date}] (total trading days: {num_trading_days})"
        )
        return sorted(valid)

    def sample_stocks(self, n=50, start_date=None, end_date=None,
                      min_trading_days=None, stock_prefix=None, seed=42):
        """
        从可用股票中随机采样 n 只。

        Args:
            n: 采样数量
            start_date: 区间起始
            end_date: 区间截止
            min_trading_days: 最少交易天数
            stock_prefix: 前缀过滤，如 ("30", "31")
            seed: 随机种子（可复现）

        Returns:
            list of stock_id strings (sorted)
        """
        pool = self.get_available_stocks(
            start_date=start_date,
            end_date=end_date,
            min_trading_days=min_trading_days,
            stock_prefix=stock_prefix,
        )

        if len(pool) <= n:
            logger.warning(
                f"StockPool: available stocks ({len(pool)}) <= requested ({n}), using all."
            )
            return pool

        rng = random.Random(seed)
        sampled = sorted(rng.sample(pool, n))
        logger.info(f"StockPool: sampled {len(sampled)} stocks (seed={seed})")
        return sampled
