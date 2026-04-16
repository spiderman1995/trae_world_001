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
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

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

        # {stock_id: set of dates} — 优先读缓存
        self.availability = self._load_or_scan()

    def _cache_path(self):
        """基于 data_dir + 日期范围 生成缓存文件路径。"""
        key = f"{self.data_dir}|{self.start_date}|{self.end_date}"
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        return os.path.join(self.data_dir, f".stock_pool_cache_{h}.pkl")

    def _load_or_scan(self):
        """有缓存则直接加载（秒级），否则扫描并保存缓存。"""
        cache = self._cache_path()
        if os.path.exists(cache):
            try:
                with open(cache, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"StockPool: loaded cache ({len(data)} stocks) from {cache}")
                return data
            except Exception as e:
                logger.warning(f"StockPool: cache corrupted, re-scanning: {e}")

        availability = self._scan()

        try:
            with open(cache, "wb") as f:
                pickle.dump(availability, f)
            logger.info(f"StockPool: saved cache to {cache}")
        except Exception as e:
            logger.warning(f"StockPool: failed to save cache: {e}")

        return availability

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

        logger.info(f"StockPool: scanning {len(filtered)} files for stock availability (8 threads)...")

        normalize = self._normalize_id

        def _scan_one(args):
            fpath, fdate = args
            try:
                df = pd.read_csv(fpath, usecols=["StockID"])
            except Exception:
                return []
            counts = df["StockID"].value_counts()
            results = []
            for raw_id, cnt in counts.items():
                if cnt != 1442:
                    continue
                stock_id = normalize(raw_id)
                if stock_id is not None:
                    results.append((stock_id, fdate))
            return results

        availability = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_scan_one, item): item for item in filtered}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning stocks"):
                for stock_id, fdate in future.result():
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
                             stock_prefix=None, min_list_days=180,
                             exclude_delisted=True, blacklist=None):
        """
        返回在 [start_date, end_date] 区间内满足条件的股票列表。

        过滤规则：
        1. stock_prefix: 代码前缀过滤（如30/31开头的创业板）
        2. min_trading_days: 区间内最少交易天数（默认90%），过滤停牌过多的股票
        3. min_list_days: 上市不满N天的排除（用首次出现日期近似IPO日期）
        4. exclude_delisted: 排除已退市股票（末次交易日远早于数据末尾）
        5. blacklist: 手动黑名单（ST/*ST等无法从数据自动识别的）

        Args:
            start_date: 区间起始日期
            end_date: 区间截止日期
            min_trading_days: 最少交易天数。None 则使用区间内所有交易日数的 90%。
            stock_prefix: 股票代码前缀过滤，如 ("30", "31")。None 不过滤。
            min_list_days: 上市最少天数。股票首次出现距 start_date 不足此天数则排除。
            exclude_delisted: 排除疑似退市/长期停牌股票（最后交易日早于数据末尾30个交易日）。
            blacklist: 要排除的股票ID集合（如ST股票），None不排除。

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

        # 数据中的最后一个交易日（用于退市判断）
        global_last_date = max(all_trading_dates) if all_trading_dates else ed
        # 退市阈值：最后交易日早于全局末尾30个交易日
        sorted_all_dates = sorted(all_trading_dates)
        delist_threshold = sorted_all_dates[-30] if len(sorted_all_dates) >= 30 else sorted_all_dates[0]

        blacklist_set = set(blacklist) if blacklist else set()

        valid = []
        excluded_reasons = {"prefix": 0, "blacklist": 0, "ipo": 0, "delisted": 0, "suspended": 0}

        for stock_id, dates in self.availability.items():
            # 1. 前缀过滤
            if stock_prefix and not any(stock_id.startswith(p) for p in stock_prefix):
                excluded_reasons["prefix"] += 1
                continue

            # 2. 黑名单（ST/*ST/手动排除）
            if stock_id in blacklist_set:
                excluded_reasons["blacklist"] += 1
                continue

            # 3. 上市不满 min_list_days 天（首次出现距 start_date 太近）
            first_date = min(dates)
            if (sd - first_date).days < min_list_days:
                excluded_reasons["ipo"] += 1
                continue

            # 4. 退市/长期停牌（最后交易日远早于数据末尾）
            if exclude_delisted:
                last_date = max(dates)
                if last_date < delist_threshold:
                    excluded_reasons["delisted"] += 1
                    continue

            # 5. 停牌过多（区间内交易天数不足）
            days_in_range = sum(1 for d in dates if sd <= d <= ed)
            if days_in_range < min_trading_days:
                excluded_reasons["suspended"] += 1
                continue

            valid.append(stock_id)

        logger.info(
            f"StockPool: {len(valid)} stocks passed all filters "
            f"in [{start_date}, {end_date}] (trading days: {num_trading_days}). "
            f"Excluded: {excluded_reasons}"
        )
        return sorted(valid)

    def sample_stocks(self, n=50, start_date=None, end_date=None,
                      min_trading_days=None, stock_prefix=None,
                      min_list_days=180, exclude_delisted=True,
                      blacklist=None, seed=42):
        """
        从可用股票中随机采样 n 只。

        Args:
            n: 采样数量
            start_date: 区间起始
            end_date: 区间截止
            min_trading_days: 最少交易天数
            stock_prefix: 前缀过滤，如 ("30", "31")
            min_list_days: 上市最少天数（排除次新股）
            exclude_delisted: 排除疑似退市/长期停牌
            blacklist: 手动黑名单（如ST股票）
            seed: 随机种子（可复现）

        Returns:
            list of stock_id strings (sorted)
        """
        pool = self.get_available_stocks(
            start_date=start_date,
            end_date=end_date,
            min_trading_days=min_trading_days,
            stock_prefix=stock_prefix,
            min_list_days=min_list_days,
            exclude_delisted=exclude_delisted,
            blacklist=blacklist,
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
