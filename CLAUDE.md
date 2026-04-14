# Project: 1D-CNN + StockViT Stock Prediction

A-share stock prediction framework using 1D-CNN feature extraction + Vision Transformer.
Trains on randomly sampled stocks from the full market (30/31 prefix), not limited to ChiNext 50.

## Architecture

- **1D-CNN** (`feature_extractor.py`): ResNet-style, `[18, 1424]` tick data -> embedding
- **StockViT** (`transformer.py`): CLS-token ViT (encoder-only), 4 prediction heads (max_value, min_value, max_day, min_day)
- **embed_dim = 1024** hardcoded in both `train.py` and `train_rolling.py` (default in model class is 10000, never used)

## Data Format

- Files: `daily_summary_YYYY-MM-DD.csv` with columns `StockID, Time, 18 features`
- Each stock has exactly **1424 ticks** per day (after auction zero trimming)
- Tensor shape per stock per day (after transpose): `[18, 1424]` i.e. `[C, L]`
- Stock filtering: `--stock_pool random` (default, samples N stocks per fold) or `--stock_pool chinext50` (fallback)
- Stock pool discovery: `src/data/stock_pool.py` scans CSVs for available stocks

## 18 Features - Critical Constraints

- **Features 0-5 (absolute values)**: `trade_count, total_trade_amt, avg_trade_price, total_buy_order_id_count, total_sell_order_id_count, active_buy_amt`
  - These get `log1p()` transform before normalization
- **Features 6-17 (ratios)**: various order amount ratios and active amount ratios
  - These are NOT log-transformed, only Z-score normalized
- **Feature index 2** (`avg_trade_price`) is the price basis for ALL labels and backtest pricing

## Price Reference

- Price = `feature[2]` (avg_trade_price), last tick (index -1 = 15:00:00)
- Last tick (15:00:00) = closing auction result = official closing price
- Ticks 14:57:00~14:59:50 (index 1423~1440) are zeros — auction period, no continuous trading
- Only index 1441 (15:00:00) has the actual auction result
- Data load auto-trims auction zeros: 1442 raw -> 1424 clean

## Labels (constructed in `dataset.py __getitem__`)

- `current_price`: `input_seq[-1, 2, -1]` (15:00:00 tick, always has auction result)
- `daily_max/min`: from trimmed data (auction zeros already removed), direct `max()/min()`
- **Value targets are returns (not ratios)**: `max_value = daily_max / current_price - 1.0`
- Inference must add back 1.0: `pred_price = current_price * (1.0 + pred_return)`

```
AUCTION_ZERO_START = 1423   # 14:57:00
AUCTION_ZERO_END   = 1441   # 14:59:50, index 1441(15:00:00) kept
trim_auction_zeros() removes indices 1423-1440
daily_max = target_seq[:, 2, :].max(axis=1)
daily_min = target_seq[:, 2, :].min(axis=1)
max_value = daily_max[max_day] / current_price - 1.0
```

## Normalization

1. First 6 features: `log1p(x)` first
2. Then Z-score: `(x - mean) / (std + 1e-6)`
3. `mean/std` computed per-fold from the sampled stocks' training data (single-pass with data loading)
4. `mean/std` are saved in checkpoint and must be reused at inference/test/backtest
5. Missing values (NaN/Inf): linear interpolation along tick axis, fallback to 0

## Loss Functions (`loss.py`)

- Only 2 classes: `MultiTaskLoss` (homoscedastic uncertainty weighting) and `PeakDayLoss` (Gaussian-smoothed KL divergence)
- Value heads: `SmoothL1Loss(beta=0.1)` — L1/L2 transition at 10% error
- Day heads: `PeakDayLoss(sigma=2.0)`
- Combined via: `MultiTaskLoss` with 4 learnable `log_vars`
- Per-task losses and log_vars logged to TensorBoard for monitoring

## Training

- Primary script: `train_rolling.py` (rolling window with warm-start across folds)
- Default: `seq_len=180, pred_len=60, train_days=480, test_days=60, step_days=10`
- Stock sampling: per-fold, randomly sample `--num_stocks 50` from available pool (seed = base_seed + fold_idx)
- DataLoader: `num_workers=4, persistent_workers=True, prefetch_factor=2` for GPU overlap
- Mixed precision: `torch.cuda.amp.autocast()` + `GradScaler()` (old-style API for PyTorch 1.10.1 compatibility)
- Checkpoint contains: `feature_extractor`, `vit`, `optimizer`, `mean`, `std`, `config`
- `model_final.pth` = copy of `model_best.pth` (best val_loss epoch, NOT last epoch)
- `train.py` is for debugging only (no validation, no early stopping)
- Constraint: `train_days >= seq_len + pred_len` (each sample needs 240 consecutive days)

## Backtest Pipeline

- `backtest.py` -> `visualization.py` (auto-triggered with `--plot`)
- Direction filter: only buy stocks where `pred_min_day < pred_max_day` (bullish pattern)
- Score = `pred_max_return - pred_min_return` (amplitude), top_k stocks, equal weight
- Transaction costs: commission (default 0.0000875 = wan 0.875, both sides) + stamp tax (sell only, default 0)
- Daily NAV recording (all trading days, not just rebalance days)
- Equal-weight benchmark for comparison

## Rebalance Logic

- Both `backtest.py` and `test_weekly.py` use calendar-based: sort all dates, pick every N-th trading day
- `rebalance_interval=5` means roughly weekly

## PyTorch Compatibility

- Branch `torch1.10.1` targets older PyTorch (1.10.1) for machine 203
- Use `torch.cuda.amp.autocast()` not `torch.amp.autocast('cuda')`
- Use `torch.cuda.amp.GradScaler()` not `torch.amp.GradScaler('cuda')`

## Key File Paths

- Data dir: configured via `.env` -> `DATA_DIR` (local: `D:/temp/0_tempdata8`, server: `E:/pre_process/version1`)
- Best checkpoint: typically last fold's `model_final.pth` (most data seen)
- Stock pool: `src/data/stock_pool.py` (discovery + sampling)
- ChiNext50 fallback: `src/data/chinext50.py`

## Don'ts

- Don't change `embed_dim` without updating both training scripts
- Don't add unused loss classes to `loss.py` (keep it minimal)
- Don't use `IncrementalStockDataset` (deleted, was unused)
- Don't assume `model_final.pth` is last epoch - it's the best epoch copy
- Don't forget value targets are returns (centered at 0), not ratios (centered at 1)
