# Project: 1D-CNN + StockViT Stock Prediction

A-share (ChiNext 50) stock prediction framework using 1D-CNN feature extraction + Vision Transformer.

## Architecture

- **1D-CNN** (`feature_extractor.py`): ResNet-style, `[18, 1442]` tick data -> embedding
- **StockViT** (`transformer.py`): CLS-token ViT, 4 prediction heads (max_value, min_value, max_day, min_day)
- **embed_dim = 1024** hardcoded in both `train.py` and `train_rolling.py` (default in model class is 10000, never used)

## Data Format

- Files: `daily_summary_YYYY-MM-DD.csv` with columns `StockID, Time, 18 features`
- Each stock has exactly **1442 ticks** per day
- Tensor shape per stock per day (after transpose): `[18, 1442]` i.e. `[C, L]`
- Stock filtering: ChiNext 50 only (`src/data/chinext50.py`)

## 18 Features - Critical Constraints

- **Features 0-5 (absolute values)**: `trade_count, total_trade_amt, avg_trade_price, total_buy_order_id_count, total_sell_order_id_count, active_buy_amt`
  - These get `log1p()` transform before normalization
- **Features 6-17 (ratios)**: various order amount ratios and active amount ratios
  - These are NOT log-transformed, only Z-score normalized
- **Feature index 2** (`avg_trade_price`) is the price basis for ALL labels and backtest pricing

## Price Reference

- Price = `feature[2]` (avg_trade_price), last tick (index `-1`)
- Last tick corresponds to A-share closing auction (14:57-15:00) = official closing price
- Used consistently in: label construction (`dataset.py`), backtest pricing (`backtest.py`), inference

## Labels (constructed in `dataset.py __getitem__`)

```
current_price = input_seq[-1, 2, -1] + 1e-6
max_value = daily_max[argmax] / current_price   (ratio)
min_value = daily_min[argmin] / current_price   (ratio)
max_day = argmax(daily_max)                     (0-based index)
min_day = argmin(daily_min)                     (0-based index)
```

## Normalization

1. First 6 features: `log1p(x)` first
2. Then Z-score: `(x - mean) / (std + 1e-6)`
3. `mean/std` are saved in checkpoint and must be reused at inference/test/backtest

## Loss Functions (`loss.py`)

- Only 2 classes: `MultiTaskLoss` (homoscedastic uncertainty weighting) and `PeakDayLoss` (Gaussian-smoothed KL divergence)
- Value heads: `SmoothL1Loss`
- Day heads: `PeakDayLoss(sigma=2.0)`
- Combined via: `MultiTaskLoss` with 4 learnable `log_vars`
- Validation loss uses same `MultiTaskLoss` weighting (aligned with train loss)

## Training

- Primary script: `train_rolling.py` (rolling window with warm-start across folds)
- `train.py` is for debugging only (no validation, no early stopping)
- Mixed precision: `torch.cuda.amp.autocast()` + `GradScaler()` (old-style API for PyTorch 1.10.1 compatibility)
- Checkpoint contains: `feature_extractor`, `vit`, `optimizer`, `mean`, `std`, `config`
- `model_final.pth` = copy of `model_best.pth` (best val_loss epoch, NOT last epoch)

## Backtest Pipeline

- `backtest.py` -> `visualization.py` (auto-triggered with `--plot`)
- Direction filter: only buy stocks where `pred_min_day < pred_max_day` (bullish pattern)
- Score = `pred_max_value - pred_min_value` (amplitude), top_k stocks, equal weight
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

- Data dir: `D:/temp/0_tempdata8` (default, configurable)
- Best checkpoint: typically `runs_rolling_v1/fold_7/model_final.pth` (last fold = most data)

## Don'ts

- Don't change `embed_dim` without updating both training scripts
- Don't add unused loss classes to `loss.py` (keep it minimal)
- Don't use `IncrementalStockDataset` (deleted, was unused)
- Don't assume `model_final.pth` is last epoch - it's the best epoch copy
