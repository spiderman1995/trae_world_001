# 1D-CNN + StockViT 股票预测框架（与当前代码一致）

本文档基于当前仓库代码，描述真实可运行的架构、数据流、训练与推理方式。

---

## 1. 项目结构（当前）

```text
project3_1dcnn_vit_trae/
├── runs_*/                          # 训练输出目录（每次实验/每个fold）
├── src/
│   ├── data/
│   │   ├── dataset.py               # 数据读取、标签构建、标准化
│   │   └── chinext50.py             # 创业板50成分股列表
│   ├── models/
│   │   ├── feature_extractor.py     # 1D-CNN ResNet特征提取器
│   │   ├── transformer.py           # StockViT时序模型 + 多任务头
│   │   └── loss.py                  # 损失函数：MultiTaskLoss + PeakDayLoss
│   ├── utils/
│   │   └── __init__.py              # 空包初始化文件
│   ├── train.py                     # 单区间训练脚本
│   ├── train_rolling.py             # 滚动窗口训练主脚本
│   ├── test_weekly.py               # 测试集每周调仓评估脚本
│   ├── inference.py                 # 推理脚本（Predictor类）
│   ├── backtest.py                  # 基于真实数据的滚动回测脚本
│   └── visualization.py             # 可视化脚本
├── MODEL_README.md
└── requirements.txt
```

---

## 2. 数据格式与18个特征

### 2.1 原始CSV格式

- 文件命名：`daily_summary_YYYY-MM-DD.csv`
- 列结构：`StockID + Time + 18个特征列`
- 每只股票每天固定 **1442 行** tick
- 单日单股票张量（转置后）：`[18, 1442]`，即 `[C, L]`

### 2.2 18个特征（列索引与语义）

| 索引 | 字段名 | 含义 |
|---|---|---|
| 0 | `trade_count` | 成交笔数 |
| 1 | `total_trade_amt` | 成交总金额 |
| 2 | `avg_trade_price` | 成交均价（标签价格基准） |
| 3 | `total_buy_order_id_count` | 总买入ID数 |
| 4 | `total_sell_order_id_count` | 总卖出ID数 |
| 5 | `active_buy_amt` | 主动买成交额 |
| 6 | `buy_mid_order_amt_ratio` | 中单买成交金额占比 |
| 7 | `buy_big_order_amt_ratio` | 大单买成交金额占比 |
| 8 | `buy_xl_order_amt_ratio` | 特大单买成交金额占比 |
| 9 | `sell_mid_order_amt_ratio` | 中单卖成交金额占比 |
| 10 | `sell_big_order_amt_ratio` | 大单卖成交金额占比 |
| 11 | `sell_xl_order_amt_ratio` | 特大单卖成交金额占比 |
| 12 | `buy_mid_active_amt_ratio` | 中单主动买成交额占比 |
| 13 | `buy_big_active_amt_ratio` | 大单主动买成交额占比 |
| 14 | `buy_xl_active_amt_ratio` | 特大单主动买成交额占比 |
| 15 | `sell_mid_active_amt_ratio` | 中单主动卖成交额占比 |
| 16 | `sell_big_active_amt_ratio` | 大单主动卖成交额占比 |
| 17 | `sell_xl_active_amt_ratio` | 特大单主动卖成交额占比 |

---

## 3. 数据预处理与标签构建（真实实现）

### 3.1 文件过滤（`_get_filtered_files`）

- 扫描 `daily_summary_*.csv`，按 `start_date / end_date` 参数筛选日期范围内的文件
- 不做合法性校验，仅筛选文件名日期

### 3.2 流式全局统计（`_compute_streaming_stats`）

当未传入 `mean/std` 时执行：

1. **预检查**：逐文件 `read_csv(nrows=5)`，读取失败则跳过（warning，继续下一文件）
2. **流式累计**：对通过预检查的文件逐一读取：
   - 过滤为创业板50股票（`get_chinext50_constituents()`）
   - 按股票分组，**仅保留 tick 数恰好为 1442 且无 NaN/Inf 的股票**（与 `_load_data` 校验口径一致）
   - 前6列做 `log1p`
   - 累加 `total_sum / total_sq_sum / total_count`
   - 读取后丢弃，不占内存
3. **计算统计量**：
   - `mean = sum / count`
   - `var = E[x²] - E[x]²`
   - `std = sqrt(max(var, 1e-10))`
4. 返回 `torch.FloatTensor(mean)`, `torch.FloatTensor(std)`

> **注意**：此阶段遇到处理异常的文件会跳过（skip），但不中断整体流程。

### 3.3 全量数据加载（`_load_data`）

分两阶段，校验行为不同：

**Phase 1 — 逐文件读取（有 try/except 保护，异常时跳过该文件继续）**：
- 缺少 `StockID` / `Time` 列 → raise ValueError → 被 except 捕获，**跳过该文件**
- 特征列数不等于18 → 同上，跳过
- 存在 NaN 值 → 同上，跳过
- 存在 Inf 值 → 同上，跳过

**Phase 2 — 逐日处理（无 try/except，异常直接中断整个加载）**：
- StockID 不是6位纯数字格式 → raise ValueError，**中断**
- 某只股票/某天的 tick 行数不等于1442 → raise ValueError，**中断**
- 某只股票/某天特征含非有限值 → raise ValueError，**中断**

通过校验后：
- 过滤，仅保留创业板50股票
- 每日每股转置为 `[18, 1442]`
- 按 `stock_id → {data: [Days, 18, 1442], dates: [Date]}` 组织内存

### 3.4 训练样本与标签（`__getitem__`）

```
input_seq  = data[start : start+seq_len]    # [seq_len, 18, 1442]
target_seq = data[start+seq_len : start+seq_len+pred_len]  # [pred_len, 18, 1442]
```

**标签计算（使用特征2：avg_trade_price）**：

```python
current_price = input_seq[-1, 2, -1] + 1e-6         # 输入最后一天最后一个tick的均价
daily_max = target_seq[:, 2, :].max(axis=1)          # 每天最高均价
daily_min = target_seq[:, 2, :].min(axis=1)          # 每天最低均价
max_day   = argmax(daily_max)                        # 最高点出现在第几天
min_day   = argmin(daily_min)                        # 最低点出现在第几天
max_value = daily_max[max_day] / current_price       # 最高价格比率
min_value = daily_min[min_day] / current_price       # 最低价格比率
```

**输入标准化（仅对 input_data，不影响标签计算）**：

1. 前6列做 `log1p`：`input_data[:, :6, :] = log1p(input_data[:, :6, :])`
2. Z-score 标准化：`(x - mean.view(1,18,1)) / (std.view(1,18,1) + 1e-6)`

**返回值**：
```python
torch.FloatTensor(input_data),   # [seq_len, 18, 1442]
{
  'max_value': float,            # 最高价格比率（相对current_price）
  'min_value': float,            # 最低价格比率
  'max_day':   long,             # 最高点天索引（0-based，< pred_len）
  'min_day':   long,             # 最低点天索引
  'current_price': float         # 基准价（含1e-6偏移）
}
```

---

## 4. 模型架构（真实实现）

### 4.1 1D-CNN 特征提取器（`feature_extractor.py`）

类定义：`FeatureExtractor(input_channels=18, output_dim=10000)`  
训练时实际使用：`output_dim=1024`（两个训练脚本均 hardcode `embed_dim=1024`）

```
Conv1d(18→64, k=7, stride=2, padding=3) + BN + ReLU + MaxPool(k=3, stride=2)
└── layer1: 2× BasicBlock1D(64→64)
└── layer2: 2× BasicBlock1D(64→128, stride=2)
└── layer3: 2× BasicBlock1D(128→256, stride=2)
└── layer4: 2× BasicBlock1D(256→512, stride=2)
AdaptiveAvgPool1d(1)
Linear(512 → output_dim)
Tanh()  →  输出范围 (-1, 1)
```

`BasicBlock1D`：标准ResNet残差块，`Conv1d(k=3) + BN + ReLU + Conv1d(k=3) + BN + skip`

**输入/输出**：
- 输入（展平后）：`[B × Seq, 18, 1442]`
- 输出：`[B × Seq, 1024]`

### 4.2 StockViT 时序建模（`transformer.py`）

类定义：`StockViT(seq_len=180, pred_len=60, embed_dim=10000, depth=4, num_heads=4, ...)`  
训练时实际使用：`embed_dim=1024`

```
输入：[B, Seq, 1024]
│
├── 拼接 CLS token → [B, Seq+1, 1024]
├── 加可学习位置编码 pos_embed[1, seq_len+1, 1024]
├── Dropout
├── depth × Block:
│     ├── LayerNorm
│     ├── Attention(多头自注意力，手动实现，兼容 PyTorch < 2.0)
│     ├── DropPath
│     ├── LayerNorm
│     └── MLP(Linear→GELU→Dropout→Linear→Dropout)
├── LayerNorm
└── 取 x[:, 0]（CLS token）→ [B, 1024]
```

**4个预测头（均接 CLS token 表征）**：

| 头 | 层 | 输出形状 | 含义 |
|---|---|---|---|
| `head_max_value` | `Linear(1024→1)` | `[B, 1]` | 最高价格比率 |
| `head_min_value` | `Linear(1024→1)` | `[B, 1]` | 最低价格比率 |
| `head_max_day` | `Linear(1024→pred_len)` | `[B, pred_len]` | 最高点天分布（logits） |
| `head_min_day` | `Linear(1024→pred_len)` | `[B, pred_len]` | 最低点天分布（logits） |

---

## 5. 损失函数（`loss.py`）

### 5.1 训练中实际使用的损失

**`PeakDayLoss`**（日期任务）：
```python
# 将真实天索引构造为高斯平滑分布（sigma=2.0）
soft = exp(-0.5 * ((idx - target_idx) / sigma)^2)
soft = soft / soft.sum(dim=1)          # 归一化为概率分布
log_probs = log_softmax(logits, dim=1) # 预测 logits → log概率
loss = KLDivLoss(batchmean)(log_probs, soft)
```

**`MultiTaskLoss`**（多任务加权）：
```python
# 4个可学习 log_vars 参数（初始化为0）
total_loss = Σ [ exp(-log_var_i) * loss_i  +  0.5 * log_var_i ]
# 基于同方差不确定性（Homoscedastic Uncertainty）自动加权
```

**训练损失组合**（`train_rolling.py`）：
```python
loss_max_value = SmoothL1Loss(pred_max_value, target_max_value)
loss_min_value = SmoothL1Loss(pred_min_value, target_min_value)
loss_max_day   = PeakDayLoss(pred_max_day_logits, target_max_day)
loss_min_day   = PeakDayLoss(pred_min_day_logits, target_min_day)
total_loss = MultiTaskLoss([loss_max_value, loss_min_value, loss_max_day, loss_min_day])
```


---

## 6. 训练流程

### 6.1 单区间训练（`train.py`）

**默认参数**：
| 参数 | 默认值 |
|---|---|
| `seq_len` | 180 |
| `pred_len` | 60 |
| `epochs` | 10 |
| `batch_size` | 2 |
| `lr` | 1e-4 |
| `depth` | 4 |
| `num_heads` | 4 |
| `day_sigma` | 2.0 |

**流程**：
1. 构建 `StockDataset`（按 `start_date/end_date` 过滤）
2. 构建模型（embed_dim=1024 hardcoded）
3. 优化器：`AdamW(lr=1e-4, weight_decay=1e-4)`，混合精度（`torch.amp.GradScaler`）
4. 每个 epoch 后保存 checkpoint：包含 `feature_extractor / vit / optimizer / mean / std / config`
5. **无验证集、无 Early Stopping**

### 6.2 滚动窗口训练（`train_rolling.py`）

**默认参数**：
| 参数 | 默认值 | 说明 |
|---|---|---|
| `seq_len` | 60 | 输入序列天数 |
| `pred_len` | 60 | 预测窗口天数 |
| `epochs` | 20 | 每个fold最大epoch数 |
| `batch_size` | 2 | |
| `lr` | 1e-4 | |
| `train_days` | 180 | 每fold训练集天数 |
| `test_days` | 60 | 每fold验证集天数 |
| `step_days` | 10 | fold滑动步长 |
| `depth` | 4 | Transformer层数 |
| `num_heads` | 4 | 注意力头数 |
| `topk` | 3 | TopK天精度容差 |
| `patience` | 5 | Early Stopping耐心值 |
| `min_delta` | 1e-3 | Early Stopping最小改善量 |
| `day_sigma` | 2.0 | PeakDayLoss 高斯平滑宽度 |
| `seed` | 42 | 随机种子（全局固定） |
| `weight_decay` | 1e-4 | |
| `drop_ratio` | 0.0 | |
| `attn_drop_ratio` | 0.0 | |
| `resume_from` | None | 第一fold的初始权重路径 |
| `start_date` | None | 数据起始日期过滤（YYYY-MM-DD） |
| `end_date` | None | 数据截止日期过滤（YYYY-MM-DD） |

**fold划分逻辑**：
```
fold_0: train[day_1 .. day_180],  val[day_181 .. day_240]
fold_1: train[day_11 .. day_190], val[day_191 .. day_250]
...（每次滑动 step_days=10 天）
```

**验证集数据范围**（实际加载比 test_range 多）：
```python
eval_range = (max(1, test_range[0] - seq_len), test_range[1])
# 向前多取 seq_len 天，确保验证集有足够的上下文构建样本
```

**训练细节**：
- 混合精度：`torch.cuda.amp.autocast()` + `GradScaler()`
- warm-start：每个fold可加载上个fold的 `model_final.pth`（`strict=False`）
- TensorBoard 写入：`Train/Loss_step`（每10步）、`Train/Loss_epoch`、`Val/Loss_epoch`、`Val/Top{K}_MaxDay_epoch`、`Val/Top{K}_MinDay_epoch`

**验证损失计算**（与训练损失口径一致）：
```python
# validate() 中同样通过 MultiTaskLoss 加权，与训练 loss 口径统一
losses_list = torch.stack([loss_max_value, loss_min_value, loss_max_day, loss_min_day])
val_loss = mean(mtl_loss_wrapper(losses_list))
```

**TopK-Day 精度**：
```python
# 预测天与真实天之差绝对值 ≤ topk，视为命中
topk_max_acc = |{pred_max_day 命中}| / total_samples
topk_min_acc = |{pred_min_day 命中}| / total_samples
```

**Early Stopping**：
```python
if val_loss < best_val_loss - min_delta:
    patience_counter = 0
    保存 model_best.pth
else:
    patience_counter += 1
    if patience_counter >= patience:
        停止当前fold训练
```

**每个fold保存文件**：
```
fold_N/
├── model_best.pth    # val_loss 最低时的快照（含 best_epoch 信息）
├── model_final.pth   # = model_best.pth 的副本（若无best则为最后epoch）
├── logs/events.out.tfevents.*
└── train_fold_N_*.txt
```

**checkpoint 内容**（`model_best.pth` / `model_final.pth`）：
```python
{
  'feature_extractor': state_dict,
  'vit': state_dict,
  'optimizer': state_dict,
  'mean': FloatTensor[18],
  'std': FloatTensor[18],
  'config': {
      'seq_len', 'pred_len', 'embed_dim',
      'depth', 'num_heads', 'input_channels'
  },
  'best_val_loss': float,
  'best_topk_max': float,
  'best_topk_min': float,
  'best_epoch': int,   # 仅 model_best.pth 有此字段
}
```

---

## 7. 测试集评估（`test_weekly.py`）

每周调仓方式的测试脚本，与 `train_rolling.py` 的验证逻辑完全独立。

**运行方式**：
```bash
python src/test_weekly.py \
  --checkpoint_path runs_rolling/fold_5/model_final.pth \
  --data_dir "D:/temp/0_tempdata8" \
  --test_start_date 2024-07-01 \
  --test_end_date   2024-09-30 \
  --rebalance_interval 5 \
  --patience 5 \
  --min_delta 1e-3 \
  --output_csv weekly_test_report.csv
```

**主要参数**：
| 参数 | 默认值 | 说明 |
|---|---|---|
| `rebalance_interval` | 5 | 每隔 N 个日历交易日调仓一次（与 backtest.py 逻辑一致） |
| `patience` | 5 | 连续几周无改善时停止 |
| `min_delta` | 1e-3 | 最小改善阈值 |

**评估逻辑**：
1. 从 checkpoint 读取 `config/mean/std`，重建模型
2. 用 `build_dataset_period()` 将测试区间向前扩展 `seq_len` 天（保证首个测试样本有完整输入）
3. 遍历数据集，每条样本计算：
   - `sample_loss = SmoothL1(pred_max_value, tgt) + SmoothL1(pred_min_value, tgt) + normalized_day_error`
   - `score = pred_max_value - pred_min_value`（调仓排名信号）
   - `realized = tgt_max_value - tgt_min_value`（实际振幅）
4. 按调仓日（`rebalance_date = dates[start_idx + seq_len]`）分组，然后按日历日期排序，每 `rebalance_interval` 个日期选取一次（与 backtest.py 调仓逻辑一致）
5. 每个调仓周计算：
   - `weekly_loss`：该周所有样本的平均 loss
   - `rank_ic`：`score` 与 `realized` 的 Spearman 相关（股票数 ≥ 2 时计算）
6. **Early Stopping**：`bad_rounds >= patience` 时停止输出

**输出 CSV 字段**：

| 字段 | 说明 |
|---|---|
| `rebalance_date` | 调仓日期 |
| `num_stocks` | 当周参与评估的股票数 |
| `weekly_loss` | 当周平均样本损失 |
| `rank_ic` | Spearman RankIC |
| `best_loss_so_far` | 历史最优 loss |
| `early_stop_counter` | 连续未改善周数 |

---

## 8. 回测框架（`backtest.py`）

**运行方式**：
```bash
python src/backtest.py \
    --checkpoint_path runs_rolling_v1/fold_7/model_final.pth \
    --data_dir "D:/temp/0_tempdata8" \
    --start_date 2024-07-01 \
    --end_date   2024-09-30 \
    --top_k 5 \
    --rebalance_interval 5 \
    --commission 0.0000875 \
    --stamp_tax 0.0 \
    --output_csv backtest_history.csv \
    --plot \
    --plot_output backtest_analysis.png
```

**主要参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `top_k` | 5 | 每期持有股票数 |
| `rebalance_interval` | 5 | 每隔 N 个交易日调仓一次（5 ≈ 周频） |
| `initial_capital` | 1,000,000 | 初始资金（元） |
| `commission` | 0.0000875 | 佣金率（双向），默认万0.875 |
| `stamp_tax` | 0.0 | 印花税率（仅卖出），默认0 |
| `batch_size` | 8 | 推理批大小 |
| `--plot` | off | 开启后自动调用 `visualization.py` 生成分析图 |
| `--plot_output` | `backtest_analysis.png` | 输出图片路径（仅 `--plot` 时生效） |

**数据格式要求**：与训练数据相同，`daily_summary_YYYY-MM-DD.csv`，自动向前扩展 `seq_len` 天历史以保证首个调仓日有完整输入窗口。

**价格参考**：使用 `avg_trade_price`（特征索引2）最后一个 tick（索引-1），对应A股收盘集合竞价（14:57-15:00）的成交价，即官方收盘价 / 清算结算参考价。

**推理 & 选股逻辑**：
1. 从 checkpoint 读取 `mean/std/config`，恢复模型
2. 每个调仓日，对每只有足够历史（≥ `seq_len` 天）的创业板50股票做推理
3. **方向过滤**：仅保留 `pred_min_day < pred_max_day`（先跌后涨，看多形态）的股票
4. 排名信号：`score = pred_max_value - pred_min_value`（振幅预测）
5. 选前 `top_k` 只股票，等权买入
6. 下一调仓日全部清仓（扣佣金+印花税），重新排名买入（扣佣金）

**交易成本模型**：
- **卖出成本**：`gross × (commission + stamp_tax)`
- **买入成本**：有效买价 = `price × (1 + commission)`，资金等分后按有效价计算可买份额
- 成本在回测报告中单独列示

**每日净值记录**：
- 回测遍历测试区间内的**所有交易日**（不仅是调仓日），每天按最新价格计算 `holdings_value + cash`
- 非调仓日仅记录净值，不做交易

**等权基准线**：
- 每日对所有有价格数据的股票计算等权平均收益率
- 基准线独立于组合，用于衡量超额收益

**生成的绩效指标**（控制台输出）：
- 总收益率、基准收益率、超额收益（年化）
- 年化收益率、年化波动率、Sharpe 比率
- 最大回撤
- 佣金总额、印花税总额、总交易成本

**输出 CSV 字段**（`backtest_history.csv`）：

| 字段 | 说明 |
|---|---|
| `portfolio_value` | 当日组合总净值（现金+持仓市值） |
| `benchmark_value` | 当日等权基准净值 |
| `cash` | 当日现金余额 |
| `num_holdings` | 当日持仓股票数 |
| `is_rebalance` | 是否为调仓日 |
| `num_bullish` | 调仓日通过方向过滤的股票数（非调仓日为 NaN） |
| `top5_stocks` | 调仓日选出的 top5 股票代码（逗号分隔） |

### 8.1 可视化（`visualization.py`）

**自动触发**：`backtest.py` 加 `--plot` 参数时自动调用。也可独立运行：
```bash
python src/visualization.py  # 默认读取 backtest_history.csv
```

**4 个子图**：
1. **净值曲线**：组合 vs 等权基准，调仓日用三角标记
2. **回撤曲线**：组合相对历史最高的回撤幅度
3. **每期收益率柱状图**：每个调仓周期的组合收益 vs 基准收益（红绿色标注正负）
4. **持仓 & 候选数**：实际持仓数 + 看涨候选股票数的阶梯图

**接口**：
```python
from src.visualization import plot_backtest_results
# 方式1：直接传 DataFrame（backtest.py 内部使用）
plot_backtest_results(df=df, output_path="backtest_analysis.png")
# 方式2：从 CSV 读取（独立使用）
plot_backtest_results(csv_path="backtest_history.csv")
```

---

## 9. 推理（`inference.py`）

提供 `Predictor` 类，封装 checkpoint 加载与批量预测。

**初始化**：从 checkpoint 的 `config` 字段自动读取 `seq_len / pred_len / embed_dim / depth / num_heads / input_channels`，若 checkpoint 无 config 则使用默认值（`seq_len=180, pred_len=60, embed_dim=1024`）。

**`predict()` 输出字段**（DataFrame）：

| 字段 | 说明 |
|---|---|
| `current_price` | 输入序列末尾基准价 |
| `pred_max_value_ratio` | 预测最高价格比率 |
| `pred_min_value_ratio` | 预测最低价格比率 |
| `pred_max_value` | 预测最高价（绝对值） |
| `pred_min_value` | 预测最低价（绝对值） |
| `pred_max_day` | 预测最高点天索引 |
| `pred_min_day` | 预测最低点天索引 |
| `target_max_value_ratio` | 真实最高价格比率 |
| `target_min_value_ratio` | 真实最低价格比率 |
| `target_max_value` | 真实最高价（绝对值） |
| `target_min_value` | 真实最低价（绝对值） |
| `target_max_day` | 真实最高点天索引 |
| `target_min_day` | 真实最低点天索引 |

> **注意**：`Predictor` 初始化时会从 checkpoint 读取 `mean/std` 并缓存为 `self.mean / self.std`，`predict()` 调用时传入 `StockDataset`，与训练时归一化参数保持一致。

---

## 10. 运行示例与参数说明

### 10.1 滚动窗口训练（主要训练方式）

```bash
python src/train_rolling.py \
  --data_dir "D:/temp/0_tempdata8" \
  --output_dir "runs_rolling_v1" \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-4 \
  --seq_len 60 \
  --pred_len 60 \
  --train_days 180 \
  --test_days 60 \
  --step_days 10 \
  --depth 4 \
  --num_heads 4 \
  --day_sigma 2.0 \
  --topk 3 \
  --weight_decay 1e-4 \
  --drop_ratio 0.0 \
  --attn_drop_ratio 0.0 \
  --patience 5 \
  --min_delta 1e-3 \
  --seed 42 \
  --resume_from "runs_rolling_v0/fold_7/model_final.pth" \
  --start_date 2024-01-01 \
  --end_date 2024-12-31
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--data_dir` | 否 | `D:\temp\0_tempdata8` | 存放 `daily_summary_YYYY-MM-DD.csv` 的数据目录 |
| `--output_dir` | 否 | `runs_rolling` | 训练输出根目录，每个 fold 在下级子目录 `fold_N/` |
| `--epochs` | 否 | 20 | 每个 fold 最大训练轮数（可被 Early Stopping 提前终止） |
| `--batch_size` | 否 | 2 | 每批样本数。受显存限制，4050(6GB) 建议 2~8 |
| `--lr` | 否 | 1e-4 | AdamW 学习率 |
| `--seq_len` | 否 | 60 | 输入序列天数（模型看到多少天历史） |
| `--pred_len` | 否 | 60 | 预测窗口天数（预测未来多少天内的最高/最低点） |
| `--train_days` | 否 | 180 | 每个 fold 训练集覆盖的交易日天数 |
| `--test_days` | 否 | 60 | 每个 fold 验证集覆盖的交易日天数 |
| `--step_days` | 否 | 10 | fold 之间滑动步长（步长越小 fold 越多，训练越慢） |
| `--depth` | 否 | 4 | Transformer 编码器层数 |
| `--num_heads` | 否 | 4 | 多头注意力头数 |
| `--day_sigma` | 否 | 2.0 | PeakDayLoss 高斯软标签宽度，越大对天误差越宽容 |
| `--topk` | 否 | 3 | TopK 天精度容差，预测天与真实天差 ≤ topk 即视为命中 |
| `--weight_decay` | 否 | 1e-4 | AdamW 权重衰减（L2 正则化强度） |
| `--drop_ratio` | 否 | 0.0 | Transformer MLP/Embedding Dropout 比例 |
| `--attn_drop_ratio` | 否 | 0.0 | 注意力权重 Dropout 比例 |
| `--patience` | 否 | 5 | Early Stopping 耐心值，连续几个 epoch 无改善则停止 |
| `--min_delta` | 否 | 1e-3 | Early Stopping 最小改善量，val_loss 降幅不足此值不算改善 |
| `--seed` | 否 | 42 | 全局随机种子，固定种子保证可复现 |
| `--resume_from` | 否 | None | 第一个 fold 的初始权重路径（后续 fold 自动继承上一个 fold） |
| `--start_date` | 否 | None | 数据起始日期过滤，格式 YYYY-MM-DD（None 表示不过滤） |
| `--end_date` | 否 | None | 数据截止日期过滤，格式 YYYY-MM-DD（None 表示不过滤） |

**输出文件**：每个 fold 保存在 `{output_dir}/fold_N/` 下，包含 `model_best.pth`、`model_final.pth`、TensorBoard 日志、训练文本日志。推荐使用**最后一个 fold 的 `model_final.pth`**（见过最多数据）。

---

### 10.2 单区间训练（调试/快速实验用）

```bash
python src/train.py \
  --data_dir "D:/temp/0_tempdata8" \
  --output_dir "runs_single" \
  --epochs 10 \
  --batch_size 2 \
  --lr 1e-4 \
  --seq_len 180 \
  --pred_len 60 \
  --depth 4 \
  --num_heads 4 \
  --day_sigma 2.0 \
  --start_date 2024-01-01 \
  --end_date 2024-12-31
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--data_dir` | 否 | `D:\temp\0_tempdata8` | 数据目录 |
| `--output_dir` | 否 | `runs` | 输出目录，每个 epoch 保存 `model_epoch_N.pth` |
| `--epochs` | 否 | 10 | 训练总轮数（无 Early Stopping，跑满全部） |
| `--batch_size` | 否 | 2 | 每批样本数 |
| `--lr` | 否 | 1e-4 | 学习率 |
| `--seq_len` | 否 | 180 | 输入序列天数 |
| `--pred_len` | 否 | 60 | 预测窗口天数 |
| `--depth` | 否 | 4 | Transformer 层数 |
| `--num_heads` | 否 | 4 | 注意力头数 |
| `--day_sigma` | 否 | 2.0 | PeakDayLoss 高斯宽度 |
| `--start_date` | 否 | `2024-01-01` | 训练数据起始日期 |
| `--end_date` | 否 | `2024-12-31` | 训练数据截止日期 |

> **注意**：此脚本无验证集、无 Early Stopping，适合快速调试，不建议用于正式训练。

---

### 10.3 测试集每周调仓评估

```bash
python src/test_weekly.py \
  --checkpoint_path runs_rolling_v1/fold_7/model_final.pth \
  --data_dir "D:/temp/0_tempdata8" \
  --test_start_date 2024-07-01 \
  --test_end_date   2024-09-30 \
  --batch_size 8 \
  --rebalance_interval 5 \
  --patience 5 \
  --min_delta 1e-3 \
  --seed 42 \
  --output_csv weekly_test_report.csv
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--checkpoint_path` | **是** | — | 模型权重路径，推荐用最后一个 fold 的 `model_final.pth` |
| `--data_dir` | **是** | — | 数据目录（与训练相同格式） |
| `--test_start_date` | **是** | — | 测试起始日期，格式 YYYY-MM-DD |
| `--test_end_date` | **是** | — | 测试截止日期，格式 YYYY-MM-DD |
| `--batch_size` | 否 | 8 | 推理批大小，不影响结果，仅影响速度 |
| `--rebalance_interval` | 否 | 5 | 调仓间隔，按日历日期排序后每 N 个交易日选一次（与 backtest.py 一致） |
| `--patience` | 否 | 5 | 连续几周 loss 无改善则提前终止评估 |
| `--min_delta` | 否 | 1e-3 | 最小改善阈值 |
| `--seed` | 否 | 42 | 随机种子 |
| `--output_csv` | 否 | `weekly_test_report.csv` | 输出 CSV 路径，含每周 loss、RankIC 等指标 |

**输出指标**：每周的 `weekly_loss`、`rank_ic`（Spearman 相关）、`early_stop_counter`。RankIC 反映模型预测的股票排名与真实振幅排名的一致性。

---

### 10.4 回测（真实数据模拟交易）

```bash
python src/backtest.py \
  --checkpoint_path runs_rolling_v1/fold_7/model_final.pth \
  --data_dir "D:/temp/0_tempdata8" \
  --start_date 2024-07-01 \
  --end_date   2024-09-30 \
  --initial_capital 1000000 \
  --top_k 5 \
  --rebalance_interval 5 \
  --commission 0.0000875 \
  --stamp_tax 0.0 \
  --batch_size 8 \
  --output_csv backtest_history.csv \
  --plot \
  --plot_output backtest_analysis.png
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--checkpoint_path` | **是** | — | 模型权重路径 |
| `--data_dir` | **是** | — | 数据目录（`daily_summary_YYYY-MM-DD.csv`） |
| `--start_date` | **是** | — | 回测起始日期（自动向前扩展 seq_len 天加载历史） |
| `--end_date` | **是** | — | 回测截止日期 |
| `--initial_capital` | 否 | 1,000,000 | 初始资金（元） |
| `--top_k` | 否 | 5 | 每期持有股票数量 |
| `--rebalance_interval` | 否 | 5 | 每隔 N 个交易日调仓一次（5 ≈ 周频） |
| `--commission` | 否 | 0.0000875 | 佣金率，双向收取（万0.875 = 0.00875%） |
| `--stamp_tax` | 否 | 0.0 | 印花税率，仅卖出收取（设0则无印花税） |
| `--batch_size` | 否 | 8 | 推理批大小 |
| `--output_csv` | 否 | `backtest_history.csv` | 输出 CSV，含每日净值、基准线、持仓数等 |
| `--plot` | 否 | off | 开启后回测完自动生成 4 子图分析图 |
| `--plot_output` | 否 | `backtest_analysis.png` | 图片输出路径 |

**选股逻辑**：先过滤方向（`pred_min_day < pred_max_day`，先跌后涨才做多），再按振幅 `score = pred_max_value - pred_min_value` 降序选前 `top_k` 只，等权买入。

**输出**：控制台打印总收益率 / 基准收益率 / 超额收益 / 年化收益 / Sharpe / 最大回撤 / 交易成本明细；保存每日明细到 CSV；`--plot` 时自动生成可视化图。

---

### 10.5 批量推理（Python API）

```python
from src.inference import Predictor

predictor = Predictor(
    checkpoint_path="runs_rolling_v1/fold_7/model_final.pth",
    data_dir="D:/temp/0_tempdata8"
)
# 全量推理
df = predictor.predict()
df.to_csv("predictions.csv", index=False)

# 单只股票推理
df_single = predictor.predict(stock_id="300750")
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `checkpoint_path` | **是** | — | 模型权重路径 |
| `data_dir` | **是** | — | 数据目录 |
| `device` | 否 | 自动检测 | 推理设备，如 `torch.device("cpu")` |
| `seq_len` | 否 | checkpoint config | 覆盖 checkpoint 中存储的 seq_len |
| `pred_len` | 否 | checkpoint config | 覆盖 checkpoint 中存储的 pred_len |
| `embed_dim` | 否 | checkpoint config | 覆盖嵌入维度 |
| `depth` | 否 | checkpoint config | 覆盖 Transformer 层数 |
| `num_heads` | 否 | checkpoint config | 覆盖注意力头数 |
| `input_channels` | 否 | checkpoint config | 覆盖输入通道数（18） |

> **推荐**：不传可选参数，全部从 checkpoint 自动读取，避免手动指定导致参数不匹配。

---

### 10.6 checkpoint 选择建议

| checkpoint | 适用场景 |
|---|---|
| `runs_rolling_v1/fold_7/model_final.pth` | **推荐**。最后一个 fold，见过最多数据，seq_len=60, pred_len=60 |
| `runs_rolling_v1/fold_N/model_final.pth` | 需要特定时间窗口的模型时选对应 fold |
| `runs/model_epoch_1.pth` | 仅供调试，单区间训练只跑了1个 epoch，无 mean/std |

---

## 11. 关键约束与已知问题

1. **价格基准**：标签与 `current_price` 均使用特征2（`avg_trade_price`）的最后一个 tick（对应收盘集合竞价，即官方收盘价）。回测中也统一使用此价格。

2. **`_load_data()` 两阶段校验行为不同**：Phase 1（逐文件读取）异常被 `except Exception` 捕获、跳过文件；Phase 2（逐日处理）异常直接 raise 中断。

3. **`model_final.pth` 实为最优模型**：当 `model_best.pth` 存在时，`model_final.pth` 是其副本（最佳epoch），而非训练结束时的最后一个 epoch。

