# 1D-CNN + StockViT 股票预测框架

基于 A 股 tick 级数据，使用 1D-CNN 特征提取 + Vision Transformer 时序建模，预测未来 60 个交易日内的最高/最低价及其出现时间。训练采用全市场随机采样（30/31 开头创业板股票），滚动窗口 + warm-start 策略。

---

## 1. 项目结构

```text
project3_1dcnn_vit_trae/
├── runs_*/                          # 训练输出目录（每次实验/每个fold）
├── src/
│   ├── data/
│   │   ├── dataset.py               # 数据读取、标签构建、标准化（单次加载+统计）
│   │   ├── stock_pool.py            # 股票池发现、风险过滤、随机采样
│   │   └── chinext50.py             # 创业板50成分股列表（回退用）
│   ├── models/
│   │   ├── feature_extractor.py     # 1D-CNN ResNet特征提取器（空间池化）
│   │   ├── transformer.py           # StockViT时序模型 + 多任务头
│   │   └── loss.py                  # 损失函数：MultiTaskLoss + PeakDayLoss
│   ├── utils/
│   │   └── __init__.py
│   ├── config.py                    # DATA_DIR 配置（读 .env）
│   ├── train_rolling.py             # 滚动窗口训练主脚本（正式训练）
│   ├── train.py                     # 单区间训练脚本（仅调试用）
│   ├── test_weekly.py               # 测试集每周调仓评估
│   ├── inference.py                 # 推理脚本（Predictor类）
│   ├── backtest.py                  # 基于真实数据的滚动回测
│   └── visualization.py             # 回测可视化（4子图）
├── CLAUDE.md                        # 开发约束文档
├── MODEL_README.md                  # 本文档
└── requirements.txt
```

---

## 2. 数据格式与 18 个特征

### 2.1 原始 CSV 格式

- 文件命名：`daily_summary_YYYY-MM-DD.csv`
- 列结构：`StockID, Time, 18个特征列`
- 每只股票每天原始 **1442 行** tick（9:30~15:00，每10秒）
- 尾盘集合竞价零行自动裁剪：index 1423~1440（14:57:00~14:59:50）为零行，移除后保留 **1424 tick**
- 裁剪后单日单股张量（转置后）：`[18, 1424]`，即 `[C, L]`

### 2.2 18 个特征（列索引与语义）

| 索引 | 字段名 | 含义 | 预处理 |
|---|---|---|---|
| 0 | `trade_count` | 成交笔数 | log1p + Z-score |
| 1 | `total_trade_amt` | 成交总金额 | log1p + Z-score |
| 2 | `avg_trade_price` | 成交均价（**标签价格基准**） | log1p + Z-score |
| 3 | `total_buy_order_id_count` | 总买入ID数 | log1p + Z-score |
| 4 | `total_sell_order_id_count` | 总卖出ID数 | log1p + Z-score |
| 5 | `active_buy_amt` | 主动买成交额 | log1p + Z-score |
| 6 | `buy_mid_order_amt_ratio` | 中单买成交金额占比 | Z-score |
| 7 | `buy_big_order_amt_ratio` | 大单买成交金额占比 | Z-score |
| 8 | `buy_xl_order_amt_ratio` | 特大单买成交金额占比 | Z-score |
| 9 | `sell_mid_order_amt_ratio` | 中单卖成交金额占比 | Z-score |
| 10 | `sell_big_order_amt_ratio` | 大单卖成交金额占比 | Z-score |
| 11 | `sell_xl_order_amt_ratio` | 特大单卖成交金额占比 | Z-score |
| 12 | `buy_mid_active_amt_ratio` | 中单主动买成交额占比 | Z-score |
| 13 | `buy_big_active_amt_ratio` | 大单主动买成交额占比 | Z-score |
| 14 | `buy_xl_active_amt_ratio` | 特大单主动买成交额占比 | Z-score |
| 15 | `sell_mid_active_amt_ratio` | 中单主动卖成交额占比 | Z-score |
| 16 | `sell_big_active_amt_ratio` | 大单主动卖成交额占比 | Z-score |
| 17 | `sell_xl_active_amt_ratio` | 特大单主动卖成交额占比 | Z-score |

**特征分组**：
- **索引 0-5（绝对值量）**：成交笔数、金额等，量级差异大 → 先 `log1p()` 再 Z-score
- **索引 6-17（比率）**：占比指标，已是 0~1 范围 → 仅 Z-score

---

## 3. 数据预处理与标签构建

### 3.1 股票池发现与过滤（`stock_pool.py`）

`StockPool` 类扫描数据目录，建立每只股票的交易日映射，并进行风险过滤：

**扫描阶段**（8线程并行，仅读 StockID 列）：
- 每只股票每天需恰好 1442 tick 才视为有效
- 结果缓存到 `.stock_pool_cache_{hash}.pkl`，再次运行秒级加载

**过滤规则**（`get_available_stocks`）：

| 过滤项 | 参数 | 默认值 | 说明 |
|---|---|---|---|
| 代码前缀 | `stock_prefix` | `("30", "31")` | 只看创业板 |
| 上市不满N天 | `min_list_days` | 180 | 首次出现距训练起始日不足180天则排除 |
| 退市/长期停牌 | `exclude_delisted` | True | 最后交易日早于数据末尾30个交易日 |
| 停牌过多 | `min_trading_days` | 区间的90% | 区间内交易天数不足则排除 |
| ST/*ST黑名单 | `blacklist` | None | 手动传入（创业板ST无法从价格自动识别） |

**采样**：`sample_stocks(n=500, seed=base_seed+fold_idx)` 从过滤后的池中随机采样。

### 3.2 数据加载与统计量计算（`dataset.py`）

`StockDataset.__init__` 流程：

```
_get_filtered_files()           # 按日期范围筛选 CSV 文件
    ↓
_load_data_and_stats()          # 单次遍历完成两件事：
    ├── 8线程并行读 CSV          #   - ThreadPoolExecutor(max_workers=8)
    ├── 按 stock_ids 过滤        #   - 只加载采样到的股票
    ├── NaN/Inf 线性插值          #   - 逐列 np.interp，兜底填0
    ├── tick 裁剪 1442→1424      #   - trim_auction_zeros()
    ├── 累加统计量（if needed）    #   - log1p后的 sum/sq_sum/count
    └── 存入 stock_data dict     #   - {stock_id: {data: [Days,18,1424], dates: [...]}}
    ↓
_build_indices()                # 滑动窗口建索引，stride=sample_stride
```

**关键设计**：数据加载和统计量计算合并为一次遍历（`_load_data_and_stats`），避免重复读取 CSV。训练集传入 `mean=None, std=None` 时自动计算；验证集复用训练集的 `mean/std`。

### 3.3 训练样本与标签（`__getitem__`）

```python
input_seq  = data[start : start+seq_len]                     # [seq_len, 18, 1424]
target_seq = data[start+seq_len : start+seq_len+pred_len]    # [pred_len, 18, 1424]
```

**标签计算（使用特征2：avg_trade_price）**：

```python
current_price = input_seq[-1, 2, -1] + 1e-6         # 输入最后一天最后tick（15:00收盘价）
daily_max = target_seq[:, 2, :].max(axis=1)          # 每天最高均价
daily_min = target_seq[:, 2, :].min(axis=1)          # 每天最低均价
max_day   = argmax(daily_max)                        # 最高点出现在第几天（0-based）
min_day   = argmin(daily_min)                        # 最低点出现在第几天
max_value = daily_max[max_day] / current_price - 1.0 # 收益率（中心在0附近）
min_value = daily_min[min_day] / current_price - 1.0 # 收益率
```

> **重要**：标签是**收益率**（return），不是价格比率（ratio）。`max_value=0.05` 表示最高价比当前价高5%。推理时还原价格：`pred_price = current_price * (1.0 + pred_return)`。

**输入标准化**（仅对 input_data，标签使用原始价格计算）：

1. 前6列做 `log1p`：`input_data[:, :6, :] = log1p(input_data[:, :6, :])`
2. Z-score：`(x - mean[1,18,1]) / (std[1,18,1] + 1e-6)`

**样本滑动步长**：`sample_stride=10`（默认），每只股票在训练窗口内每隔10天取一个样本，减少样本间相关性。

**返回值**：
```python
torch.FloatTensor(input_data),   # [seq_len, 18, 1424]
{
  'max_value': float,            # 最高收益率（相对current_price）
  'min_value': float,            # 最低收益率
  'max_day':   long,             # 最高点天索引（0-based，< pred_len）
  'min_day':   long,             # 最低点天索引
  'current_price': float         # 基准价（含1e-6偏移）
}
```

---

## 4. 模型架构

### 4.1 1D-CNN 特征提取器（`feature_extractor.py`）

类定义：`FeatureExtractor(input_channels=18, output_dim=1024)`

```
输入: [B × Seq, 18, 1424]

Conv1d(18→64, k=7, stride=2, padding=3) + BN + ReLU + MaxPool(k=3, stride=2)
└── layer1: 2× BasicBlock1D(64→64)           →  [B, 64, 356]
└── layer2: 2× BasicBlock1D(64→128, stride=2) →  [B, 128, 178]
└── layer3: 2× BasicBlock1D(128→256, stride=2) → [B, 256, 89]
└── layer4: 2× BasicBlock1D(256→512, stride=2) → [B, 512, 45]
AdaptiveAvgPool1d(pool_size=2)                  → [B, 512, 2]
Flatten                                         → [B, 1024]
Tanh()                                          → 输出范围 (-1, 1)

输出: [B × Seq, 1024]
```

**空间池化设计**：`pool_size = ceil(output_dim / 512)`。当 `output_dim=1024` 时，`512 × 2 = 1024`，无需fc层——每个输出维度天然对应 **不同通道 × 不同时间段** 的响应，避免全局池化后用 fc 硬扩维导致的特征冗余。

当 `output_dim=10000` 时：`pool_size=20`，`512×20=10240`，经一个小 `Linear(10240, 10000)` 投射。

`BasicBlock1D`：标准 ResNet 残差块，`Conv1d(k=3) + BN + ReLU + Conv1d(k=3) + BN + skip`。

### 4.2 StockViT 时序建模（`transformer.py`）

类定义：`StockViT(seq_len=180, pred_len=60, embed_dim=1024, depth=4, num_heads=4, ...)`

```
输入：[B, 180, 1024]  （180天的CNN特征序列）
│
├── 拼接 CLS token → [B, 181, 1024]
├── 加可学习位置编码 pos_embed[1, 181, 1024]
├── Dropout(p=drop_ratio)
├── depth=4 × Block:
│     ├── LayerNorm → Attention(多头自注意力，手动实现 sdpa，兼容 PyTorch < 2.0)
│     ├── DropPath
│     ├── LayerNorm → MLP(Linear→GELU→Dropout→Linear→Dropout)
│     └── DropPath
├── LayerNorm
└── 取 x[:, 0]（CLS token）→ [B, 1024]
```

**4个预测头**（均接 CLS token 表征）：

| 头 | 层 | 输出 | 含义 |
|---|---|---|---|
| `head_max_value` | `Linear(1024→1)` | `[B, 1]` | 最高收益率 |
| `head_min_value` | `Linear(1024→1)` | `[B, 1]` | 最低收益率 |
| `head_max_day` | `Linear(1024→60)` | `[B, 60]` | 最高点天分布（logits） |
| `head_min_day` | `Linear(1024→60)` | `[B, 60]` | 最低点天分布（logits） |

---

## 5. 损失函数（`loss.py`）

仅 2 个类，设计精简：

### 5.1 PeakDayLoss（日期预测）

将真实天索引构造为高斯平滑概率分布，用 KL 散度衡量：

```python
soft = exp(-0.5 * ((idx - target_idx) / sigma)^2)   # sigma=2.0
soft = soft / soft.sum(dim=1)                        # 归一化为概率分布
log_probs = log_softmax(logits, dim=1)
loss = KLDivLoss(batchmean)(log_probs, soft)
```

高斯平滑使得预测偏差±2天也有部分概率密度，避免硬标签带来的梯度稀疏。

### 5.2 MultiTaskLoss（多任务自动加权）

基于同方差不确定性（Homoscedastic Uncertainty）的自适应权重：

```python
# 4个可学习参数 log_vars（初始化为0）
total_loss = Σ [ exp(-log_var_i) * loss_i + 0.5 * log_var_i ]
```

当某任务 loss 大时，`log_var` 自动增大（降低该任务权重），防止一个任务主导训练。

### 5.3 训练损失组合

```python
loss_max_value = SmoothL1Loss(beta=0.1)(pred_max_value, target_max_value)
loss_min_value = SmoothL1Loss(beta=0.1)(pred_min_value, target_min_value)
loss_max_day   = PeakDayLoss(sigma=2.0)(pred_max_day_logits, target_max_day)
loss_min_day   = PeakDayLoss(sigma=2.0)(pred_min_day_logits, target_min_day)
total_loss = MultiTaskLoss([loss_max_value, loss_min_value, loss_max_day, loss_min_day])
```

- `SmoothL1Loss(beta=0.1)`：误差 <10% 时用 L2（精确优化），>10% 时用 L1（抗极端行情 outlier）
- TensorBoard 每10步记录 4 个子 loss 和 4 个 `log_var`，实时监控任务平衡

---

## 6. 训练流程

### 6.1 滚动窗口训练（`train_rolling.py`，正式训练）

**默认参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `seq_len` | 180 | 输入序列天数（180天历史预测60天未来） |
| `pred_len` | 60 | 预测窗口天数 |
| `epochs` | 20 | 每fold最大epoch数（有early stopping） |
| `batch_size` | 16 | |
| `lr` | 1e-4 | AdamW 学习率 |
| `train_days` | 480 | 每fold训练集天数 |
| `test_days` | 60 | 每fold验证集天数 |
| `step_days` | 10 | fold滑动步长 |
| `depth` | 4 | Transformer层数 |
| `num_heads` | 4 | 注意力头数 |
| `weight_decay` | 5e-3 | L2 正则化（较强，抑制过拟合） |
| `drop_ratio` | 0.2 | Transformer MLP/Embedding Dropout |
| `attn_drop_ratio` | 0.2 | 注意力权重 Dropout |
| `smooth_l1_beta` | 0.1 | SmoothL1 L1/L2 转换点 |
| `day_sigma` | 2.0 | PeakDayLoss 高斯宽度 |
| `topk` | 3 | TopK天精度容差 |
| `patience` | 5 | Early Stopping 耐心值 |
| `min_delta` | 1e-3 | Early Stopping 最小改善量 |
| `seed` | 42 | 全局随机种子 |
| `scheduler` | cosine | LR调度器：cosine / plateau / none |
| `max_grad_norm` | 1.0 | 梯度裁剪（0=禁用） |
| `stock_pool` | random | 股票采样策略：random / chinext50 |
| `num_stocks` | 500 | 每fold采样股票数 |
| `sample_stride` | 10 | 样本级滑动步长（天） |
| `min_list_days` | 180 | 排除上市不满N天的股票 |
| `num_workers` | 4 | DataLoader工作进程（Windows自动降为0） |
| `resume_from` | None | 第一fold初始权重路径 |
| `start_date` | None | 数据起始日期过滤 |
| `end_date` | None | 数据截止日期过滤 |

**fold 划分逻辑**：

```
fold_1: train[day_1 .. day_480],   val[day_481 .. day_540]
fold_2: train[day_11 .. day_490],  val[day_491 .. day_550]
...（每次滑动 step_days=10 天）
```

约束：`train_days >= seq_len + pred_len`（480 >= 180+60=240），每只股票每fold最多产生 `(480-240)/10 + 1 = 25` 个样本。

**每fold训练流程**：

1. **股票采样**：从 StockPool 过滤后随机采样 500 只（seed = base_seed + fold_idx）
2. **数据加载**：训练集自动计算 `mean/std`；验证集复用训练集的
3. **模型构建**：加载上一 fold 的 `model_final.pth`（warm-start，`strict=False`）
4. **训练循环**：混合精度（`autocast + GradScaler`）、梯度裁剪、cosine LR
5. **验证**：每 epoch 结束计算 val_loss + TopK 天精度
6. **Early Stopping**：连续 5 个 epoch val_loss 不降则停止
7. **保存**：`model_best.pth`（最优epoch）→ 复制为 `model_final.pth`

**验证集数据范围**（比 test_range 多取 seq_len 天上下文）：
```python
eval_range = (max(1, test_range[0] - seq_len), test_range[1])
```

**TensorBoard 监控**（每10步写入）：
- `Train/Loss_step`、`Train/Loss_epoch`
- `Train/loss_max_value`、`Train/loss_min_value`、`Train/loss_max_day`、`Train/loss_min_day`
- `Train/log_var_max_val`、`Train/log_var_min_val`、`Train/log_var_max_day`、`Train/log_var_min_day`
- `Val/Loss_epoch`、`Val/Top3_MaxDay_epoch`、`Val/Top3_MinDay_epoch`
- `Train/LR`

**TopK 天精度**：
```python
# 预测天与真实天之差绝对值 ≤ topk=3，视为命中
topk_max_acc = 命中数 / 总样本数
```

**checkpoint 内容**：
```python
{
  'feature_extractor': state_dict,
  'vit': state_dict,
  'optimizer': state_dict,
  'mean': FloatTensor[18],           # 必须在推理时复用
  'std': FloatTensor[18],
  'config': {
      'seq_len', 'pred_len', 'embed_dim',
      'depth', 'num_heads', 'input_channels'
  },
  'best_val_loss': float,
  'best_topk_max': float,
  'best_topk_min': float,
  'best_epoch': int,
}
```

**Windows 兼容性**：
- `num_workers` 在 Windows 上自动降为 0（Dataset 过大 pickle 失败）
- `persistent_workers` 和 `prefetch_factor` 仅在 `num_workers > 0` 时传入（PyTorch 1.10.1 兼容）
- 混合精度使用旧式 API：`torch.cuda.amp.autocast()` + `torch.cuda.amp.GradScaler()`

### 6.2 单区间训练（`train.py`，仅调试用）

简化版训练脚本：无验证集、无 Early Stopping、无股票采样、无 warm-start。

```bash
python src/train.py --data_dir "D:/temp/0_tempdata8" --epochs 10 --batch_size 2
```

> **不建议用于正式训练。**

---

## 7. 测试集评估（`test_weekly.py`）

每周调仓方式评估模型在测试集上的表现。

```bash
python src/test_weekly.py \
  --checkpoint_path runs_rolling_v5/fold_N/model_final.pth \
  --data_dir "E:/pre_process/version1" \
  --test_start_date 2024-07-01 \
  --test_end_date   2024-09-30 \
  --rebalance_interval 5 \
  --output_csv weekly_test_report.csv
```

**评估逻辑**：
1. 从 checkpoint 读取 `config/mean/std`，重建模型
2. 用 `build_dataset_period()` 将测试区间向前扩展 `seq_len` 天
3. 每条样本计算 loss 和 `score = pred_max_return - pred_min_return`
4. 按调仓日分组，每 `rebalance_interval` 天计算：
   - `weekly_loss`：平均样本损失
   - `rank_ic`：score 与真实振幅的 Spearman 相关

**输出 CSV 字段**：`rebalance_date, num_stocks, weekly_loss, rank_ic, best_loss_so_far, early_stop_counter`

---

## 8. 回测框架（`backtest.py`）

```bash
python src/backtest.py \
    --checkpoint_path runs_rolling_v5/fold_N/model_final.pth \
    --data_dir "E:/pre_process/version1" \
    --start_date 2024-07-01 \
    --end_date   2024-09-30 \
    --top_k 5 \
    --rebalance_interval 5 \
    --commission 0.0000875 \
    --stamp_tax 0.0 \
    --plot --plot_output backtest_analysis.png
```

**选股逻辑**：
1. 每个调仓日，对每只有足够历史的股票做推理
2. **方向过滤**：仅保留 `pred_min_day < pred_max_day`（先跌后涨，看多形态）
3. **排名信号**：`score = pred_max_return - pred_min_return`（振幅预测）
4. 选前 `top_k` 只，等权买入
5. 下一调仓日全部清仓重新排名

**交易成本**：
- 佣金：`commission = 0.0000875`（万0.875），买卖双向
- 印花税：`stamp_tax`（仅卖出），默认0

**价格参考**：`avg_trade_price`（特征索引2）最后一个 tick（15:00:00 集合竞价结果），即A股官方收盘价。

**输出**：
- 控制台：总收益率、基准收益率、超额收益、年化收益、Sharpe、最大回撤、交易成本
- CSV：每日净值明细（portfolio_value, benchmark_value, cash, num_holdings, is_rebalance等）
- `--plot`：自动调用 `visualization.py` 生成4子图（净值曲线、回撤、每期收益、持仓数）

---

## 9. 推理（`inference.py`）

```python
from src.inference import Predictor

predictor = Predictor(
    checkpoint_path="runs_rolling_v5/fold_N/model_final.pth",
    data_dir="E:/pre_process/version1"
)
df = predictor.predict()                        # 全量推理
df_single = predictor.predict(stock_id="300750") # 单股推理
```

从 checkpoint 的 `config` 字段自动读取所有模型参数，无需手动指定。

**输出 DataFrame 字段**：

| 字段 | 说明 |
|---|---|
| `current_price` | 输入序列末尾基准价 |
| `pred_max_return` | 预测最高收益率 |
| `pred_min_return` | 预测最低收益率 |
| `pred_max_value` | 预测最高价 = current_price × (1 + pred_max_return) |
| `pred_min_value` | 预测最低价 = current_price × (1 + pred_min_return) |
| `pred_max_day` | 预测最高点天索引（0-based） |
| `pred_min_day` | 预测最低点天索引 |
| `target_max_return` | 真实最高收益率 |
| `target_min_return` | 真实最低收益率 |
| `target_max_value` | 真实最高价 |
| `target_min_value` | 真实最低价 |
| `target_max_day` | 真实最高点天索引 |
| `target_min_day` | 真实最低点天索引 |

---

## 10. 运行示例

### 10.1 正式训练（服务器）

```bash
python -m src.train_rolling \
  --data_dir E:/pre_process/version1 \
  --train_days 480 --test_days 60 --step_days 10 \
  --num_stocks 500 --sample_stride 10 \
  --epochs 20 --batch_size 32 \
  --output_dir runs_rolling_v5
```

### 10.2 回测

```bash
python src/backtest.py \
  --checkpoint_path runs_rolling_v5/fold_N/model_final.pth \
  --data_dir E:/pre_process/version1 \
  --start_date 2024-07-01 --end_date 2024-09-30 \
  --top_k 5 --rebalance_interval 5 \
  --plot --plot_output backtest_analysis.png
```

### 10.3 checkpoint 选择

| 选择 | 建议 |
|---|---|
| 最后一个 fold 的 `model_final.pth` | **推荐**，见过最多数据 |
| 特定 fold 的 `model_final.pth` | 需要特定时间窗口的模型时 |

> `model_final.pth` 是 `model_best.pth`（最佳 val_loss epoch）的副本，**不是最后一个 epoch**。

---

## 11. 关键约束

1. **价格基准**：标签、推理、回测统一使用特征2（`avg_trade_price`）最后 tick（15:00:00 收盘集合竞价结果）。

2. **标签是收益率**：`max_value = price / current_price - 1.0`，中心在0附近。推理还原：`price = current_price × (1 + return)`。不要混淆为价格比率。

3. **mean/std 必须一致**：训练时自动计算并保存到 checkpoint；推理/回测时必须从 checkpoint 读取复用。不同 fold 的 mean/std 不同（采样股票不同）。

4. **Windows 限制**：DataLoader `num_workers` 自动降为 0（避免 pickle 大 Dataset 失败）；`prefetch_factor` 仅在 `num_workers > 0` 时传入（PyTorch 1.10.1 兼容）。

5. **embed_dim 协同**：`feature_extractor.output_dim` 和 `StockViT.embed_dim` 必须一致，均在 `train_rolling.py` 中 hardcode 为 1024。

6. **尾盘零行**：原始 1442 tick 中 index 1423~1440 是集合竞价零行，加载时自动裁剪为 1424 tick。index 1441（15:00:00）保留。
