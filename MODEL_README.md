# 1D-CNN + StockViT 股票预测框架

基于 A 股 tick 级数据，使用 1D-CNN 特征提取 + Vision Transformer 时序建模，预测未来 15 个交易日内的最高/最低价及其出现时间。训练采用全市场随机采样（30/31 开头创业板股票），滚动窗口 + warm-start 策略。

---

## 1. 项目结构

```text
project3_1dcnn_vit_trae/
├── runs_*/                          # 训练输出目录（每次实验/每个fold）
├── src/
│   ├── data/
│   │   ├── dataset.py               # 数据读取、标签构建、预归一化（含 GlobalDataCache 跨fold预加载）
│   │   ├── stock_pool.py            # 股票池发现、风险过滤、随机采样
│   │   └── chinext50.py             # 创业板50成分股列表（回退用）
│   ├── models/
│   │   ├── feature_extractor.py     # 1D-CNN ResNet特征提取器（RevIN + 空间池化 + BatchNorm）
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
│   ├── visualization.py             # 回测可视化（4子图）
│   └── factor_ic_test.py            # 单因子IC测试（信号衰减分析）
├── CLAUDE.md                        # 开发约束文档
├── MODEL_README.md                  # 本文档
├── update.txt                       # 版本更新记录（每次改动的详细说明）
├── inspiration.txt                  # 灵感备忘录（待验证的改进方向）
├── train_diary.txt                  # 训练日记（实验结果记录）
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

| 索引 | 字段名 | 含义 | 数据管道预处理 |
|---|---|---|---|
| 0 | `trade_count` | 成交笔数 | log1p |
| 1 | `total_trade_amt` | 成交总金额 | log1p |
| 2 | `avg_trade_price` | 成交均价（**标签价格基准**） | log1p |
| 3 | `total_buy_order_id_count` | 总买入ID数 | log1p |
| 4 | `total_sell_order_id_count` | 总卖出ID数 | log1p |
| 5 | `active_buy_amt` | 主动买成交额 | log1p |
| 6 | `buy_mid_order_amt_ratio` | 中单买成交金额占比 | 无 |
| 7 | `buy_big_order_amt_ratio` | 大单买成交金额占比 | 无 |
| 8 | `buy_xl_order_amt_ratio` | 特大单买成交金额占比 | 无 |
| 9 | `sell_mid_order_amt_ratio` | 中单卖成交金额占比 | 无 |
| 10 | `sell_big_order_amt_ratio` | 大单卖成交金额占比 | 无 |
| 11 | `sell_xl_order_amt_ratio` | 特大单卖成交金额占比 | 无 |
| 12 | `buy_mid_active_amt_ratio` | 中单主动买成交额占比 | 无 |
| 13 | `buy_big_active_amt_ratio` | 大单主动买成交额占比 | 无 |
| 14 | `buy_xl_active_amt_ratio` | 特大单主动买成交额占比 | 无 |
| 15 | `sell_mid_active_amt_ratio` | 中单主动卖成交额占比 | 无 |
| 16 | `sell_big_active_amt_ratio` | 大单主动卖成交额占比 | 无 |
| 17 | `sell_xl_active_amt_ratio` | 特大单主动卖成交额占比 | 无 |

**特征分组**：
- **索引 0-5（绝对值量）**：成交笔数、金额等，量级差异大 → 数据管道做 `log1p()` 压缩量级
- **索引 6-17（比率）**：占比指标，已是 0~1 范围 → 数据管道不做处理
- **所有18个特征**统一由模型内 **RevIN**（可逆实例归一化）做归一化，不再使用全局 Z-score

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

> **防未来信息泄漏**：每个 fold 的采样 `end_date` 只用到 `train_period[1]`（训练期末），不用 `test_period[1]`。这样测试期的退市/停牌不会参与训练时的股票筛选 — 模拟真实部署时只能看到截止当前的历史数据。

### 3.2 数据加载（`dataset.py`）

**两种加载路径**：

#### 快速路径：跨 fold 预加载（`GlobalDataCache`，主路径）

在 `main()` 启动时，一次性读取所有 fold 需要股票的全部日期数据：

```
main() 启动
    ↓
预计算所有 fold 的股票采样 → all_needed_stocks（并集）
    ↓
内存预估：若 < 128 GB 则创建 GlobalDataCache（否则退回慢速路径）
    ↓
GlobalDataCache._load()
    ├── 8线程并行读 CSV（pyarrow 引擎如可用，2-3x 更快）
    ├── NaN/Inf 插值填充（仅 read 阶段一次）
    ├── tick 裁剪 1442→1424
    └── 同时存储 raw（float32）和 normalized（log1p 应用于前6特征）
    ↓
每个 fold 的 StockDataset(global_cache=...) 从缓存切片（numpy view，零拷贝）
```

#### 慢速路径：逐 fold 读 CSV（回退，`--preload off` 或内存超限时）

```
StockDataset.__init__
    ├── _get_filtered_files()   # 按日期范围筛选 CSV
    ├── _load_data()             # 调用共享 helper _read_csvs_to_per_stock_arrays
    └── _prenormalize()          # log1p 前6特征
```

**关键设计**：
- `_read_csvs_to_per_stock_arrays` 为模块级共享 helper，`StockDataset` 和 `GlobalDataCache` 都调用它（避免代码重复）
- `__init__` 一次性完成 log1p，`__getitem__` 只做数组切片
- `mean/std` **不再计算**，直接置为 `zeros(18)/ones(18)` 占位（仅为 checkpoint 兼容保留字段，归一化完全由模型内 RevIN 处理）

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

**输入预处理**（仅对 input_data，标签使用原始价格计算）：

1. 数据管道：前6列做 `log1p`（在 `_prenormalize()` 中一次性完成）
2. 模型内：RevIN 对每个样本独立做实例归一化（减均值除标准差），无需全局 Z-score

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

RevIN(18)                                       → 每样本实例归一化（解决牛熊分布漂移）
Conv1d(18→64, k=7, stride=2, padding=3) + BN + ReLU + MaxPool(k=3, stride=2)
└── layer1: 1× BasicBlock1D(64→64)             →  [B, 64, 356]
└── layer2: 1× BasicBlock1D(64→128, stride=2)  →  [B, 128, 178]
└── layer3: 1× BasicBlock1D(128→256, stride=2) →  [B, 256, 89]
└── layer4: 1× BasicBlock1D(256→512, stride=2) →  [B, 512, 45]
AdaptiveAvgPool1d(pool_size=2)                  → [B, 512, 2]
Flatten                                         → [B, 1024]
BatchNorm1d(1024)                               → 归一化输出分布，梯度不饱和

输出: [B × Seq, 1024]
```

**参数量**：~3.1M（每stage 1个block的精简版）。

**RevIN（可逆实例归一化）**：对每个样本独立计算 `mean/var`，做 `(x - mean) / sqrt(var + eps)`，可学习 `gamma/beta`。解决不同市场环境（牛市/熊市）的分布漂移问题，替代了数据管道中的全局 Z-score。

**空间池化设计**：`pool_size = ceil(output_dim / 512)`。当 `output_dim=1024` 时，`512 × 2 = 1024`，无需fc层——每个输出维度天然对应 **不同通道 × 不同时间段** 的响应，避免全局池化后用 fc 硬扩维导致的特征冗余。

**BatchNorm1d**：替代原来的 Tanh。Tanh 在饱和区梯度接近0，阻碍深层网络学习；BatchNorm 归一化分布的同时保持梯度流通，且有可学习的缩放和偏移参数。

`BasicBlock1D`：标准 ResNet 残差块，`Conv1d(k=3) + BN + ReLU + Conv1d(k=3) + BN + skip`。

### 4.2 StockViT 时序建模（`transformer.py`）

类定义：`StockViT(seq_len=180, pred_len=15, embed_dim=1024, depth=4, num_heads=4, ...)`

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
| `head_max_day` | `Linear(1024→pred_len)` | `[B, 15]` | 最高点天分布（logits） |
| `head_min_day` | `Linear(1024→pred_len)` | `[B, 15]` | 最低点天分布（logits） |

---

## 5. 损失函数（`loss.py`）

仅 2 个类，设计精简：

### 5.1 PeakDayLoss（日期预测）

将真实天索引构造为高斯平滑概率分布，用 KL 散度衡量：

```python
soft = exp(-0.5 * ((idx - target_idx) / sigma)^2)   # sigma=1.0（默认）
soft = soft / soft.sum(dim=1)                        # 归一化为概率分布
log_probs = log_softmax(logits, dim=1)
loss = KLDivLoss(batchmean)(log_probs, soft)
```

高斯平滑使得预测偏差±1天也有部分概率密度，避免硬标签带来的梯度稀疏。sigma=1.0 配合 pred_len=15 的短期预测窗口。

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
loss_max_day   = PeakDayLoss(sigma=1.0)(pred_max_day_logits, target_max_day)
loss_min_day   = PeakDayLoss(sigma=1.0)(pred_min_day_logits, target_min_day)
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
| `seq_len` | 180 | 输入序列天数（180天历史预测15天未来） |
| `pred_len` | 15 | 预测窗口天数 |
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
| `day_sigma` | 1.0 | PeakDayLoss 高斯宽度 |
| `topk` | 1 | TopK天精度容差（±1天） |
| `patience` | 5 | Early Stopping 耐心值（v7 起从 10 降为 5） |
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
| `preload` | auto | 跨fold数据预加载：auto/on/off（auto阈值128GB） |
| `cudnn_benchmark` | False | 开启后 Conv 自动选最快算法（5-15% 提速，牺牲严格可复现） |

**fold 划分逻辑**：

```
fold_1: train[day_1 .. day_480],   val[day_481 .. day_540]
fold_2: train[day_11 .. day_490],  val[day_491 .. day_550]
...（每次滑动 step_days=10 天）
```

约束：`train_days >= seq_len + pred_len`（480 >= 180+15=195），每只股票每fold最多产生 `(480-195)/10 + 1 = 29` 个样本。

**每fold训练流程**：

1. **股票采样**：从 StockPool 过滤后随机采样 500 只，`end_date=train_period[1]` 避免未来信息泄漏（seed = base_seed + fold_idx）
2. **数据加载**：从 `GlobalDataCache` 切片（零拷贝视图），`mean/std` 为占位 zeros/ones
3. **模型构建**：加载上一 fold 的 `model_final.pth`（warm-start，`strict=False`）
4. **训练循环**：混合精度（`autocast + GradScaler`）、梯度裁剪、cosine LR、`non_blocking=True` GPU 传输
5. **验证**：每 epoch 结束计算 val_loss + TopK 天精度 + 4个子loss + Rank IC
6. **Early Stopping**：连续 5 个 epoch val_loss 不降则停止
7. **保存**：`model_best.pth`（最优epoch）→ 复制为 `model_final.pth`

**优雅停止**：

运行中按一次 `Ctrl+C`，训练会跑完当前 epoch 并保存 best model 后退出，tqdm 进度条显示 `STATUS=STOPPING`。按两次强制退出。fold 间的 `_stop_requested` 标志保证已完成 fold 的模型不会丢失。

**验证集数据范围**（比 test_range 多取 seq_len 天上下文）：
```python
eval_range = (max(1, test_range[0] - seq_len), test_range[1])
```

**TensorBoard 监控**：

训练（每10步写入）：
- `Train/Loss_step`、`Train/Loss_epoch`
- `Train/loss_max_value`、`Train/loss_min_value`、`Train/loss_max_day`、`Train/loss_min_day`
- `Train/log_var_max_val`、`Train/log_var_min_val`、`Train/log_var_max_day`、`Train/log_var_min_day`
- `Train/LR`

验证（每epoch写入）：
- `Val/Loss_epoch`、`Val/Top1_MaxDay_epoch`、`Val/Top1_MinDay_epoch`
- `Val/loss_max_value`、`Val/loss_min_value`、`Val/loss_max_day`、`Val/loss_min_day`
- `Val/RankIC_max_value`、`Val/RankIC_min_value`

**TopK 天精度**：
```python
# 预测天与真实天之差绝对值 ≤ topk=1，视为命中（±1天容差）
topk_max_acc = 命中数 / 总样本数
```

**checkpoint 内容**：
```python
{
  'feature_extractor': state_dict,
  'vit': state_dict,
  'optimizer': state_dict,
  'mean': FloatTensor[18],           # legacy，不再用于归一化（RevIN替代）
  'std': FloatTensor[18],            # legacy，保留向后兼容
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

3. **归一化由 RevIN 处理**：模型内 RevIN 对每个样本独立归一化，不依赖全局 mean/std。checkpoint 中仍保存 mean/std 字段（现在是 `zeros(18)/ones(18)` 占位，不再计算），推理/回测时也不再使用。

4. **Windows 限制**：DataLoader `num_workers` 自动降为 0（避免 pickle 大 Dataset 失败）；`prefetch_factor` 仅在 `num_workers > 0` 时传入（PyTorch 1.10.1 兼容）。

5. **embed_dim 协同**：`feature_extractor.output_dim` 和 `StockViT.embed_dim` 必须一致，均在 `train_rolling.py` 中 hardcode 为 1024。

6. **尾盘零行**：原始 1442 tick 中 index 1423~1440 是集合竞价零行，加载时自动裁剪为 1424 tick。index 1441（15:00:00）保留。

7. **无未来信息泄漏**：
   - 股票采样 `end_date=train_period[1]`（不看测试期的退市/停牌）
   - `current_price` 用输入窗口末尾 tick（预测时刻，非未来）
   - RevIN 只用当前样本自身的均值/方差（per-sample，无跨样本信息流）
   - 标签 `max/min_value/day` 从 `target_seq` 计算是预测目标，不算泄漏
   - Val 输入跨 train/test 边界是 walk-forward 标准做法（输入 = 预测时刻之前的历史）
