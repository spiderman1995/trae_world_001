# 1D-CNN + StockViT 股票预测框架（与当前代码一致）

本文档基于当前仓库代码，描述真实可运行的架构、数据流、训练与推理方式。

## 1. 项目结构（当前）

```text
project3_1dcnn_vit_trae/
├── runs_*/                       # 训练输出目录（每次实验/每个fold）
├── src/
│   ├── data/
│   │   ├── dataset.py            # 数据读取、标签构建、标准化
│   │   └── chinext50.py          # 创业板50成分股列表
│   ├── models/
│   │   ├── feature_extractor.py  # 1D-CNN ResNet特征提取器
│   │   ├── transformer.py        # StockViT时序模型 + 多任务头
│   │   └── loss.py               # PeakDayLoss / MultiTaskLoss
│   ├── train.py                  # 单区间训练脚本
│   ├── train_rolling.py          # 滚动窗口训练主脚本
│   ├── inference.py              # 推理脚本
│   └── visualization.py          # 现有可视化脚本
├── MODEL_README.md
└── requirements.txt
```

## 2. 数据格式与18个特征

### 2.1 原始CSV格式

- 文件命名：`daily_summary_YYYY-MM-DD.csv`
- 列结构：`StockID + Time + 18个特征`
- 每只股票每天固定 `1442` 行 tick
- 单日单股票张量：`[18, 1442]`

### 2.2 18个特征（索引与语义）

- 0 `trade_count`：成交笔数  
- 1 `total_trade_amt`：成交总金额  
- 2 `avg_trade_price`：成交均价  
- 3 `total_buy_order_id_count`：总买入ID数  
- 4 `total_sell_order_id_count`：总卖出ID数  
- 5 `active_buy_amt`：主动买成交额  
- 6 `buy_mid_order_amt_ratio`  
- 7 `buy_big_order_amt_ratio`  
- 8 `buy_xl_order_amt_ratio`  
- 9 `sell_mid_order_amt_ratio`  
- 10 `sell_big_order_amt_ratio`  
- 11 `sell_xl_order_amt_ratio`  
- 12 `buy_mid_active_amt_ratio`  
- 13 `buy_big_active_amt_ratio`  
- 14 `buy_xl_active_amt_ratio`  
- 15 `sell_mid_active_amt_ratio`  
- 16 `sell_big_active_amt_ratio`  
- 17 `sell_xl_active_amt_ratio`

## 3. 数据预处理与标签构建（真实实现）

### 3.1 文件过滤与质量检查

`StockDataset` 在加载时会：

- 按日期范围筛选文件（`start_date/end_date`）
- 预检查文件可读性（损坏文件跳过）
- 校验列结构、数值合法性、每股每日 `1442` 行
- 仅保留创业板50股票

### 3.2 流式全局统计（内存友好）

当未传入 `mean/std` 时，`dataset.py` 会对筛选后的文件做流式统计：

- 逐文件读取 -> 更新 `total_sum/total_sq_sum/total_count` -> 丢弃
- 前6列先做 `log1p` 再参与统计
- 计算 `mean/std`：  
  `mean = sum / count`  
  `var = E[x^2] - E[x]^2`  
  `std = sqrt(max(var, 1e-10))`

这避免了一次性拼接全部样本导致的内存峰值。

### 3.3 训练样本与标签

`__getitem__` 输出 `(input_data, targets)`：

- 输入窗口：`input_seq = [seq_len, 18, 1442]`
- 目标窗口：`target_seq = [pred_len, 18, 1442]`
- 标签计算使用 `avg_trade_price`（索引2）：
  - `current_price = input_seq[-1, 2, -1]`
  - `daily_max = target_seq[:, 2, :].max(axis=1)`
  - `daily_min = target_seq[:, 2, :].min(axis=1)`
  - `max_day = argmax(daily_max)`
  - `min_day = argmin(daily_min)`
  - `max_value = daily_max[max_day] / current_price`
  - `min_value = daily_min[min_day] / current_price`

### 3.4 输入标准化

返回给模型前执行：

1. 前6列做 `log1p`
2. 全18列做 z-score：`(x - mean) / (std + 1e-6)`

## 4. 模型架构（真实实现）

### 4.1 1D-CNN 特征提取器

文件：`src/models/feature_extractor.py`

- `Conv1d(18->64, k=7, stride=2)` + `BN` + `ReLU` + `MaxPool`
- 4层 ResNet1D (`BasicBlock1D`)
- `AdaptiveAvgPool1d(1)`
- `Linear(512 -> output_dim)`，训练中 `output_dim=1024`
- `Tanh`

输入输出：

- 输入（展平后）：`[B * Seq, 18, 1442]`
- 输出：`[B * Seq, 1024]`

### 4.2 StockViT 时序建模

文件：`src/models/transformer.py`

- 输入：`[B, Seq, 1024]`
- 加 `CLS token` + 可学习位置编码
- 多层 Transformer Block（含自注意力与MLP）
- 输出使用 `CLS` 表征
- 4个预测头：
  - `max_value: Linear(1024->1)`
  - `min_value: Linear(1024->1)`
  - `max_day: Linear(1024->pred_len)`
  - `min_day: Linear(1024->pred_len)`

## 5. 损失函数（真实实现）

文件：`src/models/loss.py`

- 幅度任务：`nn.SmoothL1Loss`（`max_value/min_value`）
- 日期任务：`PeakDayLoss`
  - 将真实日期构造成高斯平滑分布
  - 用 `KLDivLoss` 比较预测分布与平滑目标
- 总损失：`MultiTaskLoss`
  - 用可学习 `log_vars` 对4个任务自动加权

## 6. 训练流程

### 6.1 单区间训练

脚本：`src/train.py`

- 默认：`seq_len=180`, `pred_len=60`
- 保存 checkpoint 时包含：
  - `feature_extractor/vit/optimizer`
  - `mean/std`
  - `config(seq_len,pred_len,embed_dim,depth,num_heads,input_channels)`

### 6.2 滚动窗口训练

脚本：`src/train_rolling.py`

- 默认：`seq_len=60`, `pred_len=60`
- 折叠由参数动态生成：
  - `--train_days`
  - `--test_days`
  - `--step_days`
- 支持跨fold warm-start：
  - 每个fold可加载上个fold `model_final.pth`
- 使用混合精度训练（autocast + GradScaler）
- 每个fold保存：
  - `fold_N/model_final.pth`
  - `fold_N/logs/events.out.tfevents.*`
  - `fold_N/train_fold_N_*.txt`

验证指标：

- `Val/Loss`
- `TopK Max-Day Acc`
- `TopK Min-Day Acc`

## 7. 推理流程（真实实现）

脚本：`src/inference.py`

- 从 checkpoint 读取模型权重与 config
- 构建 `StockDataset` + `DataLoader`
- 输出字段包括：
  - `pred_max_value_ratio/pred_min_value_ratio`
  - `pred_max_value/pred_min_value`
  - `pred_max_day/pred_min_day`
  - 以及对应目标值（若标签可用）

注意：当前推理脚本默认写在 `if __name__ == "__main__"` 中手工指定路径。

## 8. 运行示例

### 8.1 滚动训练

```bash
python src/train_rolling.py \
  --data_dir "D:/temp/0_tempdata8" \
  --output_dir "runs_rolling_v1" \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4 \
  --seq_len 60 \
  --pred_len 60 \
  --train_days 120 \
  --test_days 60 \
  --step_days 10 \
  --weight_decay 1e-4 \
  --drop_ratio 0.0 \
  --attn_drop_ratio 0.0
```

### 8.2 单区间训练

```bash
python src/train.py \
  --data_dir "D:/temp/0_tempdata8" \
  --output_dir "runs_single" \
  --epochs 10 \
  --batch_size 2
```

### 8.3 推理

编辑 `src/inference.py` 末尾 checkpoint/data_dir 后运行：

```bash
python src/inference.py
```

## 9. 当前实现的关键约束

- 目标价格基准已切换为 `avg_trade_price`（特征2），不是特征0
- 训练/验证标准化参数是按当前训练集统计并在该fold验证复用
- `src` 目录当前没有 `backtest.py`，文档不再描述不存在的回测主脚本
