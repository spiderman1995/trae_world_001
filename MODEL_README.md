
# 1D-CNN + ViT 股票预测与回测框架

本文档详细说明了该代码库的功能、项目结构、以及如何运行模型训练、推理和策略回测。

## 1. 项目结构

```
.project3_1dcnn_vit_trae/
├── runs_*/                  # 存放所有训练运行的日志、模型和结果
├── src/
│   ├── data/                 # 数据加载和预处理模块 (dataset.py)
│   ├── models/               # 模型定义 (transformer.py, feature_extractor.py)
│   ├── utils/                # 工具函数
│   ├── train_rolling.py      # 滚动窗口训练主脚本
│   ├── inference.py          # 模型推理脚本
│   ├── backtest.py           # 策略回测系统主脚本
│   └── visualization.py      # 回测结果可视化脚本
├── backtest_history.csv      # 回测生成的历史净值数据
├── backtest_analysis.png     # 可视化脚本生成的分析图表
├── MODEL_README.md           # 本文档
└── requirements.txt          # Python 依赖
```

---

## 2. 核心功能与用法

### 2.1 模型训练 (命令已修正)

通过运行 `train_rolling.py` 脚本，可以启动滚动窗口训练流程。该脚本接受丰富的命令行参数以控制训练过程。

**运行命令示例:**
```bash
# 这是一个更完整的示例，展示了常用参数
python src/train_rolling.py \
    --data_dir "/path/to/your/data" \
    --output_dir "./runs_rolling_exp1" \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4 \
    --seq_len 60 \
    --pred_len 60
```

*   `--data_dir`: **(必需)** 存放原始 `daily_summary_*.csv` 文件的目录。
*   `--output_dir`: 训练日志和模型文件的保存目录。
*   其他参数如 `--epochs`, `--batch_size`, `--lr` 等用于控制训练超参数。

### 2.2 模型推理 (用法已修正)

`inference.py` 脚本用于加载一个训练好的模型，并对数据进行预测。**注意：该脚本不接受命令行参数。**

**使用方法:**

1.  **编辑 `src/inference.py` 文件**: 在文件底部的 `if __name__ == "__main__":` 部分，修改以下两个变量：
    ```python
    if __name__ == "__main__":
        predictor = Predictor(
            checkpoint_path="./runs_rolling/fold_1/model_final.pth",  # <--- 修改为您的模型路径
            data_dir=r"/path/to/your/data"                     # <--- 修改为您的数据目录
        )
        df = predictor.predict()
        df.to_csv("predictions.csv", index=False)
    ```
2.  **运行脚本**:
    ```bash
    python src/inference.py
    ```
*   预测结果将被保存在项目根目录的 `predictions.csv` 文件中。

### 2.3 策略回测与可视化

这是一个两步流程，用于评估模型策略并将其可视化。这两个脚本均**不接受命令行参数**，所有配置都在脚本内部完成。

**第1步：运行回测**

`backtest.py` 会执行回测计算，输出详细日志，并生成 `backtest_history.csv` 文件。

**运行命令:**
```bash
python src/backtest.py
```
*   **如何配置**: 在 `backtest.py` 文件底部的 `if __name__ == '__main__':` 代码块中修改回测参数（如起止时间、初始资金、模型路径等）。

**第2步：生成可视化图表**

`visualization.py` 会读取 `backtest_history.csv` 文件，并生成分析图表 `backtest_analysis.png`。

**运行命令:**
```bash
python src/visualization.py
```

---

## 3. 模型架构

该模型采用混合神经网络架构，结合用于局部特征提取的 **1D-CNN (ResNet)** 和用于长期时间序列建模的 **序列 Transformer (StockViT)**。

*   **特征提取器 (1D-CNN)**: 将每天的高频原始数据压缩成一个高维特征向量。
*   **时序建模 (StockViT)**: 处理由特征提取器生成的“天级别”特征序列，捕捉跨天的时间依赖关系。
*   **多任务预测头**: 模型同时预测未来一段时间内的最大值发生日、最小值发生日、最大值幅度和最小值幅度。

---

## 4. 数据与预处理

*   **输入**: 每日高频交易数据，每个文件代表一只股票一天的数据。
*   **特征**: 模型使用18个预定义的特征，包括成交量、成交额、订单ID数、不同规模订单的金额占比等。
*   **预处理**: 对部分特征应用 `log1p` 变换，然后对所有特征进行全局 Z-Score 标准化。
