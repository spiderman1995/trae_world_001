
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
│   └── visualization.py      # 回测结果可视化脚本 (新增)
├── backtest_history.csv      # 回测生成的历史净值数据 (新增)
├── backtest_analysis.png     # 可视化脚本生成的分析图表 (新增)
├── MODEL_README.md           # 本文档
└── requirements.txt          # Python 依赖
```

---

## 2. 核心功能与用法

### 2.1 模型训练

通过运行 `train_rolling.py` 脚本，可以启动滚动窗口训练流程。

**运行命令:**
```bash
python src/train_rolling.py --data_path /path/to/your/data --save_dir ./runs_rolling/ --epochs 50
```

### 2.2 模型推理

`inference.py` 脚本用于加载一个训练好的模型，并对新的数据进行预测。

**运行命令:**
```bash
python src/inference.py --model_path ./runs_rolling/fold_1/model_final.pth --data_path /path/to/new/data
```

### 2.3 策略回测与可视化 (更新)

这是一个两步流程，用于评估模型策略并将其可视化。

**第1步：运行回测**

`backtest.py` 提供了一个完整的策略回测框架，用于评估模型在历史数据上的表现。它会输出详细的周期性日志，并在结束后生成 `backtest_history.csv` 文件。

**运行命令:**
```bash
python src/backtest.py
```

**如何配置:**

回测的所有参数都在 `backtest.py` 文件底部的 `if __name__ == '__main__':` 代码块中进行配置，包括起止时间、初始资金、模型路径和策略参数（如`k`值）。

**第2步：生成可视化图表**

`visualization.py` 脚本会读取上一步生成的 `backtest_history.csv` 文件，并创建一个包含净值曲线和回撤曲线的分析图表 `backtest_analysis.png`。

**运行命令:**
```bash
python src/visualization.py
```

**重要提示：** 当前回测系统默认使用**模拟数据**和**模拟预测**。要获得有意义的结果，您需要：
1.  修改 `DataLoader` 类以加载您的**真实历史数据**。
2.  修改 `TopKStrategy` 中的 `generate_signals` 方法，接入**真实的模型推理逻辑**。

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
