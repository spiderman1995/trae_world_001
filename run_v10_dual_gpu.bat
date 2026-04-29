@echo off
REM =============================================================================
REM v10 双卡 V100 训练脚本 (Windows 版)
REM 两张卡跑不同实验配置，零代码改动，同时出结果直接对比
REM =============================================================================
REM
REM 使用方法：
REM   方式1 - 同时启动两张卡（推荐用两个终端分别启动，可独立监控）：
REM     终端1: run_v10_dual_gpu.bat gpu0
REM     终端2: run_v10_dual_gpu.bat gpu1
REM
REM   方式2 - 单个终端同时启动（后台运行，日志写文件）：
REM     run_v10_dual_gpu.bat both
REM
REM TensorBoard 监控：
REM   tensorboard --logdir runs_v10_A:runs_v10_A,runs_v10_B:runs_v10_B --port 6006
REM =============================================================================

set DATA_DIR=E:/pre_process/version1

set COMMON_ARGS=--data_dir %DATA_DIR% --seq_len 180 --pred_len 15 --train_days 480 --test_days 60 --step_days 15 --epochs 20 --patience 5 --min_delta 1e-3 --lr 1e-4 --weight_decay 5e-3 --drop_ratio 0.2 --attn_drop_ratio 0.2 --smooth_l1_beta 0.1 --day_sigma 1.0 --scheduler cosine --max_grad_norm 1.0 --warm_start_mode full --stock_pool random --sample_stride 15 --min_list_days 180 --num_workers 4 --preload auto --cudnn_benchmark --seed 42

if "%1"=="" set "1=both"

if "%1"=="gpu0" goto :gpu0
if "%1"=="gpu1" goto :gpu1
if "%1"=="both" goto :both
echo Usage: %~nx0 [gpu0^|gpu1^|both]
exit /b 1

REM =============================================================================
REM 实验 A（GPU 0）：大配置 — 全部合格股票, embed_dim=384, depth=4
REM 参数量: CNN 1.76M + Proj 0.2M + ViT ~9.5M = 11.5M
REM =============================================================================
:gpu0
echo ========== Starting Experiment A on GPU 0 ==========
set CUDA_VISIBLE_DEVICES=0
python -m src.train_rolling %COMMON_ARGS% --output_dir runs_v10_A --cnn_dim 512 --embed_dim 384 --depth 4 --num_heads 4 --num_stocks 9999 --batch_size 32
echo ========== Experiment A finished ==========
goto :eof

REM =============================================================================
REM 实验 B（GPU 1）：小配置 — 全部合格股票, embed_dim=256, depth=3
REM 参数量: CNN 1.76M + Proj 0.13M + ViT ~2.8M = 4.7M
REM =============================================================================
:gpu1
echo ========== Starting Experiment B on GPU 1 ==========
set CUDA_VISIBLE_DEVICES=1
python -m src.train_rolling %COMMON_ARGS% --output_dir runs_v10_B --cnn_dim 512 --embed_dim 256 --depth 3 --num_heads 4 --num_stocks 9999 --batch_size 32
echo ========== Experiment B finished ==========
goto :eof

REM =============================================================================
REM 同时启动两张卡（各自写日志）
REM =============================================================================
:both
echo ========== Starting both experiments ==========
start "Experiment A - GPU 0" cmd /c "%~f0 gpu0 > gpu0.log 2>&1"
start "Experiment B - GPU 1" cmd /c "%~f0 gpu1 > gpu1.log 2>&1"
echo GPU 0: Experiment A started (log: gpu0.log)
echo GPU 1: Experiment B started (log: gpu1.log)
echo Use "type gpu0.log" or "type gpu1.log" to check progress.
echo TensorBoard: tensorboard --logdir runs_v10_A:runs_v10_A,runs_v10_B:runs_v10_B
goto :eof
