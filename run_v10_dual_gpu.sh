#!/bin/bash
# =============================================================================
# v10 双卡 V100 训练脚本
# 两张卡跑不同实验配置，零代码改动，同时出结果直接对比
# =============================================================================
#
# 使用方法（服务器上）：
#   chmod +x run_v10_dual_gpu.sh
#   nohup bash run_v10_dual_gpu.sh > dual_gpu.log 2>&1 &
#
# 或分别启动（推荐，可独立监控）：
#   nohup bash run_v10_dual_gpu.sh gpu0 > gpu0.log 2>&1 &
#   nohup bash run_v10_dual_gpu.sh gpu1 > gpu1.log 2>&1 &
#
# TensorBoard 监控：
#   tensorboard --logdir runs_v10_A:runs_v10_A,runs_v10_B:runs_v10_B --port 6006
# =============================================================================

DATA_DIR="E:/pre_process/version1"

# ---- 公共参数 ----
COMMON_ARGS="
    --data_dir ${DATA_DIR}
    --seq_len 180
    --pred_len 15
    --train_days 480
    --test_days 60
    --step_days 15
    --epochs 20
    --patience 5
    --min_delta 1e-3
    --lr 1e-4
    --weight_decay 5e-3
    --drop_ratio 0.2
    --attn_drop_ratio 0.2
    --smooth_l1_beta 0.1
    --day_sigma 1.0
    --scheduler cosine
    --max_grad_norm 1.0
    --warm_start_mode full
    --stock_pool random
    --sample_stride 15
    --min_list_days 180
    --num_workers 4
    --preload auto
    --cudnn_benchmark
    --seed 42
"

# =============================================================================
# 实验 A（GPU 0）：大配置 — 全部合格股票, embed_dim=384, depth=4
# 参数量: CNN 1.76M + Proj 0.2M + ViT ~9.5M ≈ 11.5M
# =============================================================================
run_gpu0() {
    echo "========== Starting Experiment A on GPU 0 =========="
    CUDA_VISIBLE_DEVICES=0 python -m src.train_rolling \
        ${COMMON_ARGS} \
        --output_dir runs_v10_A \
        --cnn_dim 512 \
        --embed_dim 384 \
        --depth 4 \
        --num_heads 4 \
        --num_stocks 9999 \
        --batch_size 32
    echo "========== Experiment A finished =========="
}

# =============================================================================
# 实验 B（GPU 1）：小配置 — 全部合格股票, embed_dim=256, depth=3
# 参数量: CNN 1.76M + Proj 0.13M + ViT ~2.8M ≈ 4.7M
# =============================================================================
run_gpu1() {
    echo "========== Starting Experiment B on GPU 1 =========="
    CUDA_VISIBLE_DEVICES=1 python -m src.train_rolling \
        ${COMMON_ARGS} \
        --output_dir runs_v10_B \
        --cnn_dim 512 \
        --embed_dim 256 \
        --depth 3 \
        --num_heads 4 \
        --num_stocks 9999 \
        --batch_size 32
    echo "========== Experiment B finished =========="
}

# =============================================================================
# 启动逻辑
# =============================================================================
case "${1:-both}" in
    gpu0) run_gpu0 ;;
    gpu1) run_gpu1 ;;
    both)
        run_gpu0 &
        PID0=$!
        run_gpu1 &
        PID1=$!
        echo "GPU 0 PID: ${PID0}, GPU 1 PID: ${PID1}"
        echo "Waiting for both experiments to finish..."
        wait ${PID0}
        echo "Experiment A (GPU 0) done."
        wait ${PID1}
        echo "Experiment B (GPU 1) done."
        echo "All experiments finished. Compare results:"
        echo "  tensorboard --logdir runs_v10_A:runs_v10_A,runs_v10_B:runs_v10_B"
        ;;
    *)
        echo "Usage: $0 [gpu0|gpu1|both]"
        exit 1
        ;;
esac
