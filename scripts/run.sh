#!/bin/bash

# 退出脚本，如果任何命令失败
set -e

# 设置随机种子 (作业要求)
SEED=42

# 硬件要求 (作业要求)
# 建议：NVIDIA GPU (e.g., T4, V100, A100) 至少 16GB VRAM
# 如果 VRAM 不足，请在 src/config.py 中减小 BATCH_SIZE 或 MAX_SEQ_LEN

echo "============================================="
echo "Mid-Term Assignment: Transformer From Scratch"
echo "Student: 占普剑 (25110166)"
echo "Running main training script..."
echo "Seed: $SEED"
echo "============================================="

# 创建必要的目录
mkdir -p ./models
mkdir -p ./results

# 运行训练脚本
# (所有配置都在 src/config.py 中，所以这里很干净)
python -u src/train.py

echo "============================================="
echo "Training finished."
echo "Results:"
echo "  - Log:       ./results/training_log.txt"
echo "  - Loss Plot: ./results/loss_curve.png"
echo "  - Model:     ./models/transformer_checkpoint.pth"
echo "============================================="