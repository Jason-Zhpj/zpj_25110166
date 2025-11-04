# src/config.py
import torch

# --- 数据集配置 ---
DATASET_NAME = "ccdv/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
# 作业要求：使用 10k-20k 子集进行本地训练
# 我们选择 10k 训练, 2k 验证
TRAIN_SUBSET_SIZE = 10000
VALID_SUBSET_SIZE = 2000
SOURCE_COL = 'article'
TARGET_COL = 'highlights'
# 建议使用一个预训练的分词器，这不违反“手工搭建 Transformer”的要求
# 它只负责将文本转为 ID
TOKENIZER_NAME = "t5-small" 

# --- 模型超参数 (小型) ---
# 确保 d_model % n_heads == 0
D_MODEL = 256
N_LAYERS = 6  # Encoder 和 Decoder 各 3 层
N_HEADS = 8
D_K = D_MODEL // N_HEADS # 32
D_V = D_MODEL // N_HEADS # 32
D_FF = 1024  # 4 * D_MODEL
DROPOUT = 0.1
MAX_SEQ_LEN = 512 # CNN/DM 文章很长, 但我们先截断到 512

# --- 训练超参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
ADAMW_BETAS = (0.9, 0.98)
ADAMW_EPS = 1e-9
WEIGHT_DECAY = 0.01
# 梯度裁剪 (作业"进阶"要求)
CLIP_GRAD_NORM = 1.0
# 学习率调度 (作业"进阶"要求)
SCHEDULER_WARMUP_STEPS = 1000
SEED = 42

# --- 文件路径 ---
MODEL_SAVE_PATH = "./models/transformer_checkpoint_8head.pth"
LOSS_PLOT_PATH = "./results/loss_curve_8head.png"
LOG_FILE_PATH = "./results/training_log_8head.txt"

# --- 消融实验控制 (作业要求) ---
# 你可以在这里添加布尔开关来控制消融实验
USE_POSITIONAL_ENCODING = True
USE_RESIDUALS = True