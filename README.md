# 大模型基础与应用 - 期中作业

本项目是 "大模型基础与应用" 课程的期中作业，内容是**从零开始实现一个完整的 Transformer 模型**。

该实现包含一个标准的 Encoder-Decoder 架构，并在**cnn-dailymail**上进行了训练和验证。

## 项目结构

本仓库遵循标准的机器学习项目结构：

```
zpj_25110166/
|-- README.md               # 本项目说明
|-- requirements.txt        # Python 依赖包列表
|-- models/
|   +-- iwslt_transformer.pt  # 训练好的模型权重 (由 run.sh 生成)
|-- results/
|   |-- train_loss_curve.png  # 训练损失曲线 (由 run.sh 生成)
|   +-- ablation_loss_curve.png # (消融实验图表示例)
|-- scripts/
|   +-- run.sh              # 一键复现训练的BASH脚本
+-- src/
    |-- train.py            # 主训练脚本 (入口: python -m src.train)
    |-- model.py            # Transformer 完整模型 (Encoder, Decoder)
    |-- modules.py          # 核心模块 (MultiHeadAttention, FFN, PositionalEncoding等)
    |-- data.py             # 数据集处理、Spacy分词、DataLoader构建
    +-- utils.py            # 辅助函数 (掩码生成、AdamW/Scheduler等)
```

## 环境配置与安装

我们建议使用 `conda` 创建独立的 Python 虚拟环境。

1. **创建并激活 Conda 环境**

   ```
   conda create -n llm_midterm python=3.10
   conda activate llm_midterm
   ```
   
2. **安装 Python 依赖**

   克隆仓库后，使用 requirements.txt 文件一键安装所有依赖：

   ```
   git clone https://github.com/YourUsername/BJTU-LLMs-Mid-Term-main.git
   cd BJTU-LLMs-Mid-Term-main
   pip install -r requirements.txt
   ```

## 如何运行 (复现实验)

本项目提供了bash脚本复现实验结果，包含作业要求的固定随机种子 (`--seed 42`)。

### 使用BASH脚本

作业中要求的 `scripts/run.sh` 脚本 已包含所有必要的参数，是复现实验的最便捷方式。

```
# 确保脚本有执行权限
chmod +x scripts/run.sh

# 运行脚本
./scripts/run.sh
```

该脚本将自动执行以下操作：

1. 使用t5-small分词器
2. 创建 `results/` 和 `models/` 目录。
3. 按照指定的超参数开始训练。
4. 将最终模型保存到 `models/iwslt_transformer.pt`。
5. 将训练损失曲线图保存到 `results/train_loss_curve.png`。

## 实验超参数

复现所用的关键超参数配置如下，均定义在 `scripts/run.sh` 中：

| **参数**     | **值** | **描述**             |
| ------------ | ------ | -------------------- |
| `d_model`    | 256    | 模型隐藏层维度       |
| `n_heads`    | 8      | 多头注意力头数       |
| `n_layers`   | 3      | Encoder/Decoder 层数 |
| `d_ff`       | 512    | FFN 内部维度         |
| `batch_size` | 32     | 批量大小             |
| `lr`         | 3e-4   | 学习率               |
| `epochs`     | 15     | 训练轮数             |
| `seed`       | 42     | 全局随机种子         |

## 硬件要求

- **CPU**: 可运行，但训练会非常缓慢。
- **GPU**: 推荐使用。本实验在单块 V100 (32GB) 上完成，完成20个epoch约耗时 50-70 分钟。对于 `d_model=256` 的小模型，一块 VRAM > 8GB 的消费级显卡（如 RTX 3060 / 4060）即可满足训练要求。

## 实验结果

训练完成后，`results/` 目录下将生成训练损失曲线图。