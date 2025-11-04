# src/utils.py
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def set_seed(seed: int):
    """
    设置全局随机种子 (作业要求)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global random seed set to {seed}")

def count_parameters(model: nn.Module) -> int:
    """
    统计模型参数 (作业要求)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_loss_plot(train_losses, val_losses, filepath: str):
    """
    保存训练/验证损失曲线图 (作业要求)
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 确保 results 目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    print(f"Loss plot saved to {filepath}")

def save_checkpoint(model, optimizer, epoch, loss, filepath: str):
    """
    保存模型检查点 (作业"进阶"要求)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved to {filepath} (Epoch {epoch})")

def load_checkpoint(filepath: str, model, optimizer):
    """
    加载模型检查点 (作业"进阶"要求)
    """
    if not os.path.exists(filepath):
        print("No checkpoint found, starting from scratch.")
        return 0, float('inf')
        
    checkpoint = torch.load(filepath, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath} (Epoch {epoch}, Loss {loss:.4f})")
    return epoch, loss