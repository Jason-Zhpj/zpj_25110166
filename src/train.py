# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import time
import os

import config # 超参数
from model import Transformer
from data import get_dataloaders
from utils import set_seed, count_parameters, save_loss_plot, save_checkpoint, load_checkpoint

def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device, clip_norm):
    model.train() # 切换到训练模式
    epoch_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Training Epoch")
    for batch in pbar:
        # 1. 数据
        src = batch['src']
        tgt_input = batch['tgt_input']
        tgt_target = batch['tgt_target']
        src_mask = batch['src_mask']
        tgt_mask = batch['tgt_mask']
        
        # 2. 前向传播
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        # 3. 计算损失
        # Logits: (B, L_tgt, Vocab) -> (B * L_tgt, Vocab)
        # Target: (B, L_tgt) -> (B * L_tgt)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)), 
            tgt_target.reshape(-1)
        )
        
        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 5. 梯度裁剪 (作业"进阶"要求)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        
        # 6. 更新参数
        optimizer.step()
        
        # 7. 更新学习率 (作业"进阶"要求)
        scheduler.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval() # 切换到评估模式
    epoch_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating")
        for batch in pbar:
            # 1. 数据
            src = batch['src']
            tgt_input = batch['tgt_input']
            tgt_target = batch['tgt_target']
            src_mask = batch['src_mask']
            tgt_mask = batch['tgt_mask']

            # 2. 前向传播
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # 3. 计算损失
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), 
                tgt_target.reshape(-1)
            )
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    return epoch_loss / len(dataloader)


def main():
    # --- 1. 设置与初始化 ---
    set_seed(config.SEED)
    device = config.DEVICE
    os.makedirs(os.path.dirname(config.LOG_FILE_PATH), exist_ok=True)
    log_file = open(config.LOG_FILE_PATH, 'w')
    
    def log_print(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log_print(f"Using device: {device}")
    log_print(f"CONFIG: d_model={config.D_MODEL}, n_layers={config.N_LAYERS}, n_heads={config.N_HEADS}")
    log_print(f"CONFIG: Ablation: USE_PE={config.USE_POSITIONAL_ENCODING}, USE_RES={config.USE_RESIDUALS}")

    # --- 2. 加载数据 ---
    train_loader, val_loader, tokenizer = get_dataloaders()
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    log_print(f"Vocabulary size: {vocab_size}")
    log_print(f"PAD token ID: {pad_token_id}")

    # --- 3. 构建模型 ---
    model = Transformer(
        enc_vocab_size=vocab_size,
        dec_vocab_size=vocab_size,
        d_model=config.D_MODEL,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        max_seq_len=config.MAX_SEQ_LEN
    ).to(device)
    
    log_print(f"Model initialized. Total parameters: {count_parameters(model):,}") # 参数统计

    # --- 4. 设置优化器和损失函数 ---
    
    # 使用 AdamW (作业"进阶"要求)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        betas=config.ADAMW_BETAS,
        eps=config.ADAMW_EPS,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 损失函数, 忽略 padding (重要!)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    # 学习率调度 (作业"进阶"要求)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.SCHEDULER_WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # --- 5. 加载检查点 (可选) ---
    # start_epoch, _ = load_checkpoint(config.MODEL_SAVE_PATH, model, optimizer)
    start_epoch = 0 # 强制从 0 开始
    
    # --- 6. 训练循环 ---
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    log_print("--- Starting Training ---")
    for epoch in range(start_epoch, config.EPOCHS):
        start_time = time.time()
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, config.CLIP_GRAD_NORM
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        log_print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
        log_print(f'\tTrain Loss: {train_loss:.4f} | Val. Loss: {val_loss:.4f}')
        
        # 保存最佳模型 (作业"进阶"要求)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config.MODEL_SAVE_PATH)
            log_print(f'\tBest model saved (Val. Loss: {best_val_loss:.4f}).')
            
    log_print("--- Finished Training ---")
    log_file.close()

    # --- 7. 保存结果 (作业要求) ---
    save_loss_plot(train_losses, val_losses, config.LOSS_PLOT_PATH)

if __name__ == "__main__":
    #import debugpy
    ##保证host和端口一致，listen可以只设置端口。则为localhost,否则设置成(host,port)
    #debugpy.listen(21830)
    #print('wait debugger')
    #debugpy.wait_for_client()
    #print("Debugger Attached")
    main()