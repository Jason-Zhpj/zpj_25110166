# src/modules.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    实现固定（非学习）的正弦位置编码 (Section 3.5)
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # PE 矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 广播机制 (max_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数维度
        
        # 增加 batch 维度 (1, max_len, d_model)
        # register_buffer 确保 pe 矩阵是模型状态的一部分，但不作为参数被优化
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        """
        # 截取所需长度的 PE 并加到 x 上
        # self.pe (1, max_len, d_model) -> (1, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def scaled_dot_product_attention(Q: torch.Tensor, 
                                 K: torch.Tensor, 
                                 V: torch.Tensor, 
                                 mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
    """
    实现缩放点积注意力 (Section 3.2.1, Eq. 1)
    
    Args:
        Q (B, n_heads, seq_len_q, d_k)
        K (B, n_heads, seq_len_k, d_k)
        V (B, n_heads, seq_len_v, d_v) (seq_len_k == seq_len_v)
        mask (B, 1, seq_len_q, seq_len_k) or (B, 1, 1, seq_len_k)
    
    Returns:
        output (B, n_heads, seq_len_q, d_v)
        attn_weights (B, n_heads, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)
    
    # 1. QK^T / sqrt(d_k)
    # (B, h, L_q, d_k) @ (B, h, d_k, L_k) -> (B, h, L_q, L_k)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. Masking
    if mask is not None:
        # mask (B, 1, L_q, L_k) or (B, 1, 1, L_k)
        # PyTorch 的 masked_fill_：mask 中为 True 的地方被填充
        scores = scores.masked_fill(mask, -1e9)
    
    # 3. Softmax
    attn_weights = torch.softmax(scores, dim=-1) # (B, h, L_q, L_k)
    
    # 4. (Softmax * V)
    # (B, h, L_q, L_k) @ (B, h, L_v, d_v) -> (B, h, L_q, d_v) (L_k == L_v)
    output = attn_weights @ V
    
    return output, attn_weights