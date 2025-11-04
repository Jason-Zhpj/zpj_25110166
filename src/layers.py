# src/layers.py
import torch
import torch.nn as nn
from modules import scaled_dot_product_attention
import config # 导入超参数

class MultiHeadAttention(nn.Module):
    """
    实现多头注意力 (Section 3.2.2)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads # d_k 和 d_v 通常相等
        self.d_v = d_model // n_heads
        
        # 定义 W_q, W_k, W_v, W_o 线性层
        self.W_q = nn.Linear(d_model, d_model) # W_i^Q
        self.W_k = nn.Linear(d_model, d_model) # W_i^K
        self.W_v = nn.Linear(d_model, d_model) # W_i^V
        self.W_o = nn.Linear(d_model, d_model) # W^O
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        ...
        """
        batch_size = Q.size(0)
        
        q_proj = self.W_q(Q)
        k_proj = self.W_k(K)
        v_proj = self.W_v(V)
        
        q_proj = q_proj.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = k_proj.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = v_proj.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_output, _ = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        
        output = self.W_o(attn_output) # (B, L_q, d_model)
        
        return output

class PositionwiseFeedForward(nn.Module):
    """
    实现逐点前馈网络 (Section 3.3, Eq. 2)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # W_1
        self.w_2 = nn.Linear(d_ff, d_model) # W_2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, L, d_model)
        # (B, L, d_model) -> (B, L, d_ff) -> (B, L, d_model)
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """
    实现单个 Encoder 层 (Section 3.1)
    采用 Post-Norm 结构: x + Sublayer(x) -> LayerNorm
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Add & Norm 1
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Add & Norm 2
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x (B, L_src, d_model)
        # mask (B, 1, 1, L_src)
        
        # 1. Multi-Head Attention (Sublayer 1)
        _x = x
        attn_output = self.self_attn(Q=x, K=x, V=x, mask=mask)
        
        # 2. Add & Norm 1
        if config.USE_RESIDUALS:
            x = _x + self.dropout1(attn_output)
        else:
            x = self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 3. Feed Forward (Sublayer 2)
        _x = x
        ffn_output = self.ffn(x)
        
        # 4. Add & Norm 2
        if config.USE_RESIDUALS:
            x = _x + self.dropout2(ffn_output)
        else:
            x = self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x


class DecoderLayer(nn.Module):
    """
    实现单个 Decoder 层 (Section 3.1)
    采用 Post-Norm 结构
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                encoder_output: torch.Tensor, 
                src_mask: torch.Tensor, 
                tgt_mask: torch.Tensor) -> torch.Tensor:
        # x (B, L_tgt, d_model)
        # encoder_output (B, L_src, d_model)
        # src_mask (B, 1, 1, L_src)
        # tgt_mask (B, 1, L_tgt, L_tgt)
        
        # 1. Masked Multi-Head Attention (Sublayer 1)
        _x = x
        attn_output = self.masked_self_attn(Q=x, K=x, V=x, mask=tgt_mask)
        if config.USE_RESIDUALS:
            x = _x + self.dropout1(attn_output)
        else:
            x = self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 2. Cross-Attention (Sublayer 2)
        _x = x
        # Q=x (来自 Decoder), K=encoder_output, V=encoder_output (来自 Encoder)
        cross_attn_output = self.cross_attn(Q=x, K=encoder_output, V=encoder_output, mask=src_mask)
        if config.USE_RESIDUALS:
            x = _x + self.dropout2(cross_attn_output)
        else:
            x = self.dropout2(cross_attn_output)
        x = self.norm2(x)
        
        # 3. Feed Forward (Sublayer 3)
        _x = x
        ffn_output = self.ffn(x)
        if config.USE_RESIDUALS:
            x = _x + self.dropout3(ffn_output)
        else:
            x = self.dropout3(ffn_output)
        x = self.norm3(x)
        
        return x