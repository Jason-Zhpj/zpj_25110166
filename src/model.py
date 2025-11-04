# src/model.py
import torch
import torch.nn as nn
import math
from layers import EncoderLayer, DecoderLayer
from modules import PositionalEncoding
import config

class Encoder(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 n_layers: int, 
                 n_heads: int, 
                 d_ff: int, 
                 dropout: float, 
                 max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model) # 最终的 LayerNorm

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # src (B, L_src)
        # src_mask (B, 1, 1, L_src)
        
        # 1. Embedding + Positional Encoding
        # (B, L_src) -> (B, L_src, d_model)
        src_embed = self.token_embedding(src) * math.sqrt(self.d_model)
        
        if config.USE_POSITIONAL_ENCODING:
            src_embed = self.pos_encoding(src_embed)
        
        # 2. 逐层 Encoder
        output = src_embed
        for layer in self.layers:
            output = layer(output, src_mask)
            
        # 3. 最终 Norm (原版 Transformer 在 Encoder 最后有一个 Norm)
        output = self.norm(output)
        return output # (B, L_src, d_model)

class Decoder(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 n_layers: int, 
                 n_heads: int, 
                 d_ff: int, 
                 dropout: float, 
                 max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model) # 最终的 LayerNorm
        
    def forward(self, 
                tgt: torch.Tensor, 
                encoder_output: torch.Tensor, 
                src_mask: torch.Tensor, 
                tgt_mask: torch.Tensor) -> torch.Tensor:
        # tgt (B, L_tgt)
        # encoder_output (B, L_src, d_model)
        # src_mask (B, 1, 1, L_src)
        # tgt_mask (B, 1, L_tgt, L_tgt)
        
        # 1. Embedding + Positional Encoding
        # (B, L_tgt) -> (B, L_tgt, d_model)
        tgt_embed = self.token_embedding(tgt) * math.sqrt(self.d_model)
        if config.USE_POSITIONAL_ENCODING:
            tgt_embed = self.pos_encoding(tgt_embed)
            
        # 2. 逐层 Decoder
        output = tgt_embed
        for layer in self.layers:
            output = layer(output, encoder_output, src_mask, tgt_mask)
            
        # 3. 最终 Norm
        output = self.norm(output)
        return output # (B, L_tgt, d_model)

class Transformer(nn.Module):
    def __init__(self, 
                 enc_vocab_size: int, 
                 dec_vocab_size: int, 
                 d_model: int, 
                 n_layers: int, 
                 n_heads: int, 
                 d_ff: int, 
                 dropout: float, 
                 max_seq_len: int):
        super().__init__()
        
        self.encoder = Encoder(enc_vocab_size, d_model, n_layers, n_heads, 
                               d_ff, dropout, max_seq_len)
        
        self.decoder = Decoder(dec_vocab_size, d_model, n_layers, n_heads, 
                               d_ff, dropout, max_seq_len)
        
        # 最终的线性层 (Logits)
        self.final_proj = nn.Linear(d_model, dec_vocab_size)
        
        # 初始化参数
        self._initialize_weights()
        
    def _initialize_weights(self):
        # 使用 Xavier/Glorot 初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor, 
                src_mask: torch.Tensor, 
                tgt_mask: torch.Tensor) -> torch.Tensor:
        # src (B, L_src)
        # tgt (B, L_tgt)
        # src_mask (B, 1, 1, L_src)
        # tgt_mask (B, 1, L_tgt, L_tgt)
        
        # 1. Encoder
        encoder_output = self.encoder(src, src_mask) # (B, L_src, d_model)
        
        # 2. Decoder
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask) # (B, L_tgt, d_model)
        
        # 3. Final Projection
        logits = self.final_proj(decoder_output) # (B, L_tgt, dec_vocab_size)
        
        return logits