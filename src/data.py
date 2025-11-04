# src/data.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer
import config

class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, split, subset_size):
        self.dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split=split, streaming=False)
        # 作业要求：使用小数据集
        if subset_size > 0:
            self.dataset = self.dataset.select(range(subset_size))
            
        self.tokenizer = tokenizer
        self.max_len = config.MAX_SEQ_LEN

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        article = item[config.SOURCE_COL]
        summary = item[config.TARGET_COL]

        # 1. Tokenize Encoder Input (Article)
        encoder_input = self.tokenizer(
            article, 
            max_length=self.max_len, 
            truncation=True, 
            padding=False,
            return_tensors=None # 返回 list
        )
        encoder_input_ids = encoder_input['input_ids']

        # 2. Tokenize Decoder Input & Target (Summary)
        # Decoder Input: [SOS] + summary_text
        # Decoder Target: summary_text + [EOS]
        decoder_input = self.tokenizer(
            summary, 
            max_length=self.max_len - 1, # 为 SOS/EOS 留空间
            truncation=True, 
            padding=False,
            return_tensors=None
        )
        
        decoder_input_ids = [self.tokenizer.pad_token_id] + decoder_input['input_ids']
        decoder_target_ids = decoder_input['input_ids'] + [self.tokenizer.eos_token_id]

        return {
            'encoder_input_ids': torch.tensor(encoder_input_ids, dtype=torch.long),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'decoder_target_ids': torch.tensor(decoder_target_ids, dtype=torch.long),
        }

def create_look_ahead_mask(seq_len):
    """
    创建后续掩码 (Look-Ahead Mask)
    [[False,  True,  True],
     [False, False,  True],
     [False, False, False]]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask # (seq_len, seq_len)

def collate_fn(batch):
    """
    自定义 Collate_fn, 用于 Dataloader
    1. 动态填充 (Padding)
    2. 创建掩码 (Masking)
    """
    # 注意：T5 使用 pad_token_id (0) 作为解码器的起始符
    pad_token_id = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME).pad_token_id

    # 1. 填充 (在 CPU 上)
    encoder_inputs = pad_sequence(
        [item['encoder_input_ids'] for item in batch], 
        batch_first=True, 
        padding_value=pad_token_id
    ) # (B, L_src)
    
    decoder_inputs = pad_sequence(
        [item['decoder_input_ids'] for item in batch], 
        batch_first=True, 
        padding_value=pad_token_id
    ) # (B, L_tgt)
    
    decoder_targets = pad_sequence(
        [item['decoder_target_ids'] for item in batch], 
        batch_first=True, 
        padding_value=pad_token_id
    ) # (B, L_tgt)
    
    encoder_inputs = encoder_inputs.to(config.DEVICE)
    decoder_inputs = decoder_inputs.to(config.DEVICE)
    decoder_targets = decoder_targets.to(config.DEVICE)

    src_padding_mask = (encoder_inputs == pad_token_id).unsqueeze(1).unsqueeze(2)

    tgt_padding_mask = (decoder_inputs == pad_token_id).unsqueeze(1).unsqueeze(-1)

    L_tgt = decoder_inputs.size(1)
    tgt_look_ahead_mask = create_look_ahead_mask(L_tgt).unsqueeze(0).unsqueeze(1).to(config.DEVICE)
    
    tgt_mask = tgt_padding_mask | tgt_look_ahead_mask

    return {
        'src': encoder_inputs,      
        'tgt_input': decoder_inputs, 
        'tgt_target': decoder_targets, 
        'src_mask': src_padding_mask,  
        'tgt_mask': tgt_mask,        
    }

def get_dataloaders():
    print(f"Loading tokenizer: {config.TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    print(f"Loading dataset: {config.DATASET_NAME} (Train: {config.TRAIN_SUBSET_SIZE}, Val: {config.VALID_SUBSET_SIZE})")
    train_dataset = SummarizationDataset(tokenizer, "train", config.TRAIN_SUBSET_SIZE)
    val_dataset = SummarizationDataset(tokenizer, "validation", config.VALID_SUBSET_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, tokenizer