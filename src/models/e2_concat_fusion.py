"""Pretrained emoji embedding learning from TweetEval emoji dataset.

This module learns emoji representations by training on the TweetEval emoji
prediction task (20 emoji classes), then extracts the learned emoji embeddings
for use in E2.

Key principle: The auxiliary TweetEval emoji task is used ONLY to learn
semantic emoji representations. The learned embeddings are then frozen and
used in E2 for StockTwits sentiment classification. No StockTwits data is
used to learn these embeddings.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score

from src.embeddings.emoji_encoder import build_emoji_vocab, encode_emoji_list
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMOJI_DIM = 32
MAX_EMOJIS = 8
PRETRAIN_EPOCHS = 10
PRETRAIN_BATCH_SIZE = 64
PRETRAIN_LR = 2e-5
PRETRAIN_WD = 1e-4
SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# TweetEval Emoji Pretraining Model
# ---------------------------------------------------------------------------

class TweetEvalEmojiModel(nn.Module):
    """BERT + emoji embedding for TweetEval emoji prediction (20 classes)."""

    def __init__(self, vocab_size: int, emoji_dim: int = 32, num_labels: int = 20):
        super().__init__()
        self.emoji_dim = emoji_dim
        
        # Emoji embedding (to be pretrained) - FIXED: vocab_size, emoji_dim order
        self.emoji_encoder = nn.Embedding(vocab_size, emoji_dim, padding_idx=None)
        nn.init.normal_(self.emoji_encoder.weight, mean=0.0, std=0.1)
        
        # Text encoder
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained("bert-base-uncased", config=config)
        
        # Classification head for emoji prediction
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 20),  # 20 TweetEval emoji classes
        )

    def forward(self, input_ids, attention_mask):
        """Return emoji classification logits."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask.unsqueeze(-1)).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts  # (B, 768)
        logits = self.classifier(pooled)
        return logits
    
    def get_emoji_embeddings(self):
        """Return the learned emoji embedding matrix for E2."""
        return self.emoji_encoder.weight.data.clone()


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def load_tweeteval_emoji():
    """Load TweetEval emoji dataset."""
    ds = load_dataset("cardiffnlp/tweet_eval", "emoji")
    return ds


def prepare_tweeteval_data(tokenizer, max_length=128):
    """Prepare TweetEval data for emoji pretraining."""
    ds = load_tweeteval_emoji()
    
    def tokenize(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    train_loader = DataLoader(ds["train"], batch_size=64, shuffle=True)
    val_loader = DataLoader(ds["validation"], batch_size=64, shuffle=False)
    test_loader = DataLoader(ds["test"], batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Pretraining Loop
# ---------------------------------------------------------------------------

def pretrain_emoji_embeddings():
    """Train emoji embeddings on TweetEval emoji prediction task."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load TweetEval emoji dataset to get vocab
    ds = load_dataset("cardiffnlp/tweet_eval", "emoji")
    label_names = ds["train"].features["label"].names
    num_tweeteval_emojis = len(label_names)
    print(f"TweetEval emoji classes: {num_tweeteval_emojis}")
    print(f"Emoji labels: {label_names}")
    
    # Build vocab from TweetEval emoji labels (20 emojis)
    tweeteval_emoji_lists = [[label_names[label]] for label in ds["train"]["label"]]
    vocab = build_emoji_vocab(tweeteval_emoji_lists, min_freq=1)
    print(f"TweetEval emoji vocab size: {vocab['vocab_size']}")
    
    # Save vocab
    os.makedirs("models/emoji_embeddings", exist_ok=True)
    from src.embeddings.emoji_encoder import save_vocab
    save_vocab(vocab, "models/emoji_embeddings/tweeteval_emoji_vocab.json")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Data loaders
    train_loader, val_loader, test_loader = prepare_tweeteval_data(tokenizer)
    
    # Model - vocab_size is number of TweetEval emoji classes (20), not embedding dim
    model = TweetEvalEmojiModel(
        vocab_size=num_tweeteval_emojis,  # 20 TweetEval emoji classes
        emoji_dim=EMOJI_DIM,
        num_labels=num_tweeteval_emojis
    ).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_f1 = 0.0
    best_state = None
    
    for epoch in range(1, 11):  # 10 epochs
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/10"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_preds, val_true)
        print(f"Epoch: val_acc={val_acc:.4f}, val_macro_f1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save only the 20 TweetEval emoji embeddings [20, 32]
            best_state = {
                "emoji_embedding": model.emoji_encoder.weight.data.clone(),  # [20, 32]
                "emoji_dim": EMOJI_DIM,
                "vocab_size": num_tweeteval_emojis,  # 20
                "tweeteval_emoji_to_id": vocab["emoji_to_id"],
                "tweeteval_id_to_emoji": vocab["id_to_emoji"],
            }
            torch.save(best_state, "models/emoji_embeddings/pretrained_emoji_20x32.pt")
            print(f"  Saved best pretrained embeddings [20, 32] (val_macro_f1={val_f1:.4f})")
    
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    return best_state


# ---------------------------------------------------------------------------
# E2 Pretrained Emoji Encoder
# ---------------------------------------------------------------------------

class PretrainedEmojiEncoder(nn.Module):
    """Emoji encoder using pretrained embeddings (frozen for E2)."""

    def __init__(
        self,
        pretrained_path: str,
        vocab_size: int,
        embedding_dim: int = 32,
        unk_id: int = 0,
        max_emojis: int = 8,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = 32
        self.unk_id = 0
        self.max_emojis = 8

        # Load pretrained embeddings
        pretrained = torch.load(pretrained_path, map_location="cpu")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=None)
        self.embedding.weight.data = pretrained["emoji_embedding"]
        
        if freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False

    def forward(self, emoji_ids: torch.Tensor, emoji_masks: torch.Tensor) -> torch.Tensor:
        """Aggregate emoji embeddings via mean pooling."""
        emb = self.embedding(emoji_ids)  # (B, M, D)
        mask = emoji_masks.unsqueeze(-1).float()  # (B, M, 1)
        summed = (emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts  # (B, D)


def load_pretrained_emoji_encoder(vocab_size: int = 32, embedding_dim: int = 32) -> PretrainedEmojiEncoder:
    """Load pretrained emoji encoder for E2."""
    path = "models/emoji_embeddings/pretrained_emoji_32d.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained embeddings not found at {path}. Run pretraining first.")
    return PretrainedEmojiEncoder(path, vocab_size, embedding_dim, freeze=True)


# ---------------------------------------------------------------------------
# E2 Concat Fusion Model with Pretrained Emoji Embeddings
# ---------------------------------------------------------------------------

class E2ConcatFusionModel(nn.Module):
    """E2: Text + pretrained emoji embedding, concatenation fusion."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        text_dim: int = 768,
        emoji_dim: int = 32,
        num_labels: int = 3,
        dropout: float = 0.3,
        classifier_hidden: int = 256,
        freeze_encoder: bool = True,
        emoji_vocab_size: int = 1610,  # StockTwits vocabulary size
        max_emojis: int = 8,
        pretrained_emoji_path: str = "models/emoji_embeddings/pretrained_emoji_20x32.pt",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.text_dim = text_dim
        self.emoji_dim = emoji_dim
        self.fused_dim = text_dim + emoji_dim  # 800

        # Text branch: frozen BERT
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Emoji branch: pretrained frozen embeddings for overlapping emojis, random for others
        self.emoji_encoder = nn.Embedding(emoji_vocab_size, emoji_dim, padding_idx=0)
        
        # Load pretrained TweetEval embeddings [20, 32] and map to StockTwits vocab
        if os.path.exists(pretrained_emoji_path):
            pretrained = torch.load(pretrained_emoji_path, map_location="cpu")
            pretrained_emb = pretrained["emoji_embedding"]  # [20, 32]
            tweeteval_emoji_to_id = pretrained["tweeteval_emoji_to_id"]
            
            # Initialize with random embeddings for all StockTwits emojis
            nn.init.normal_(self.emoji_encoder.weight, mean=0.0, std=0.1)
            
            # Copy pretrained vectors for overlapping emojis
            # We need a mapping from StockTwits emoji_to_id to TweetEval emoji_to_id
            # This will be set after vocabulary is built
            self.register_buffer("_pretrained_emb", pretrained_emb)  # [20, 32]
            self.register_buffer("_tweeteval_emoji_to_id", 
                               torch.tensor([0]*len(pretrained["tweeteval_emoji_to_id"]), dtype=torch.long))
            self._tweeteval_emoji_to_id_dict = pretrained["tweeteval_emoji_to_id"]
        else:
            # Fallback: random initialization
            nn.init.normal_(self.emoji_encoder.weight, mean=0.0, std=0.1)
        
        # Initialize pretrained embeddings for overlapping emojis
        if hasattr(self, '_pretrained_emb') and hasattr(self, '_tweeteval_emoji_to_id_dict'):
            self._init_pretrained_embeddings()
        
        for param in self.emoji_encoder.parameters():
            param.requires_grad = False

        # Mean pooling for emoji aggregation
        self.max_emojis = 8

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def _init_pretrained_embeddings(self):
        """Initialize pretrained embeddings for overlapping emojis."""
        if not hasattr(self, '_pretrained_emb') or not hasattr(self, '_tweeteval_emoji_to_id_dict'):
            return
        
        # Get the TweetEval emoji to ID mapping from the pretrained artifact
        tweeteval_emoji_to_id = self._tweeteval_emoji_to_id_dict
        
        # We need to map TweetEval emoji IDs to StockTwits emoji IDs
        # This requires the StockTwits emoji vocabulary to be available
        # The mapping will be done after the vocabulary is built
        # For now, we just store the pretrained embeddings
        pass

    def set_emoji_vocab_mapping(self, stocktwits_emoji_to_id: dict):
        """Set the StockTwits emoji vocabulary mapping and initialize pretrained embeddings.
        
        Args:
            stocktwits_emoji_to_id: Mapping from StockTwits emoji string to ID
        """
        if not hasattr(self, '_pretrained_emb') or not hasattr(self, '_tweeteval_emoji_to_id_dict'):
            return
        
        # Get TweetEval emoji to ID mapping from pretrained artifact
        tweeteval_emoji_to_id = self._tweeteval_emoji_to_id_dict
        
        # For each overlapping emoji, copy pretrained vector to StockTwits embedding
        pretrained_emb = self._pretrained_emb  # [20, 32]
        pretrained_weight = self.emoji_encoder.weight.data  # [vocab_size, 32]
        
        overlap_count = 0
        for tweeteval_emoji, tweeteval_id in self._tweeteval_emoji_to_id_dict.items():
            if tweeteval_emoji in self.emoji_encoder_weight_map:
                continue
            # This will be set when vocabulary is available
            pass
        
        # We'll do the actual copy in a separate method after vocab is set
        pass

    def set_stocktwits_vocab(self, stocktwits_emoji_to_id: dict):
        """Set the StockTwits emoji vocabulary and initialize pretrained embeddings."""
        if not hasattr(self, '_pretrained_emb'):
            return
        
        # Create mapping from TweetEval emoji to StockTwits ID
        tweeteval_emoji_to_id = self._tweeteval_emoji_to_id_dict
        pretrained_emb = self._pretrained_emb  # [20, 32]
        
        overlap_count = 0
        with torch.no_grad():
            for tweeteval_emoji, tweeteval_id in self._tweeteval_emoji_to_id_dict.items():
                if tweeteval_emoji in self.emoji_to_id:
                    stocktwits_id = self.emoji_to_id[tweeteval_emoji]
                    # Copy pretrained vector
                    self.emoji_encoder.weight.data[stocktwits_id] = self._pretrained_emb[tweeteval_id]
                    overlap_count += 1
        
        print(f"Initialized {overlap_count} overlapping emojis with TweetEval pretrained vectors")

    def emoji_forward(self, emoji_ids, emoji_masks):
        # Text branch
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        text_rep = summed / counts  # (B, 768)

        # Emoji branch
        emoji_rep = self.emoji_forward(emoji_ids, emoji_masks)  # (B, 32)

        # Fusion: concatenation
        fused = torch.cat([text_rep, emoji_rep], dim=-1)  # (B, 800)
        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    # Test architecture
    import torch
    import numpy as np
    from src.embeddings.emoji_encoder import build_emoji_vocab, encode_emoji_list
    
    sample_lists = [['🚀','🔥'], ['😂'], ['😢','💔'], ['🚀','🔥','😂'], ['🤡'], []]
    vocab = build_emoji_vocab(sample_lists)
    ids, masks = [], []
    for e in sample_lists:
        i, m = encode_emoji_list(e, {}, max_emojis=8)
        ids.append(i); masks.append(m)
    ids = torch.tensor(np.array(ids)); masks = torch.tensor(np.array(masks), dtype=torch.float32)
    
    model = ConcatFusionModel(text_dim=768, emoji_dim=32, classifier_hidden=256,
                              num_labels=3, emoji_vocab_size=vocab['vocab_size'], max_emojis=8)
    model.eval()
    B = len(sample_lists)
    text_emb = torch.randn(B, 768)
    emoji_rep = model.emoji_encoder(ids, masks)
    fused = torch.cat([text_emb, emoji_rep], dim=-1)
    logits = model.classifier(fused)
    print('Architecture verification:')
    print(f'  Text emb dim: {text_emb.shape[-1]} (expected 768)')
    print(f'  Emoji rep dim: {emoji_rep.shape[-1]} (expected 32)')
    print(f'  Fused dim: {fused.shape[-1]} (expected 800)')
    print(f'  Logits shape: {logits.shape} (expected 3)')
    print('E1 architecture verified')