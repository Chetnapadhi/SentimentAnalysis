"""Pretrain emoji embeddings on TweetEval emoji prediction task.

This script trains emoji embeddings on the TweetEval emoji prediction task
(20 emoji classes) and saves the learned embeddings for use in E2.

The learned emoji embeddings are trained ONLY on TweetEval data, completely
separate from the StockTwits sentiment data used in E0/E1/E2.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import accuracy_score, f1_score

from src.embeddings.emoji_encoder import build_emoji_vocab
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMOJI_DIM = 32
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
        
        # Emoji embedding (to be pretrained)
        # Note: vocab_size here is the TweetEval emoji vocabulary size (20)
        self.emoji_encoder = nn.Embedding(vocab_size, emoji_dim, padding_idx=None)
        nn.init.normal_(self.emoji_encoder.weight, mean=0.0, std=0.1)
        
        # Text encoder
        config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(MODEL_NAME, config=config)
        
        # Classification head for emoji prediction (20 TweetEval emoji classes)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),  # 20 TweetEval emoji classes
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
        summed = (last_hidden * mask).sum(dim=1)
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
            max_length=max_length,
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
    set_seed(SEED)
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Data loaders
    train_loader, val_loader, test_loader = prepare_tweeteval_data(tokenizer)
    
    # Model
    model = TweetEvalEmojiModel(
        vocab_size=num_tweeteval_emojis,  # 20 TweetEval emoji classes
        emoji_dim=EMOJI_DIM,
        num_labels=num_tweeteval_emojis
    ).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_f1 = 0.0
    best_state = None
    
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Pretrain Epoch {epoch}/{PRETRAIN_EPOCHS}"):
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
        print(f"Epoch {epoch}/{PRETRAIN_EPOCHS}: val_acc={val_acc:.4f}, val_macro_f1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                "emoji_embedding": model.emoji_encoder.weight.data.clone(),
                "emoji_dim": EMOJI_DIM,
                "vocab_size": num_tweeteval_emojis,
                "label_names": label_names,
            }
            torch.save(best_state, "models/emoji_embeddings/pretrained_emoji_32d.pt")
            print(f"  Saved best pretrained embeddings (val_macro_f1={val_f1:.4f})")
    
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    return best_state


if __name__ == "__main__":
    print("Starting TweetEval emoji pretraining...")
    pretrain_emoji_embeddings()
    print("Pretraining complete!")