"""Produce the missing E2 emoji-embedding artifact (1610 x 32) for E2/E4/E5.

Why this file exists
====================
``src/train_e2.py``, ``src/train_e4.py`` and ``src/train_e5.py`` all load

    models/emoji_embeddings/stocktwits_emoji_embedding_e2_1610x32.pt

but **nothing in the repository ever wrote it**. ``src/train_e2_pretrain.py``
saves a different file (``pretrained_emoji_32d.pt``, shape ``[20, 32]``) and no
step maps the 20 TweetEval emoji vectors into the 1610-entry StockTwits emoji
vocabulary. Without this artifact E2, E4 and E5 cannot run at all.

Two defects in ``train_e2_pretrain.py`` are corrected here
==========================================================
1. **The emoji embedding never received a gradient.** In that script
   ``self.emoji_encoder`` is constructed and then read back out to be saved,
   but it is never referenced in ``forward()`` — which uses only
   ``encoder -> mean-pool -> classifier``. The saved "pretrained" matrix was
   therefore bit-identical to its random initialisation (verified: gradient is
   ``None`` and the weights are unchanged after optimisation steps). E2 and E4
   would have been comparing "pretrained" against "random" embeddings that were
   both random, making the E1-vs-E2 and E3-vs-E4 contrasts vacuous.

   Here the emoji embedding **is** the output layer: the model scores a tweet
   against each emoji by a dot product between a learned 32-d projection of the
   frozen text vector and that emoji's 32-d embedding. The embedding is in the
   computational graph, so it genuinely learns a space in which emojis used in
   similar contexts sit close together — which is what "pretrained emoji
   embedding" is supposed to mean.

2. **Full-BERT fine-tuning was computationally infeasible here.** The original
   optimises ``model.parameters()`` (all 110M BERT weights) for 10 epochs over
   45,000 tweets padded to a fixed 128 tokens — on CPU that is well over a day.
   This script keeps BERT **frozen** and caches its outputs, which also matches
   the frozen-encoder methodology used everywhere else in the project, and runs
   in minutes.

Transfer to the StockTwits vocabulary
=====================================
TweetEval emoji has 20 classes, each named by an emoji character. The
StockTwits emoji vocabulary has 1610 entries (id 0 = ``<UNK>``). Entries whose
emoji appears in TweetEval receive the learned vector; all others are randomly
initialised with the same ``N(0, 0.1)`` distribution the models use. Counts of
each are recorded in the artifact so the split is auditable.

Usage:
    python -m src.data.build_e2_emoji_artifact
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.seed import set_seed

EMOJI_DIM = 32
SEED = 42
MODEL_NAME = "bert-base-uncased"
OUT_PATH = "models/emoji_embeddings/stocktwits_emoji_embedding_e2_1610x32.pt"
VOCAB_PATH = "models/emoji_embeddings/emoji_vocab.json"
CACHE_DIR = "results/_shared_embeddings/tweeteval_emoji"


class EmojiEmbeddingLearner(nn.Module):
    """Score text against emoji embeddings by dot product.

    ``logits[i, e] = <proj(text_i), emoji_embedding[e]> / sqrt(d)``

    Because the emoji embedding forms the output layer, every emoji vector
    receives gradient on every batch — which is precisely what the original
    pretraining script failed to do.
    """

    def __init__(self, num_emojis: int, text_dim: int = 768, emoji_dim: int = EMOJI_DIM):
        super().__init__()
        self.emoji_dim = emoji_dim
        self.proj = nn.Sequential(
            nn.Linear(text_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, emoji_dim),
        )
        self.emoji_embedding = nn.Parameter(torch.empty(num_emojis, emoji_dim))
        nn.init.normal_(self.emoji_embedding, mean=0.0, std=0.1)

    def forward(self, text_vec: torch.Tensor) -> torch.Tensor:
        z = self.proj(text_vec)                                    # (B, D)
        return z @ self.emoji_embedding.t() / (self.emoji_dim ** 0.5)


def _featurize(texts, tokenizer, encoder, device, batch_size=64, max_length=64):
    out = []
    encoder.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Featurize"):
            enc = tokenizer(
                texts[i : i + batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            hidden = encoder(**enc, return_dict=True).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(
                ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
            )
    return np.concatenate(out, axis=0)


def _cached_features(split, texts, tokenizer, encoder, device):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{split}.npy")
    if os.path.exists(path):
        print(f"  [{split}] cached features loaded")
        return np.load(path)
    feats = _featurize(texts, tokenizer, encoder, device)
    np.save(path, feats)
    return feats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default=OUT_PATH)
    args = p.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"Device: {device}")

    ds = load_dataset("cardiffnlp/tweet_eval", "emoji")
    label_names = ds["train"].features["label"].names
    print(f"TweetEval emoji classes ({len(label_names)}): {' '.join(label_names)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    for param in encoder.parameters():
        param.requires_grad = False

    print("Featurizing TweetEval with frozen BERT (cached)...")
    splits = {}
    for split in ["train", "validation", "test"]:
        feats = _cached_features(split, list(ds[split]["text"]), tokenizer, encoder, device)
        splits[split] = (feats, np.array(ds[split]["label"]))
        print(f"  {split:<11} {feats.shape}")

    model = EmojiEmbeddingLearner(len(label_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    def loader(split, shuffle):
        x, y = splits[split]
        return DataLoader(
            TensorDataset(torch.tensor(x, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.long)),
            batch_size=args.batch_size, shuffle=shuffle,
        )

    train_loader, val_loader = loader("train", True), loader("validation", False)
    before = model.emoji_embedding.detach().clone()
    best_f1, best_emb, best_epoch = -1.0, None, -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.extend(model(xb.to(device)).argmax(-1).cpu().numpy().tolist())
                trues.extend(yb.numpy().tolist())
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average="macro", zero_division=0)
        print(f"  epoch {epoch:>2}/{args.epochs} loss {total/len(train_loader):.4f} "
              f"val_acc {acc:.4f} val_macroF1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_emb = model.emoji_embedding.detach().clone()

    drift = (best_emb - before).abs().max().item()
    print(f"\nBest val macro-F1 {best_f1:.4f} (epoch {best_epoch})")
    print(f"Max |change| in emoji embedding vs init: {drift:.4f}")
    if drift < 1e-6:
        raise RuntimeError(
            "Emoji embedding did not move from its initialisation — it is not "
            "in the computational graph. Refusing to save a fake artifact."
        )

    # ---- transfer into the StockTwits vocabulary -------------------------
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    emoji_to_id, vocab_size = vocab["emoji_to_id"], vocab["vocab_size"]

    generator = torch.Generator().manual_seed(SEED)
    transferred = torch.empty(vocab_size, EMOJI_DIM)
    nn.init.normal_(transferred, mean=0.0, std=0.1, generator=generator)

    matched, matched_emojis = 0, []
    for idx, name in enumerate(label_names):
        target = emoji_to_id.get(name)
        if target is not None:
            transferred[target] = best_emb[idx].cpu()
            matched += 1
            matched_emojis.append(name)

    transferred[0] = 0.0  # id 0 is <UNK>/pad; keep it a clean zero vector

    artifact = {
        "emoji_embedding": transferred,
        "vocab_size": vocab_size,
        "embedding_dim": EMOJI_DIM,
        "seed": SEED,
        "tweet_eval_pretrained_count": matched,
        "random_initialized_count": vocab_size - matched,
        "matched_emojis": matched_emojis,
        "pretrain_val_macro_f1": best_f1,
        "pretrain_best_epoch": best_epoch,
        "source": "TweetEval emoji (20-class) via dot-product emoji embedding, frozen BERT",
        "provenance": (
            "Built by src/data/build_e2_emoji_artifact.py. Replaces "
            "src/train_e2_pretrain.py, whose emoji embedding was never used in "
            "forward() and so was saved at its random initialisation."
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(artifact, args.out)

    print(f"\nTransferred {matched}/{len(label_names)} TweetEval emojis into the "
          f"{vocab_size}-entry StockTwits vocabulary")
    print(f"  matched: {' '.join(matched_emojis)}")
    print(f"  randomly initialised: {vocab_size - matched}")
    print(f"Saved {tuple(transferred.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
