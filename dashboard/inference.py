"""Cached model loading and live inference engine for E0, E3, and E5."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer
import streamlit as st

from dashboard.data import get_project_root
from src.data.preprocessing import extract_emojis, remove_emojis
from src.embeddings.emoji_encoder import encode_emoji_list, load_vocab


# Primary sentiment labels + intuitive financial aliases
SENTIMENT_NAMES = {0: "Negative (Bearish)", 1: "Neutral", 2: "Positive (Bullish)"}
SHORT_SENTIMENT_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_MAP = {0: "Bearish", 1: "Neutral", 2: "Bullish"}


@st.cache_resource
def load_cached_tokenizer():
    """Load and cache the BERT tokenizer."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@st.cache_resource
def load_cached_emoji_vocab() -> dict:
    """Load and cache the training-derived emoji vocabulary."""
    path = get_project_root() / "models" / "emoji_embeddings" / "emoji_vocab.json"
    return load_vocab(str(path))


def get_inference_device() -> torch.device:
    """Return cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model_e3():
    """Load and cache the E3 Attention Fusion Model."""
    from src.models.e3_attention_fusion import AttentionFusionModel
    root = get_project_root()
    vocab = load_cached_emoji_vocab()
    device = get_inference_device()

    model = AttentionFusionModel(
        model_name="bert-base-uncased",
        text_dim=768,
        emoji_dim=32,
        num_labels=3,
        dropout=0.3,
        classifier_hidden=256,
        freeze_encoder=True,
        emoji_vocab_size=vocab["vocab_size"],
        max_emojis=8,
        init_mode="random",
    )

    ckpt_path = root / "results" / "E3" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing E3 checkpoint at {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_model_e0():
    """Load and cache the E0 Text-Only Baseline Model."""
    from src.models.text_only import TextOnlyModel
    root = get_project_root()
    device = get_inference_device()

    model = TextOnlyModel(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_labels=3,
        dropout=0.1,
        freeze_encoder=True,
    )

    ckpt_path = root / "results" / "E0" / "best_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_model_e5():
    """Load and cache the E5 Gated Fusion Model."""
    from src.models.e5_gated_fusion import GatedFusionModel
    root = get_project_root()
    vocab = load_cached_emoji_vocab()
    device = get_inference_device()

    model = GatedFusionModel(
        model_name="bert-base-uncased",
        text_dim=768,
        emoji_dim=32,
        num_labels=3,
        dropout=0.3,
        classifier_hidden=256,
        freeze_encoder=True,
        emoji_vocab_size=vocab["vocab_size"],
        max_emojis=8,
    )

    ckpt_path = root / "results" / "E5" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing E5 checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_live_inference(
    text: str,
    model_name: str = "E3",
) -> dict[str, Any]:
    """Run full live inference pipeline on user input text."""
    if not text or not text.strip():
        raise ValueError("Please enter a social-media post.")

    # 1. Preprocessing (exact canonical logic)
    emojis = extract_emojis(text)
    text_no_emoji = remove_emojis(text)
    num_emojis = len(emojis)

    # 2. Tokenize text
    tokenizer = load_cached_tokenizer()
    enc = tokenizer(
        text_no_emoji,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    device = get_inference_device()
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 3. Encode emojis
    vocab = load_cached_emoji_vocab()
    e_ids, e_mask = encode_emoji_list(emojis, vocab["emoji_to_id"], max_emojis=8)
    emoji_ids_t = torch.tensor([e_ids], dtype=torch.long, device=device)
    emoji_masks_t = torch.tensor([e_mask], dtype=torch.float32, device=device)

    # 4. Model execution
    with torch.no_grad():
        if model_name == "E0":
            model = load_model_e0()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        elif model_name == "E5":
            model = load_model_e5()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                emoji_ids=emoji_ids_t,
                emoji_masks=emoji_masks_t,
            )
        else:  # Default to E3
            model = load_model_e3()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                emoji_ids=emoji_ids_t,
                emoji_masks=emoji_masks_t,
            )

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

    return {
        "original_text": text,
        "text_without_emoji": text_no_emoji,
        "emojis": emojis,
        "num_emojis": num_emojis,
        "model": model_name,
        "pred_label": LABEL_MAP[pred_idx],
        "sentiment_label": SENTIMENT_NAMES[pred_idx],
        "short_label": SHORT_SENTIMENT_NAMES[pred_idx],
        "pred_idx": pred_idx,
        "probs": {
            "Bearish": float(probs[0]),
            "Neutral": float(probs[1]),
            "Bullish": float(probs[2]),
        },
        "confidence": float(probs[pred_idx]),
    }
