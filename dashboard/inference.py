"""Cached model loading and live inference for the Phase 2 emotion study.

Emotion inference is a hybrid: the trained EM0/EM3 model, the hand-labelled
emoji lexicon, and a sarcasm detector (see ``src/lexicon/``). Model loading
and batching live in ``src/inference.py`` so the dashboard and the offline
evaluation run identical code; this module only adds Streamlit caching.
"""

from __future__ import annotations

import json
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
from src.inference import EMOTION_BACKBONE, EMOTION_LABELS  # noqa: F401  (re-exported)


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
    vocab = load_vocab(str(path))
    vocab.setdefault("vocab_size", len(vocab["emoji_to_id"]))
    return vocab


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


# ---------------------------------------------------------------------------
# Phase 2: hybrid emotion inference
# ---------------------------------------------------------------------------

FUSION_CONFIG_PATH = get_project_root() / "data" / "lexicon" / "fusion_config.json"


@st.cache_resource
def load_fusion_config() -> dict:
    """Parameters chosen on validation by ``src/analysis/evaluate_hybrid.py``."""
    if FUSION_CONFIG_PATH.exists():
        return json.loads(FUSION_CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_resource
def load_emotion_model(dataset: str = "goemotions", model_key: str = "EM3"):
    """Trained Phase 2 checkpoint + the exact emoji vocabulary it was trained with."""
    from src.inference import load_emotion_model as _load
    return _load(dataset, model_key)


@st.cache_resource
def load_sarcasm_detector():
    """Pretrained irony classifier + rules; falls back to rules if unavailable."""
    from src.lexicon.sarcasm import SarcasmDetector
    cfg = load_fusion_config().get("sarcasm", {})
    return SarcasmDetector(
        backend=os.environ.get("SARCASM_BACKEND", "auto"),
        threshold=cfg.get("threshold", 0.5),
        rule_weight=cfg.get("rule_weight", 0.6),
    )


def run_emotion_inference(
    text: str, dataset: str = "goemotions", model_key: str = "EM3", mode: str = "hybrid",
) -> dict[str, Any]:
    """Predict an emotion.

    ``mode="hybrid"`` (default) fuses the trained model with the emoji lexicon
    and sarcasm detection. ``mode="model"`` returns the trained model alone.
    Both readings are always included in the result so the UI can show how
    the lexicon and sarcasm layers changed the answer.
    """
    from src.inference import predict_proba
    from src.lexicon.hybrid import FusionConfig, HybridEmotionPredictor

    if not text or not text.strip():
        raise ValueError("Please enter a sentence or social-media post.")
    labels = EMOTION_LABELS[dataset]
    load_emotion_model(dataset, model_key)                  # warm the Streamlit cache
    model_probs = predict_proba([text], dataset, model_key)[0]

    tuned = load_fusion_config().get("fusion", {}).get(f"{dataset}/{model_key}", {})
    sarcasm_cfg = load_fusion_config().get("sarcasm", {})
    cfg = FusionConfig(
        emoji_weight=tuned.get("emoji_weight", 1.5),
        sarcasm_shift=tuned.get("sarcasm_shift", 0.85),
        sarcasm_threshold=sarcasm_cfg.get("threshold", 0.5),
    )
    if mode == "model":
        cfg = FusionConfig(emoji_weight=0.0, sarcasm_shift=0.0)

    result = HybridEmotionPredictor(dataset, load_sarcasm_detector(), cfg).predict(text, model_probs)
    sarcasm = result.sarcasm
    detected = bool(sarcasm.is_sarcastic and mode == "hybrid")
    # Headline: the intended emotion when sarcasm is corroborated ("detected",
    # ~92% precise on held-out TweetEval irony), else the validated hybrid.
    # Measured cost on the emotion test sets: 0.2-0.4 accuracy points, because
    # their annotators often labelled the literal emotion of sarcastic posts.
    final = result.intended_probs if detected and result.intended_probs is not None else result.probs
    predicted = int(np.argmax(final))
    return {
        "original_text": text,
        "text_without_emoji": remove_emojis(text),
        "emojis": extract_emojis(text),
        "num_emojis": result.evidence.n_occurrences,
        "dataset": dataset,
        "model": model_key,
        "mode": mode,
        "labels": [label.title() for label in labels],
        # final answer
        "pred_label": labels[predicted].title(),
        "pred_idx": predicted,
        "probs": {label.title(): float(final[i]) for i, label in enumerate(labels)},
        "confidence": float(final[predicted]),
        # the trained model on its own, for comparison
        "model_label": result.model_label.title(),
        "model_probs": {label.title(): float(result.model_probs[i]) for i, label in enumerate(labels)},
        "surface_label": result.surface_label.title(),
        "validated_label": result.label.title(),
        # explanations
        "sarcasm": {
            "probability": float(sarcasm.probability),
            "level": sarcasm.level if mode == "hybrid" else "none",
            "is_sarcastic": detected,
            "reasons": sarcasm.reasons,
            "backend": sarcasm.backend,
        },
        "emoji_evidence": {
            "strength": float(result.evidence.strength),
            "mixed": bool(result.evidence.mixed),
            "top_label": result.evidence.top_label,
            "breakdown": result.evidence.breakdown,
        },
        "notes": result.notes,
    }
