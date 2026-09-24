"""Streamlit-free loading and batched inference for the Phase 2 emotion models.

Shared by the dashboard (``dashboard/inference.py`` wraps these in
``st.cache_resource``) and by offline evaluation
(``src/analysis/evaluate_hybrid.py``), so both run exactly the same code.

Emoji-vocabulary safety
-----------------------
An EM3/EM5 checkpoint is only meaningful with the exact ``emoji -> id``
mapping it was trained with. Previously the dashboard fell back to
``emoji_vocab_emotion_unified.json`` for GoEmotions; only 1 of the 117 emojis
shared with the real training vocabulary had the same id there, so nearly
every emoji reached the model as the wrong embedding row. Resolution order
is now:

1. ``emoji_to_id`` stored inside the checkpoint (new training runs save it)
2. ``models/emoji_embeddings/emoji_vocab_<dataset>.json`` -- the train-split
   vocabulary, rebuilt deterministically by ``build_vocab``
3. refuse to load if the vocabulary size disagrees with the checkpoint's
   embedding table, instead of silently scrambling emoji ids.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.data.preprocessing import extract_emojis, remove_emojis
from src.embeddings.emoji_encoder import encode_emoji_list, load_vocab

ROOT = Path(__file__).resolve().parents[1]
EMOTION_BACKBONE = "cardiffnlp/twitter-roberta-base"
EMOTION_LABELS = {
    "goemotions": ["anger", "disgust", "fear", "joy", "sadness", "surprise"],
    "tweeteval": ["anger", "joy", "optimism", "sadness"],
}


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(dataset: str, model_key: str, mode: str = "finetune") -> Path:
    return ROOT / "results" / "emotion" / dataset / f"{model_key}_{mode}" / "best_model.pt"


def resolve_emoji_vocab(dataset: str, checkpoint: dict | None = None) -> dict:
    """Return the emoji->id mapping a checkpoint was trained with."""
    if checkpoint and checkpoint.get("emoji_to_id"):
        mapping = checkpoint["emoji_to_id"]
        return {"emoji_to_id": mapping, "vocab_size": len(mapping), "source": "checkpoint"}

    path = ROOT / "models" / "emoji_embeddings" / f"emoji_vocab_{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Rebuild it deterministically with:\n"
            f"  python -c \"from src.emotion_pipeline import build_vocab; "
            f"from src.data.emotion_adapter import load_emotion_splits; "
            f"build_vocab(load_emotion_splits('{dataset}')[0], '{dataset}')\""
        )
    vocab = load_vocab(str(path))
    vocab["vocab_size"] = len(vocab["emoji_to_id"])
    vocab["source"] = path.name
    return vocab


@lru_cache(maxsize=1)
def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(EMOTION_BACKBONE)


@lru_cache(maxsize=8)
def load_emotion_model(dataset: str, model_key: str = "EM3", mode: str = "finetune"):
    """Load a trained Phase 2 checkpoint. Returns (model, vocab)."""
    from src.emotion_pipeline import unfreeze_top_layers
    from src.models.e3_attention_fusion import AttentionFusionModel
    from src.models.e5_gated_fusion import GatedFusionModel
    from src.models.text_only import TextOnlyModel

    path = checkpoint_path(dataset, model_key, mode)
    if not path.exists():
        raise FileNotFoundError(f"Missing emotion checkpoint: {path}")
    ckpt = torch.load(path, map_location=device(), weights_only=False)
    state = ckpt.get("trainable_state_dict") or ckpt["model_state_dict"]

    vocab = resolve_emoji_vocab(dataset, ckpt) if model_key != "EM0" else {"emoji_to_id": {}, "vocab_size": 1}
    if "emoji_encoder.weight" in state:
        rows = state["emoji_encoder.weight"].shape[0]
        if rows != vocab["vocab_size"]:
            raise ValueError(
                f"{path.parent.name}: checkpoint has {rows} emoji rows but vocabulary "
                f"'{vocab.get('source')}' has {vocab['vocab_size']}. Refusing to load: "
                "emoji ids would be scrambled."
            )

    labels = EMOTION_LABELS[dataset]
    common = dict(model_name=EMOTION_BACKBONE, num_labels=len(labels), dropout=0.3, freeze_encoder=True)
    if model_key == "EM0":
        model = TextOnlyModel(hidden_size=768, **common)
    elif model_key == "EM3":
        model = AttentionFusionModel(text_dim=768, emoji_dim=32, classifier_hidden=256,
                                     emoji_vocab_size=vocab["vocab_size"], max_emojis=8,
                                     init_mode="random", **common)
    elif model_key == "EM5":
        model = GatedFusionModel(text_dim=768, emoji_dim=32, classifier_hidden=256,
                                 emoji_vocab_size=vocab["vocab_size"], max_emojis=8,
                                 freeze_emoji=False, **common)
    else:
        raise ValueError(f"Unknown emotion model: {model_key}")

    if mode == "finetune":
        unfreeze_top_layers(model.encoder, 2)
    model.load_state_dict(state, strict=False)
    model.to(device()).eval()
    return model, vocab


def predict_proba(texts: list[str], dataset: str, model_key: str = "EM3",
                  mode: str = "finetune", batch_size: int = 32) -> np.ndarray:
    """Model-only class probabilities, shape (N, n_labels).

    Inputs are prepared exactly as in training: the text branch sees the
    emoji-free sentence, the emoji branch sees the distinct emoji list.
    """
    model, vocab = load_emotion_model(dataset, model_key, mode)
    tok = load_tokenizer()
    dev = device()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok([remove_emojis(t) for t in batch], padding=True, truncation=True,
                      max_length=64, return_tensors="pt").to(dev)
            if model_key == "EM0":
                logits = model(enc["input_ids"], enc["attention_mask"])
            else:
                ids, masks = zip(*(encode_emoji_list(extract_emojis(t), vocab["emoji_to_id"], 8)
                                   for t in batch))
                logits = model(enc["input_ids"], enc["attention_mask"],
                               torch.tensor(ids, dtype=torch.long, device=dev),
                               torch.tensor(masks, dtype=torch.float32, device=dev))
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)
