"""E5 — text + pretrained emoji embedding, gated fusion.

MIRRORS E0–E4 (seed, class weights, optimizer, batch, max_length, metrics,
model-selection criterion) except it uses a learned gate to modulate emoji
contribution before concatenation.

Controlled-experiment contract:
- Same train/validation/test examples and labels as E0–E4.
- Text branch receives ONLY ``text_without_emoji``.
- Emoji branch receives ONLY ``emoji_list``.
- ``original_text`` is never passed to BERT.
- Emoji embeddings are PRETRAINED on TweetEval emoji task and FROZEN.
- Gate values are reported as interpretability signals, not causal explanations.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from src.common_pipeline import (
    EmojiDataset,
    encode_emojis,
    evaluate,
    get_or_cache_bert_embeddings,
    prepare_experiment,
    train_loop,
)
from src.models.e5_gated_fusion import GatedFusionModel

PRETRAINED_EMOJI_PATH = "models/emoji_embeddings/stocktwits_emoji_embedding_e2_1610x32.pt"


def _forward_fn(model, text_emb, emoji_ids, emoji_masks):
    return model.forward_from_cache(text_emb, emoji_ids, emoji_masks)


def _save_gate_analysis(model, test_ds, test_df, device, out_dir):
    """Collect and save 32-dimensional gate value analysis from the test set."""
    from torch.utils.data import DataLoader

    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    all_gates = []

    model.eval()
    with torch.no_grad():
        for text_emb, e_ids, e_mask, _ in loader:
            text_emb = text_emb.to(device)
            e_ids, e_mask = e_ids.to(device), e_mask.to(device)
            _ = model.forward_from_cache(text_emb, e_ids, e_mask)
            all_gates.append(model.last_gate_values.cpu().numpy())

    gates = np.concatenate(all_gates, axis=0)  # (N, 32) - do not squeeze to scalar
    assert gates.ndim == 2 and gates.shape[1] == 32, f"Expected (N, 32), got {gates.shape}"

    # Per-class gate statistics
    labels = test_df["label"].to_numpy()
    label_names = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    gate_stats = {
        "gate_dim": 32,
        "overall_mean": float(gates.mean()),
        "overall_std": float(gates.std()),
        "overall_median": float(np.median(gates)),
        "dimension_means": gates.mean(axis=0).tolist(),
    }
    for c in [0, 1, 2]:
        mask = labels == c
        if mask.sum() > 0:
            cls_gates = gates[mask]
            gate_stats[f"{label_names[c]}_mean"] = float(cls_gates.mean())
            gate_stats[f"{label_names[c]}_std"] = float(cls_gates.std())
            gate_stats[f"{label_names[c]}_median"] = float(np.median(cls_gates))
            gate_stats[f"{label_names[c]}_dimension_means"] = cls_gates.mean(axis=0).tolist()

    # Gate histogram bins across all gate elements
    hist_counts, hist_edges = np.histogram(gates, bins=20, range=(0, 1))
    gate_stats["histogram_counts"] = hist_counts.tolist()
    gate_stats["histogram_edges"] = hist_edges.tolist()

    path = os.path.join(out_dir, "gate_analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gate_stats, f, indent=2)
    print(f"Saved 32-d gate analysis -> {path}")

    # Print summary
    print(f"\n=== E5 32-d Gate Analysis ===")
    print(f"Overall gate: mean={gate_stats['overall_mean']:.4f} "
          f"std={gate_stats['overall_std']:.4f} "
          f"median={gate_stats['overall_median']:.4f}")
    for c in [0, 1, 2]:
        name = label_names[c]
        print(f"  {name}: mean={gate_stats[f'{name}_mean']:.4f} "
              f"std={gate_stats[f'{name}_std']:.4f}")


def main() -> None:
    ctx = prepare_experiment("E5")
    cfg = ctx["cfg"]
    device = ctx["device"]
    vocab = ctx["vocab"]
    class_weights = ctx["class_weights"]
    out_dir = ctx["out_dir"]
    train_df, val_df, test_df = ctx["train_df"], ctx["val_df"], ctx["test_df"]

    # Verify pretrained emoji artifact exists
    if not os.path.exists(PRETRAINED_EMOJI_PATH):
        raise FileNotFoundError(
            f"Pretrained emoji embedding not found at {PRETRAINED_EMOJI_PATH}. "
            "Run the E2 embedding transfer script first."
        )

    # Build model (pretrained frozen emoji + gated fusion)
    model = GatedFusionModel(
        model_name=cfg["text_model"]["name"],
        text_dim=768,
        emoji_dim=cfg["emoji"]["embedding_dim"],
        num_labels=3,
        dropout=cfg["classifier_head"]["dropout"],
        classifier_hidden=cfg["classifier_head"]["classifier_hidden"],
        freeze_encoder=True,
        emoji_vocab_size=vocab["vocab_size"],
        max_emojis=8,
        pretrained_emoji_path=PRETRAINED_EMOJI_PATH,
    )

    # Featurize text (frozen BERT, cached)
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_model"]["name"])
    encoder = model.encoder.to(device)
    cache_dir = os.path.join(out_dir, "embeddings_cache")

    X_train = get_or_cache_bert_embeddings(
        train_df, "train", tokenizer, encoder,
        cfg["text_model"]["max_length"], cfg["training"]["batch_size"],
        device, cache_dir,
    )
    X_val = get_or_cache_bert_embeddings(
        val_df, "validation", tokenizer, encoder,
        cfg["text_model"]["max_length"], cfg["training"]["batch_size"],
        device, cache_dir,
    )
    X_test = get_or_cache_bert_embeddings(
        test_df, "test", tokenizer, encoder,
        cfg["text_model"]["max_length"], cfg["training"]["batch_size"],
        device, cache_dir,
    )

    # Encode emojis
    e_train_ids, e_train_mask = encode_emojis(train_df, vocab["emoji_to_id"])
    e_val_ids, e_val_mask = encode_emojis(val_df, vocab["emoji_to_id"])
    e_test_ids, e_test_mask = encode_emojis(test_df, vocab["emoji_to_id"])

    # Datasets
    train_ds = EmojiDataset(X_train, e_train_ids, e_train_mask, train_df["label"].to_numpy())
    val_ds = EmojiDataset(X_val, e_val_ids, e_val_mask, val_df["label"].to_numpy())
    test_ds = EmojiDataset(X_test, e_test_ids, e_test_mask, test_df["label"].to_numpy())

    # Train
    train_result = train_loop(
        model, _forward_fn, train_ds, val_ds,
        cfg, device, class_weights, out_dir,
    )

    # Evaluate
    evaluate(
        model, _forward_fn, test_ds, test_df, train_result,
        cfg, device, out_dir, "E5",
        train_df=train_df, val_df=val_df,
    )

    # Gate analysis (E5-specific)
    _save_gate_analysis(model, test_ds, test_df, device, out_dir)


if __name__ == "__main__":
    if os.environ.get("RUN_E5") == "1":
        main()
    else:
        print("E5 is prepared but disabled. Set RUN_E5=1 to train (after approval).")
