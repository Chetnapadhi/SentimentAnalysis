"""E3 — text + random emoji embedding, attention-based fusion.

MIRRORS E0/E1 exactly (seed, class weights, optimizer, batch, max_length,
metrics, model-selection criterion) except it uses text-conditioned attention
fusion with randomly-initialized emoji embeddings.

Controlled-experiment contract:
- Same train/validation/test examples and labels as E0/E1.
- Text branch receives ONLY ``text_without_emoji``.
- Emoji branch receives ONLY ``emoji_list``.
- ``original_text`` is never passed to BERT.
- Emoji embeddings are RANDOMLY INITIALIZED (no pretrained embeddings).
"""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer

from src.common_pipeline import (
    EmojiDataset,
    compute_class_weights,
    encode_emojis,
    evaluate,
    get_or_cache_bert_embeddings,
    load_config,
    load_splits,
    prepare_experiment,
    train_loop,
)
from src.models.e3_attention_fusion import AttentionFusionModel


def _forward_fn(model, text_emb, emoji_ids, emoji_masks):
    return model.forward_from_cache(text_emb, emoji_ids, emoji_masks)


def main() -> None:
    ctx = prepare_experiment("E3")
    cfg = ctx["cfg"]
    device = ctx["device"]
    vocab = ctx["vocab"]
    class_weights = ctx["class_weights"]
    out_dir = ctx["out_dir"]
    train_df, val_df, test_df = ctx["train_df"], ctx["val_df"], ctx["test_df"]

    # Build model (random emoji init)
    model = AttentionFusionModel(
        model_name=cfg["text_model"]["name"],
        text_dim=768,
        emoji_dim=cfg["emoji"]["embedding_dim"],
        num_labels=3,
        dropout=cfg["classifier_head"]["dropout"],
        classifier_hidden=cfg["classifier_head"]["classifier_hidden"],
        freeze_encoder=True,
        emoji_vocab_size=vocab["vocab_size"],
        max_emojis=8,
        init_mode="random",
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
        cfg, device, out_dir, "E3",
        train_df=train_df, val_df=val_df,
    )


if __name__ == "__main__":
    if os.environ.get("RUN_E3") == "1":
        main()
    else:
        print("E3 is prepared but disabled. Set RUN_E3=1 to train (after approval).")