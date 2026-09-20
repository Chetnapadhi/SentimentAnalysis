"""E2 — text + PRETRAINED frozen emoji embedding, concatenation fusion.

Why this file exists alongside ``src/train_e2.py``
--------------------------------------------------
``src/train_e2.py`` prepares the datasets and then ends with a hard
``assert False`` — it never calls a training loop or an evaluation routine, so
E2 produces no ``metrics.json``, no ``predictions.csv`` and no checkpoint. It
cannot be completed by flipping an environment variable.

This module supplies the missing training/evaluation path, mirroring
``src/train_e3.py`` exactly (same seed, class weights, optimizer, batch size,
max_length, metrics and model-selection criterion) so E2 stays comparable with
the rest of the tournament. ``src/train_e2.py`` is left untouched.

Controlled-experiment contract (identical to E1 except the emoji init):
- Same train/validation/test rows and labels as E0-E5.
- Text branch receives ONLY ``text_without_emoji``.
- Emoji branch receives ONLY ``emoji_list``.
- Emoji embeddings are PRETRAINED (TweetEval-derived) and FROZEN, which is the
  single difference from E1's randomly-initialised trainable embeddings.

Requires ``models/emoji_embeddings/stocktwits_emoji_embedding_e2_1610x32.pt``,
built by ``python -m src.data.build_e2_emoji_artifact``.
"""

from __future__ import annotations

import os

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
from src.models.e2_concat_fusion import E2ConcatFusionModel

PRETRAINED_EMOJI_PATH = "models/emoji_embeddings/stocktwits_emoji_embedding_e2_1610x32.pt"


def _forward_fn(model, text_emb, emoji_ids, emoji_masks):
    return model.forward_from_cache(text_emb, emoji_ids, emoji_masks)


def main() -> None:
    if not os.path.exists(PRETRAINED_EMOJI_PATH):
        raise FileNotFoundError(
            f"Missing pretrained emoji artifact: {PRETRAINED_EMOJI_PATH}\n"
            "Build it first:  python -m src.data.build_e2_emoji_artifact"
        )

    ctx = prepare_experiment("E2")
    cfg = ctx["cfg"]
    device = ctx["device"]
    vocab = ctx["vocab"]
    out_dir = ctx["out_dir"]
    train_df, val_df, test_df = ctx["train_df"], ctx["val_df"], ctx["test_df"]

    artifact = torch.load(PRETRAINED_EMOJI_PATH, map_location="cpu", weights_only=False)
    pretrained = artifact["emoji_embedding"]
    assert pretrained.shape == (vocab["vocab_size"], cfg["emoji"]["embedding_dim"]), (
        f"Artifact shape {tuple(pretrained.shape)} does not match vocab "
        f"({vocab['vocab_size']}, {cfg['emoji']['embedding_dim']})"
    )
    print(
        f"Pretrained emoji artifact: {tuple(pretrained.shape)} | "
        f"{artifact.get('tweet_eval_pretrained_count', '?')} transferred from TweetEval, "
        f"{artifact.get('random_initialized_count', '?')} randomly initialised"
    )

    model = E2ConcatFusionModel(
        model_name=cfg["text_model"]["name"],
        text_dim=768,
        emoji_dim=cfg["emoji"]["embedding_dim"],
        num_labels=3,
        dropout=cfg["classifier_head"]["dropout"],
        classifier_hidden=cfg["classifier_head"]["classifier_hidden"],
        freeze_encoder=True,
        emoji_vocab_size=vocab["vocab_size"],
        max_emojis=8,
        pretrained_emoji_path="",  # weights are copied in explicitly below
    )
    model.emoji_encoder.weight.data.copy_(pretrained)
    model.emoji_encoder.weight.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} (BERT and emoji embedding frozen)")

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_model"]["name"])
    encoder = model.encoder.to(device)
    cache_dir = os.path.join(out_dir, "embeddings_cache")

    splits = {}
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        splits[name] = get_or_cache_bert_embeddings(
            df, name, tokenizer, encoder,
            cfg["text_model"]["max_length"], cfg["training"]["batch_size"],
            device, cache_dir,
        )

    e_ids, e_masks = {}, {}
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        e_ids[name], e_masks[name] = encode_emojis(df, vocab["emoji_to_id"])

    train_ds = EmojiDataset(splits["train"], e_ids["train"], e_masks["train"],
                            train_df["label"].to_numpy())
    val_ds = EmojiDataset(splits["validation"], e_ids["validation"], e_masks["validation"],
                          val_df["label"].to_numpy())
    test_ds = EmojiDataset(splits["test"], e_ids["test"], e_masks["test"],
                           test_df["label"].to_numpy())

    train_result = train_loop(
        model, _forward_fn, train_ds, val_ds,
        cfg, device, ctx["class_weights"], out_dir,
    )
    evaluate(
        model, _forward_fn, test_ds, test_df, train_result,
        cfg, device, out_dir, "E2",
        train_df=train_df, val_df=val_df,
    )


if __name__ == "__main__":
    if os.environ.get("RUN_E2") == "1":
        main()
    else:
        print("E2 is prepared but disabled. Set RUN_E2=1 to train.")
