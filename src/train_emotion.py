"""Phase 2 emotion experiment runner (EM0 / EM3 / EM5).

Usage
-----
    python -m src.train_emotion --dataset goemotions --model em0 --mode frozen
    python -m src.train_emotion --dataset goemotions --model em0 --mode finetune --unfreeze-layers 2

Models
------
``em0``  text-only baseline (no emoji branch at all)
``em3``  text-conditioned attention fusion over emoji embeddings
``em5``  32-d gated fusion over mean-pooled emoji embeddings

Modes
-----
``frozen``    backbone frozen, text embeddings cached once, only fusion + head
              train. EM0/EM3/EM5 then share a byte-identical text
              representation, so any difference between them is attributable
              to the emoji branch and nothing else.
``finetune``  top ``--unfreeze-layers`` encoder blocks train end-to-end at
              ``--encoder-lr`` alongside the head.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.common_pipeline import get_or_cache_bert_embeddings
from src.emotion_pipeline import (
    DEFAULT_BACKBONE,
    TokenizedEmotionDataset,
    encode_emojis,
    evaluate_model,
    make_collate_fn,
    prepare,
    trainable_report,
    train_model,
    unfreeze_top_layers,
)
from src.models.e3_attention_fusion import AttentionFusionModel
from src.models.e5_gated_fusion import GatedFusionModel
from src.models.text_only import TextOnlyModel

MODEL_CHOICES = ("em0", "em3", "em5")


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(
    model_key: str, *, backbone: str, num_labels: int, vocab_size: int,
    dropout: float, freeze_encoder: bool,
):
    """Instantiate the requested architecture with an N-way output head."""
    if model_key == "em0":
        return TextOnlyModel(
            model_name=backbone, hidden_size=768, num_labels=num_labels,
            dropout=dropout, freeze_encoder=freeze_encoder,
        )
    if model_key == "em3":
        return AttentionFusionModel(
            model_name=backbone, text_dim=768, emoji_dim=32,
            num_labels=num_labels, dropout=dropout, classifier_hidden=256,
            freeze_encoder=freeze_encoder, emoji_vocab_size=vocab_size,
            max_emojis=8, init_mode="random",
        )
    if model_key == "em5":
        return GatedFusionModel(
            model_name=backbone, text_dim=768, emoji_dim=32,
            num_labels=num_labels, dropout=dropout, classifier_hidden=256,
            freeze_encoder=freeze_encoder, emoji_vocab_size=vocab_size,
            max_emojis=8, freeze_emoji=False,
        )
    raise ValueError(f"Unknown model {model_key!r}")


# ---------------------------------------------------------------------------
# Forward adapters
# ---------------------------------------------------------------------------

def make_forward_fn(model_key: str, mode: str):
    """Return ``fn(model, *inputs) -> logits`` matching the batch layout."""
    if mode == "frozen":
        # batch inputs = (text_embed, emoji_ids, emoji_masks)
        if model_key == "em0":
            return lambda m, text_emb, e_ids, e_mask: m.classifier(text_emb)
        return lambda m, text_emb, e_ids, e_mask: m.forward_from_cache(
            text_emb, e_ids, e_mask
        )
    # finetune: batch inputs = (input_ids, attention_mask, emoji_ids, emoji_masks)
    if model_key == "em0":
        return lambda m, ids, attn, e_ids, e_mask: m(ids, attn)
    return lambda m, ids, attn, e_ids, e_mask: m(ids, attn, e_ids, e_mask)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

class CachedEmotionDataset(torch.utils.data.Dataset):
    """(cached_text_embedding, emoji_ids, emoji_masks, label)."""

    def __init__(self, embeds, emoji_ids, emoji_masks, labels):
        self.embeds = torch.tensor(embeds, dtype=torch.float32)
        self.emoji_ids = torch.tensor(emoji_ids, dtype=torch.long)
        self.emoji_masks = torch.tensor(emoji_masks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeds[i], self.emoji_ids[i], self.emoji_masks[i], self.labels[i]


def build_loaders(args, ctx, model, tokenizer):
    """Build train/val/test DataLoaders for the selected mode."""
    dfs = [ctx["train_df"], ctx["val_df"], ctx["test_df"]]
    emoji = [encode_emojis(df, ctx["vocab"]["emoji_to_id"]) for df in dfs]

    if getattr(args, "ablate_emoji", False):
        # Capacity-matched control: keep the architecture and parameter count
        # identical, but blank the emoji input so every example takes the
        # zero-emoji path. Any gap between this and the real run is
        # attributable to emoji content alone -- comparing EM3 against EM0
        # would instead confound the emoji branch with a different head size.
        emoji = [(np.zeros_like(ids), np.zeros_like(masks)) for ids, masks in emoji]
        print("ABLATION: emoji ids/masks zeroed — every example uses the no-emoji path")

    if args.mode == "frozen":
        cache_dir = os.path.join("results/emotion", args.dataset, "_text_cache")
        encoder = model.encoder.to(ctx["device"])
        embeds = [
            get_or_cache_bert_embeddings(
                df, name, tokenizer, encoder, args.max_length,
                args.batch_size, ctx["device"], cache_dir,
            )
            for df, name in zip(dfs, ["train", "validation", "test"])
        ]
        datasets = [
            CachedEmotionDataset(e, ei, em, df["label"].to_numpy())
            for e, (ei, em), df in zip(embeds, emoji, dfs)
        ]
        collate = None
    else:
        datasets = [
            TokenizedEmotionDataset(
                df["text_without_emoji"].tolist(), ei, em, df["label"].to_numpy()
            )
            for df, (ei, em) in zip(dfs, emoji)
        ]
        collate = make_collate_fn(tokenizer, args.max_length)

    train_loader = DataLoader(
        datasets[0], batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    eval_loaders = [
        DataLoader(d, batch_size=args.eval_batch_size, shuffle=False,
                   collate_fn=collate, num_workers=0)
        for d in datasets[1:]
    ]
    return train_loader, eval_loaders[0], eval_loaders[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["goemotions", "tweeteval", "irony"], required=True)
    p.add_argument("--model", choices=MODEL_CHOICES, required=True)
    p.add_argument("--mode", choices=["frozen", "finetune"], default="frozen")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--unfreeze-layers", type=int, default=2,
                   help="Top encoder blocks to train (finetune mode only).")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--encoder-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablate-emoji", action="store_true",
                   help="Zero every emoji input: capacity-matched no-emoji control.")
    p.add_argument("--tag", default="", help="Suffix for the results directory.")
    args = p.parse_args()

    suffix = args.mode if not args.tag else f"{args.mode}_{args.tag}"
    experiment = f"{args.model.upper()}_{suffix}"
    ctx = prepare(args.dataset, experiment, seed=args.seed)

    if args.mode == "finetune":
        head_lr = args.head_lr if args.head_lr < 1e-3 else 1e-4
    else:
        head_lr = args.head_lr

    model = build_model(
        args.model,
        backbone=args.backbone,
        num_labels=ctx["num_labels"],
        vocab_size=ctx["vocab"]["vocab_size"],
        dropout=args.dropout,
        freeze_encoder=True,
    )

    unfrozen: list[str] = []
    if args.mode == "finetune":
        unfrozen = unfreeze_top_layers(model.encoder, args.unfreeze_layers)
        print(f"Unfroze top {args.unfreeze_layers} encoder blocks "
              f"({len(unfrozen)} tensors, encoder_lr={args.encoder_lr})")

    report = trainable_report(model)
    print(f"Trainable: {report['trainable_params']:,} / {report['total_params']:,} "
          f"({report['trainable_pct']}%)")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    train_loader, val_loader, test_loader = build_loaders(args, ctx, model, tokenizer)
    forward_fn = make_forward_fn(args.model, args.mode)

    train_result = train_model(
        model, forward_fn, train_loader, val_loader,
        device=ctx["device"],
        class_weights=ctx["class_weights"],
        out_dir=ctx["out_dir"],
        epochs=args.epochs,
        head_lr=head_lr,
        encoder_lr=args.encoder_lr if args.mode == "finetune" else None,
        weight_decay=args.weight_decay,
        names=ctx["names"],
        # Store the exact emoji->id mapping with the weights, so inference can
        # never pair this checkpoint with a different vocabulary file.
        checkpoint_meta={
            "emoji_to_id": ctx["vocab"]["emoji_to_id"],
            "backbone": args.backbone,
            "dataset": args.dataset,
        },
    )

    evaluate_model(
        model, forward_fn, test_loader, ctx["test_df"], ctx["train_df"],
        train_result,
        device=ctx["device"],
        out_dir=ctx["out_dir"],
        experiment=experiment,
        names=ctx["names"],
        extra={
            "config": {
                "dataset": args.dataset,
                "model": args.model,
                "mode": args.mode,
                "backbone": args.backbone,
                "unfreeze_layers": args.unfreeze_layers if args.mode == "finetune" else 0,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "head_lr": head_lr,
                "encoder_lr": args.encoder_lr if args.mode == "finetune" else None,
                "dropout": args.dropout,
                "seed": args.seed,
                "ablate_emoji": bool(args.ablate_emoji),
                "emoji_vocab_size": ctx["vocab"]["vocab_size"],
            },
            "trainable_params": report,
        },
    )


if __name__ == "__main__":
    main()
