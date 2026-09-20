"""Featurize the StockTwits splits once and share the cache across E0-E5.

Why this exists
---------------
E0-E5 each cache frozen-BERT text embeddings under their own
``results/E<n>/embeddings_cache/`` directory. The featurization itself is
identical in all six (frozen ``bert-base-uncased``, mask-weighted mean pooling
over ``text_without_emoji``, dynamic padding), but because the cache is
per-experiment, running the suite end-to-end featurizes the same 123,763 texts
six times. On CPU that is ~2.6 h per pass, i.e. roughly 15 h of duplicated work.

This script computes each split once and then places the result at every path
the six scripts will look for, using **hard links** so the six "copies" share
one set of blocks on disk (~380 MB total instead of ~2.3 GB).

Cache key conventions differ between the experiments, and both are reproduced
exactly:

* ``E0``      ``embeds_<split>_<model>_<H8>.npy``
* ``E1, E2``  ``text_<split>_<H8>.npy``
  where ``H = sha1(concat(train) + concat(val) + concat(test))`` — one global
  hash over all three splits.
* ``E3-E5``   ``text_<split>_<h8>.npy``
  where ``h = sha1(concat(texts_of_that_split))`` — a per-split hash
  (``src/common_pipeline.get_or_cache_bert_embeddings``).

If a target already exists it is left alone, so this is safe to re-run.

Usage:
    python -m src.data.precompute_embeddings
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.common_pipeline import load_config, load_splits

SPLITS = ["train", "validation", "test"]


def featurize(texts, tokenizer, encoder, max_length, batch_size, device) -> np.ndarray:
    """Frozen BERT + mask-weighted mean pooling. Matches E0-E5 exactly."""
    out = []
    encoder.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Featurizing"):
            enc = tokenizer(
                texts[i : i + batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            hidden = encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                return_dict=True,
            ).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0)


def _link_or_copy(src: str, dst: str) -> str:
    """Hard-link ``src`` to ``dst``; fall back to a copy if links are refused."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return "exists"
    try:
        os.link(src, dst)
        return "linked"
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(src, dst)
        return "copied"


def target_paths(split: str, global_h8: str, split_h8: str, model_name: str) -> list[str]:
    """Every cache path the six experiments will probe for this split."""
    safe_model = model_name.replace("/", "_")
    paths = [f"results/E0/embeddings_cache/embeds_{split}_{safe_model}_{global_h8}.npy"]
    for exp in ("E1", "E2"):
        paths.append(f"results/{exp}/embeddings_cache/text_{split}_{global_h8}.npy")
    for exp in ("E3", "E4", "E5"):
        paths.append(f"results/{exp}/embeddings_cache/text_{split}_{split_h8}.npy")
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared-dir", default="results/_shared_embeddings")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    cfg = load_config()
    model_name = cfg["text_model"]["name"]
    max_length = cfg["text_model"]["max_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"Device: {device} | threads: {torch.get_num_threads()} | model: {model_name}")

    train_df, val_df, test_df = load_splits()
    frames = {"train": train_df, "validation": val_df, "test": test_df}
    texts = {s: frames[s]["text_without_emoji"].tolist() for s in SPLITS}
    for s in SPLITS:
        print(f"  {s:<11} {len(texts[s]):>7} rows")

    # Global hash (E0/E1/E2) — concatenation of all three splits, in order.
    global_h = hashlib.sha1(
        "".join(texts["train"] + texts["validation"] + texts["test"]).encode("utf-8")
    ).hexdigest()[:8]
    print(f"Global cache hash (E0/E1/E2): {global_h}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    for param in encoder.parameters():
        param.requires_grad = False

    os.makedirs(args.shared_dir, exist_ok=True)

    for split in SPLITS:
        split_h = hashlib.sha1("".join(texts[split]).encode("utf-8")).hexdigest()[:8]
        shared = os.path.join(args.shared_dir, f"{split}_{split_h}.npy")

        if os.path.exists(shared):
            print(f"\n[{split}] shared cache present -> {shared}")
        else:
            print(f"\n[{split}] featurizing {len(texts[split])} texts "
                  f"(per-split hash {split_h})...")
            embeds = featurize(
                texts[split], tokenizer, encoder, max_length, args.batch_size, device
            )
            np.save(shared, embeds)
            print(f"[{split}] saved {embeds.shape} -> {shared} "
                  f"({os.path.getsize(shared) / 1e6:.0f} MB)")

        for dst in target_paths(split, global_h, split_h, model_name):
            status = _link_or_copy(shared, dst)
            print(f"    {status:>7}  {dst}")

    print("\nDone. E0-E5 will now load these caches instead of re-featurizing.")


if __name__ == "__main__":
    main()
