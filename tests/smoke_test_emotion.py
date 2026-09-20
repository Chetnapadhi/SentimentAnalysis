"""Smoke tests for the Phase 2 emotion models (EM0 / EM3 / EM5).

Mirrors ``tests/smoke_test_e3_e4_e5.py`` but asserts the Phase 2 contract:

1. **N-way head.** EM0 / EM3 / EM5 emit ``[B, num_labels]`` for an arbitrary
   ``num_labels`` (6 for GoEmotions Ekman-6, 4 for TweetEval), not a hardcoded 3.
2. **Partial unfreezing.** ``unfreeze_top_layers(encoder, n)`` leaves exactly
   the top ``n`` transformer blocks trainable and everything below frozen --
   the supervisor's "unfreeze the top 2 layers" instruction.
3. **Zero-emoji safety carries over.** The learned ``emoji_absent`` fallback
   (EM3) and the clean zero vector (EM5) still hold at the new label count,
   which matters because the emotion corpora are mostly emoji-free.
4. **Ekman mapping integrity.** All 28 GoEmotions labels map into the 6 Ekman
   families plus neutral, and the 6 ids are dense and ordered.

Run: ``.venv/Scripts/python.exe tests/smoke_test_emotion.py``
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.emotion_adapter import EKMAN6_NAMES, EKMAN6_TO_ID, EKMAN_MAP
from src.emotion_pipeline import trainable_report, unfreeze_top_layers
from src.models.e3_attention_fusion import AttentionFusionModel
from src.models.e5_gated_fusion import GatedFusionModel
from src.models.text_only import TextOnlyModel

BACKBONE = "cardiffnlp/twitter-roberta-base"
B, EMOJI_DIM, MAX_EMOJIS, VOCAB = 3, 32, 8, 200

# Example 0: 2 emojis | Example 1: 1 emoji | Example 2: 0 emojis
EMOJI_IDS = torch.tensor([
    [10, 20, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=torch.long)
EMOJI_MASKS = torch.tensor([
    [1.0, 1.0, 0, 0, 0, 0, 0, 0],
    [1.0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=torch.float32)


def test_ekman_mapping():
    print("=" * 70)
    print("TESTING Ekman-6 label mapping")
    print("=" * 70)

    families = set(EKMAN_MAP.values())
    assert families == set(EKMAN6_NAMES) | {"neutral"}, f"Unexpected families: {families}"
    print(f"  [OK] 28 GoEmotions labels collapse to {len(EKMAN6_NAMES)} families + neutral")

    assert len(EKMAN_MAP) == 28, f"Expected 28 source labels, got {len(EKMAN_MAP)}"
    print("  [OK] Every one of the 28 source labels has a mapping")

    assert sorted(EKMAN6_TO_ID.values()) == list(range(6)), "Ids must be dense 0..5"
    assert "neutral" not in EKMAN6_TO_ID, "neutral must be excluded from the 6-class space"
    print(f"  [OK] Label ids dense 0..5, neutral excluded: {EKMAN6_NAMES}")
    print()


def test_num_labels(model_key: str, num_labels: int):
    print("=" * 70)
    print(f"TESTING {model_key.upper()} head with num_labels={num_labels}")
    print("=" * 70)

    if model_key == "em0":
        model = TextOnlyModel(
            model_name=BACKBONE, hidden_size=768, num_labels=num_labels,
            dropout=0.3, freeze_encoder=True,
        )
    elif model_key == "em3":
        model = AttentionFusionModel(
            model_name=BACKBONE, text_dim=768, emoji_dim=EMOJI_DIM,
            num_labels=num_labels, dropout=0.3, classifier_hidden=256,
            freeze_encoder=True, emoji_vocab_size=VOCAB, max_emojis=MAX_EMOJIS,
            init_mode="random",
        )
    else:
        model = GatedFusionModel(
            model_name=BACKBONE, text_dim=768, emoji_dim=EMOJI_DIM,
            num_labels=num_labels, dropout=0.3, classifier_hidden=256,
            freeze_encoder=True, emoji_vocab_size=VOCAB, max_emojis=MAX_EMOJIS,
            freeze_emoji=False,
        )
    model.eval()

    text_rep = torch.randn(B, 768)
    if model_key == "em0":
        logits = model.classifier(text_rep)
    else:
        logits = model.forward_from_cache(text_rep, EMOJI_IDS, EMOJI_MASKS)

    assert logits.shape == (B, num_labels), f"logits shape {logits.shape} != ({B}, {num_labels})"
    print(f"  [OK] logits: {list(logits.shape)} == [B, {num_labels}]")
    assert not torch.isnan(logits).any(), "logits contain NaN"
    print("  [OK] No NaN in logits (includes the zero-emoji row)")

    if model_key == "em3":
        assert model.last_fused.shape == (B, 800)
        assert torch.allclose(model.last_attention_weights[2], torch.zeros(MAX_EMOJIS)), \
            "zero-emoji row must have all-zero attention"
        assert torch.allclose(model.last_emoji_context[2], model.emoji_absent), \
            "zero-emoji row must fall back to learned emoji_absent"
        print("  [OK] Zero-emoji row falls back to learned emoji_absent (no softmax over -inf)")
        assert torch.isclose(model.last_attention_weights[0, :2].sum(), torch.tensor(1.0), atol=1e-5)
        print("  [OK] Valid attention weights still sum to 1.0")

    if model_key == "em5":
        g = model.last_gate_values
        assert g.shape == (B, EMOJI_DIM), f"gate shape {g.shape}"
        assert (g >= 0).all() and (g <= 1).all(), "gate values must lie in [0, 1]"
        print(f"  [OK] Gate: {list(g.shape)} == [B, 32], values in [0, 1]")
        assert torch.allclose(model.last_emoji_rep[2], torch.zeros(EMOJI_DIM)), \
            "zero-emoji row must mean-pool to a clean zero vector"
        print("  [OK] Zero-emoji row mean-pools to a clean zero vector (no div-by-zero)")
        assert model.emoji_encoder.weight.requires_grad, \
            "Phase 2 EM5 uses freeze_emoji=False, so the embedding must be trainable"
        print("  [OK] freeze_emoji=False makes the emoji embedding trainable")

    print()
    return model


def test_partial_unfreezing():
    print("=" * 70)
    print("TESTING partial unfreezing (supervisor: 'unfreeze the top 2 layers')")
    print("=" * 70)

    model = TextOnlyModel(
        model_name=BACKBONE, hidden_size=768, num_labels=6,
        dropout=0.3, freeze_encoder=True,
    )
    frozen_report = trainable_report(model)
    print(f"  Fully frozen encoder: {frozen_report['trainable_params']:,} trainable "
          f"({frozen_report['trainable_pct']}%)")

    blocks = model.encoder.encoder.layer
    n_blocks = len(blocks)
    unfreeze_top_layers(model.encoder, 2)

    for i, block in enumerate(blocks):
        should_train = i >= n_blocks - 2
        for name, param in block.named_parameters():
            assert param.requires_grad == should_train, (
                f"block {i} param {name}: requires_grad={param.requires_grad}, "
                f"expected {should_train}"
            )
    print(f"  [OK] Exactly blocks {n_blocks - 2}-{n_blocks - 1} of {n_blocks} are trainable")

    assert not model.encoder.embeddings.word_embeddings.weight.requires_grad, \
        "word embeddings must stay frozen"
    print("  [OK] Embeddings and all lower blocks remain frozen")

    after = trainable_report(model)
    assert after["trainable_params"] > frozen_report["trainable_params"], \
        "unfreezing must increase the trainable count"
    print(f"  [OK] Trainable grew {frozen_report['trainable_params']:,} -> "
          f"{after['trainable_params']:,} ({after['trainable_pct']}% of total)")

    # n_layers=0 must be a clean no-op that leaves the encoder fully frozen.
    model2 = TextOnlyModel(
        model_name=BACKBONE, hidden_size=768, num_labels=6,
        dropout=0.3, freeze_encoder=True,
    )
    assert unfreeze_top_layers(model2.encoder, 0) == []
    assert not any(p.requires_grad for p in model2.encoder.parameters()), \
        "n_layers=0 must leave the encoder fully frozen"
    print("  [OK] unfreeze_top_layers(encoder, 0) is a clean no-op")
    print()


if __name__ == "__main__":
    test_ekman_mapping()
    # 6 classes = GoEmotions Ekman-6, 4 classes = TweetEval emotion
    for key in ("em0", "em3", "em5"):
        test_num_labels(key, num_labels=6)
    test_num_labels("em3", num_labels=4)
    test_partial_unfreezing()
    print("=" * 70)
    print("ALL PHASE 2 EMOTION SMOKE TESTS PASSED")
    print("=" * 70)
