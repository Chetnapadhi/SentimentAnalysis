"""Smoke test script for E3, E4, and E5 architecture and contract compliance.

Tests:
1. E3 Attention Fusion:
   - Intermediate and final tensor shapes:
     text_rep = [B, 768]
     emoji_emb = [B, 8, 32]
     Q = [B, 32]
     K = [B, 8, 32]
     V = [B, 8, 32]
     attention_weights = [B, 8]
     emoji_context = [B, 32]
     fused = [B, 800]
     logits = [B, 3]
   - Explicit example with 2 emojis:
     * padding attention == 0
     * valid attention weights sum to 1.0
   - Explicit example with 1 emoji:
     * padding attention == 0
     * valid attention weight == 1.0
   - Explicit example with 0 emojis:
     * no NaN
     * attention weights are all 0
     * emoji_context is exactly the learned fallback parameter (emoji_absent)
   - Parameter trainability:
     * BERT frozen (requires_grad = False)
     * emoji_encoder trainable (requires_grad = True)
     * Wq/Wk/Wv trainable (requires_grad = True)
     * emoji_absent trainable (requires_grad = True)
     * classifier trainable (requires_grad = True)

2. E4 Attention Fusion (Pretrained):
   - Loads E2 pretrained artifact (shape [1610, 32])
   - Emoji embedding frozen (requires_grad = False)
   - Attention behavior identical to E3
   - Zero-emoji handling works

3. E5 Gated Fusion:
   - Loads E2 pretrained artifact (shape [1610, 32])
   - Emoji embedding frozen
   - Gate shape = [B, 32]
   - All gate values in [0, 1]
   - Zero-emoji handling does not produce NaN
   - fused shape = [B, 800]
   - logits = [B, 3]
"""

from __future__ import annotations

import os
import sys
import torch
import numpy as np

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import numpy as np

# In environments where transformers cannot load (e.g. Python 3.14 on Windows),
# provide a compliant mock BERT encoder so full PyTorch architecture tests run.
try:
    import transformers
except Exception:
    from unittest.mock import MagicMock
    class MockBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(10, 768))
        def forward(self, input_ids=None, attention_mask=None, return_dict=True):
            B = input_ids.shape[0] if input_ids is not None else 1
            M = input_ids.shape[1] if input_ids is not None else 8
            res = MagicMock()
            res.last_hidden_state = torch.randn(B, M, 768)
            return res

    mock_tf = MagicMock()
    mock_tf.AutoConfig.from_pretrained = MagicMock(return_value={})
    mock_tf.AutoModel.from_pretrained = MagicMock(side_effect=lambda *args, **kwargs: MockBert())
    sys.modules["transformers"] = mock_tf

from src.models.e3_attention_fusion import AttentionFusionModel
from src.models.e5_gated_fusion import GatedFusionModel


def setup_mock_e2_artifact():
    """Ensure mock E2 artifact exists for smoke testing E4 and E5."""
    path = os.path.join(ROOT, "models", "emoji_embeddings", "stocktwits_emoji_embedding_e2_1610x32.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        gen = torch.Generator().manual_seed(42)
        mock_data = {
            "emoji_embedding": torch.randn(1610, 32, generator=gen),
            "vocab_size": 1610,
            "embedding_dim": 32,
            "seed": 42,
            "tweet_eval_pretrained_count": 19,
            "random_initialized_count": 1591,
        }
        torch.save(mock_data, path)
        print(f"[TEST SETUP] Created test E2 embedding artifact at {path}")
    return path


def test_e3():
    print("=" * 70)
    print("TESTING E3: Attention Fusion with Random Emoji Embeddings")
    print("=" * 70)

    B = 3
    text_dim = 768
    emoji_dim = 32
    max_emojis = 8
    vocab_size = 1610
    num_labels = 3

    model = AttentionFusionModel(
        model_name="bert-base-uncased",
        text_dim=text_dim,
        emoji_dim=emoji_dim,
        num_labels=num_labels,
        dropout=0.3,
        classifier_hidden=256,
        freeze_encoder=True,
        emoji_vocab_size=vocab_size,
        max_emojis=max_emojis,
        init_mode="random",
    )
    model.eval()

    # 1. Parameter trainability check
    print("\n[E3.1] Checking Parameter Trainability:")
    for name, param in model.encoder.named_parameters():
        assert not param.requires_grad, f"BERT param {name} should be frozen!"
    print("  [OK] BERT encoder is FROZEN (all requires_grad=False)")

    assert model.emoji_encoder.weight.requires_grad, "E3 emoji embedding must be trainable!"
    print("  [OK] E3 emoji_encoder is TRAINABLE (requires_grad=True)")

    assert model.Wq.weight.requires_grad, "Wq must be trainable!"
    assert model.Wk.weight.requires_grad, "Wk must be trainable!"
    assert model.Wv.weight.requires_grad, "Wv must be trainable!"
    print("  [OK] Wq, Wk, Wv are TRAINABLE (requires_grad=True)")

    assert model.emoji_absent.requires_grad, "emoji_absent must be trainable!"
    assert model.emoji_absent.shape == (emoji_dim,), f"emoji_absent shape must be ({emoji_dim},), got {model.emoji_absent.shape}"
    print(f"  [OK] emoji_absent is TRAINABLE (shape: {tuple(model.emoji_absent.shape)})")

    for name, param in model.classifier.named_parameters():
        assert param.requires_grad, f"Classifier param {name} must be trainable!"
    print("  [OK] Classifier head is TRAINABLE (requires_grad=True)")

    # 2. Test examples:
    # Example 0: 2 emojis (indices 10, 20)
    # Example 1: 1 emoji (index 5)
    # Example 2: 0 emojis (all pad)
    text_rep = torch.randn(B, text_dim)
    emoji_ids = torch.tensor([
        [10, 20, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.long)
    emoji_masks = torch.tensor([
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)

    logits = model.forward_from_cache(text_rep, emoji_ids, emoji_masks)

    # 3. Shape verification
    print("\n[E3.2] Verifying Tensor Shapes:")
    assert model.last_text_rep.shape == (B, 768), f"text_rep shape mismatch: {model.last_text_rep.shape}"
    print(f"  [OK] text_rep: {list(model.last_text_rep.shape)} == [B, 768]")

    assert model.last_emoji_emb.shape == (B, 8, 32), f"emoji_emb shape mismatch: {model.last_emoji_emb.shape}"
    print(f"  [OK] emoji_emb sequence: {list(model.last_emoji_emb.shape)} == [B, 8, 32]")

    assert model.last_Q.shape == (B, 32), f"Q shape mismatch: {model.last_Q.shape}"
    print(f"  [OK] Q: {list(model.last_Q.shape)} == [B, 32]")

    assert model.last_K.shape == (B, 8, 32), f"K shape mismatch: {model.last_K.shape}"
    print(f"  [OK] K: {list(model.last_K.shape)} == [B, 8, 32]")

    assert model.last_V.shape == (B, 8, 32), f"V shape mismatch: {model.last_V.shape}"
    print(f"  [OK] V: {list(model.last_V.shape)} == [B, 8, 32]")

    assert model.last_attention_weights.shape == (B, 8), f"alpha shape mismatch: {model.last_attention_weights.shape}"
    print(f"  [OK] attention_weights: {list(model.last_attention_weights.shape)} == [B, 8]")

    assert model.last_emoji_context.shape == (B, 32), f"emoji_context shape mismatch: {model.last_emoji_context.shape}"
    print(f"  [OK] emoji_context: {list(model.last_emoji_context.shape)} == [B, 32]")

    assert model.last_fused.shape == (B, 800), f"fused shape mismatch: {model.last_fused.shape}"
    print(f"  [OK] fused: {list(model.last_fused.shape)} == [B, 800]")

    assert logits.shape == (B, 3), f"logits shape mismatch: {logits.shape}"
    print(f"  [OK] logits: {list(logits.shape)} == [B, 3]")

    # 4. Check attention behavior per example
    print("\n[E3.3] Checking Attention Weight Behavior:")
    alpha = model.last_attention_weights

    # Example 0: 2 emojis
    print(f"  Example 0 (2 emojis) attention weights: {alpha[0].tolist()[:4]}")
    assert torch.allclose(alpha[0, 2:], torch.zeros(6)), "Padding positions must have 0 attention!"
    assert torch.isclose(alpha[0, :2].sum(), torch.tensor(1.0), atol=1e-5), f"Valid attention must sum to 1, got {alpha[0, :2].sum()}"
    print("  [OK] Example 0 (2 emojis): padding attention == 0, valid weights sum to 1.0")

    # Example 1: 1 emoji
    print(f"  Example 1 (1 emoji) attention weights: {alpha[1].tolist()[:4]}")
    assert torch.allclose(alpha[1, 1:], torch.zeros(7)), "Padding positions must have 0 attention!"
    assert torch.isclose(alpha[1, 0], torch.tensor(1.0), atol=1e-5), f"Single emoji attention must be 1.0, got {alpha[1, 0]}"
    print("  [OK] Example 1 (1 emoji): padding attention == 0, valid weight == 1.0")

    # Example 2: 0 emojis
    print(f"  Example 2 (0 emojis) attention weights: {alpha[2].tolist()}")
    assert torch.allclose(alpha[2], torch.zeros(8)), "Zero-emoji example must have 0 attention everywhere!"
    assert not torch.isnan(alpha[2]).any(), "Zero-emoji attention has NaN!"
    assert not torch.isnan(model.last_emoji_context[2]).any(), "Zero-emoji context has NaN!"
    assert not torch.isnan(logits[2]).any(), "Zero-emoji logits has NaN!"
    assert torch.allclose(model.last_emoji_context[2], model.emoji_absent), "Zero-emoji context must equal learned fallback parameter emoji_absent!"
    print("  [OK] Example 2 (0 emojis): no NaN, attention all 0, context equals learned emoji_absent fallback")

    print("\n>>> E3 SMOKE TEST PASSED! <<<\n")


def test_e4(artifact_path):
    print("=" * 70)
    print("TESTING E4: Attention Fusion with Pretrained Emoji Embeddings")
    print("=" * 70)

    B = 3
    text_dim = 768
    emoji_dim = 32
    max_emojis = 8
    vocab_size = 1610
    num_labels = 3

    model = AttentionFusionModel(
        model_name="bert-base-uncased",
        text_dim=text_dim,
        emoji_dim=emoji_dim,
        num_labels=num_labels,
        dropout=0.3,
        classifier_hidden=256,
        freeze_encoder=True,
        emoji_vocab_size=vocab_size,
        max_emojis=max_emojis,
        init_mode="pretrained",
        pretrained_emoji_path=artifact_path,
    )
    model.eval()

    # 1. Verification of pretrained loading and frozen state
    print("\n[E4.1] Verifying Pretrained Weights & Freeze:")
    artifact = torch.load(artifact_path, map_location="cpu")
    expected_emb = artifact["emoji_embedding"]
    assert tuple(expected_emb.shape) == (1610, 32), f"Expected (1610, 32), got {expected_emb.shape}"
    print(f"  [OK] Pretrained artifact verified: shape={tuple(expected_emb.shape)}")

    assert torch.allclose(model.emoji_encoder.weight, expected_emb), "Model weights did not match pretrained artifact!"
    print("  [OK] Pretrained weights correctly copied to emoji_encoder")

    assert not model.emoji_encoder.weight.requires_grad, "E4 emoji embedding MUST be frozen!"
    print("  [OK] E4 emoji_encoder is FROZEN (requires_grad=False)")

    # 2. Forward pass with mixed examples
    text_rep = torch.randn(B, text_dim)
    emoji_ids = torch.tensor([
        [15, 30, 0, 0, 0, 0, 0, 0],
        [8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.long)
    emoji_masks = torch.tensor([
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)

    logits = model.forward_from_cache(text_rep, emoji_ids, emoji_masks)

    # 3. Shape and zero-emoji checks
    print("\n[E4.2] Verifying Shapes & Zero-Emoji Fallback:")
    assert logits.shape == (B, 3), f"Logits shape mismatch: {logits.shape}"
    assert model.last_fused.shape == (B, 800)
    assert not torch.isnan(logits).any()

    # Verify zero-emoji fallback
    assert torch.allclose(model.last_emoji_context[2], model.emoji_absent)
    print("  [OK] Zero-emoji example uses learned fallback emoji_absent")
    print(f"  [OK] Logits shape: {list(logits.shape)}")
    print(f"  [OK] Fused shape: {list(model.last_fused.shape)}")

    print("\n>>> E4 SMOKE TEST PASSED! <<<\n")


def test_e5(artifact_path):
    print("=" * 70)
    print("TESTING E5: Gated Fusion with Pretrained Emoji Embeddings (32-d Gate)")
    print("=" * 70)

    B = 3
    text_dim = 768
    emoji_dim = 32
    max_emojis = 8
    vocab_size = 1610
    num_labels = 3

    model = GatedFusionModel(
        model_name="bert-base-uncased",
        text_dim=text_dim,
        emoji_dim=emoji_dim,
        num_labels=num_labels,
        dropout=0.3,
        classifier_hidden=256,
        freeze_encoder=True,
        emoji_vocab_size=vocab_size,
        max_emojis=max_emojis,
        pretrained_emoji_path=artifact_path,
    )
    model.eval()

    # 1. Check frozen embedding
    print("\n[E5.1] Checking Frozen Pretrained Embedding:")
    assert not model.emoji_encoder.weight.requires_grad, "E5 emoji embedding MUST be frozen!"
    print("  [OK] E5 emoji_encoder is FROZEN (requires_grad=False)")

    # 2. Check gate architecture: Linear(800 -> 32) + Sigmoid
    print("\n[E5.2] Checking Gate Architecture:")
    gate_linear = model.gate[0]
    assert isinstance(gate_linear, torch.nn.Linear)
    assert gate_linear.in_features == 800, f"Expected 800 in_features, got {gate_linear.in_features}"
    assert gate_linear.out_features == 32, f"Expected 32 out_features, got {gate_linear.out_features}"
    print(f"  [OK] Gate linear layer: Linear({gate_linear.in_features} -> {gate_linear.out_features})")
    assert isinstance(model.gate[1], torch.nn.Sigmoid), "Gate must end with Sigmoid!"
    print("  [OK] Gate activation: Sigmoid")

    # 3. Forward pass with 2 emojis, 1 emoji, and 0 emojis
    text_rep = torch.randn(B, text_dim)
    emoji_ids = torch.tensor([
        [15, 30, 0, 0, 0, 0, 0, 0],
        [8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.long)
    emoji_masks = torch.tensor([
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)

    logits = model.forward_from_cache(text_rep, emoji_ids, emoji_masks)

    # 4. Check gate properties
    print("\n[E5.3] Checking Gate Outputs & Values:")
    g = model.last_gate_values
    assert g is not None
    assert g.shape == (B, 32), f"Gate shape must be ({B}, 32), got {g.shape}"
    print(f"  [OK] Gate shape: {list(g.shape)} == [B, 32]")

    assert (g >= 0.0).all() and (g <= 1.0).all(), "All gate values must be in [0, 1]!"
    print(f"  [OK] Gate values bounded in [0, 1] (min={g.min():.4f}, max={g.max():.4f})")

    # 5. Check zero-emoji handling and no NaNs
    print("\n[E5.4] Checking Zero-Emoji Safety & Fused Shapes:")
    assert not torch.isnan(logits).any(), "Logits contain NaN!"
    assert not torch.isnan(g).any(), "Gate contains NaN!"
    assert not torch.isnan(model.last_fused).any(), "Fused representation contains NaN!"
    print("  [OK] Zero-emoji example produces NO NaNs")

    assert model.last_fused.shape == (B, 800), f"Fused shape must be ({B}, 800), got {model.last_fused.shape}"
    print(f"  [OK] Fused shape: {list(model.last_fused.shape)} == [B, 800]")

    assert logits.shape == (B, 3), f"Logits shape must be ({B}, 3), got {logits.shape}"
    print(f"  [OK] Logits shape: {list(logits.shape)} == [B, 3]")

    print("\n>>> E5 SMOKE TEST PASSED! <<<\n")


if __name__ == "__main__":
    artifact_path = setup_mock_e2_artifact()
    test_e3()
    test_e4(artifact_path)
    test_e5(artifact_path)
    print("=" * 70)
    print("ALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
