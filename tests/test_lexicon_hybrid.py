"""Tests for the emoji lexicon, sarcasm rules and hybrid fusion.

Pure-logic tests: no model weights, no network, runs in about a second.

    python tests/test_lexicon_hybrid.py        # or: pytest tests/test_lexicon_hybrid.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import emoji as emoji_lib

from src.data.preprocessing import extract_emoji_occurrences, extract_emojis
from src.lexicon.emoji_lexicon import (
    MANUAL_PATH, UNIFIED_LABELS, aggregate, load_manual, lookup,
)
from src.lexicon.hybrid import DATASET_LABELS, FusionConfig, _PROJECTIONS, fuse, project
from src.lexicon.sarcasm import SarcasmDetector, SarcasmResult

RULES = SarcasmDetector(backend="rules")


def ev(text):
    return aggregate(extract_emoji_occurrences(text), text)


# ---------------------------------------------------------------------------
# Lexicon integrity
# ---------------------------------------------------------------------------

def test_manual_rows_are_valid():
    rows = list(csv.DictReader(open(MANUAL_PATH, encoding="utf-8")))
    assert len(rows) >= 250
    seen = set()
    for r in rows:
        e = r["emoji"]
        assert emoji_lib.is_emoji(e), f"not an emoji: {e!r}"
        assert e not in seen, f"duplicate row: {e}"
        seen.add(e)
        total = sum(float(r[k]) for k in UNIFIED_LABELS)
        assert abs(total - 1) < 1e-6, f"{e}: emotion weights sum to {total}"
        assert -1 <= float(r["polarity"]) <= 1
        assert 0 <= float(r["intensity"]) <= 1
        assert 0 <= float(r["sarcasm_cue"]) <= 1


def test_every_unicode_emoji_gets_a_label():
    fq = [e for e, d in emoji_lib.EMOJI_DATA.items()
          if d.get("status") == emoji_lib.STATUS["fully_qualified"]]
    assert len(fq) > 3000
    for e in fq:
        entry = lookup(e)
        assert abs(sum(entry.dist) - 1) < 1e-6
        assert entry.source in {"manual", "derived", "default"}


def test_normalisation_skin_tone_zwj_and_variation_selector():
    assert lookup("👍🏽").source == "manual" and lookup("👍🏽").dist == lookup("👍").dist
    assert lookup("🤦‍♂️").dist == lookup("🤦").dist
    assert lookup("❤").dist == lookup("❤️").dist
    # exact ZWJ entries must win over their first component
    assert lookup("❤️‍🔥").notes.startswith("heart on fire")


def test_key_emojis_have_sensible_labels():
    assert lookup("😭").top_label() == "sadness"
    assert lookup("😡").top_label() == "anger"
    assert lookup("🤮").top_label() == "disgust"
    assert lookup("😱").top_label() == "fear"
    assert lookup("🙄").sarcasm_cue >= 0.8
    assert lookup("🤡").sarcasm_cue >= 0.8
    assert lookup("❤️").sarcasm_cue < 0.1


# ---------------------------------------------------------------------------
# Multi-emoji aggregation
# ---------------------------------------------------------------------------

def test_occurrences_keep_repeats_but_training_extractor_does_not():
    assert len(extract_emoji_occurrences("😂😂😂")) == 3
    assert len(extract_emojis("😂😂😂")) == 1   # training contract unchanged


def test_repetition_increases_strength_with_diminishing_returns():
    s1, s3, s9 = ev("ugh 😭").strength, ev("ugh 😭😭😭").strength, ev("ugh " + "😭" * 9).strength
    assert s1 < s3 < s9
    assert (s9 - s3) < (s3 - s1)


def test_neutral_emojis_carry_almost_no_evidence():
    assert ev("🇮🇳 🚗 📎").strength < 0.2
    assert ev("no emojis here").strength == 0.0


def test_mixed_emojis_are_flagged():
    assert ev("I laughed so hard I cried 😂💔").mixed
    assert not ev("😂😂").mixed


# ---------------------------------------------------------------------------
# Sarcasm rules
# ---------------------------------------------------------------------------

def test_sarcasm_positive_cases():
    for text in [
        "I just love waiting 3 hours at the DMV 🙄🙄",
        "Great job management, losing half my money 🤡💸",
        "Oh wonderful, my flight got cancelled again 🙃",
        "I enjoy doing laundry /s",
    ]:
        assert RULES.score(text).is_sarcastic, text


def test_ambiguous_emoji_alone_is_not_sarcasm_evidence():
    """😂 👏 🙂 are sincere as often as not; they only count with other context."""
    from src.lexicon.sarcasm import rule_features
    from src.lexicon.text_cues import analyse_text
    for text in ["That joke was hilarious 😂😂", "Great presentation 👏", "Nice day at the beach 🙂"]:
        score, _ = rule_features(analyse_text(text), ev(text))
        assert score == 0.0, (text, score)
    # ...but the same emoji does count once the context is sarcastic
    t = "Love waiting in traffic for two hours 👏"
    assert rule_features(analyse_text(t), ev(t))[0] > 1.5


def test_two_tier_verdict_requires_corroboration_for_the_learned_model():
    from src.lexicon.sarcasm import decide
    # learned model confident, no rule evidence -> only "possible"
    assert decide(0.93, 0.0, threshold=0.4, has_model=True) == "possible"
    # learned model confident AND a real cue -> "detected"
    assert decide(0.93, 1.5, threshold=0.4, has_model=True) == "detected"
    # rules alone confident -> "detected" even if the model disagrees
    assert decide(0.10, 3.5, threshold=0.4, has_model=True) == "detected"
    assert decide(0.20, 0.0, threshold=0.4, has_model=True) == "none"


def test_possible_sarcasm_never_changes_the_emotion():
    p = np.array([0.05, 0.85, 0.05, 0.05])
    maybe = SarcasmResult(0.9, False, 0.1, 0.9, "test", level="possible")
    _, final, _ = fuse(p, "tweeteval", ev("I am so happy today"), maybe, FusionConfig())
    assert np.allclose(final, p)


def test_single_text_cue_is_ranked_up_but_not_flagged_by_rules_alone():
    """Rules require two converging cues. One phrase with no emoji raises the
    score but stays under threshold -- the pretrained irony classifier is what
    handles plain-text sarcasm. This documents that limit rather than hiding it."""
    sarcastic = RULES.score("Yeah right like the bus is ever on time")
    sincere = RULES.score("The bus was on time today")
    assert sarcastic.probability > sincere.probability
    assert not sarcastic.is_sarcastic


def test_sarcasm_negative_controls():
    for text in [
        "That joke was hilarious 😂😂",
        "Happy anniversary to my best friend ❤️",
        "My dog passed away this morning 😭💔",
        "Can't wait for the concert tonight 🤩",
        "Happy to see you again!",
        "We won the championship!!! 🎉🎉🔥🚀",
    ]:
        assert not RULES.score(text).is_sarcastic, text


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def test_projection_rows_sum_to_one():
    for dataset, proj in _PROJECTIONS.items():
        assert set(proj) == set(UNIFIED_LABELS) - {"neutral"}
        for u, row in proj.items():
            assert abs(sum(row.values()) - 1) < 1e-9, (dataset, u)
            assert set(row) <= set(DATASET_LABELS[dataset])
        q = project({"neutral": 1.0}, dataset, smoothing=0.0)
        assert np.allclose(q, 1 / len(DATASET_LABELS[dataset]))


def test_no_emoji_no_sarcasm_returns_model_output_exactly():
    """The no-regression guarantee: plain text is never altered."""
    for dataset in DATASET_LABELS:
        p = np.random.default_rng(0).dirichlet(np.ones(len(DATASET_LABELS[dataset])))
        quiet = SarcasmResult(0.1, False, 0.1, None, "test")
        after, final, notes = fuse(p, dataset, ev("just a plain sentence"), quiet, FusionConfig())
        assert np.allclose(final, p) and np.allclose(after, p) and not notes


def test_emojis_can_overrule_an_emoji_blind_model():
    labels = DATASET_LABELS["goemotions"]
    p = np.array([0.1, 0.05, 0.05, 0.1, 0.1, 0.6])       # model says surprise
    _, final, _ = fuse(p, "goemotions", ev("I can't believe you did this 😡🤬"), None, FusionConfig())
    assert labels[int(np.argmax(final))] == "anger"


def test_sarcasm_moves_mass_away_from_positive_labels():
    labels = DATASET_LABELS["tweeteval"]
    p = np.array([0.05, 0.85, 0.05, 0.05])                # model says joy
    loud = SarcasmResult(0.95, True, 0.95, None, "test")
    _, final, notes = fuse(p, "tweeteval", ev("love waiting in line 🙄"), loud, FusionConfig())
    assert final[labels.index("joy")] < 0.5
    assert labels[int(np.argmax(final))] in {"anger", "sadness"}
    assert notes


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"  [OK] {name}")
    print(f"\nALL {len(tests)} LEXICON / SARCASM / FUSION TESTS PASSED")
