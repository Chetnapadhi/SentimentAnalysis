"""Inference-time fusion: trained model + emoji lexicon + sarcasm.

    p_final(d)  ∝  p_model(d) · q_emoji(d) ^ (λ · S)

``p_model``  the fine-tuned EM0/EM3 distribution over the dataset's labels
``q_emoji``  the aggregated emoji-lexicon distribution, projected onto the
             same labels and lightly smoothed
``S``        emoji evidence strength in [0, 1): 0 when there are no emojis
             or only uninformative ones (flags, objects)
``λ``        emoji weight, chosen on the validation split

This is a product of experts. Its key property: **with S = 0 the output is
exactly p_model**, so sentences without emotional emojis are never changed by
the emoji term -- the trained model is only overruled where it is known to be
blind.

Sarcasm then moves probability from the positive labels to the negative ones,
in proportion to P(sarcasm). The negative target follows the emoji evidence
when there is some (🤡 -> disgust/anger, 😭 -> sadness), else a default.

Both the surface reading ("joy") and the intended reading ("anger") are
returned, since for sarcasm both are true in different senses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.preprocessing import extract_emoji_occurrences
from src.lexicon.emoji_lexicon import UNIFIED_LABELS, EmojiEvidence, aggregate
from src.lexicon.sarcasm import SarcasmDetector, SarcasmResult

# ---------------------------------------------------------------------------
# Unified -> dataset label-space projections
# ---------------------------------------------------------------------------
# Rows: unified label. Columns: dataset label. "neutral" maps to uniform on
# purpose: an emoji that carries no emotion must not push any class.

DATASET_LABELS: dict[str, list[str]] = {
    "goemotions": ["anger", "disgust", "fear", "joy", "sadness", "surprise"],
    "tweeteval": ["anger", "joy", "optimism", "sadness"],
}

_PROJECTIONS: dict[str, dict[str, dict[str, float]]] = {
    # Ekman grouping from the GoEmotions paper puts love under joy.
    "goemotions": {
        "joy": {"joy": 1.0},
        "love": {"joy": 1.0},
        "sadness": {"sadness": 1.0},
        "anger": {"anger": 1.0},
        "fear": {"fear": 1.0},
        "surprise": {"surprise": 1.0},
        "disgust": {"disgust": 1.0},
    },
    # TweetEval has no fear/disgust/surprise/love: map each to its closest.
    "tweeteval": {
        "joy": {"joy": 0.75, "optimism": 0.25},
        "love": {"joy": 0.8, "optimism": 0.2},
        "sadness": {"sadness": 1.0},
        "anger": {"anger": 1.0},
        "fear": {"sadness": 0.7, "anger": 0.3},
        "surprise": {"joy": 0.4, "optimism": 0.2, "anger": 0.2, "sadness": 0.2},
        "disgust": {"anger": 1.0},
    },
}

POSITIVE_LABELS = {"goemotions": {"joy"}, "tweeteval": {"joy", "optimism"}}
DEFAULT_SARCASM_TARGET = {
    "goemotions": {"anger": 0.45, "disgust": 0.25, "sadness": 0.30},
    "tweeteval": {"anger": 0.65, "sadness": 0.35},
}
NEGATIVE_UNIFIED = ("sadness", "anger", "fear", "disgust")


def project(unified: dict[str, float], dataset: str, smoothing: float = 0.1) -> np.ndarray:
    """Project a unified-label distribution onto a dataset's label space."""
    labels = DATASET_LABELS[dataset]
    proj = _PROJECTIONS[dataset]
    out = np.zeros(len(labels))
    for u, mass in unified.items():
        if u == "neutral":
            out += mass / len(labels)
            continue
        for d, share in proj[u].items():
            out[labels.index(d)] += mass * share
    out = out / out.sum() if out.sum() > 0 else np.full(len(labels), 1 / len(labels))
    return (1 - smoothing) * out + smoothing / len(labels)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    emoji_weight: float = 1.5        # λ
    sarcasm_shift: float = 0.85      # fraction of positive mass moved at P(sarcasm)=1
    sarcasm_threshold: float = 0.5   # below this, no shift is applied


# When sarcasm is detected, the "likely intended" reading is shown next to the
# headline prediction. Validation showed that shifting the headline itself does
# not improve benchmark accuracy (the datasets' annotators often labelled the
# literal emotion), so this shift is presented as an interpretation only.
INTERPRETIVE_SARCASM_SHIFT = 0.85


@dataclass
class HybridResult:
    labels: list[str]
    probs: np.ndarray                 # final (validated configuration)
    model_probs: np.ndarray           # trained model alone
    after_emoji_probs: np.ndarray     # model x emoji, before any sarcasm shift
    evidence: EmojiEvidence
    sarcasm: SarcasmResult
    notes: list[str] = field(default_factory=list)
    intended_probs: np.ndarray | None = None   # sarcasm-adjusted interpretation

    @property
    def intended_label(self) -> str | None:
        if self.intended_probs is None:
            return None
        return self.labels[int(np.argmax(self.intended_probs))]

    @property
    def label(self) -> str:
        return self.labels[int(np.argmax(self.probs))]

    @property
    def surface_label(self) -> str:
        """What the words (plus emojis) literally say, before the sarcasm shift."""
        return self.labels[int(np.argmax(self.after_emoji_probs))]

    @property
    def model_label(self) -> str:
        return self.labels[int(np.argmax(self.model_probs))]


def fuse(model_probs: np.ndarray, dataset: str, evidence: EmojiEvidence,
         sarcasm: SarcasmResult | None, cfg: FusionConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pure function: returns (after_emoji, final, notes). No I/O, easy to test."""
    labels = DATASET_LABELS[dataset]
    p = np.asarray(model_probs, dtype=float)
    p = p / p.sum()
    notes: list[str] = []

    # 1) emoji evidence (product of experts)
    exponent = cfg.emoji_weight * evidence.strength
    if exponent > 0:
        q = project(evidence.as_dict(), dataset)
        fused = p * np.power(q, exponent)
        fused /= fused.sum()
        if labels[int(np.argmax(fused))] != labels[int(np.argmax(p))]:
            notes.append(f"emojis changed the prediction from {labels[int(np.argmax(p))]} "
                         f"to {labels[int(np.argmax(fused))]}")
    else:
        fused = p.copy()
    after_emoji = fused.copy()

    # 2) sarcasm: move mass from positive labels to the intended negative ones.
    #    Only a corroborated ("detected") verdict may rewrite the emotion.
    if sarcasm is not None and sarcasm.is_sarcastic and cfg.sarcasm_shift > 0:
        pos_idx = [labels.index(l) for l in POSITIVE_LABELS[dataset]]
        pos_mass = fused[pos_idx].sum()
        moved = cfg.sarcasm_shift * sarcasm.probability * pos_mass
        if moved > 1e-6:
            neg_unified = {u: evidence.as_dict()[u] for u in NEGATIVE_UNIFIED}
            if evidence.strength > 0.2 and sum(neg_unified.values()) > 0.15:
                target = project(neg_unified, dataset, smoothing=0.0)
                target[pos_idx] = 0.0
                target = target / target.sum() if target.sum() > 0 else None
            else:
                target = None
            if target is None:
                target = np.zeros(len(labels))
                for l, v in DEFAULT_SARCASM_TARGET[dataset].items():
                    target[labels.index(l)] = v
            fused[pos_idx] *= (1 - moved / pos_mass)
            fused += moved * target
            fused /= fused.sum()
            notes.append(f"sarcasm ({sarcasm.probability:.0%}) moved {moved:.0%} of the "
                         f"probability from {'/'.join(sorted(POSITIVE_LABELS[dataset]))} "
                         f"to the intended negative emotion")
    return after_emoji, fused, notes


class HybridEmotionPredictor:
    """Wraps any `predict_proba(text) -> np.ndarray` model function."""

    def __init__(self, dataset: str, sarcasm: SarcasmDetector | None = None,
                 cfg: FusionConfig | None = None):
        if dataset not in DATASET_LABELS:
            raise ValueError(f"Unknown dataset {dataset!r}")
        self.dataset = dataset
        self.labels = DATASET_LABELS[dataset]
        self.sarcasm = sarcasm if sarcasm is not None else SarcasmDetector(backend="rules")
        self.cfg = cfg or FusionConfig()

    def predict(self, text: str, model_probs: np.ndarray,
                sarcasm_model_prob: float | None = None) -> HybridResult:
        evidence = aggregate(extract_emoji_occurrences(text), text)
        sarc = self.sarcasm.score(text, model_prob=sarcasm_model_prob, evidence=evidence)
        after_emoji, final, notes = fuse(model_probs, self.dataset, evidence, sarc, self.cfg)
        intended = None
        if sarc.is_sarcastic:
            interp = FusionConfig(self.cfg.emoji_weight, INTERPRETIVE_SARCASM_SHIFT,
                                  self.cfg.sarcasm_threshold)
            _, intended, _ = fuse(model_probs, self.dataset, evidence, sarc, interp)
        return HybridResult(self.labels, final, np.asarray(model_probs, dtype=float),
                            after_emoji, evidence, sarc, notes, intended)
