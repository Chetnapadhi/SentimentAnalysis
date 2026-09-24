"""Sarcasm detection without training anything in this project.

None of the trained models has a sarcasm output: TweetEval emotion and
GoEmotions contain no sarcasm label. Before this module, "I just love waiting
3 hours at the DMV 🙄🙄" came out as joy 0.98.

Two evidence sources are combined in logit space:

1. **Rules** (always available, fully explainable)
   - explicit markers: ``/s``, ``#sarcasm``, ``#not``
   - sarcastic phrases: "yeah right", "oh great", "just what I needed" ...
   - positive words about a negative situation (Riloff et al., 2013)
   - sarcasm-marker emojis from the manual lexicon: 🙄 🙃 🤡 😒 😏 🤦
   - emoji/text polarity conflict: "Great news 😡", "car got towed 🙂"

2. **A pretrained irony classifier** (optional; used when it can be loaded)
   ``cardiffnlp/twitter-roberta-base-irony`` -- fine-tuned by Cardiff NLP on
   TweetEval-irony *train*. It is an external off-the-shelf component, not a
   model trained by this project. It reads subtle irony far better than rules
   but has no special knowledge of emoji, so rule evidence is added on top.

``SARCASM_MODEL`` (environment variable) can point at any other Hugging Face
sequence classifier whose label 1 means "irony". If none can be loaded
(offline, low disk), the detector silently falls back to rules only and
reports that in ``backend``.

Scores are probabilities; ``explain`` lists every cue that fired, so a user
can see *why* a sentence was flagged.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from src.data.preprocessing import extract_emoji_occurrences
from src.lexicon.emoji_lexicon import EmojiEvidence, aggregate
from src.lexicon.text_cues import TextCues, analyse_text

ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_IRONY = "cardiffnlp/twitter-roberta-base-irony"

# Rule weights (logit units). The intercept means one weak cue alone is never
# enough; sarcasm needs at least two converging signals.
W_INTERCEPT = -2.5
W_EXPLICIT = 4.0
W_PHRASE_CAP = 2.5
W_INCONGRUITY = 1.5
W_EMOJI_CUE = 2.6
W_TEXT_POS_EMOJI_NEG = 1.4
W_BAD_NEWS_HAPPY_EMOJI = 1.0
W_STYLE = 0.4

# Emojis at or above this sarcasm cue (🙄 🤡 🙃 😒 😏 🤦) are markers in their own
# right. Below it (😂 👏 🙂 👍 😉 💀 ✨) they are just as often sincere, so they
# only add evidence when some other cue already suggests sarcasm.
STRONG_EMOJI_CUE = 0.6

# A learned-model verdict needs at least this much rule evidence behind it to
# become "detected"; otherwise it is reported as "possible". The pretrained
# irony classifier was trained on ~48% ironic tweets and scores plain sincere
# enthusiasm highly ("I am so happy today" -> 0.93), so on its own it is not
# trusted to rewrite anything.
CORROBORATION_MIN = 1.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


@dataclass
class SarcasmResult:
    probability: float
    is_sarcastic: bool                 # "detected": corroborated, allowed to change the emotion
    rule_probability: float
    model_probability: float | None
    backend: str
    reasons: list[str] = field(default_factory=list)
    level: str = "none"                # "detected" | "possible" | "none"
    rule_score: float = 0.0


def decide(prob: float, rule_score: float, threshold: float, has_model: bool) -> str:
    """Two-tier verdict.

    detected  rules alone are confident (>= 2 converging cues), or the combined
              score clears the threshold AND rule evidence corroborates it
    possible  the combined score clears the threshold on the learned model alone
    none      otherwise
    """
    if _sigmoid(W_INTERCEPT + rule_score) >= 0.5:
        return "detected"
    if prob >= threshold:
        if not has_model or rule_score >= CORROBORATION_MIN:
            return "detected"
        return "possible"
    return "none"


def rule_features(cues: TextCues, ev: EmojiEvidence) -> tuple[float, list[str]]:
    """Sum of rule evidence in logit units (excluding the intercept)."""
    score, reasons = 0.0, []

    if cues.explicit_marker:
        score += W_EXPLICIT
        reasons.append("explicit sarcasm marker (/s, #sarcasm, #not)")

    if cues.phrases:
        phrase_score = min(W_PHRASE_CAP, sum(w for _, w in cues.phrases))
        score += phrase_score
        reasons.append("sarcastic phrasing: " + ", ".join(f"'{p}'" for p, _ in cues.phrases))

    if cues.positive_hits and cues.negative_situations:
        score += W_INCONGRUITY
        reasons.append(
            f"positive words ({', '.join(sorted(set(cues.positive_hits))[:3])}) about a "
            f"negative situation ({', '.join(cues.negative_situations[:3])})"
        )

    # Context from anything other than the emojis themselves.
    has_context = bool(cues.explicit_marker or cues.phrases or cues.negative_situations
                       or (ev.strength > 0.3 and cues.polarity > 0.15 and ev.polarity < -0.25))
    if ev.sarcasm_cue >= STRONG_EMOJI_CUE:
        score += W_EMOJI_CUE * ev.sarcasm_cue
        markers = [b["emoji"] for b in ev.breakdown if b["sarcasm_cue"] >= STRONG_EMOJI_CUE]
        reasons.append(f"sarcasm-marker emoji {' '.join(markers)}")
    elif ev.sarcasm_cue > 0 and has_context:
        score += W_EMOJI_CUE * ev.sarcasm_cue
        markers = [b["emoji"] for b in ev.breakdown if b["sarcasm_cue"] >= 0.3]
        if markers:
            reasons.append(f"often-ironic emoji {' '.join(markers)} in a sarcastic context")

    if ev.strength > 0.3:
        if cues.polarity > 0.15 and ev.polarity < -0.25:
            score += W_TEXT_POS_EMOJI_NEG
            reasons.append("positive words but negative emojis")
        bad_news = cues.polarity < -0.15 or cues.negative_situations
        if bad_news and ev.polarity > 0.3 and ev.sarcasm_cue >= 0.3:
            score += W_BAD_NEWS_HAPPY_EMOJI
            reasons.append("bad news delivered with a smiling or laughing emoji")

    style = []
    if cues.scare_quotes:
        style.append("scare quotes")
    if cues.elongation and (cues.phrases or cues.negative_situations):
        style.append("elongated word")
    if style:
        score += W_STYLE * len(style)
        reasons.append("style: " + ", ".join(style))

    return score, reasons


class SarcasmDetector:
    """Rules + (optional) learned irony classifier, blended in logit space.

    ``rule_weight`` scales how much rule evidence is added on top of the
    learned model's logit. With no learned model, the rules alone decide.
    """

    def __init__(self, backend: str = "auto", threshold: float = 0.5,
                 rule_weight: float = 0.6, device: str = "cpu"):
        self.threshold = threshold
        self.rule_weight = rule_weight
        self.device = device
        self._model = None
        self._tokenizer = None
        self.backend = "rules"
        if backend in ("auto", "pretrained"):
            self._try_load_pretrained(required=backend == "pretrained")

    # -- learned backend -----------------------------------------------------
    def _try_load_pretrained(self, required: bool) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            name = os.environ.get("SARCASM_MODEL", PRETRAINED_IRONY)
            self._tokenizer = AutoTokenizer.from_pretrained(name)
            self._model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device)
            self._model.eval()
            self.backend = f"pretrained:{name}"
        except Exception as exc:              # offline, no disk, etc.
            if required:
                raise
            self._model = None
            self.backend = f"rules (pretrained unavailable: {type(exc).__name__})"

    def model_probabilities(self, texts: list[str], batch_size: int = 32) -> list[float] | None:
        """P(irony) from the learned model, or None if no model is loaded."""
        if self._model is None:
            return None
        import torch
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self._tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                      max_length=128, return_tensors="pt").to(self.device)
                probs = torch.softmax(self._model(**enc).logits, dim=-1)[:, 1]
                out.extend(probs.cpu().tolist())
        return out

    # -- scoring -------------------------------------------------------------
    def score(self, text: str, model_prob: float | None = None,
              evidence: EmojiEvidence | None = None) -> SarcasmResult:
        cues = analyse_text(text)
        ev = evidence if evidence is not None else aggregate(extract_emoji_occurrences(text), text)
        rule_score, reasons = rule_features(cues, ev)
        rule_prob = _sigmoid(W_INTERCEPT + rule_score)

        if model_prob is None and self._model is not None:
            model_prob = self.model_probabilities([text])[0]

        if model_prob is None:
            prob = rule_prob
        else:
            prob = _sigmoid(_logit(model_prob) + self.rule_weight * rule_score)
            if model_prob >= 0.5:
                reasons.insert(0, f"irony classifier: {model_prob:.0%}")

        level = decide(prob, rule_score, self.threshold, has_model=model_prob is not None)
        if level == "possible":
            reasons.append("irony classifier only -- no textual or emoji cue backs it up, "
                           "so the emotion is left unchanged")
        return SarcasmResult(
            probability=prob,
            is_sarcastic=level == "detected",
            rule_probability=rule_prob,
            model_probability=model_prob,
            backend=self.backend,
            reasons=reasons,
            level=level,
            rule_score=rule_score,
        )
