"""Manually labelled emoji -> emotion lexicon, with full Unicode coverage.

Why this exists
---------------
The trained emoji branch (EM3/EM5) learned almost nothing: the emotion corpora
it was trained on carry emojis in only 2-23% of rows. Probing the fine-tuned
models showed an emoji-only input like "😂😂😂😂" scoring ~0.27 on its top class
(near uniform), and 🤡 🤬 🤮 🚀 falling outside the training vocabulary
entirely. Retraining is out of scope, so emoji knowledge is supplied
explicitly, at inference time, from a hand-curated lexicon.

Label sources, in priority order
--------------------------------
``manual``   288 emojis hand-labelled in ``data/lexicon/emoji_lexicon_manual.csv``:
             every face, heart, hand gesture, and the emotionally loaded
             symbols and finance emojis (🚀 📉 💸 💎 🐂 🐻 ...).
``derived``  Any other emoji whose official CLDR name contains an emotional
             keyword (``heart``, ``crying``, ``party`` ...).
``default``  Everything else (flags, keycaps, most objects) -> neutral with
             near-zero intensity, so it contributes almost no evidence.

``python -m src.lexicon.emoji_lexicon --export`` writes every emoji in Unicode
with its label and source to ``data/lexicon/emoji_lexicon_full.csv`` so a
human can review or override any row.

Emotion space
-------------
Unified labels: joy, love, sadness, anger, fear, surprise, disgust, neutral.
Sarcasm is deliberately NOT an emotion column. It is a pragmatic layer
(``sarcasm_cue`` in [0, 1]) that the sarcasm detector consumes separately.
"""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import emoji as emoji_lib

UNIFIED_LABELS: list[str] = [
    "joy", "love", "sadness", "anger", "fear", "surprise", "disgust", "neutral",
]

ROOT = Path(__file__).resolve().parents[2]
MANUAL_PATH = ROOT / "data" / "lexicon" / "emoji_lexicon_manual.csv"
FULL_PATH = ROOT / "data" / "lexicon" / "emoji_lexicon_full.csv"

VS16 = "️"
ZWJ = "‍"
SKIN_TONES = {chr(c) for c in range(0x1F3FB, 0x1F400)}

# Trailing emojis ("...ugh 😭😭") usually summarise the whole sentence, so they
# count for a little more than emojis mid-sentence.
TRAILING_BOOST = 1.25
# Evidence saturates: strength = 1 - exp(-K * total_weight).
EVIDENCE_K = 1.2


@dataclass(frozen=True)
class EmojiEntry:
    emoji: str
    dist: tuple[float, ...]          # over UNIFIED_LABELS, sums to 1
    polarity: float                  # -1 (negative) .. +1 (positive)
    intensity: float                 # 0 .. 1
    sarcasm_cue: float               # 0 .. 1
    source: str                      # manual | derived | default
    notes: str = ""

    @property
    def emotional_share(self) -> float:
        """Probability mass NOT on 'neutral' -- how much emotion it carries."""
        return 1.0 - self.dist[UNIFIED_LABELS.index("neutral")]

    def top_label(self) -> str:
        return UNIFIED_LABELS[max(range(len(self.dist)), key=self.dist.__getitem__)]


def _normalise(values: list[float]) -> tuple[float, ...]:
    total = sum(values)
    if total <= 0:
        out = [0.0] * len(UNIFIED_LABELS)
        out[UNIFIED_LABELS.index("neutral")] = 1.0
        return tuple(out)
    return tuple(v / total for v in values)


# ---------------------------------------------------------------------------
# Manual lexicon
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_manual(path: str | os.PathLike = MANUAL_PATH) -> dict[str, EmojiEntry]:
    entries: dict[str, EmojiEntry] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            e = row["emoji"]
            entries[e] = EmojiEntry(
                emoji=e,
                dist=_normalise([float(row[k]) for k in UNIFIED_LABELS]),
                polarity=float(row["polarity"]),
                intensity=float(row["intensity"]),
                sarcasm_cue=float(row["sarcasm_cue"]),
                source="manual",
                notes=row.get("notes", ""),
            )
    return entries


# ---------------------------------------------------------------------------
# Derived labels for every other emoji, from its official CLDR name
# ---------------------------------------------------------------------------

def _d(**kw) -> tuple[float, ...]:
    return _normalise([kw.get(k, 0.0) for k in UNIFIED_LABELS])


# First match wins. Keywords are matched against the CLDR short name,
# e.g. ':smiling_face_with_heart-eyes:'.
_KEYWORD_RULES: list[tuple[tuple[str, ...], tuple[float, ...], float, float]] = [
    (("broken_heart",), _d(sadness=0.9, love=0.1), -0.8, 0.8),
    (("heart", "kiss", "love"), _d(love=0.7, joy=0.3), 0.7, 0.5),
    (("crying", "tears"), _d(sadness=0.9, neutral=0.1), -0.7, 0.6),
    (("angry", "pouting", "rage", "enraged", "cursing"), _d(anger=0.9, disgust=0.1), -0.8, 0.8),
    (("fear", "scream", "anxious", "worried"), _d(fear=0.9, surprise=0.1), -0.7, 0.7),
    (("vomit", "nauseated"), _d(disgust=0.9, anger=0.1), -0.8, 0.7),
    (("party", "confetti", "celebrat", "trophy", "medal", "firework", "sparkler"),
     _d(joy=0.9, surprise=0.1), 0.8, 0.7),
    (("grinning", "smiling", "laugh", "beaming", "grin"), _d(joy=0.8, neutral=0.2), 0.6, 0.5),
    (("frowning", "sad", "disappointed", "pensive"), _d(sadness=0.8, neutral=0.2), -0.6, 0.5),
    (("skull", "coffin", "headstone"), _d(sadness=0.5, fear=0.3, neutral=0.2), -0.5, 0.5),
    (("flag",), _d(neutral=1.0), 0.0, 0.0),
]


def _cldr_name(e: str) -> str:
    data = emoji_lib.EMOJI_DATA.get(e)
    if data:
        return data.get("en", "").strip(":").lower()
    return emoji_lib.demojize(e).strip(":").lower()


_KEYWORD_RE_CACHE: dict[str, re.Pattern] = {}


def _keyword_matches(name: str, keyword: str) -> bool:
    """Whole-word match (plus simple plural) against a CLDR name.

    CLDR names use ``_``/``-`` as separators, e.g. ``person_with_skullcap`` or
    ``tear-off_calendar``. A naive ``keyword in name`` substring check makes
    "beverage_box" match "rage", "boxing_glove" and "gloves" match "love",
    "person_with_skullcap" match "skull", and "spouting_whale" match
    "pouting" -- none of which are the intended emotion. Word-boundary
    matching (allowing a trailing "s" for plurals like "fireworks") avoids
    that while still catching the real target names.
    """
    rx = _KEYWORD_RE_CACHE.get(keyword)
    if rx is None:
        rx = re.compile(r"\b" + re.escape(keyword) + r"s?\b")
        _KEYWORD_RE_CACHE[keyword] = rx
    normalised = name.replace("_", " ").replace("-", " ")
    return bool(rx.search(normalised))


def _derive(e: str) -> EmojiEntry:
    name = _cldr_name(e)
    for keywords, dist, polarity, intensity in _KEYWORD_RULES:
        if any(_keyword_matches(name, k) for k in keywords):
            return EmojiEntry(e, dist, polarity, intensity, 0.0, "derived", name)
    return EmojiEntry(e, _d(neutral=1.0), 0.0, 0.05, 0.0, "default", name)


# ---------------------------------------------------------------------------
# Lookup with normalisation
# ---------------------------------------------------------------------------

def _candidates(e: str) -> list[str]:
    """Progressively simpler forms of an emoji to try against the lexicon.

    👍🏽 -> 👍 (skin tone), ❤ <-> ❤️ (variation selector),
    🤦‍♂️ -> 🤦 (gendered ZWJ sequence), while exact ZWJ entries such as
    ❤️‍🔥 or 😮‍💨 still match first.
    """
    out = [e]
    no_tone = "".join(ch for ch in e if ch not in SKIN_TONES)
    no_vs = no_tone.replace(VS16, "")
    out += [no_tone, no_vs, no_vs + VS16]
    if ZWJ in no_vs:
        head = no_vs.split(ZWJ)[0]
        out += [head, head + VS16]
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


@lru_cache(maxsize=4096)
def lookup(e: str) -> EmojiEntry:
    manual = load_manual()
    for cand in _candidates(e):
        if cand in manual:
            entry = manual[cand]
            if cand != e:
                return EmojiEntry(e, entry.dist, entry.polarity, entry.intensity,
                                  entry.sarcasm_cue, "manual", f"normalised to {cand}; {entry.notes}")
            return entry
    return _derive(e)


# ---------------------------------------------------------------------------
# Multi-emoji aggregation
# ---------------------------------------------------------------------------

@dataclass
class EmojiEvidence:
    dist: tuple[float, ...]            # aggregated distribution over UNIFIED_LABELS
    strength: float                    # 0 (no evidence) .. ~1 (strong)
    polarity: float                    # weighted mean polarity
    sarcasm_cue: float                 # strongest sarcasm cue present, repetition-boosted
    mixed: bool                        # strong positive AND strong negative emojis together
    n_occurrences: int
    breakdown: list[dict] = field(default_factory=list)

    @property
    def top_label(self) -> str:
        return UNIFIED_LABELS[max(range(len(self.dist)), key=self.dist.__getitem__)]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(UNIFIED_LABELS, self.dist))


EMPTY_EVIDENCE = EmojiEvidence(
    dist=_d(neutral=1.0), strength=0.0, polarity=0.0, sarcasm_cue=0.0,
    mixed=False, n_occurrences=0, breakdown=[],
)


def _is_trailing(text: str, end: int) -> bool:
    """True if nothing but emojis/space/punctuation follows this position."""
    rest = emoji_lib.replace_emoji(text[end:], replace="")
    return not any(ch.isalnum() for ch in rest)


def aggregate(occurrences: list[dict], text: str = "") -> EmojiEvidence:
    """Combine every emoji occurrence into one piece of evidence.

    * Repetition raises weight with diminishing returns: an emoji seen ``c``
      times counts ``1 + log2(c)`` times (😭 x4 -> 3x, not 4x).
    * Trailing emojis get ``TRAILING_BOOST``.
    * Each emoji's weight is ``intensity * emotional_share`` so neutral-ish
      emojis (🤖, flags, keycaps) barely move anything.
    """
    if not occurrences:
        return EMPTY_EVIDENCE

    groups: dict[str, dict] = {}
    for occ in occurrences:
        g = groups.setdefault(occ["emoji"], {"count": 0, "trailing": False})
        g["count"] += 1
        if text and _is_trailing(text, occ["end"]):
            g["trailing"] = True

    acc = [0.0] * len(UNIFIED_LABELS)
    total_w = pol_w = 0.0
    breakdown, max_cue, max_cue_count = [], 0.0, 0
    has_pos = has_neg = False

    for e, g in groups.items():
        entry = lookup(e)
        rep = 1.0 + math.log2(g["count"])
        pos = TRAILING_BOOST if g["trailing"] else 1.0
        w = entry.intensity * entry.emotional_share * rep * pos
        for i, v in enumerate(entry.dist):
            acc[i] += w * v
        total_w += w
        pol_w += w * entry.polarity
        if entry.sarcasm_cue > max_cue:
            max_cue, max_cue_count = entry.sarcasm_cue, g["count"]
        if w > 0.15:
            has_pos |= entry.polarity > 0.4
            has_neg |= entry.polarity < -0.4
        breakdown.append({
            "emoji": e,
            "count": g["count"],
            "label": entry.top_label(),
            "polarity": entry.polarity,
            "sarcasm_cue": entry.sarcasm_cue,
            "weight": round(w, 3),
            "source": entry.source,
            "notes": entry.notes,
        })

    if total_w <= 1e-9:
        # Only neutral emojis (flags, objects): present but uninformative.
        return EmojiEvidence(_d(neutral=1.0), 0.0, 0.0, max_cue, False,
                             len(occurrences), breakdown)

    dist = tuple(v / total_w for v in acc)
    strength = 1.0 - math.exp(-EVIDENCE_K * total_w)
    # Repeating a sarcasm marker (🙄🙄) makes the cue a little more certain.
    cue = min(1.0, max_cue * min(1.0, 0.85 + 0.1 * max_cue_count))
    breakdown.sort(key=lambda b: -b["weight"])
    return EmojiEvidence(dist, strength, pol_w / total_w, cue,
                         has_pos and has_neg, len(occurrences), breakdown)


# ---------------------------------------------------------------------------
# Export: every emoji in Unicode with its label
# ---------------------------------------------------------------------------

def export_full(path: str | os.PathLike = FULL_PATH) -> dict[str, int]:
    """Write a label row for EVERY fully-qualified emoji known to the library."""
    rows, counts = [], {"manual": 0, "derived": 0, "default": 0}
    for e, data in emoji_lib.EMOJI_DATA.items():
        if data.get("status") != emoji_lib.STATUS["fully_qualified"]:
            continue
        entry = lookup(e)
        counts[entry.source] += 1
        rows.append([e, _cldr_name(e), *[f"{v:.3f}" for v in entry.dist],
                     f"{entry.polarity:.2f}", f"{entry.intensity:.2f}",
                     f"{entry.sarcasm_cue:.2f}", entry.top_label(), entry.source])
    rows.sort(key=lambda r: ({"manual": 0, "derived": 1, "default": 2}[r[-1]], r[1]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["emoji", "cldr_name", *UNIFIED_LABELS, "polarity", "intensity",
                    "sarcasm_cue", "top_label", "source"])
        w.writerows(rows)
    return counts


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", action="store_true", help="Write the full-coverage CSV.")
    p.add_argument("--lookup", nargs="*", default=[], help="Emojis to look up.")
    args = p.parse_args()

    if args.export:
        c = export_full()
        total = sum(c.values())
        print(f"Wrote {total} emojis -> {FULL_PATH}")
        for k, v in c.items():
            print(f"  {k:<8} {v:>5}  ({100 * v / total:.1f}%)")
    for e in args.lookup:
        entry = lookup(e)
        top = sorted(zip(UNIFIED_LABELS, entry.dist), key=lambda t: -t[1])[:3]
        print(f"{e}  [{entry.source}]  top={top}  polarity={entry.polarity:+.2f}  "
              f"sarcasm_cue={entry.sarcasm_cue:.2f}  {entry.notes}")
