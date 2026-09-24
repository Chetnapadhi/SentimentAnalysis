"""Regression test for the derived-label keyword matcher in emoji_lexicon.py.

Kept in its own file (not test_lexicon_hybrid.py) to avoid clobbering
concurrent edits to that file.

    python tests/test_emoji_lexicon_keywords.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.lexicon.emoji_lexicon import lookup

# Each of these previously matched a keyword as a substring of an unrelated
# CLDR name and came out emotionally mislabelled:
#   beverage_box ~ "rage", boxing_glove/gloves ~ "love", person_with_skullcap
#   ~ "skull", spouting_whale ~ "pouting", tear-off_calendar ~ "tear",
#   slovenia ~ "love".
_FALSE_POSITIVES = ["🧃", "🥊", "🧤", "👲", "🐳", "📆", "🇸🇮"]

# These rely on the same keywords matching for real and must keep working.
_TRUE_POSITIVES = {
    "🎆": "joy",       # fireworks (plural of "firework")
    "❤️": "love",      # heart
    "😭": "sadness",   # loudly crying face
    "😠": "anger",      # angry face
    "🤮": "disgust",   # face vomiting
    "😱": "fear",       # face screaming in fear
}


def test_keyword_matcher_ignores_embedded_substrings():
    for e in _FALSE_POSITIVES:
        entry = lookup(e)
        assert entry.top_label() == "neutral", (e, entry.top_label(), entry.notes)
        assert entry.source in {"derived", "default"}, (e, entry.source)


def test_keyword_matcher_still_catches_real_matches():
    for e, expected in _TRUE_POSITIVES.items():
        entry = lookup(e)
        assert entry.top_label() == expected, (e, entry.top_label())


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"  [OK] {name}")
    print(f"\nALL {len(tests)} EMOJI-LEXICON KEYWORD TESTS PASSED")
