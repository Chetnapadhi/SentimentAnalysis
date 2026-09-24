"""Hand-curated text cues used by the sarcasm detector.

These are deliberately small and conservative. The fine-tuned transformer is
far better at reading plain text than any word list, so text cues are NOT
used to override its emotion prediction. They exist for one job the model
cannot do: noticing *incongruity* -- positive words wrapped around a bad
situation ("I just love waiting 3 hours at the DMV"), which is the classic
signature of sarcasm (Riloff et al., 2013, "Sarcasm as Contrast between a
Positive Sentiment and Negative Situation").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Polarity words
# ---------------------------------------------------------------------------

POSITIVE_WORDS = frozenset("""
love loved loving loves adore adored like liked enjoy enjoyed enjoying great greatest
good best better awesome amazing wonderful fantastic fabulous excellent perfect lovely
brilliant beautiful nice glad happy happiest excited exciting thrilled delighted fun
favorite favourite blessed lucky grateful thankful thanks thank yay hooray woohoo wow
cool sweet superb incredible impressive genius smart stellar magnificent terrific
joy joyful cheerful pleased proud win won winning winner success successful
""".split())

NEGATIVE_WORDS = frozenset("""
hate hated hating hates awful terrible horrible worst bad worse sad angry mad upset
annoyed annoying furious pissed disgusting gross nasty sick tired exhausted miserable
depressed depressing lonely scared afraid terrified anxious worried nervous stressed
cry crying cried hurt pain painful broken ruined disaster fail failed failure lose lost
losing loser stupid dumb useless pathetic ridiculous disappointed disappointing boring
unfair cruel ugly sucks suck sucked damn hell crap wtf ugh smh meh
""".split())

NEGATORS = frozenset("""
not no never nothing nobody none neither nor cannot can't cant won't wont don't dont
doesn't doesnt didn't didnt isn't isnt aren't arent wasn't wasnt shouldn't wouldn't
hardly barely
""".split())

INTENSIFIERS = frozenset("""
so very really extremely super totally absolutely truly incredibly soo sooo
""".split())

# ---------------------------------------------------------------------------
# Negative situations: bad circumstances described WITHOUT a sentiment word
# ---------------------------------------------------------------------------

NEGATIVE_SITUATION_PATTERNS = [
    # "can't wait" is excitement, not a bad situation -- exclude it.
    r"(?<!can't )(?<!cant )(?<!cannot )\bwait(?:ing|ed)?\b",
    r"\bqueue\b", r"\bin line\b", r"\bdmv\b",
    r"\bcancel+ed\b", r"\bdelay(?:ed|s)?\b", r"\bstuck\b", r"\btraffic\b",
    r"\bmondays?\b", r"\bhomework\b", r"\bexams?\b", r"\btaxes\b", r"\bovertime\b",
    r"\bdeadlines?\b", r"\bmeetings?\b", r"\bdentist\b", r"\bhospital\b",
    r"\bsurgery\b", r"\bfuneral\b", r"\bdied\b", r"\bdeath\b", r"\bpassed away\b",
    r"\blos(?:t|ing|e)\b", r"\bbroke\b", r"\bbroken\b", r"\bcrash(?:ed|ing)?\b",
    r"\bdumped\b", r"\bfired\b", r"\blaid off\b", r"\brejected\b", r"\bsick\b",
    r"\bflu\b", r"\bheadache\b", r"\bhangover\b", r"\brain(?:ing|y)?\b",
    r"\bflat tire\b", r"\bdead battery\b", r"\bspilled\b", r"\bmissed\b",
    r"\blate again\b", r"\bbills?\b", r"\bdebt\b", r"\brent\b", r"\bcrutches\b",
    r"\bno sleep\b", r"\binsomnia\b", r"\b[3-5] ?am\b", r"\balarm\b",
    r"\bpower (?:cut|outage)\b", r"\bwifi\b.*\bdown\b", r"\binternet\b.*\bdown\b",
    r"\bchores\b", r"\blaundry\b", r"\bparking ticket\b", r"\btowed\b",
    # finance / StockTwits
    # Bare "red" would fire on "red roses"; only the finance idioms count.
    r"\b(?:in the|all|deep|sea of) red\b", r"\bdump(?:ed|ing)?\b", r"\bbag ?hold(?:er|ing)\b", r"\bmargin call\b",
    r"\bliquidat(?:ed|ion)\b", r"\brug ?pull(?:ed)?\b", r"\bscam\b", r"\bdown \d+%",
    r"-\d+(?:\.\d+)?%", r"\bhalf my\b", r"\bmy (?:money|savings|portfolio)\b",
    # Bare "again" would fire on "see you again!"; only a bad thing recurring counts.
    r"\b(?:cancel+ed|broke|crashed|down|late|failed|lost|sick|delayed) again\b",
]
_NEG_SITUATION_RE = [re.compile(p, re.I) for p in NEGATIVE_SITUATION_PATTERNS]

# ---------------------------------------------------------------------------
# Sarcasm phrases
# ---------------------------------------------------------------------------

EXPLICIT_MARKERS = [
    r"(?:^|\s)/s\b", r"#sarcas(?:m|tic)\b", r"#not\b", r"#iron(?:y|ic)\b",
    r"#yeahright\b", r"#jk\b",
]

# (pattern, weight, needs_negative_situation)
SARCASM_PHRASES: list[tuple[str, float, bool]] = [
    (r"\byeah,? (?:right|sure)\b", 1.8, False),
    (r"\boh,? (?:great|wonderful|joy|perfect|fantastic|lovely|brilliant|nice|good|awesome|cool|goody)\b", 1.6, False),
    (r"\bjust what i (?:needed|wanted)\b", 1.8, False),
    (r"\bthanks? (?:a lot|so much)\b", 0.6, True),
    (r"\bthanks for nothing\b", 2.0, False),
    (r"\b(?:i|we) (?:just )?love (?:it )?(?:when|how)\b", 1.2, False),
    (r"\blove (?:that|this) for me\b", 1.6, False),
    (r"\bso glad\b", 0.8, True),
    (r"\bwhat a (?:surprise|shock|shocker|treat|joy|great idea)\b", 1.6, False),
    (r"\bbig surprise\b", 1.4, False),
    (r"\bshocker\b", 1.2, False),
    (r"\bsaid no one ever\b", 2.5, False),
    (r"\bas if\b", 1.0, False),
    (r"\bsure,? (?:jan|buddy|pal|thing)\b", 1.6, False),
    (r"\bcool story\b", 1.6, False),
    (r"\bwow,? just wow\b", 1.2, False),
    (r"\bhow (?:nice|lovely|original|convenient|thoughtful)\b", 1.2, False),
    (r"\bnothing (?:like|beats)\b", 0.8, True),
    (r"\bcan'?t wait\b", 0.9, True),
    (r"\bmy favou?rite\b", 0.9, True),
    (r"\bbest (?:day|thing|idea|decision) ever\b", 1.0, True),
    (r"\bgreat job\b", 1.0, True),
    (r"\bwell done\b", 0.8, True),
    (r"\bslow clap\b", 1.8, False),
    (r"\bbecause that (?:makes sense|went well|worked)\b", 1.8, False),
    (r"\bwhat could (?:possibly )?go wrong\b", 2.0, False),
    (r"\bperfect timing\b", 1.2, True),
    (r"\blucky me\b", 1.4, False),
    (r"\bstory of my life\b", 1.0, False),
    (r"\b(?:yay|hooray|woohoo)\b", 0.9, True),
    (r"\bexactly what\b", 0.7, True),
    (r"\b(?:clearly|obviously|totally|definitely)\b", 0.4, False),
    (r"\bto the moon\b", 0.5, True),
]
_EXPLICIT_RE = [re.compile(p, re.I) for p in EXPLICIT_MARKERS]


def strip_explicit_markers(text: str) -> str:
    """Remove /s, #sarcasm, #irony, #not ... from a text.

    Needed for EVALUATION on TweetEval irony: SemEval-2018 Task 3 collected its
    ironic tweets by searching for exactly these hashtags, and the Hugging Face
    release keeps them -- every one of the 311 ironic test tweets contains one.
    Scoring with them present measures hashtag lookup, not sarcasm detection.
    At inference time on real posts the markers are genuine signal and are
    left in.
    """
    for rx in _EXPLICIT_RE:
        text = rx.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
_PHRASE_RE = [(re.compile(p, re.I), w, needs) for p, w, needs in SARCASM_PHRASES]

_WORD_RE = re.compile(r"[a-z']+")
_SCARE_QUOTES_RE = re.compile(r"[\"“”]\s*\w+(?:\s\w+)?\s*[\"“”]")
_ELONGATION_RE = re.compile(r"\b\w*([a-z])\1{2,}\w*\b", re.I)


@dataclass
class TextCues:
    polarity: float                    # -1 .. +1
    positive_hits: list[str] = field(default_factory=list)
    negative_hits: list[str] = field(default_factory=list)
    negative_situations: list[str] = field(default_factory=list)
    explicit_marker: bool = False
    phrases: list[tuple[str, float]] = field(default_factory=list)
    scare_quotes: bool = False
    elongation: bool = False


def analyse_text(text: str) -> TextCues:
    """Extract polarity (with simple negation handling) and sarcasm cues."""
    lower = text.lower()
    tokens = _WORD_RE.findall(lower)

    score, pos_hits, neg_hits = 0.0, [], []
    for i, tok in enumerate(tokens):
        val = 1.0 if tok in POSITIVE_WORDS else -1.0 if tok in NEGATIVE_WORDS else 0.0
        if not val:
            continue
        window = tokens[max(0, i - 3):i]
        if any(w in NEGATORS for w in window):
            val = -val * 0.8                      # "not good" is weaker than "bad"
        if any(w in INTENSIFIERS for w in window):
            val *= 1.5
        score += val
        (pos_hits if val > 0 else neg_hits).append(tok)
    polarity = math.tanh(score / 2.0)

    situations = []
    for rx in _NEG_SITUATION_RE:
        m = rx.search(lower)
        if m:
            situations.append(m.group(0))

    phrases = []
    for rx, w, needs_situation in _PHRASE_RE:
        m = rx.search(lower)
        if m and (not needs_situation or situations):
            phrases.append((m.group(0), w))

    return TextCues(
        polarity=polarity,
        positive_hits=pos_hits,
        negative_hits=neg_hits,
        negative_situations=situations,
        explicit_marker=any(rx.search(lower) for rx in _EXPLICIT_RE),
        phrases=phrases,
        scare_quotes=bool(_SCARE_QUOTES_RE.search(text)),
        elongation=bool(_ELONGATION_RE.search(lower)),
    )
