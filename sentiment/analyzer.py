import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# ── Phrase normalisation ───────────────────────────────────────────────────────
# VADER scores individual words without context. "lower" = −1.2 in VADER's
# lexicon, so "lower cost / lower taxes / lower tariffs" score as negative even
# though they describe a positive outcome. We replace these known false-negative
# patterns with semantically equivalent phrases that VADER handles correctly.
_PHRASE_SUBS = [
    # "lower/reduce cost|price|tax|tariff|inflation|burden|rate" → positive framing
    (re.compile(
        r'\b(lower(?:ing)?|reduc(?:e|ing|tion\s+of))\s+'
        r'(cost|costs|price|prices|tax|taxes|tariff|tariffs|inflation|deficit|burden|rate|rates)\b',
        re.IGNORECASE,
    ), r'ease \2 beneficially'),
    # "lower cost of doing business" — full phrase
    (re.compile(r'\blower\s+cost\s+of\s+doing\s+business\b', re.IGNORECASE),
     'improve ease of doing business'),
    # "ease of doing business" — already positive but make it explicit
    (re.compile(r'\bease\s+of\s+doing\s+business\b', re.IGNORECASE),
     'excellent business environment'),
]


def _normalize(text: str) -> str:
    """Apply phrase-level substitutions before VADER scoring."""
    for pattern, replacement in _PHRASE_SUBS:
        text = pattern.sub(replacement, text)
    return text

# Add India-specific political terms to VADER lexicon
CUSTOM_LEXICON = {
    "atmanirbhar": 2.0, "viksit": 1.5, "growth": 1.0, "boost": 1.5,
    "scam": -2.5, "corruption": -2.5, "controversy": -1.5, "slammed": -1.5,
    "praised": 2.0, "lauded": 2.0, "historic": 1.5, "milestone": 1.5,
    "declined": -1.0, "slump": -1.5, "deficit": -1.0, "slowdown": -1.5,
    "surge": 1.5, "record": 1.0, "relief": 1.5, "setback": -1.5,
    "allegation": -2.0, "arrested": -2.0, "resign": -2.0, "protest": -1.0,
    "achievement": 2.0, "success": 2.0, "initiative": 1.0, "reform": 1.0,
    "opposition": -0.5, "accused": -2.0, "violated": -2.0, "crisis": -2.0,
}
_analyzer.lexicon.update(CUSTOM_LEXICON)

# ── Subject-aware sentiment adjustment ────────────────────────────────────────
# When the Minister is the CRITIC (attacking/questioning others), the story is
# not negative for him — it's normal political activity. Only flag as negative
# when the Minister is the TARGET of criticism.

_MIN = r"(?:piyush\s+goyal|goyal|union\s+minister|commerce\s+minister|minister\s+goyal)"

# Minister is the grammatical SUBJECT performing a critical action on someone else
_MINISTER_AS_CRITIC = re.compile(
    r"(?<!\w)" + _MIN +
    r".{0,50}"
    r"(criticis|slams?|attacks?|blasts?|questions?|warns?|urges?|accuses?|dismisses?|rejects?|opposes?|counters?|rebukes?|condemns?)",
    re.IGNORECASE,
)

# Someone/something is criticising the Minister (Minister is TARGET)
_MINISTER_AS_TARGET = re.compile(
    r"(criticis|slams?|attacks?|blasts?|accuses?|targets?|opposes?|rebukes?|condemns?)"
    r".{0,40}" + _MIN,
    re.IGNORECASE,
)

# Minister himself is in trouble (always genuinely negative)
_MINISTER_TROUBLE = re.compile(
    _MIN +
    r".{0,30}(resign|arrest|bail|probe|scam|corruption|scandal|allegation|charge|fir\b|under\s+fire|under\s+scrutiny)",
    re.IGNORECASE,
)


def adjust_for_subject(text: str, raw_compound: float) -> tuple:
    """
    Re-evaluate sentiment direction based on who is subject vs target.
    Returns (adjusted_compound, adjustment_note).
    """
    if raw_compound >= -0.05:
        return raw_compound, ""          # Already non-negative — no change needed

    t = text.lower()
    is_critic  = bool(_MINISTER_AS_CRITIC.search(t))
    is_target  = bool(_MINISTER_AS_TARGET.search(t))
    is_trouble = bool(_MINISTER_TROUBLE.search(t))

    if is_critic and not is_target and not is_trouble:
        # Minister is criticising someone else — dampen to neutral
        adjusted = max(raw_compound * 0.15, -0.04)
        return adjusted, "reclassified: minister as critic, not subject of criticism"

    return raw_compound, ""


def score(text: str, apply_subject_adjustment: bool = False) -> dict:
    """Return VADER scores + label for a piece of text."""
    if not text or not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "neutral", "adjustment": ""}
    scores = _analyzer.polarity_scores(_normalize(text))
    c = scores["compound"]
    note = ""
    if apply_subject_adjustment:
        c, note = adjust_for_subject(text, c)
    label = "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    return {**scores, "compound": c, "label": label, "adjustment": note}


def score_with_engagement(text: str, likes: int = 0, comments: int = 0, shares: int = 0) -> dict:
    """Sentiment weighted by engagement — high-engagement posts carry more signal."""
    base = score(text)
    engagement = likes + (comments * 2) + (shares * 3)
    base["engagement"] = engagement
    base["weighted_compound"] = base["compound"] * (1 + min(engagement / 10000, 2.0))
    base["weighted_compound"] = max(-1.0, min(1.0, base["weighted_compound"]))
    return base


def aggregate_sentiment(items: list) -> dict:
    """
    Aggregate a list of scored items into a summary dict.
    Each item must have a 'sentiment_score' (compound) key.
    """
    if not items:
        return {"score": 0.0, "label": "neutral", "positive": 0, "negative": 0, "neutral": 0, "total": 0}
    scores = [i.get("sentiment_score", 0.0) for i in items]
    avg = sum(scores) / len(scores)
    pos = sum(1 for s in scores if s >= 0.05)
    neg = sum(1 for s in scores if s <= -0.05)
    neu = len(scores) - pos - neg
    label = "positive" if avg >= 0.05 else ("negative" if avg <= -0.05 else "neutral")
    return {
        "score": round(avg, 4),
        "label": label,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "total": len(items),
        "pct_positive": round(pos / len(items) * 100, 1),
        "pct_negative": round(neg / len(items) * 100, 1),
        "pct_neutral": round(neu / len(items) * 100, 1),
    }
