import re
import json
import logging
from typing import Optional, Tuple
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

logger = logging.getLogger(__name__)

# ── VADER setup (used as fallback when LLM is unavailable) ────────────────────
_analyzer = SentimentIntensityAnalyzer()

_PHRASE_SUBS = [
    (re.compile(
        r'\b(lower(?:ing)?|reduc(?:e|ing|tion\s+of))\s+'
        r'(cost|costs|price|prices|tax|taxes|tariff|tariffs|inflation|deficit|burden|rate|rates)\b',
        re.IGNORECASE,
    ), r'ease \2 beneficially'),
    (re.compile(r'\blower\s+cost\s+of\s+doing\s+business\b', re.IGNORECASE),
     'improve ease of doing business'),
    (re.compile(r'\bease\s+of\s+doing\s+business\b', re.IGNORECASE),
     'excellent business environment'),
]

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


def _normalize(text: str) -> str:
    for pattern, replacement in _PHRASE_SUBS:
        text = pattern.sub(replacement, text)
    return text


# ── Subject-aware helpers (still used by VADER fallback path) ─────────────────
_MIN = r"(?:piyush\s+goyal|goyal|union\s+minister|commerce\s+minister|minister\s+goyal)"

_MINISTER_AS_CRITIC = re.compile(
    r"(?<!\w)" + _MIN +
    r".{0,50}"
    r"(criticis|slams?|attacks?|blasts?|questions?|warns?|urges?|accuses?|dismisses?|rejects?|opposes?|counters?|rebukes?|condemns?)",
    re.IGNORECASE,
)
_MINISTER_AS_TARGET = re.compile(
    r"(criticis|slams?|attacks?|blasts?|accuses?|targets?|opposes?|rebukes?|condemns?)"
    r".{0,40}" + _MIN,
    re.IGNORECASE,
)
_MINISTER_TROUBLE = re.compile(
    _MIN +
    r".{0,30}(resign|arrest|bail|probe|scam|corruption|scandal|allegation|charge|fir\b|under\s+fire|under\s+scrutiny)",
    re.IGNORECASE,
)


def adjust_for_subject(text: str, raw_compound: float) -> tuple:
    if raw_compound >= -0.05:
        return raw_compound, ""
    t = text.lower()
    is_critic  = bool(_MINISTER_AS_CRITIC.search(t))
    is_target  = bool(_MINISTER_AS_TARGET.search(t))
    is_trouble = bool(_MINISTER_TROUBLE.search(t))
    if is_critic and not is_target and not is_trouble:
        adjusted = max(raw_compound * 0.15, -0.04)
        return adjusted, "reclassified: minister as critic, not subject of criticism"
    return raw_compound, ""


# ── LLM-based scoring (primary path) ─────────────────────────────────────────
_llm_client = None
_llm_client_attempted = False

_SYSTEM_PROMPT = """You analyse Indian news articles or headlines about Union Commerce & Industry Minister Piyush Goyal.

Score sentiment strictly from MINISTER GOYAL'S perspective — how does this news reflect on him and his ministry?

POSITIVE: He is achieving goals, leading negotiations, advocating for India, making policy progress, being praised, or successfully pushing back against critics/foreign agencies.
NEGATIVE: He is being directly criticised, facing personal controversy, under scrutiny, or associated with failure or scandal.
NEUTRAL: Factual reporting with no clear positive or negative slant toward him.

CRITICAL RULE — ADVOCACY IS NOT NEGATIVE:
If Goyal is the one SPEAKING (criticising rating agencies, challenging foreign policies, defending India's interests, calling for debates, urging reforms), this reflects POSITIVELY or NEUTRALLY on him. He is doing his job. Do NOT score these as negative.
Only score NEGATIVE when Goyal himself is the TARGET of the criticism or in serious trouble.

Respond with JSON only, no extra text:
{"label": "positive"|"neutral"|"negative", "compound": <float -1.0 to 1.0>, "reason": "<brief phrase>"}"""


def _get_llm_client():
    global _llm_client, _llm_client_attempted
    if _llm_client_attempted:
        return _llm_client
    _llm_client_attempted = True
    try:
        import anthropic
        _llm_client = anthropic.Anthropic()
    except Exception as e:
        logger.warning("Could not initialise Anthropic client: %s", e)
    return _llm_client


def _derive_vader_like(compound: float, label: str) -> Tuple[float, float, float]:
    """Synthesise pos/neu/neg probabilities (summing to ~1) from a compound score."""
    abs_c = abs(compound)
    if label == "positive":
        pos = max(0.3, min(1.0, 0.3 + abs_c * 0.7))
        neg = 0.0
        neu = round(1.0 - pos, 3)
        return round(pos, 3), neu, neg
    if label == "negative":
        neg = max(0.3, min(1.0, 0.3 + abs_c * 0.7))
        pos = 0.0
        neu = round(1.0 - neg, 3)
        return pos, neu, round(neg, 3)
    # neutral
    return 0.1, 0.8, 0.1


def _score_with_llm(text: str) -> Optional[dict]:
    """Call Claude Haiku to score sentiment. Returns None on any failure."""
    client = _get_llm_client()
    if client is None:
        return None
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=128,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text[:2000]}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        label = result.get("label", "neutral")
        if label not in ("positive", "neutral", "negative"):
            label = "neutral"
        c = max(-1.0, min(1.0, float(result.get("compound", 0.0))))
        # Re-align compound sign with label to guard against model inconsistencies
        if label == "positive" and c < 0:
            c = abs(c)
        elif label == "negative" and c > 0:
            c = -abs(c)
        pos, neu, neg = _derive_vader_like(c, label)
        return {
            "compound": round(c, 4),
            "pos": pos,
            "neu": neu,
            "neg": neg,
            "label": label,
            "adjustment": f"llm: {result.get('reason', '')}",
        }
    except Exception as e:
        logger.debug("LLM scoring failed (%s); falling back to VADER", e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def score(text: str, apply_subject_adjustment: bool = False) -> dict:
    """
    Return sentiment scores for a piece of text.

    Tries Claude Haiku first (subject-aware, understands advocacy/quote attribution).
    Falls back to VADER + subject-adjustment if the API is unavailable.
    Return dict is always: {compound, pos, neu, neg, label, adjustment}
    """
    if not text or not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "neutral", "adjustment": ""}

    result = _score_with_llm(text)
    if result is not None:
        return result

    # VADER fallback
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
    """Aggregate a list of scored items into a summary dict."""
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
