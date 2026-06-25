"""
Searches for third-party mentions of Piyush Goyal on Twitter/X and LinkedIn.

These differ from the main social collectors (which fetch the Minister's own
posts).  Here we look for what OTHER people are saying about him, so sentiment
analysis applies directly as a measure of public / professional opinion.

Twitter/X  — via nitter open-source mirrors (RSS search, no API key needed).
LinkedIn   — via Google News RSS filtered to site:linkedin.com (returns public
             LinkedIn Pulse articles and posts that Google has indexed).
"""
import feedparser
import hashlib
import re
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
from urllib.parse import quote
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sentiment.analyzer import score

SEARCH_QUERY = "Piyush Goyal"

# Nitter public instances — tried in order, first successful one wins.
# Availability changes frequently; this list is extended but best-effort.
NITTER_INSTANCES = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.1d4.us",
    "https://nitter.lunar.icu",
    "https://nitter.nl",
    "https://nitter.it",
    "https://nitter.tiekoetter.com",
    "https://nitter.cz",
    "https://nitter.42l.fr",
    "https://nitter.unixfox.eu",
    "https://nitter.moomoo.me",
    "https://nitter.fdn.fr",
    "https://nitter.sethforprivacy.com",
]

# Minister's own handles — skip these in mention results
_OWN_HANDLES = {"piyushgoyal", "piyushgoyaloffc"}

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; feedparser)"}

# Regex to detect X profile / list pages (not individual tweets)
_PROFILE_PAGE_RE = re.compile(
    r"(/ Posts / X\s*[-–]\s*x\.com|/ X - x\.com)\s*$"
    r"|^@\w+\s*/",
    re.IGNORECASE,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"&[#a-zA-Z0-9]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_recent(pub_str: str, hours: int) -> bool:
    try:
        dt = dateparser.parse(pub_str)
        if dt is None:
            return True
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= datetime.now(timezone.utc) - timedelta(hours=hours)
    except Exception:
        return True


def _parse_dt(pub_str: str) -> datetime:
    try:
        dt = dateparser.parse(pub_str)
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt or datetime.now(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


# ── Twitter/X mentions via nitter ─────────────────────────────────────────────

def _parse_nitter_entries(entries: list, hours: int) -> list:
    """Convert nitter RSS entries into normalised mention dicts."""
    results   = []
    seen_ids: set = set()
    for entry in entries:
        raw_author = entry.get("author", "")
        author = raw_author.lstrip("@").lower().split("@")[-1].strip()
        if author in _OWN_HANDLES:
            continue
        text = _clean(entry.get("summary", "") or entry.get("title", ""))
        if not text or not _is_recent(entry.get("published", ""), hours):
            continue
        link = entry.get("link", "")
        eid  = hashlib.md5((link or text[:80]).encode()).hexdigest()[:12]
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        pub  = _parse_dt(entry.get("published", ""))
        sent = score(text, apply_subject_adjustment=False)
        results.append({
            "id": eid, "platform": "Twitter/X", "source_type": "mention",
            "handle": f"@{author}" if author else "",
            "text": text[:500], "summary": "",
            "published": pub.isoformat(),
            "published_display": pub.strftime("%d %b %Y, %I:%M %p"),
            "url": link, "likes": 0, "comments": 0, "retweets": 0,
            "sentiment_score": sent["compound"], "sentiment_label": sent["label"],
            "sentiment_basis": "mention_text", "adjustment": "", "category": "mention",
        })
    return results


def _fetch_via_nitter(hours: int) -> list:
    """Try each nitter instance for RSS search results."""
    for instance in NITTER_INSTANCES:
        try:
            url  = f"{instance}/search/rss?q={quote(SEARCH_QUERY)}&f=tweets"
            feed = feedparser.parse(url, request_headers=_HEADERS)
            # A valid nitter RSS feed will have entries AND real tweet content
            if feed.entries and any(
                e.get("summary") or e.get("title") for e in feed.entries[:3]
            ):
                results = _parse_nitter_entries(feed.entries, hours)
                if results:
                    print(f"[mentions] Twitter/X (nitter): {len(results)} from {instance}")
                    return results
        except Exception:
            continue
    return []


def _fetch_via_google_xcom(hours: int) -> list:
    """
    Fallback: search Google News for site:x.com mentions.
    Google indexes many public X posts and tweet-quoting news items.
    """
    try:
        url  = (
            "https://news.google.com/rss/search"
            f"?q={quote(SEARCH_QUERY + ' site:x.com')}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        feed = feedparser.parse(url)
        if not feed.entries:
            return []
        print(f"[mentions] Twitter/X (Google/x.com): {len(feed.entries)} raw")
    except Exception as exc:
        print(f"[mentions] Twitter/X (Google/x.com): failed: {exc}")
        return []

    results   = []
    seen_ids: set = set()
    for entry in feed.entries:
        title = entry.get("title", "").strip()
        if not title or not _is_recent(entry.get("published", ""), hours):
            continue
        # Skip profile / list index pages
        if _PROFILE_PAGE_RE.search(title):
            continue
        # Strip " - x.com" suffix if present
        clean_title = re.sub(r"\s*-\s*x\.com\s*$", "", title, flags=re.IGNORECASE).strip()
        if len(clean_title) < 15:          # too short to be a real tweet
            continue
        eid = hashlib.md5(clean_title.encode()).hexdigest()[:12]
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        pub  = _parse_dt(entry.get("published", ""))
        sent = score(clean_title, apply_subject_adjustment=False)
        results.append({
            "id": eid, "platform": "Twitter/X", "source_type": "mention",
            "handle": "",
            "text": clean_title[:500], "summary": "",
            "published": pub.isoformat(),
            "published_display": pub.strftime("%d %b %Y, %I:%M %p"),
            "url": entry.get("link", ""),
            "likes": 0, "comments": 0, "retweets": 0,
            "sentiment_score": sent["compound"], "sentiment_label": sent["label"],
            "sentiment_basis": "mention_text", "adjustment": "", "category": "mention",
        })

    print(f"[mentions] Twitter/X (Google/x.com): {len(results)} after filtering")
    return results[:60]


def fetch_twitter_mentions(hours: int = 48) -> list:
    """
    Search for tweets mentioning Piyush Goyal.

    Strategy (tried in order):
    1. Nitter RSS search — real tweet text with author handles.
       Works when a nitter instance is available.
    2. Google News site:x.com — tweet text indexed by Google.
       Reliable fallback; handles are not available but content is.
    """
    results = _fetch_via_nitter(hours)
    if results:
        return results

    print("[mentions] Twitter/X: nitter unavailable — trying Google News/x.com fallback")
    return _fetch_via_google_xcom(hours)


# ── LinkedIn mentions via Google News ─────────────────────────────────────────

def fetch_linkedin_mentions(hours: int = 24) -> list:
    """
    Find LinkedIn content mentioning Piyush Goyal via Google News RSS
    (query filtered to site:linkedin.com).  Returns LinkedIn Pulse articles
    and public posts that have been indexed by Google.
    """
    try:
        url = (
            "https://news.google.com/rss/search"
            f"?q={quote(SEARCH_QUERY + ' site:linkedin.com')}"
            "&hl=en-IN&gl=IN&ceid=IN:en"
        )
        feed    = feedparser.parse(url)
        entries = feed.entries
        print(f"[mentions] LinkedIn: {len(entries)} raw results from Google News")
    except Exception as exc:
        print(f"[mentions] LinkedIn: search failed: {exc}")
        return []

    results  = []
    seen_ids: set = set()

    for entry in entries:
        raw_title = entry.get("title", "").strip()
        if not raw_title:
            continue
        if not _is_recent(entry.get("published", ""), hours):
            continue

        # Google News appends " - Author" — strip if short enough to be a byline
        clean_title = raw_title
        if " - " in raw_title:
            parts = raw_title.rsplit(" - ", 1)
            if len(parts[1]) < 80:
                clean_title = parts[0].strip()

        eid = hashlib.md5(clean_title.encode()).hexdigest()[:12]
        if eid in seen_ids:
            continue
        seen_ids.add(eid)

        summary  = _clean(entry.get("summary", "") or "")
        text_for_score = clean_title + ". " + summary[:300]
        pub  = _parse_dt(entry.get("published", ""))
        sent = score(text_for_score, apply_subject_adjustment=False)

        results.append({
            "id":                eid,
            "platform":          "LinkedIn",
            "source_type":       "mention",
            "handle":            "",
            "text":              clean_title,
            "summary":           summary[:400],
            "published":         pub.isoformat(),
            "published_display": pub.strftime("%d %b %Y, %I:%M %p"),
            "url":               entry.get("link", ""),
            "likes":    0,
            "comments": 0,
            "retweets": 0,
            "sentiment_score":   sent["compound"],
            "sentiment_label":   sent["label"],
            "sentiment_basis":   "mention_text",
            "adjustment":        "",
            "category":          "mention",
        })

    print(f"[mentions] LinkedIn: {len(results)} mentions")
    return results
