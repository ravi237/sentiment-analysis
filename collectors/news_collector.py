import feedparser
import hashlib
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sentiment.analyzer import score

_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","was","are","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","must",
    "that","this","these","those","it","its","as","not","no","nor",
}

# Words so ubiquitous in this dataset they don't help discriminate topics.
# Removing them lets Jaccard focus on the actual story (e.g. "uk", "fta", "visit").
_SUBJECT_WORDS = {
    "piyush", "goyal", "minister", "union", "commerce", "ministry",
    "india", "indian",
}

CUTOFF_HOURS = 24

# RSS feeds — Google News searches + major Indian outlets
FEEDS = [
    # Google News — Minister
    {
        "url": "https://news.google.com/rss/search?q=%22Piyush+Goyal%22&hl=en-IN&gl=IN&ceid=IN:en",
        "category": "minister", "source": "Google News"
    },
    # Google News — Ministry
    {
        "url": "https://news.google.com/rss/search?q=%22Ministry+of+Commerce%22+India&hl=en-IN&gl=IN&ceid=IN:en",
        "category": "ministry", "source": "Google News"
    },
    {
        "url": "https://news.google.com/rss/search?q=DPIIT+India&hl=en-IN&gl=IN&ceid=IN:en",
        "category": "ministry", "source": "Google News"
    },
    {
        "url": "https://news.google.com/rss/search?q=%22Commerce+Ministry%22+India&hl=en-IN&gl=IN&ceid=IN:en",
        "category": "ministry", "source": "Google News"
    },
    # Economic Times
    {
        "url": "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
        "category": "general", "source": "Economic Times", "filter": ["Piyush Goyal", "Commerce Ministry", "DPIIT", "trade policy"]
    },
    # NDTV
    {
        "url": "https://feeds.feedburner.com/ndtvnews-india-news",
        "category": "general", "source": "NDTV", "filter": ["Piyush Goyal", "Commerce Ministry", "DPIIT"]
    },
    # Hindustan Times
    {
        "url": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
        "category": "general", "source": "Hindustan Times", "filter": ["Piyush Goyal", "Commerce Ministry"]
    },
    # Business Standard
    {
        "url": "https://www.business-standard.com/rss/politics-current-affairs-1.rss",
        "category": "general", "source": "Business Standard", "filter": ["Piyush Goyal", "Commerce Ministry", "DPIIT"]
    },
    {
        "url": "https://www.business-standard.com/rss/economy-policy-2.rss",
        "category": "ministry", "source": "Business Standard", "filter": ["Commerce", "trade", "DPIIT", "export", "import"]
    },
    # LiveMint
    {
        "url": "https://www.livemint.com/rss/politics",
        "category": "general", "source": "LiveMint", "filter": ["Piyush Goyal", "Commerce Ministry"]
    },
]


def _parse_date(entry) -> datetime:
    """Parse published date from feed entry, return UTC datetime."""
    for attr in ("published", "updated", "created"):
        raw = getattr(entry, attr, None) or entry.get(attr)
        if raw:
            try:
                dt = dateparser.parse(str(raw))
                if dt and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def _is_recent(dt: datetime, hours: int = CUTOFF_HOURS) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt >= cutoff


def _matches_filter(entry, keywords: list) -> bool:
    text = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
    return any(kw.lower() in text for kw in keywords)


def _extract_source_and_clean_title(entry, fallback_source: str) -> tuple:
    """
    For Google News entries the real publisher is in entry.source['title']
    and also appended to the title as ' - Publisher Name'.
    Returns (clean_title, actual_source, source_url).
    """
    raw_title = entry.get("title", "").strip()

    # 1. Prefer the structured source field (most reliable)
    entry_source = entry.get("source", {})
    actual_source = entry_source.get("title", "").strip()
    source_url    = entry_source.get("href", "")

    # 2. If structured source unavailable, parse from title suffix
    if not actual_source and " - " in raw_title:
        parts = raw_title.rsplit(" - ", 1)
        if len(parts) == 2 and len(parts[1]) < 60:   # sanity: source names are short
            actual_source = parts[1].strip()

    # 3. Strip the publisher suffix from the display title
    clean_title = raw_title
    if actual_source and raw_title.endswith(" - " + actual_source):
        clean_title = raw_title[: -(len(actual_source) + 3)].strip()

    return clean_title, (actual_source or fallback_source), source_url


def _make_item(entry, category: str, source: str) -> dict:
    clean_title, actual_source, source_url = _extract_source_and_clean_title(entry, source)
    summary = entry.get("summary", "").strip()
    url = entry.get("link", "")
    published = _parse_date(entry)
    text_for_sentiment = clean_title + ". " + summary[:300]
    # Apply subject-awareness only for minister articles (not ministry policy news)
    sentiment = score(text_for_sentiment, apply_subject_adjustment=(category == "minister"))
    return {
        "id": hashlib.md5(clean_title.encode()).hexdigest()[:12],
        "title": clean_title,
        "summary": summary[:400] if summary else "",
        "source": actual_source,
        "source_url": source_url,
        "url": url,
        "published": published.isoformat(),
        "published_display": published.strftime("%d %b %Y, %I:%M %p"),
        "category": category,
        "sentiment_score": sentiment["compound"],
        "sentiment_label": sentiment["label"],
        "sentiment_detail": {k: sentiment[k] for k in ("pos", "neu", "neg")},
        "adjustment": sentiment.get("adjustment", ""),
    }


def _jaccard(a: str, b: str) -> float:
    """
    Topic-focused Jaccard similarity.
    Excludes stopwords and subject-context words (minister name, ministry, etc.)
    so similarity reflects the actual story, not shared byline boilerplate.
    Includes 2-char words (uk, eu, us, pm, fd…) which carry real topic signal.
    """
    def _topic(s: str) -> set:
        return {w for w in s.lower().split()
                if w not in _STOPWORDS and w not in _SUBJECT_WORDS and len(w) >= 2}
    wa, wb = _topic(a), _topic(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def group_similar(items: list, threshold: float = 0.38) -> list:
    """
    Cluster similar news stories. Returns items with two extra fields:
      - group_size: total articles covering this story
      - group_sources: list of source names in the cluster
    """
    groups = []
    assigned = [False] * len(items)
    for i, item in enumerate(items):
        if assigned[i]:
            continue
        cluster = [item]
        for j in range(i + 1, len(items)):
            if assigned[j]:
                continue
            if _jaccard(item["title"], items[j]["title"]) >= threshold:
                cluster.append(items[j])
                assigned[j] = True
        assigned[i] = True
        # Use the most negative or highest-scoring item as representative
        rep = sorted(cluster, key=lambda x: x["sentiment_score"])[0]
        rep["group_size"] = len(cluster)
        rep["group_sources"] = list({c["source"] for c in cluster})
        rep["group_items"] = [c for c in cluster if c["id"] != rep["id"]]
        groups.append(rep)
    return groups


def fetch_news(hours: int = CUTOFF_HOURS) -> list:
    """Fetch and score all news items from the last `hours` hours."""
    seen_ids = set()
    items = []

    for feed_cfg in FEEDS:
        try:
            feed = feedparser.parse(feed_cfg["url"])
            keywords = feed_cfg.get("filter")
            category = feed_cfg["category"]
            source = feed_cfg["source"]

            for entry in feed.entries:
                if keywords and not _matches_filter(entry, keywords):
                    continue
                published = _parse_date(entry)
                if not _is_recent(published, hours):
                    continue
                # Resolve "general" category before scoring so subject-aware
                # sentiment adjustment is applied correctly at score time.
                effective_category = category
                if category == "general":
                    entry_text = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
                    effective_category = "minister" if "piyush goyal" in entry_text else "ministry"
                item = _make_item(entry, effective_category, source)
                if item["id"] not in seen_ids and item["title"]:
                    seen_ids.add(item["id"])
                    items.append(item)
        except Exception as e:
            print(f"[news_collector] Failed feed {feed_cfg['url']}: {e}")

    return sorted(items, key=lambda x: x["published"], reverse=True)
