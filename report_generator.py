"""
Assembles the full daily sentiment report from all collectors.
Saves as JSON to data/reports/. Called by scheduler at 9 AM or on-demand.
"""
import json
import os
from datetime import datetime, timezone
from typing import Optional

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "data", "reports")


def generate_report(hours: int = 24) -> dict:
    """Collect all data, run sentiment, save and return the report dict."""
    from collectors.news_collector import fetch_news
    from collectors.twitter_collector import fetch_tweets
    from collectors.instagram_collector import fetch_posts as ig_posts
    from collectors.facebook_collector import fetch_posts as fb_posts
    from collectors.follower_tracker import collect_all_counts
    from sentiment.analyzer import aggregate_sentiment

    ts = datetime.now(timezone.utc)
    print(f"[report] Generating report for last {hours}h at {ts.isoformat()}")

    # ── Collect ────────────────────────────────────────────────────────────────
    print("[report] Fetching news...")
    news = fetch_news(hours)
    print(f"[report] News: {len(news)} articles in last {hours}h")

    print("[report] Fetching tweets...")
    tweets = fetch_tweets(hours)

    print("[report] Fetching Instagram posts...")
    ig = ig_posts(hours)

    print("[report] Fetching Facebook posts...")
    fb = fb_posts(hours)

    print("[report] Collecting follower counts...")
    followers = collect_all_counts()

    print("[report] Fetching Twitter/X mentions...")
    from collectors.mentions_collector import (fetch_twitter_mentions,
                                               fetch_linkedin_mentions)
    # Mentions are published less frequently than news; use a wider window
    # (48 h) so infrequent LinkedIn posts are not missed.
    mention_hours = max(hours, 48)
    tw_mentions_raw = fetch_twitter_mentions(mention_hours)
    print("[report] Fetching LinkedIn mentions...")
    li_mentions_raw = fetch_linkedin_mentions(mention_hours)
    all_mentions = tw_mentions_raw + li_mentions_raw
    print(f"[report] Mentions: {len(tw_mentions_raw)} Twitter/X, "
          f"{len(li_mentions_raw)} LinkedIn")

    # ── Sentiment aggregations ─────────────────────────────────────────────────
    minister_news = [n for n in news if n["category"] == "minister"]
    ministry_news = [n for n in news if n["category"] == "ministry"]
    social_posts  = tweets + ig + fb

    # Main scores: news articles only (minister's own posts excluded)
    minister_sentiment = aggregate_sentiment(minister_news)
    ministry_sentiment = aggregate_sentiment(ministry_news)

    # Mentions score: third-party opinions about the minister
    mentions_sentiment = aggregate_sentiment(all_mentions) if all_mentions else {
        "score": 0.0, "label": "neutral",
        "positive": 0, "negative": 0, "neutral": 0, "total": 0,
        "pct_positive": 0.0, "pct_negative": 0.0, "pct_neutral": 0.0,
    }

    # Source breakdown
    source_breakdown = {}
    for item in minister_news + social_posts:
        src = item.get("source") or item.get("platform", "Unknown")
        source_breakdown[src] = source_breakdown.get(src, 0) + 1

    report = {
        "generated_at": ts.isoformat(),
        "generated_display": ts.strftime("%d %b %Y, %I:%M %p UTC"),
        "period_hours": hours,
        "minister": {
            "name": "Piyush Goyal",
            "title": "Union Cabinet Minister, Commerce & Industry",
            "sentiment": minister_sentiment,
            "news": minister_news,
            "tweets": tweets,
            "instagram": ig,
            "facebook": fb,
            "mentions": all_mentions,
            "mentions_sentiment": mentions_sentiment,
            "source_breakdown": source_breakdown,
        },
        "ministry": {
            "name": "Ministry of Commerce & Industry",
            "sentiment": ministry_sentiment,
            "news": ministry_news,
        },
        "followers": followers,
        "stats": {
            "total_news": len(news),
            "minister_news": len(minister_news),
            "ministry_news": len(ministry_news),
            "total_tweets": len(tweets),
            "total_instagram": len(ig),
            "total_facebook": len(fb),
            "total_tw_mentions": len(tw_mentions_raw),
            "total_li_mentions": len(li_mentions_raw),
        },
    }

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = f"report_{ts.strftime('%Y%m%d_%H%M')}.json"
    filepath = os.path.join(REPORTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(REPORTS_DIR, "latest.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[report] Saved → {filepath}")

    return report


def load_latest_report() -> Optional[dict]:
    path = os.path.join(REPORTS_DIR, "latest.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def list_reports() -> list:
    """Return list of saved report filenames, newest first."""
    if not os.path.exists(REPORTS_DIR):
        return []
    files = [f for f in os.listdir(REPORTS_DIR) if f.startswith("report_") and f.endswith(".json")]
    return sorted(files, reverse=True)
