"""
Twitter/X public profile scraper.
Uses Twitter's syndication API (the same endpoint used by Twitter's
own embedded timelines) — no developer account or login required.
"""
import requests
import json
import hashlib
import re
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sentiment.analyzer import score_with_engagement

HANDLES = ["PiyushGoyal", "PiyushGoyalOffc"]

SYNDICATION_URL = "https://syndication.twitter.com/srv/timeline-profile/screen-name/{handle}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_timeline(handle: str) -> list:
    """Fetch tweets from Twitter's syndication timeline endpoint."""
    url = SYNDICATION_URL.format(handle=handle)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"[twitter] Syndication HTTP {resp.status_code} for @{handle}")
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        script = soup.find("script", {"id": "__NEXT_DATA__"})
        if not script:
            print(f"[twitter] No __NEXT_DATA__ for @{handle}")
            return []
        data = json.loads(script.string)
        entries = data["props"]["pageProps"]["timeline"]["entries"]
        return entries
    except Exception as e:
        print(f"[twitter] Failed for @{handle}: {e}")
        return []


def _parse_entry(entry: dict, handle: str) -> Optional[dict]:
    tweet = entry.get("content", {}).get("tweet", {})
    if not tweet:
        return None
    text = tweet.get("full_text") or tweet.get("text", "")
    if not text:
        return None
    likes     = tweet.get("favorite_count", 0) or 0
    retweets  = tweet.get("retweet_count", 0) or 0
    replies   = tweet.get("reply_count", 0) or 0
    tweet_id  = str(tweet.get("id_str", ""))
    created   = tweet.get("created_at", "")
    try:
        dt = dateparser.parse(created).replace(tzinfo=timezone.utc)
    except Exception:
        dt = datetime.now(timezone.utc)
    sentiment = score_with_engagement(text, likes=likes, comments=replies, shares=retweets)
    return {
        "id": tweet_id or hashlib.md5(text.encode()).hexdigest()[:12],
        "platform": "Twitter/X",
        "handle": f"@{handle}",
        "text": text,
        "published": dt.isoformat(),
        "published_display": dt.strftime("%d %b %Y, %I:%M %p"),
        "likes": likes,
        "retweets": retweets,
        "replies": replies,
        "url": f"https://x.com/{handle}/status/{tweet_id}" if tweet_id else f"https://x.com/{handle}",
        "sentiment_score": sentiment["compound"],
        "sentiment_label": sentiment["label"],
        "engagement": sentiment["engagement"],
        # Syndication API does not expose reply text — scored on post content only.
        "sentiment_basis": "post_content",
    }


def _is_recent(iso_str: str, hours: int = 24) -> bool:
    try:
        dt = dateparser.parse(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= datetime.now(timezone.utc) - timedelta(hours=hours)
    except Exception:
        return True


def fetch_tweets(hours: int = 24) -> list:
    """Fetch recent tweets from all tracked handles."""
    all_tweets = []
    for handle in HANDLES:
        entries = _fetch_timeline(handle)
        tweets = []
        for entry in entries:
            parsed = _parse_entry(entry, handle)
            if parsed and _is_recent(parsed["published"], hours):
                tweets.append(parsed)
        print(f"[twitter] @{handle}: {len(tweets)} tweets in last {hours}h")
        all_tweets.extend(tweets)
    return sorted(all_tweets, key=lambda x: x["published"], reverse=True)


def get_follower_count() -> dict:
    """Extract follower count from the syndication page."""
    url = SYNDICATION_URL.format(handle="PiyushGoyal")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return {"platform": "Twitter/X", "handle": "@PiyushGoyal", "followers": None}
        soup = BeautifulSoup(resp.text, "lxml")
        script = soup.find("script", {"id": "__NEXT_DATA__"})
        if not script:
            return {"platform": "Twitter/X", "handle": "@PiyushGoyal", "followers": None}
        data = json.loads(script.string)
        user = data["props"]["pageProps"].get("profile", {})
        count = (
            user.get("followers_count")
            or data["props"]["pageProps"].get("timeline", {})
               .get("user", {}).get("followers_count")
        )
        if count:
            return {"platform": "Twitter/X", "handle": "@PiyushGoyal", "followers": int(count), "source": "syndication"}
    except Exception as e:
        print(f"[twitter] Follower count error: {e}")
    return {"platform": "Twitter/X", "handle": "@PiyushGoyal", "followers": None, "source": "unavailable"}
