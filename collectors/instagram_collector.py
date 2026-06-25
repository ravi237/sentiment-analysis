"""
Instagram public profile collector.
Uses Instagram's mobile API (i.instagram.com/api/v1) — same endpoint the
official Instagram app uses. Works on public profiles without login.
"""
import requests
import hashlib
from datetime import datetime, timezone, timedelta
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sentiment.analyzer import score_with_engagement

HANDLE = "piyushgoyalofficial"

# Public mobile app credentials used by Instagram's own Android app
MOBILE_HEADERS = {
    "User-Agent": "Instagram 219.0.0.12.117 Android (28/9; 420dpi; 1080x2093; "
                  "samsung; SM-G975U; beyond1q; qcom; en_US; 302733750)",
    "X-IG-App-ID": "936619743392459",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

PROFILE_URL = "https://i.instagram.com/api/v1/users/web_profile_info/?username={handle}"


def _fetch_profile() -> dict:
    url = PROFILE_URL.format(handle=HANDLE)
    resp = requests.get(url, headers=MOBILE_HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json().get("data", {}).get("user", {})


def fetch_posts(hours: int = 24) -> list:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    posts = []
    try:
        user = _fetch_profile()
        edges = user.get("edge_owner_to_timeline_media", {}).get("edges", [])
        for edge in edges:
            node = edge.get("node", {})
            taken_at = node.get("taken_at_timestamp")
            if not taken_at:
                continue
            post_dt = datetime.fromtimestamp(taken_at, tz=timezone.utc)
            if post_dt < cutoff:
                continue
            # Extract caption
            caption_edges = node.get("edge_media_to_caption", {}).get("edges", [])
            caption = caption_edges[0]["node"]["text"] if caption_edges else ""
            likes = node.get("edge_liked_by", {}).get("count", 0) or 0
            comments = node.get("edge_media_to_comment", {}).get("count", 0) or 0
            shortcode = node.get("shortcode", "")
            sentiment = score_with_engagement(caption, likes=likes, comments=comments)
            posts.append({
                "id": shortcode or hashlib.md5(caption[:50].encode()).hexdigest()[:12],
                "platform": "Instagram",
                "handle": f"@{HANDLE}",
                "text": caption[:500],
                "published": post_dt.isoformat(),
                "published_display": post_dt.strftime("%d %b %Y, %I:%M %p"),
                "likes": likes,
                "comments": comments,
                "url": f"https://www.instagram.com/p/{shortcode}/" if shortcode else f"https://instagram.com/{HANDLE}",
                "media_type": node.get("__typename", ""),
                "thumbnail": node.get("thumbnail_src", ""),
                "sentiment_score": sentiment["compound"],
                "sentiment_label": sentiment["label"],
                "engagement": sentiment["engagement"],
                # Mobile API returns comment count but not comment text.
                # Sentiment scored on caption (post content) only.
                "sentiment_basis": "post_content",
            })
    except requests.exceptions.HTTPError as e:
        print(f"[instagram] HTTP error {e.response.status_code}: {e}")
    except Exception as e:
        print(f"[instagram] Error: {e}")
    print(f"[instagram] @{HANDLE}: {len(posts)} posts in last {hours}h")
    return posts


def get_follower_count() -> dict:
    try:
        user = _fetch_profile()
        followers = user.get("edge_followed_by", {}).get("count")
        following = user.get("edge_follow", {}).get("count")
        posts_count = user.get("edge_owner_to_timeline_media", {}).get("count")
        return {
            "platform": "Instagram",
            "handle": f"@{HANDLE}",
            "followers": followers,
            "following": following,
            "posts_count": posts_count,
            "source": "instagram_mobile_api",
        }
    except Exception as e:
        print(f"[instagram] Follower count failed: {e}")
        return {"platform": "Instagram", "handle": f"@{HANDLE}", "followers": None, "source": "unavailable"}
