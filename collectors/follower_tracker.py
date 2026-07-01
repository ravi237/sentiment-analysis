"""
Aggregates follower counts from all platforms and tracks history.
"""
import json
import os
from datetime import datetime, timezone

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "follower_history.json")

PLATFORM_URLS = {
    "Twitter/X":    "https://x.com/PiyushGoyal",
    "Twitter/X (Office)": "https://x.com/PiyushGoyalOffc",
    "Instagram":    "https://www.instagram.com/piyushgoyalofficial/",
    "Facebook":     "https://www.facebook.com/PiyushGoyalOfficial/",
    "YouTube":      "https://www.youtube.com/@PiyushGoyal",
    "LinkedIn":     "https://www.linkedin.com/in/piyushgoyalofficial/",
}

# Known approximate counts as fallback when scraping fails
FALLBACK_COUNTS = {
    "Twitter/X":    12_100_000,
    "Twitter/X (Office)": 850_000,
    "Instagram":    2_000_000,
    "Facebook":     10_000_000,
    "YouTube":      180_000,
    "LinkedIn":     95_000,
}


def _load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_history(history: dict):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _reading_near(readings: list, target: datetime):
    """Return the count from the reading whose timestamp is closest to *target*."""
    best_count, best_diff = None, None
    for r in readings:
        try:
            ts = datetime.fromisoformat(r["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            diff = abs((ts - target).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_count = r["count"]
        except Exception:
            pass
    return best_count


def _trend_deltas(readings: list, current: int, now_dt: datetime) -> dict:
    """
    Compute follower change over three windows by finding the reading
    closest to each anchor point in the history.
    Only returns a delta if a reading exists within 1.5× the window
    (e.g. a 'weekly' anchor must have a reading within 10.5 days).
    """
    from datetime import timedelta
    windows = {"daily": 1, "weekly": 7, "monthly": 30}
    out = {}
    for key, days in windows.items():
        anchor = now_dt - timedelta(days=days)
        ref = _reading_near(readings, anchor)
        if ref is None:
            out[key] = None
            continue
        # Reject if the closest reading is too far from the anchor
        best_diff = None
        for r in readings:
            try:
                ts = datetime.fromisoformat(r["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                diff = abs((ts - anchor).total_seconds())
                if best_diff is None or diff < best_diff:
                    best_diff = diff
            except Exception:
                pass
        max_gap = days * 1.5 * 86400   # 1.5× the window in seconds
        if best_diff is not None and best_diff > max_gap:
            out[key] = None
        else:
            out[key] = (current - ref) if ref else None
    return out


def collect_all_counts() -> list:
    """Collect follower counts from all platforms."""
    from collectors.twitter_collector import get_follower_count as tw_count
    from collectors.instagram_collector import get_follower_count as ig_count
    from collectors.facebook_collector import get_follower_count as fb_count

    now_dt = datetime.now(timezone.utc)
    now    = now_dt.isoformat()
    history = _load_history()
    results = []

    fetchers = {
        "Twitter/X": tw_count,
        "Instagram": ig_count,
        "Facebook": fb_count,
    }

    for platform, fetcher in fetchers.items():
        try:
            data = fetcher()
        except Exception as e:
            print(f"[follower_tracker] {platform} failed: {e}")
            data = {}

        count = data.get("followers")
        if count is None:
            count = FALLBACK_COUNTS.get(platform)
            is_live = False
        else:
            is_live = True

        readings = history.get(platform, {}).get("history", [])

        # Compute daily/weekly/monthly deltas BEFORE appending today's reading
        trends = _trend_deltas(readings, count, now_dt) if count else \
                 {"daily": None, "weekly": None, "monthly": None}

        entry = {
            "platform": platform,
            "handle": data.get("handle", ""),
            "url": PLATFORM_URLS.get(platform, ""),
            "followers": count,
            "followers_display": _format_count(count),
            "delta": trends["daily"],
            "delta_display": _format_delta(trends["daily"]),
            "delta_daily": trends["daily"],
            "delta_weekly": trends["weekly"],
            "delta_monthly": trends["monthly"],
            "is_live": is_live,
            "as_of": now,
        }
        results.append(entry)

        # Persist history
        if platform not in history:
            history[platform] = {"history": []}
        history[platform]["latest"] = count
        history[platform]["history"].append({"count": count, "ts": now})
        history[platform]["history"] = history[platform]["history"][-90:]  # 3 months

    # Manually-tracked platforms (no live scraper)
    for platform in ["Twitter/X (Office)", "YouTube", "LinkedIn"]:
        count = FALLBACK_COUNTS.get(platform)
        results.append({
            "platform": platform,
            "handle": "",
            "url": PLATFORM_URLS.get(platform, ""),
            "followers": count,
            "followers_display": _format_count(count),
            "delta": None,
            "delta_display": "—",
            "delta_daily": None,
            "delta_weekly": None,
            "delta_monthly": None,
            "is_live": False,
            "as_of": now,
        })

    _save_history(history)
    return results


def _format_count(n) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _format_delta(d) -> str:
    if d is None:
        return "—"
    sign = "+" if d >= 0 else ""
    return f"{sign}{_format_count(abs(d))}" if abs(d) >= 1000 else f"{sign}{d}"


def load_history_df():
    """Return history as a dict of {platform: [(ts, count), ...]} for charting."""
    import pandas as pd
    history = _load_history()
    dfs = {}
    for platform, data in history.items():
        rows = data.get("history", [])
        if rows:
            df = pd.DataFrame(rows)
            df["ts"] = pd.to_datetime(df["ts"])
            dfs[platform] = df
    return dfs
