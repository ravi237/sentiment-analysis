"""
YouTube subscriber count scraper for Piyush Goyal's official channel.
Extracts from ytInitialData embedded in the channel page — no API key needed.
"""
import re
import requests

CHANNEL_HANDLE = "@PiyushGoyalOfficial"
CHANNEL_URL = f"https://www.youtube.com/{CHANNEL_HANDLE}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_count(text: str):
    """Convert '30.9 million', '137K', '1.2M' etc. to an integer."""
    if not text:
        return None
    text = text.lower().replace(",", "").strip()
    m = re.search(r"([\d\.]+)\s*(million|m|k|b)?", text)
    if not m:
        return None
    num = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    if suffix in ("million", "m"):
        return int(num * 1_000_000)
    if suffix == "k":
        return int(num * 1_000)
    if suffix == "b":
        return int(num * 1_000_000_000)
    return int(num)


def get_subscriber_count() -> dict:
    """
    Fetch the live subscriber count from the official YouTube channel.
    Returns {"platform": "YouTube", "handle": "...", "followers": int | None}.
    """
    try:
        resp = requests.get(CHANNEL_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        text = resp.text

        # ytInitialData is embedded as: var ytInitialData = {...};
        start = text.find("var ytInitialData = ")
        if start == -1:
            print("[youtube] ytInitialData not found in page")
            return {"platform": "YouTube", "handle": CHANNEL_HANDLE, "followers": None}

        start += len("var ytInitialData = ")
        raw = text[start: text.find(";</script>", start)]

        # Primary pattern: accessibility label (most human-readable)
        # e.g. "30.9 million subscribers"
        m = re.search(
            r'"subscriberCountText"\s*:\s*\{[^}]*"label"\s*:\s*"([^"]+)"',
            raw, re.S
        )
        if m:
            count = _parse_count(m.group(1))
            if count:
                print(f"[youtube] {CHANNEL_HANDLE}: {count:,} subscribers")
                return {
                    "platform": "YouTube",
                    "handle": CHANNEL_HANDLE,
                    "followers": count,
                    "source": "ytInitialData",
                }

        # Fallback pattern: simpleText e.g. "30.9m subscribers"
        m2 = re.search(
            r'"subscriberCountText"\s*:\s*\{[^}]*"simpleText"\s*:\s*"([^"]+)"',
            raw, re.S
        )
        if m2:
            count = _parse_count(m2.group(1))
            if count:
                print(f"[youtube] {CHANNEL_HANDLE}: {count:,} subscribers (simpleText)")
                return {
                    "platform": "YouTube",
                    "handle": CHANNEL_HANDLE,
                    "followers": count,
                    "source": "ytInitialData_simple",
                }

        print("[youtube] No subscriber count pattern matched")
    except Exception as e:
        print(f"[youtube] Error: {e}")

    return {"platform": "YouTube", "handle": CHANNEL_HANDLE, "followers": None}
