"""
Facebook public page collector.
- Follower count: via Open Graph meta tags (reliable, no login needed).
- Posts: via Playwright headless Chromium (handles JS rendering).
  Without login, Facebook shows ~3-5 most recent posts with text only;
  like/comment counts are not exposed to unauthenticated viewers.
"""
import re
import hashlib
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sentiment.analyzer import score_with_engagement

PAGE_HANDLE = "PiyushGoyalOfficial"
PAGE_URL = f"https://www.facebook.com/{PAGE_HANDLE}"

OG_HEADERS = {
    "User-Agent": "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
}


# ── Follower count (Open Graph, always reliable) ───────────────────────────────

def get_follower_count() -> dict:
    """Parse OG description tag for follower/like count. Works without login."""
    try:
        resp = requests.get(PAGE_URL, headers=OG_HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")
        desc_tag = soup.find("meta", property="og:description")
        if desc_tag:
            desc = desc_tag.get("content", "")
            # Handles Hindi: "1,03,44,383 पसंद" and English: "10,344,383 likes"
            m = re.search(r"([\d,]+)\s*(पसंद|likes?|followers?)", desc, re.IGNORECASE)
            if m:
                count = int(m.group(1).replace(",", ""))
                return {
                    "platform": "Facebook",
                    "handle": f"/{PAGE_HANDLE}",
                    "followers": count,
                    "source": "og_meta",
                }
    except Exception as e:
        print(f"[facebook] Follower count error: {e}")
    return {"platform": "Facebook", "handle": f"/{PAGE_HANDLE}", "followers": None, "source": "unavailable"}


# ── Posts (Playwright headless, handles JS) ────────────────────────────────────

def _parse_relative_time(label: str) -> Optional[datetime]:
    """
    Convert 'about 2 hours ago', 'Yesterday at 3pm', etc. to a UTC datetime.
    Returns None if unparseable.
    """
    if not label:
        return None
    label = re.sub(r"(?i)comment\s+by\s+\S+\s+", "", label).strip()
    label = re.sub(r"(?i)\s*ago\s*$", "", label).strip()
    now = datetime.now(timezone.utc)
    # "X hours", "X minutes", "X days"
    m = re.search(r"(\d+)\s+(second|minute|hour|day|week)", label, re.IGNORECASE)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        delta_map = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800}
        return now - timedelta(seconds=n * delta_map.get(unit, 0))
    try:
        return dateparser.parse(label, settings={"RETURN_AS_TIMEZONE_AWARE": True})
    except Exception:
        return None


def fetch_posts(hours: int = 24) -> list:
    """Scrape public Facebook posts using Playwright headless Chromium."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[facebook] Playwright not installed. Run: pip install playwright && playwright install chromium")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    posts = []

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                viewport={"width": 1280, "height": 900},
            )
            page = ctx.new_page()
            page.goto(PAGE_URL, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(3000)

            # Scroll to trigger lazy-loaded posts
            for _ in range(5):
                page.keyboard.press("End")
                page.wait_for_timeout(1200)

            raw_posts = page.evaluate("""() => {
                const results = [];
                const articles = document.querySelectorAll("[role='article']");
                articles.forEach(el => {
                    // Post text from auto-direction divs
                    const textEls = el.querySelectorAll("div[dir='auto']");
                    const texts = Array.from(textEls)
                        .map(e => e.innerText.trim())
                        .filter(t => t.length > 40);
                    if (!texts.length) return;

                    // Relative timestamp from aria-label spans
                    const spans = el.querySelectorAll("span[aria-label]");
                    let timeStr = "";
                    spans.forEach(s => {
                        const lbl = s.getAttribute("aria-label") || "";
                        if (/\\d+\\s*(second|minute|hour|day|week)/i.test(lbl)) timeStr = lbl;
                    });

                    // Post URL
                    const linkEl = el.querySelector(
                        "a[href*='/posts/'], a[href*='/photos/'], a[href*='/videos/']"
                    );
                    const url = linkEl ? linkEl.href : "";

                    // Reaction aria-labels (may not be present without login)
                    let reactions = 0;
                    spans.forEach(s => {
                        const lbl = s.getAttribute("aria-label") || "";
                        const m = lbl.match(/(\\d[\\d,]*)\\s*(reaction|like)/i);
                        if (m) reactions = parseInt(m[1].replace(/,/g, ""), 10);
                    });

                    results.push({ text: texts[0], time_label: timeStr, url, reactions });
                });
                return results;
            }""")

            browser.close()

            seen = set()
            for item in raw_posts:
                text = item.get("text", "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)

                dt = _parse_relative_time(item.get("time_label", ""))
                if dt is None:
                    dt = datetime.now(timezone.utc)  # fallback: treat as now
                if dt < cutoff:
                    continue

                likes = item.get("reactions", 0)
                sentiment = score_with_engagement(text, likes=likes)
                posts.append({
                    "id": hashlib.md5(text[:80].encode()).hexdigest()[:12],
                    "platform": "Facebook",
                    "handle": f"/{PAGE_HANDLE}",
                    "text": text[:500],
                    "published": dt.isoformat(),
                    "published_display": dt.strftime("%d %b %Y, %I:%M %p"),
                    "likes": likes,
                    "comments": 0,
                    "url": item.get("url") or PAGE_URL,
                    "sentiment_score": sentiment["compound"],
                    "sentiment_label": sentiment["label"],
                    "engagement": sentiment["engagement"],
                    "note": "likes unavailable without login" if likes == 0 else "",
                    # Comment text not accessible without Facebook login / Graph API.
                    # Sentiment scored on post content only.
                    "sentiment_basis": "post_content",
                })

    except Exception as e:
        print(f"[facebook] Playwright error: {e}")

    print(f"[facebook] /{PAGE_HANDLE}: {len(posts)} posts in last {hours}h")
    return posts
