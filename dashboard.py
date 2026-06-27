import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json, os, html as _html
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
import pytz


@st.cache_data(show_spinner=False)
def _build_wordcloud_cached(report_ts: str, report: dict):
    return _build_wordcloud(report)


@st.cache_data(show_spinner=False)
def _generate_pdfs_cached(report_ts: str, report: dict):
    from pdf_generator import generate_executive_summary, generate_full_report
    return generate_executive_summary(report), generate_full_report(report)


@st.cache_data(show_spinner=False)
def _cached_group_similar(cache_key: str, news: list):
    from collectors.news_collector import group_similar
    return sorted(group_similar(news), key=lambda x: x["sentiment_score"])


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_live_mentions(hours: int = 48):
    """Fetch Twitter/X and LinkedIn mentions live. Cached for 1 hour."""
    try:
        from collectors.mentions_collector import (fetch_twitter_mentions,
                                                    fetch_linkedin_mentions)
        from sentiment.analyzer import aggregate_sentiment
        tw = fetch_twitter_mentions(hours)
        li = fetch_linkedin_mentions(hours)
        items = tw + li
        sent  = aggregate_sentiment(items) if items else {
            "score": 0.0, "label": "neutral",
            "positive": 0, "negative": 0, "neutral": 0, "total": 0,
        }
        return items, sent, len(tw), len(li)
    except Exception as exc:
        return [], {}, 0, 0

def _e(text) -> str:
    """HTML-escape dynamic text to prevent broken markup."""
    return _html.escape(str(text or ""))

st.set_page_config(
    page_title="Piyush Goyal — Daily Media Brief",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

IST = pytz.timezone("Asia/Kolkata")

# ── Palette ────────────────────────────────────────────────────────────────────
POS    = "#1a7f37"   # 4.82:1 on white ✓
NEG    = "#b91c1c"   # 6.14:1 on white ✓
NEU    = "#5c5956"   # 6.10:1 on white ✓  (was #78716c which failed 4.5:1)
GOLD   = "#b45309"   # 4.58:1 on white ✓
BG     = "#f5f4f0"
CARD   = "#ffffff"
BORDER = "#e8e6e0"
ACCENT = "#1e3a5f"   # 11.3:1 on white ✓

# WCAG-safe muted text colours (all ≥4.5:1 on white)
MUTED  = "#6d6a66"   # 4.90:1 — replaces #a8a29e / #b0ada8
XFAINT = "#5e5b58"   # 5.93:1 — replaces #9e9b96

# Darker WCAG-safe variants of platform brand colours for use as foreground text
PLATFORM_TEXT_COLORS = {
    "Twitter/X":           "#0a6ea8",   # 5.50:1 ✓
    "Twitter/X (Office)":  "#0a6ea8",   # 5.50:1 ✓
    "Instagram":           "#a3105a",   # 7.55:1 ✓
    "Facebook":            "#1150ae",   # 5.10:1 ✓
    "YouTube":             "#b30000",   # 5.00:1 ✓
    "LinkedIn":            "#0a66c2",   # 5.68:1 ✓
}

PLATFORM_COLORS = {
    "Twitter/X": "#0f86c7",           # darkened from #1d9bf0 → 3.71:1 on white (non-text ≥3:1 ✓)
    "Twitter/X (Office)": "#0d8ecf",  # 3.49:1 ✓
    "Instagram": "#e1306c",           # 4.30:1 ✓
    "Facebook": "#1877f2",            # 4.13:1 ✓
    "YouTube": "#ff0000",             # 3.99:1 ✓
    "LinkedIn": "#0a66c2",            # 5.68:1 ✓
}

# SVG path data (d=) for each platform — used as low-opacity watermarks.
# Multiple sub-paths are separated by spaces; fill-rule="evenodd" handles holes.
_PLATFORM_SVG = {
    "Twitter/X": (
        "M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231"
        "-5.401 6.231H2.74l7.73-8.835L1.254 2.25H8.08l4.713 6.231z"
        "m-1.161 17.52h1.833L7.084 4.126H5.117z"
    ),
    "Twitter/X (Office)": (
        "M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231"
        "-5.401 6.231H2.74l7.73-8.835L1.254 2.25H8.08l4.713 6.231z"
        "m-1.161 17.52h1.833L7.084 4.126H5.117z"
    ),
    "Instagram": (
        "M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919"
        " 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069"
        " 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07"
        "-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699"
        "-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583"
        ".07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645"
        "-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2"
        "-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259"
        ".014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058"
        " 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2"
        " 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259"
        "-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281"
        "-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162"
        " 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403"
        "-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209"
        " 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796"
        " 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645"
        " 1.439-1.44s-.644-1.44-1.439-1.44z"
    ),
    "Facebook": (
        "M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388"
        " 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007"
        " 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83"
        "c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385"
        "C19.612 23.027 24 18.062 24 12.073z"
    ),
    "YouTube": (
        "M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545"
        " 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07"
        " 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871"
        ".505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0"
        " 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814z"
        "M9.545 15.568V8.432L15.818 12l-6.273 3.568z"
    ),
    "LinkedIn": (
        "M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853"
        " 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9"
        " 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286z"
        "M5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063"
        " 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065"
        "-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452z"
        "M22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24"
        " 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2"
        " 0 22.222 0h.003z"
    ),
}
SENT_COLOR = {"positive": POS, "negative": NEG, "neutral": NEU}
SENT_ICON  = {"positive": "▲", "negative": "▼", "neutral": "●"}

# ── Inject global CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Skip navigation (WCAG 2.4.1) ── */
  .skip-link {
    position: absolute;
    top: -999px;
    left: 0;
    background: #1e3a5f;
    color: #ffffff;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 600;
    text-decoration: none;
    z-index: 9999;
    border-radius: 0 0 4px 0;
  }
  .skip-link:focus { top: 0; outline: 3px solid #f5a623; outline-offset: 2px; }

  /* ── Base ── */
  [data-testid="stAppViewContainer"] {
    background: #eeebe5;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', system-ui, sans-serif;
  }
  [data-testid="stSidebar"] {
    background: #0f1623;
    color: white;
    border-right: 1px solid #1c2333;
  }
  .block-container { padding: 2rem 2.8rem 3.5rem; max-width: 1480px; }
  h1, h2, h3 { font-family: 'Georgia', 'Cambria', serif; letter-spacing: -0.025em; }

  /* ── Focus indicators (WCAG 2.4.7, 2.4.11) ── */
  :focus-visible {
    outline: 3px solid #1e3a5f !important;
    outline-offset: 2px !important;
  }
  a:focus-visible, button:focus-visible, [role="button"]:focus-visible {
    outline: 3px solid #1e3a5f !important;
    outline-offset: 3px !important;
    border-radius: 2px;
  }

  /* ── Dividers ── */
  .divider {
    border: none;
    border-top: 1px solid #e0ddd7;
    margin: 2rem 0;
  }

  /* ── Premium cards ── */
  .card {
    background: #ffffff;
    border: 1px solid #e0ddd7;
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 12px;
    box-shadow:
      0 1px 2px rgba(0,0,0,0.04),
      0 4px 12px rgba(0,0,0,0.06),
      0 16px 40px rgba(0,0,0,0.04);
  }

  /* ── Sentiment item cards ── */
  .neg-card {
    background: #fff9f9;
    border: 1px solid #f2d0d0;
    border-left: 4px solid #b91c1c;
    border-radius: 12px;
    padding: 15px 20px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(185,28,28,0.06);
  }
  .pos-card {
    background: #f7fdf8;
    border: 1px solid #c4e8cc;
    border-left: 4px solid #1a7f37;
    border-radius: 12px;
    padding: 15px 20px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(26,127,55,0.06);
  }
  .neu-card {
    background: #f9f8f5;
    border: 1px solid #e0ddd7;
    border-radius: 12px;
    padding: 15px 20px;
    margin-bottom: 10px;
  }

  /* ── Badges (contrast verified ≥4.5:1) ── */
  .badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
  }
  .badge-neg { background: #fde8e8; color: #9b1c1c; }   /* 6.48:1 ✓ */
  .badge-pos { background: #d6f5df; color: #145728; }   /* 6.59:1 ✓ */
  .badge-neu { background: #efede9; color: #5c5956; }   /* 5.28:1 ✓ */

  /* ── Score display ── */
  .score-big {
    font-size: 40px;
    font-weight: 700;
    font-family: 'Georgia', serif;
    letter-spacing: -0.04em;
    line-height: 1;
  }

  /* ── Misc — muted text now #5c5956 (6.1:1) replacing #7c7975 (3.9:1) ── */
  .meta-text { font-size: 12px; color: #5c5956; line-height: 1.65; }
  .platform-pill { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; color:white; }
  .adjusted-badge { font-size:10px; color:#92400e; background:#fef3c7; padding:1px 6px; border-radius:8px; }

  /* ── Section labels ── */
  .section-label {
    font-size: 15px;
    font-weight: 700;
    color: #2c2a28;
    letter-spacing: -0.01em;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e0ddd7;
  }

  /* ── Compact PDF download buttons (main content only, not sidebar) ── */
  :not([data-testid="stSidebar"]) [data-testid="stDownloadButton"] > button {
    padding: 0.2rem 0.55rem !important;
    font-size: 11px !important;
    line-height: 1.4 !important;
    min-height: auto !important;
  }

  /* ── Follower panel ── */
  .follower-panel {
    background: #faf8f4;
    border: 1px solid #e0ddd7;
    border-radius: 18px;
    padding: 20px 18px;
    box-shadow:
      0 1px 2px rgba(0,0,0,0.03),
      0 4px 12px rgba(0,0,0,0.05);
  }
  .follower-panel-title {
    font-size: 15px;
    font-weight: 700;
    color: #2c2a28;
    letter-spacing: -0.01em;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e0ddd7;
  }

  /* ── Main content buttons ── */
  :not([data-testid="stSidebar"]) [data-testid="stButton"] > button {
    background-color: #ffffff !important;
    color: #1e3a5f !important;
    border: 1.5px solid #c8c4bc !important;
    font-weight: 600 !important;
  }
  :not([data-testid="stSidebar"]) [data-testid="stButton"] > button:hover {
    background-color: #1e3a5f !important;
    color: #ffffff !important;
    border-color: #1e3a5f !important;
  }
</style>
""", unsafe_allow_html=True)

# Skip-navigation anchor (WCAG 2.4.1)
st.markdown(
    '<a class="skip-link" href="#main-content">Skip to main content</a>',
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _to_ist(iso: str) -> datetime:
    dt = dateparser.parse(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


def report_period_header(report: dict) -> str:
    gen_iso = report.get("generated_at", "")
    hours   = report.get("period_hours", 24)
    try:
        gen_ist    = _to_ist(gen_iso)
        date_str   = gen_ist.strftime("%d %B %Y")
        period_str = f"Last {hours} hours"
        generated  = gen_ist.strftime("%d %b %Y, %I:%M %p IST")
        return date_str, period_str, generated
    except Exception:
        return "—", "—", "—"


def sentiment_bar_html(pos: int, neu: int, neg: int) -> str:
    total = pos + neu + neg or 1
    pp  = pos / total * 100
    np_ = neu / total * 100
    ngp = neg / total * 100
    aria = (f"Sentiment breakdown: {pos} positive ({pp:.0f}%), "
            f"{neu} neutral ({np_:.0f}%), {neg} negative ({ngp:.0f}%)")
    # Bar uses role="img" so screen readers get the aria-label summary instead
    # of interpreting empty decorative divs (WCAG 1.3.1, 1.4.1).
    return f"""
    <div role="img" aria-label="{aria}"
         style="display:flex;gap:2px;height:8px;border-radius:6px;overflow:hidden;margin:12px 0 10px">
      <div style="background:{POS};width:{pp:.1f}%;min-width:{('2px' if pp > 0 else '0')}" aria-hidden="true"></div>
      <div style="background:#78716c;width:{np_:.1f}%;min-width:{('2px' if np_ > 0 else '0')}" aria-hidden="true"></div>
      <div style="background:{NEG};width:{ngp:.1f}%;min-width:{('2px' if ngp > 0 else '0')}" aria-hidden="true"></div>
    </div>
    <div style="display:flex;gap:20px;font-size:12px;color:#5c5956" aria-hidden="true">
      <span><span style="color:{POS};font-weight:700">{pos}</span> <span>positive ({pp:.0f}%)</span></span>
      <span><span style="color:#5c5956;font-weight:700">{neu}</span> <span>neutral ({np_:.0f}%)</span></span>
      <span><span style="color:{NEG};font-weight:700">{neg}</span> <span>negative ({ngp:.0f}%)</span></span>
    </div>"""


def score_label(s: float) -> str:
    if s >= 0.3:   return "Strongly Positive"
    if s >= 0.05:  return "Positive"
    if s <= -0.3:  return "Strongly Negative"
    if s <= -0.05: return "Negative"
    return "Neutral"


def _md_escape(text: str) -> str:
    """Escape Markdown special characters so titles render as plain text in links."""
    for ch in r"\`*_{}[]()#+-.!|$":
        text = text.replace(ch, "\\" + ch)
    return text


def _score_badge(label: str, sc: float) -> str:
    """Safe badge HTML — contains only constants (hex colors) and a float. No user text."""
    color    = SENT_COLOR[label]
    icon     = SENT_ICON[label]
    aria_lbl = f"{label} sentiment, score {sc:+.2f}"
    return (
        f"<div style='text-align:right;padding-top:4px'>"
        f"<span role='img' aria-label='{aria_lbl}' "
        f"style='background:{color}12;border:1.5px solid {color}55;color:{color};"
        f"padding:4px 12px;border-radius:20px;font-size:12px;font-weight:700;"
        f"white-space:nowrap;font-family:Georgia,serif;letter-spacing:0.3px'>"
        f"<span aria-hidden='true'>{icon}</span> {sc:+.2f}</span></div>"
    )


def render_news_card(item: dict, show_summary: bool = False):
    """Render a news card using native Streamlit components. No user text goes into HTML."""
    label = item["sentiment_label"]
    sc    = item["sentiment_score"]
    color = SENT_COLOR[label]

    c1, c2 = st.columns([11, 2])
    with c1:
        title = _md_escape(item.get("title", ""))
        url   = item.get("url", "#")
        st.markdown(f"**[{title}]({url})**")

        group_items = item.get("group_items", [])
        meta_parts = [f"`{item.get('source','')}`", item.get("published_display", "")]
        # Show "+N similar" only for old reports that lack group_items detail
        if item.get("group_size", 1) > 1 and not group_items:
            meta_parts.append(f"*+{item['group_size']-1} similar*")
        if item.get("adjustment"):
            meta_parts.append("*⚙ reclassified*")
        st.caption("  ·  ".join(meta_parts))

        if show_summary and item.get("summary"):
            st.caption(item["summary"])

    with c2:
        st.markdown(_score_badge(label, sc), unsafe_allow_html=True)

    if group_items:
        n = len(group_items)
        with st.expander(f"+ {n} more source{'s' if n > 1 else ''} covering this story"):
            for g in group_items:
                g_label = g["sentiment_label"]
                g_sc    = g["sentiment_score"]
                ga, gb = st.columns([11, 2])
                with ga:
                    gt = _md_escape(g.get("title", ""))
                    gu = g.get("url", "#")
                    st.markdown(f"[{gt}]({gu})")
                    g_meta = [f"`{g.get('source','')}`", g.get("published_display", "")]
                    if g.get("adjustment"):
                        g_meta.append("*⚙ reclassified*")
                    st.caption("  ·  ".join(g_meta))
                with gb:
                    st.markdown(_score_badge(g_label, g_sc), unsafe_allow_html=True)
                st.markdown(
                    "<hr style='margin:2px 0 8px;border:none;border-top:1px solid #f5f4f0'>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        "<hr style='margin:4px 0 10px;border:none;border-top:1px solid #f0efec'>",
        unsafe_allow_html=True,
    )


def render_mention_card(item: dict):
    """Render a Twitter/LinkedIn mention with sentiment score and engagement."""
    label    = item.get("sentiment_label", "neutral")
    sc       = item.get("sentiment_score", 0.0)
    platform = item.get("platform", "")
    pc       = PLATFORM_COLORS.get(platform, "#6b7280")
    url      = item.get("url", "")
    total_eng = _total_engagement(item)

    c1, c2 = st.columns([11, 2])
    with c1:
        handle_str = _e(item.get("handle", ""))
        date_str   = _e(item.get("published_display", ""))
        meta = f"{handle_str} · {date_str}" if handle_str else date_str
        st.markdown(
            f"<span style='background:#1a5cdb;color:white;padding:2px 8px;border-radius:4px;"
            f"font-size:11px;font-weight:600'>{platform}</span>"
            f"&nbsp;<span style='font-size:12px;color:#5c5956'>{meta}</span>",
            unsafe_allow_html=True,
        )
        text = item.get("text", "") or item.get("title", "")
        st.markdown(
            f"<p style='color:#1a1917;font-size:14px;line-height:1.5;margin:4px 0 6px'>{_e(text[:300])}</p>",
            unsafe_allow_html=True,
        )
        caption_parts = []
        if item.get("summary"):
            caption_parts.append(item["summary"][:200])
        if total_eng > 0:
            likes = item.get("likes", 0) or 0
            cmt   = item.get("comments", 0) or 0
            rt    = item.get("retweets") or item.get("shares") or 0
            caption_parts.append(f"❤ {likes:,}  💬 {cmt:,}  🔁 {rt:,}  · {total_eng:,} total engagement")
        if url:
            caption_parts.append(f"[View source post on {platform}]({url})")
        if caption_parts:
            st.caption("  ·  ".join(caption_parts))
    with c2:
        st.markdown(_score_badge(label, sc), unsafe_allow_html=True)

    st.markdown(
        "<hr style='margin:4px 0 10px;border:none;border-top:1px solid #f0efec'>",
        unsafe_allow_html=True,
    )


def _total_engagement(post: dict) -> int:
    """Raw engagement total: likes + comments + retweets/shares."""
    return (
        (post.get("likes", 0) or 0)
        + (post.get("comments", 0) or 0)
        + (post.get("retweets") or post.get("shares") or 0)
    )


def _top_post_card_html(post: dict, rank: int) -> str:
    """HTML card for the Top Posts grid."""
    pc       = PLATFORM_COLORS.get(post.get("platform", ""), "#6b7280")
    platform = post.get("platform", "")
    likes    = post.get("likes", 0) or 0
    cmt      = post.get("comments", 0) or 0
    rt       = post.get("retweets") or post.get("shares") or 0
    total    = likes + cmt + rt
    url      = post.get("url", "")
    text     = post.get("text", "")
    preview  = _e(text[:150]) + ("…" if len(text) > 150 else "")
    rank_lbl = {1: "Ranked 1st", 2: "Ranked 2nd", 3: "Ranked 3rd"}.get(rank, f"Rank {rank}")
    medal    = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
    link_lbl = f"{rank_lbl}, {platform}, {total:,} total engagement (opens in new tab)"

    wrap_open  = (f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
                  f'aria-label="{link_lbl}" style="text-decoration:none">') if url else "<div>"
    wrap_close = "</a>" if url else "</div>"

    return f"""
    {wrap_open}
    <div style="background:#fff;border:1px solid #e8e6e0;border-top:3px solid {pc};
                border-radius:14px;padding:18px 20px;margin-bottom:8px;
                box-shadow:0 1px 4px rgba(0,0,0,0.05)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <span aria-hidden="true" style="font-size:20px;line-height:1">{medal}</span>
        <span style="background:#1a5cdb;color:white;padding:2px 9px;border-radius:4px;
                     font-size:10px;font-weight:700;letter-spacing:0.3px">{platform}</span>
      </div>
      <div style="font-size:30px;font-weight:700;color:#1a1917;font-family:Georgia,serif;
                  letter-spacing:-0.03em;line-height:1">{total:,}</div>
      <div style="font-size:9px;font-weight:700;color:#6d6a66;letter-spacing:1.2px;
                  text-transform:uppercase;margin-bottom:10px">Total Engagement</div>
      <div style="display:flex;gap:14px;font-size:11px;color:#5c5956;margin-bottom:12px">
        <span><span aria-hidden="true">❤</span> <span aria-label="likes">{likes:,}</span></span>
        <span><span aria-hidden="true">💬</span> <span aria-label="comments">{cmt:,}</span></span>
        <span><span aria-hidden="true">🔁</span> <span aria-label="retweets or shares">{rt:,}</span></span>
      </div>
      <div style="font-size:12px;color:#374151;line-height:1.5;border-top:1px solid #f0efec;
                  padding-top:10px;display:-webkit-box;-webkit-line-clamp:3;
                  -webkit-box-orient:vertical;overflow:hidden">{preview}</div>
      <div style="font-size:10px;color:#6d6a66;margin-top:8px">
        {_e(post.get('handle',''))} &nbsp;·&nbsp; {_e(post.get('published_display',''))}
      </div>
    </div>
    {wrap_close}"""


def render_social_card(post: dict):
    """Render a social media post card with total-engagement badge."""
    pc       = PLATFORM_COLORS.get(post.get("platform", ""), "#6b7280")
    likes    = post.get("likes", 0) or 0
    cmt      = post.get("comments", 0) or 0
    rt       = post.get("retweets") or post.get("shares") or 0
    total    = likes + cmt + rt
    url      = post.get("url", "")
    platform = post.get("platform", "")

    c1, c2 = st.columns([11, 2])
    with c1:
        st.markdown(
            f"<span style='background:#1a5cdb;color:white;padding:2px 8px;border-radius:4px;"
            f"font-size:11px;font-weight:600'>{platform}</span>"
            f"&nbsp;<span style='font-size:12px;color:#5c5956'>"
            f"{_e(post.get('handle',''))} · {_e(post.get('published_display',''))}</span>",
            unsafe_allow_html=True,
        )
        st.write(post.get("text", "")[:300])

        eng = f"❤ {likes:,}   💬 {cmt:,}   🔁 {rt:,}"
        if post.get("note"):
            eng += "   *(likes hidden — no login)*"
        if url:
            eng += f"   [View post on {platform}]({url})"
        st.caption(eng)

    with c2:
        if total > 0:
            ptc = PLATFORM_TEXT_COLORS.get(platform, ACCENT)
            st.markdown(
                f"<div style='text-align:right;padding-top:4px'>"
                f"<span style='background:{pc}14;border:1.5px solid {pc}55;color:{ptc};"
                f"padding:4px 10px;border-radius:20px;font-size:12px;font-weight:700;"
                f"white-space:nowrap;font-family:Georgia,serif'>{total:,}</span>"
                f"<div style='font-size:9px;color:#6d6a66;margin-top:3px;text-align:right;"
                f"letter-spacing:0.5px'>engagement</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<hr style='margin:4px 0 10px;border:none;border-top:1px solid #f0efec'>",
        unsafe_allow_html=True,
    )


def _strip_html(text: str) -> str:
    """
    Remove HTML markup, URLs, HTML entities, and non-Latin characters so the
    word cloud only sees meaningful English terms.
    """
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"&[#a-zA-Z0-9]+;", " ", text)
    text = re.sub(r'\b\w*="[^"]*"', " ", text)
    text = re.sub(r"[^a-zA-Z\s'\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _pos_filter(corpus: str, stop_words: set) -> dict:
    """
    POS-tag the corpus with NLTK and return a {word: frequency} dict with
    ALL verbs removed (tags VB, VBD, VBG, VBN, VBP, VBZ).
    Falls back to a plain frequency count if NLTK is unavailable.
    """
    import re
    words = re.findall(r"[a-zA-Z']{2,}", corpus)
    try:
        import nltk
        try:
            nltk.pos_tag(["test"])
        except LookupError:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            nltk.download("punkt_tab", quiet=True)
        tagged = nltk.pos_tag(words)
        freq = {}
        for word, tag in tagged:
            w = word.lower()
            if tag.startswith("VB"):          # skip all verb forms
                continue
            if w in stop_words or len(w) < 2:
                continue
            freq[w] = freq.get(w, 0) + 1
        return freq
    except Exception:
        # Fallback without verb filtering
        freq = {}
        for w in (w.lower() for w in words):
            if w not in stop_words and len(w) >= 2:
                freq[w] = freq.get(w, 0) + 1
        return freq


def _build_wordcloud(report: dict):
    """
    Generate a word cloud PNG from news headlines/summaries and the text of
    the top-10 most-engaged social posts.  Returns PNG bytes or None.
    """
    try:
        from wordcloud import WordCloud, STOPWORDS
    except ImportError:
        return None

    news   = report["minister"]["news"] + report["ministry"]["news"]
    social = (report["minister"]["tweets"]
              + report["minister"]["instagram"]
              + report["minister"]["facebook"])

    top_social = sorted(social, key=_total_engagement, reverse=True)[:10]

    parts: list[str] = []

    # News titles (clean) + stripped summaries
    for item in news:
        parts.append(_strip_html(item.get("title", "")))
        parts.append(_strip_html(item.get("summary") or ""))

    # Top engaged social posts — repeated by engagement tier
    for post in top_social:
        text   = _strip_html(post.get("text", ""))
        eng    = _total_engagement(post)
        weight = 1 + min(eng // 3000, 4)   # 1× … 5× for > 12 k engagement
        parts.extend([text] * weight)

    corpus = " ".join(p for p in parts if p.strip())
    if len(corpus.split()) < 10:
        return None

    # Verbs are now removed automatically by NLTK POS tagging in _pos_filter().
    # This list covers only non-verb noise: subject boilerplate, generic
    # uninformative nouns/adjectives, time words, and HTML residue.
    stop_words = set(STOPWORDS)
    stop_words.update({
        # Subject boilerplate
        "piyush", "goyal", "minister", "union", "commerce", "ministry",
        "india", "indian", "government", "bjp",

        # Vague adjectives with no topic signal
        "amazing", "global", "strong", "stronger", "key", "major",
        "important", "significant", "great", "good", "better", "best",
        "high", "higher", "big", "large", "broad", "robust", "dynamic",
        "strategic", "comprehensive", "bilateral", "mutual", "critical",
        "crucial", "vital", "landmark", "historic", "innovative",
        "transformative", "ambitious", "effective", "various", "several",
        "certain", "specific", "particular", "current", "recent", "latest",
        "overall", "general", "common", "possible", "potential", "similar",
        "different", "additional", "further", "full", "whole", "entire",
        "wide", "long", "short",

        # Generic uninformative nouns
        "pact", "post", "progress", "growth", "move", "step", "way",
        "place", "part", "role", "work", "focus", "level", "number",
        "area", "side", "effort", "efforts", "goal", "goals",
        "plan", "plans", "result", "results", "impact", "outcome",
        "development", "opportunity", "opportunities", "issue", "issues",
        "sector", "sectors", "market", "markets", "need", "needs",
        "availability", "situation", "condition", "conditions",
        "context", "position", "approach", "process", "framework",
        "stage", "phase", "point", "basis", "course", "view",
        "terms", "term", "form", "forms", "aspect", "aspects",
        "statement", "statements", "report", "reports",
        "session", "conference", "summit", "event", "programme", "program",
        "initiative", "initiatives", "measure", "measures",
        "action", "actions", "response", "responses",

        # Time / quantity noise
        "new", "one", "two", "three", "four", "five",
        "year", "years", "today", "day", "week", "time", "mr", "ms",
        "per", "rs", "lakh", "crore", "cent", "news", "article",
        "recently", "soon", "early", "late", "next", "last", "first",
        "decade", "decades", "still", "yet", "already", "always",
        "ever", "never", "just", "even", "well", "back", "around",
        "across", "among", "between", "within", "without", "whether",
        "while", "though", "although", "however", "therefore", "thus",

        # Single letters / short noise
        "s", "t", "u", "re", "ve", "ll",

        # HTML attribute residue
        "href", "src", "alt", "rel", "class", "nbsp",
    })

    freq = _pos_filter(corpus, stop_words)
    if not freq:
        return None

    wc = WordCloud(
        width=1600,
        height=520,
        background_color="#f5f4f0",
        colormap="Blues",
        max_words=120,
        collocations=False,
        min_font_size=11,
        max_font_size=130,
        prefer_horizontal=0.80,
    ).generate_from_frequencies(freq)

    import io
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _fmt_delta(val) -> str:
    """Format a follower delta as a coloured HTML snippet with ARIA label."""
    if val is None:
        return "<span style='color:#6d6a66' aria-label='no change today'>—</span>"
    sign   = "+" if val >= 0 else ""
    color  = "#1a7f37" if val >= 0 else "#b91c1c"
    arrow  = "▲" if val >= 0 else "▼"
    dirstr = "increased" if val >= 0 else "decreased"
    n = abs(val)
    txt = f"{n/1_000_000:.1f}M" if n >= 1_000_000 else \
          f"{n/1_000:.1f}K"     if n >= 1_000      else str(n)
    return (f"<span style='color:{color};font-weight:700' "
            f"aria-label='Today {dirstr} by {sign}{txt}'>"
            f"<span aria-hidden='true'>{arrow}</span>&nbsp;{sign}{txt}</span>")


def follower_pill(f: dict) -> str:
    platform  = f["platform"]
    pc        = PLATFORM_COLORS.get(platform, "#6b7280")
    ptc       = PLATFORM_TEXT_COLORS.get(platform, ACCENT)
    is_live   = f["is_live"]
    live_txt  = "live" if is_live else "cached"
    url       = f["url"]
    svg_d     = _PLATFORM_SVG.get(platform, "")
    followers = f["followers_display"]
    link_lbl  = f"{platform} — {followers} followers (opens in new tab)"

    if svg_d:
        fill_rule = 'fill-rule="evenodd"' if platform == "Instagram" else ""
        # aria-hidden: purely decorative watermark (WCAG 1.1.1)
        watermark = (
            f'<div aria-hidden="true" style="position:absolute;right:-10px;bottom:-10px;'
            f'width:76px;height:76px;opacity:0.07;pointer-events:none">'
            f'<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" '
            f'focusable="false" style="width:100%;height:100%;fill:{pc}">'
            f'<path {fill_rule} d="{svg_d}"/>'
            f'</svg></div>'
        )
    else:
        watermark = ""

    d_day      = _fmt_delta(f.get("delta_daily"))
    delta_html = (
        f'<div style="border-top:1px solid #f0efec;padding-top:6px;margin-top:6px">'
        f'<div style="font-size:8px;color:#6d6a66;letter-spacing:0.8px;'
        f'text-transform:uppercase;margin-bottom:2px" aria-hidden="true">Today</div>'
        f'<div style="font-size:11px">{d_day}</div>'
        f'</div>'
    )

    live_indicator = (
        f'<span aria-label="{live_txt}" style="margin-right:3px">'
        f'{"●" if is_live else "○"}</span>'
    )

    return (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
        f'aria-label="{link_lbl}" style="text-decoration:none">'
        f'<div style="position:relative;display:inline-block;background:#fff;'
        f'border:1px solid #e8e6e0;border-top:3px solid {pc};'
        f'border-radius:12px;padding:16px 20px 14px;text-align:center;'
        f'min-width:130px;margin:4px;overflow:hidden;vertical-align:top;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.05)">'
        f'{watermark}'
        f'<div style="position:relative;z-index:1">'
        f'<div style="font-size:10px;color:{ptc};font-weight:700;'
        f'letter-spacing:0.5px;margin-bottom:6px">{live_indicator}{platform}</div>'
        f'<div style="font-size:26px;font-weight:700;color:#1a1917;'
        f'font-family:Georgia,serif;letter-spacing:-0.03em">'
        f'{followers}</div>'
        f'<div style="font-size:9px;color:#6d6a66;margin-top:2px;letter-spacing:0.3px">'
        f'followers</div>'
        f'{delta_html}'
        f'</div>'
        f'</div></a>'
    )


def follower_card_vertical(f: dict) -> str:
    """Ultra-compact follower row for the vertical right panel (~32 px each)."""
    platform  = f["platform"]
    pc        = PLATFORM_COLORS.get(platform, "#6b7280")
    ptc       = PLATFORM_TEXT_COLORS.get(platform, ACCENT)
    is_live   = f["is_live"]
    live_txt  = "live" if is_live else "cached"
    url       = f["url"]
    svg_d     = _PLATFORM_SVG.get(platform, "")
    followers = f["followers_display"]
    link_lbl  = f"{platform} — {followers} followers (opens in new tab)"

    if svg_d:
        fill_rule = 'fill-rule="evenodd"' if platform == "Instagram" else ""
        # aria-hidden: decorative icon at 11% opacity (WCAG 1.1.1)
        icon_html = (
            f'<div aria-hidden="true" style="width:18px;height:18px;flex-shrink:0;opacity:0.11">'
            f'<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" '
            f'focusable="false" style="width:100%;height:100%;fill:{pc}">'
            f'<path {fill_rule} d="{svg_d}"/></svg></div>'
        )
    else:
        icon_html = '<div aria-hidden="true" style="width:18px;flex-shrink:0"></div>'

    d_day = f.get("delta_daily")
    if d_day is not None:
        sign    = "+" if d_day >= 0 else ""
        d_color = "#1a7f37" if d_day >= 0 else "#b91c1c"
        dirstr  = "increased" if d_day >= 0 else "decreased"
        n = abs(d_day)
        d_txt = (f"{n/1_000_000:.1f}M" if n >= 1_000_000
                 else f"{n/1_000:.1f}K"  if n >= 1_000 else str(n))
        delta_html = (
            f'<span style="font-size:11px;font-weight:700;color:{d_color};'
            f'letter-spacing:-0.01em;flex-shrink:0" '
            f'aria-label="Today {dirstr} by {sign}{d_txt}">{sign}{d_txt}</span>'
        )
    else:
        delta_html = (
            '<span style="font-size:11px;color:#6d6a66;flex-shrink:0" '
            'aria-label="no change today">—</span>'
        )

    live_indicator = (
        f'<span aria-label="{live_txt}" style="margin-right:2px">'
        f'{"●" if is_live else "○"}</span>'
    )

    return (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
        f'aria-label="{link_lbl}" style="text-decoration:none;display:block">'
        f'<div style="display:flex;align-items:center;gap:8px;padding:5px 11px;'
        f'background:#ffffff;border:1px solid #e0ddd7;border-left:2.5px solid {pc};'
        f'border-radius:8px;margin-bottom:5px;'
        f'box-shadow:0 1px 2px rgba(0,0,0,0.04),0 2px 6px rgba(0,0,0,0.04)">'
        f'{icon_html}'
        f'<div style="flex:1;min-width:0">'
        f'<div style="font-size:8.5px;color:{ptc};font-weight:700;letter-spacing:0.5px;'
        f'text-transform:uppercase;line-height:1">{live_indicator}{platform}</div>'
        f'<div style="font-size:15px;font-weight:700;color:#18171a;'
        f'font-family:Georgia,serif;letter-spacing:-0.02em;line-height:1.2">'
        f'{followers}</div>'
        f'</div>'
        f'{delta_html}'
        f'</div></a>'
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='color:#e2e8f0;font-size:15px;letter-spacing:0.5px;margin-bottom:16px'>⚙ Controls</h3>", unsafe_allow_html=True)
    if st.button("🔄 Generate Report Now", type="primary", use_container_width=True):
        with st.spinner("Collecting data… (60–90 sec)"):
            try:
                from report_generator import generate_report
                report = generate_report(hours=24)
                st.session_state["report"] = report
                st.success("Done!")
            except Exception as e:
                st.error(str(e))

    from report_generator import load_latest_report, list_reports
    # Always check if a newer report is available on disk and auto-load it.
    _latest_path = os.path.join("data", "reports", "latest.json")
    _latest_mtime = os.path.getmtime(_latest_path) if os.path.exists(_latest_path) else 0
    if _latest_mtime != st.session_state.get("_report_mtime"):
        r = load_latest_report()
        if r:
            st.session_state["report"] = r
            st.session_state["_report_mtime"] = _latest_mtime

    saved = list_reports()
    if saved:
        st.markdown("<p style='color:#94a3b8;font-size:11px;letter-spacing:1px;text-transform:uppercase;margin-top:20px;margin-bottom:6px'>Archive</p>", unsafe_allow_html=True)
        sel = st.selectbox("", saved, label_visibility="collapsed")
        if st.button("Load selected", use_container_width=True):
            with open(os.path.join("data", "reports", sel)) as f:
                st.session_state["report"] = json.load(f)

    st.markdown("<hr style='border-color:#2d3548;margin:20px 0'>", unsafe_allow_html=True)

    # PDF downloads — generated once per report via @st.cache_data
    if st.session_state.get("report"):
        try:
            _r    = st.session_state["report"]
            _r_ts = _r.get("generated_at", "")
            _pdf_exec_sb, _pdf_full_sb = _generate_pdfs_cached(_r_ts, _r)
            _ts = datetime.now(IST).strftime("%Y%m%d_%H%M")
            st.download_button(
                label="⬇ Executive Summary",
                data=_pdf_exec_sb,
                file_name=f"executive_summary_{_ts}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.download_button(
                label="⬇ Full Report",
                data=_pdf_full_sb,
                file_name=f"full_report_{_ts}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as _pdf_err:
            st.caption(f"PDF unavailable: {_pdf_err}")

    st.markdown("<hr style='border-color:#2d3548;margin:20px 0'>", unsafe_allow_html=True)
    show_summary = st.toggle("Show article summaries", value=False)


    st.markdown("""
    <div style='font-size:11px;color:#94a3b8;margin-top:16px;line-height:1.8'>
    <b style='color:#cbd5e1;letter-spacing:0.5px'>Handles tracked</b><br>
    🐦 @PiyushGoyal · @PiyushGoyalOffc<br>
    📸 @piyushgoyalofficial<br>
    👤 /PiyushGoyalOfficial<br>
    ▶ YouTube · 💼 LinkedIn<br><br>
    <span style='color:#64748b'>Auto-report: daily 9 AM IST</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
report = st.session_state.get("report")

# ── Empty state ────────────────────────────────────────────────────────────────
if not report:
    st.markdown(f"""
    <div style="border-top:3px solid {ACCENT};padding-top:20px;margin-bottom:28px">
      <h1 style="margin:0;font-size:28px;color:#1a1917;font-family:'Georgia',serif">Piyush Goyal — Daily Media Brief</h1>
    </div>
    """, unsafe_allow_html=True)
    st.info("Click **Generate Report Now** in the sidebar (☰) to fetch live data.")
    st.markdown("""
**What this dashboard tracks:**
- News from Google News, Economic Times, NDTV, Business Standard, Hindustan Times, LiveMint
- Posts from Twitter/X (@PiyushGoyal, @PiyushGoyalOffc), Instagram, Facebook
- Separate sentiment stream for Ministry of Commerce & Industry
- Follower counts across all official social media handles
""")
    st.stop()

# ── Report period metadata ─────────────────────────────────────────────────────
date_str, period_str, generated = report_period_header(report)
stats    = report.get("stats", {})
ms       = report["minister"]["sentiment"]
m2       = report["ministry"]["sentiment"]
followers = report.get("followers", [])

# ── Header ─────────────────────────────────────────────────────────────────────
# Build stats line — only include non-zero mention counts to avoid showing "0"
_stat_parts = [
    f"{stats.get('total_news', 0)} news",
    f"{stats.get('total_tweets', 0)} tweets",
    f"{stats.get('total_instagram', 0)} IG",
    f"{stats.get('total_facebook', 0)} FB",
]
if stats.get('total_tw_mentions'):
    _stat_parts.append(f"{stats['total_tw_mentions']} 🐦 mentions")
if stats.get('total_li_mentions'):
    _stat_parts.append(f"{stats['total_li_mentions']} 💼 mentions")
_header_stats = " &nbsp;·&nbsp; ".join(_stat_parts)

st.markdown(f"""
<div id="main-content" role="banner"
     style="border-top:3px solid {ACCENT};padding-top:18px;border-bottom:1px solid #e8e6e0;padding-bottom:16px;margin-bottom:24px">
  <h1 style="margin:0;font-size:28px;color:#1a1917;font-family:'Georgia',serif;letter-spacing:-0.02em">Piyush Goyal — Daily Media Brief</h1>
  <div style="font-size:14px;color:#5c5956;margin-top:6px">
    <span aria-hidden="true">📅</span> <b style="color:#1a1917">{date_str}</b> &nbsp;·&nbsp;
    <span style="color:#5c5956">{period_str}</span>
  </div>
  <div style="font-size:12px;color:#5c5956;margin-top:4px">
    Generated: {generated} &nbsp;·&nbsp; {_header_stats}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sentiment overview — 3 equal columns, scores on −100 to +100 scale ────────
# Order: Minister (news) | Minister (social mentions) | Ministry (news)
_sa, _sb, _sc_col = st.columns(3, gap="medium")

# Fetch social mentions data before rendering (needed for middle card)
_rep_mm    = report["minister"].get("mentions_sentiment", {})
_rep_ments = report["minister"].get("mentions", [])
if _rep_mm.get("total", 0) > 0:
    sm_sent = _rep_mm
    sm_tw_n = sum(1 for m in _rep_ments if m["platform"] == "Twitter/X")
    sm_li_n = sum(1 for m in _rep_ments if m["platform"] == "LinkedIn")
else:
    sm_sent, sm_tw_n, sm_li_n = {}, 0, 0
sm_total = sm_sent.get("total", 0)
sm_sc    = sm_sent.get("score", 0.0) if sm_total > 0 else 0.0
sm_color = POS if sm_sc >= 0.05 else (NEG if sm_sc <= -0.05 else NEU)

_SCALE_NOTE = (
    "<div style='font-size:9px;color:#6d6a66;letter-spacing:0.3px;margin-top:3px'>"
    "Scale: −100 (most negative) to +100 (most positive)</div>"
)

# ── Card A: Minister sentiment based on news ──────────────────────────────────
with _sa:
    sc    = ms["score"]
    color = POS if sc >= 0.05 else (NEG if sc <= -0.05 else NEU)
    disp  = sc * 100
    st.markdown(f"""
    <div class="card" style="border-left:4px solid {color};margin-bottom:10px">
      <div style="font-size:9.5px;font-weight:700;color:#5e5b58;letter-spacing:1.5px;
                  text-transform:uppercase;margin-bottom:10px">Minister — Piyush Goyal</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px">
        <span class="score-big" style="color:{color}">{disp:+.1f}</span>
        <span style="font-size:14px;font-weight:600;color:{color}">{score_label(sc)}</span>
      </div>
      {_SCALE_NOTE}
      {sentiment_bar_html(ms['positive'], ms['neutral'], ms['negative'])}
      <div class="meta-text" style="margin-top:5px">
        {ms['total']} news articles &nbsp;·&nbsp;
        <span style="color:#92400e;font-style:italic">News-based only</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Card B: Minister sentiment based on social media mentions ─────────────────
with _sb:
    if sm_total > 0:
        sm_disp = sm_sc * 100
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {sm_color};margin-bottom:10px">
          <div style="font-size:9.5px;font-weight:700;color:#5e5b58;letter-spacing:1.5px;
                      text-transform:uppercase;margin-bottom:10px">Minister — Social Mentions</div>
          <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px">
            <span class="score-big" style="color:{sm_color}">{sm_disp:+.1f}</span>
            <span style="font-size:14px;font-weight:600;color:{sm_color}">{score_label(sm_sc)}</span>
          </div>
          {_SCALE_NOTE}
          {sentiment_bar_html(sm_sent.get('positive',0), sm_sent.get('neutral',0), sm_sent.get('negative',0))}
          <div class="meta-text" style="margin-top:5px">
            {sm_total} mentions &nbsp;·&nbsp; 🐦 {sm_tw_n} X &nbsp;·&nbsp; 💼 {sm_li_n} LI
            &nbsp;·&nbsp; <span style="color:#92400e;font-style:italic">Third-party only</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card" style="border-left:4px solid #e0ddd7;margin-bottom:10px">
          <div style="font-size:9.5px;font-weight:700;color:#5e5b58;letter-spacing:1.5px;
                      text-transform:uppercase;margin-bottom:10px">Minister — Social Mentions</div>
          <div style="font-size:28px;font-weight:700;color:#d0cdc8;font-family:Georgia,serif;
                      margin-bottom:4px">—</div>
          {_SCALE_NOTE}
          <div class="meta-text" style="margin-top:8px">
            Twitter/X &amp; LinkedIn mentions &nbsp;·&nbsp;
            <span style="color:#92400e;font-style:italic">Third-party only</span>
          </div>
          <div class="meta-text">Open 💬 Mentions tab or regenerate report</div>
        </div>
        """, unsafe_allow_html=True)

# ── Card C: Ministry of Commerce & Industry (news) ────────────────────────────
with _sc_col:
    sc2    = m2["score"]
    color2 = POS if sc2 >= 0.05 else (NEG if sc2 <= -0.05 else NEU)
    disp2  = sc2 * 100
    st.markdown(f"""
    <div class="card" style="border-left:4px solid {color2};margin-bottom:10px">
      <div style="font-size:9.5px;font-weight:700;color:#5e5b58;letter-spacing:1.5px;
                  text-transform:uppercase;margin-bottom:10px">Ministry of Commerce &amp; Industry</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px">
        <span class="score-big" style="color:{color2}">{disp2:+.1f}</span>
        <span style="font-size:14px;font-weight:600;color:{color2}">{score_label(sc2)}</span>
      </div>
      {_SCALE_NOTE}
      {sentiment_bar_html(m2['positive'], m2['neutral'], m2['negative'])}
      <div class="meta-text" style="margin-top:5px">
        {m2['total']} news articles &nbsp;·&nbsp;
        <span style="color:#92400e;font-style:italic">News-based only</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Alert section — negative news + negative mentions ─────────────────────────
all_news   = report["minister"]["news"] + report["ministry"]["news"]
all_social = report["minister"]["tweets"] + report["minister"]["instagram"] + report["minister"]["facebook"]

neg_news = sorted(
    [n for n in all_news if n["sentiment_label"] == "negative"],
    key=lambda x: x["sentiment_score"],
)

# Negative mentions: prefer report data; fall back to cached live fetch.
_report_mentions = report["minister"].get("mentions", [])

neg_mentions = sorted(
    [m for m in _report_mentions if m["sentiment_label"] == "negative"],
    # Primary: highest engagement first; secondary: most negative first
    key=lambda m: (-_total_engagement(m), m.get("sentiment_score", 0)),
)

if neg_news or neg_mentions:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if neg_news and neg_mentions:
        _alert_col, _ment_col = st.columns([6, 5])
    elif neg_news:
        _alert_col, _ment_col = st.columns([1]), None
        _alert_col = _alert_col[0]
    else:
        _alert_col, _ment_col = None, st.columns([1])[0]

    if _alert_col and neg_news:
        with _alert_col:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px">
              <div style="background:#fde8e8;border-radius:10px;width:42px;height:42px;display:flex;
                          align-items:center;justify-content:center;flex-shrink:0;font-size:20px">⚠️</div>
              <div>
                <div style="font-size:16px;font-weight:700;color:#9b1c1c;font-family:'Georgia',serif;letter-spacing:-0.01em">News Items Needing Attention</div>
                <div style="font-size:12px;color:#5c5956;margin-top:2px">Most negative news articles — for media strategy review</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            for item in neg_news[:5]:
                render_news_card(item, show_summary)

    if _ment_col and neg_mentions:
        with _ment_col:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px">
              <div style="background:#fde8e8;border-radius:10px;width:42px;height:42px;display:flex;
                          align-items:center;justify-content:center;flex-shrink:0;font-size:20px">⚠️</div>
              <div>
                <div style="font-size:16px;font-weight:700;color:#9b1c1c;font-family:'Georgia',serif;letter-spacing:-0.01em">Negative Mentions</div>
                <div style="font-size:12px;color:#5c5956;margin-top:2px">Twitter/X &amp; LinkedIn — sorted by engagement</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            for m in neg_mentions[:5]:
                render_mention_card(m)

# ── Top Posts by Engagement ────────────────────────────────────────────────────
top_social = [p for p in sorted(all_social, key=_total_engagement, reverse=True)
              if _total_engagement(p) > 0][:3]

if top_social:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Top Posts by Engagement</div>",
                unsafe_allow_html=True)
    _eng_cols = st.columns(3)
    for i, post in enumerate(top_social):
        with _eng_cols[i]:
            st.markdown(_top_post_card_html(post, rank=i + 1),
                        unsafe_allow_html=True)

# ── Word Cloud ─────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-label'>Topics in Focus — News Headlines &amp; Top Engaged Posts</div>",
    unsafe_allow_html=True,
)
_wc_bytes = _build_wordcloud_cached(report.get("generated_at", ""), report)
if _wc_bytes:
    import base64 as _b64
    _wc_src = _b64.b64encode(_wc_bytes).decode()
    _wc_alt = (
        "Word cloud of the most prominent topics from the last 24 hours of "
        "news headlines and top social media posts. Larger words represent "
        "higher frequency."
    )
    st.markdown(
        f'<img src="data:image/png;base64,{_wc_src}" '
        f'alt="{_wc_alt}" '
        f'style="width:100%;height:auto;border-radius:8px;display:block">',
        unsafe_allow_html=True,
    )
else:
    st.caption("Not enough text to generate a word cloud — run a fresh report.")

# ── Follower counts — below word cloud ────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Official Handles — Follower Count</div>",
            unsafe_allow_html=True)
pills_html = "".join(follower_pill(f) for f in followers)
st.markdown(f"<div style='display:flex;flex-wrap:wrap;gap:6px'>{pills_html}</div>",
            unsafe_allow_html=True)

# Trend toggle
st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
if "show_follower_trend" not in st.session_state:
    st.session_state.show_follower_trend = False
if st.button(
    "📈 Social Media Follower Trends" if not st.session_state.show_follower_trend
    else "▲ Hide Follower Trends",
    key="btn_follower_trend",
):
    st.session_state.show_follower_trend = not st.session_state.show_follower_trend

# ── Follower trend chart ───────────────────────────────────────────────────────
if st.session_state.show_follower_trend:
    try:
        from collectors.follower_tracker import load_history_df
        _hist = load_history_df()
        _live_platforms = [p for p in _hist if len(_hist[p]) >= 2]
        if _live_platforms:
            _fig = go.Figure()
            for _p in _live_platforms:
                _df = _hist[_p].sort_values("ts")
                _pc = PLATFORM_COLORS.get(_p, "#6b7280")
                _fig.add_trace(go.Scatter(
                    x=_df["ts"], y=_df["count"],
                    mode="lines+markers",
                    name=_p,
                    line=dict(color=_pc, width=2),
                    marker=dict(size=5, color=_pc),
                    hovertemplate=(
                        f"<b>{_p}</b><br>"
                        "%{x|%d %b %Y}<br>"
                        "<b>%{y:,.0f}</b> followers"
                        "<extra></extra>"
                    ),
                ))
            _fig.update_layout(
                height=280,
                margin=dict(l=0, r=10, t=10, b=0),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#eeebe5",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11),
                ),
                xaxis=dict(
                    showgrid=True, gridcolor="#f0efec",
                    tickformat="%d %b", tickfont=dict(size=10),
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="#f0efec",
                    tickfont=dict(size=10), tickformat=",.0f",
                ),
                hovermode="x unified",
                font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif"),
            )
            st.plotly_chart(_fig, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.caption("Not enough history yet — run a few daily reports to build the trend.")
    except Exception as _e:
        st.caption(f"Trend chart unavailable: {_e}")

# ── Main tabs ──────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
tab_news, tab_social, tab_mentions, tab_ministry = st.tabs(
    ["📰 News Feed", "📲 Social Media", "💬 Mentions", "🏛 Ministry News"]
)

# ── Tab 1: News feed ──────────────────────────────────────────────────────────
with tab_news:
    minister_news = report["minister"]["news"]
    grouped_sorted = _cached_group_similar(
        report.get("generated_at", "") + ":minister", minister_news
    )

    f1, f2, f3 = st.columns([1, 1, 4])
    with f1:
        sent_filter = st.selectbox("Sentiment", ["All", "Negative", "Positive", "Neutral"], key="nf_sent")
    with f2:
        src_options = ["All"] + sorted({i["source"] for i in minister_news})
        src_filter  = st.selectbox("Source", src_options, key="nf_src")

    if sent_filter != "All":
        grouped_sorted = [i for i in grouped_sorted if i["sentiment_label"] == sent_filter.lower()]
    if src_filter != "All":
        grouped_sorted = [i for i in grouped_sorted if i["source"] == src_filter]

    pos_n = sum(1 for i in grouped_sorted if i["sentiment_label"] == "positive")
    neg_n = sum(1 for i in grouped_sorted if i["sentiment_label"] == "negative")
    st.caption(f"Showing {len(grouped_sorted)} stories ({len(minister_news)} total articles)  ·  🟢 {pos_n}  🔴 {neg_n}")

    if not grouped_sorted:
        st.info("No articles match the current filter.")
    for item in grouped_sorted:
        render_news_card(item, show_summary)

# ── Tab 2: Social media ────────────────────────────────────────────────────────
with tab_social:
    # Sort by total engagement (highest first) throughout this tab
    all_posts = sorted(all_social, key=_total_engagement, reverse=True)

    sub1, sub2, sub3, sub4 = st.tabs(["All", "🐦 Twitter/X", "📸 Instagram", "👤 Facebook"])

    def _render_social_tab(posts, platform=None):
        items = [p for p in posts if not platform or p["platform"] == platform]
        if not items:
            st.info(f"No posts found{f' on {platform}' if platform else ''}.")
            return
        st.caption(f"{len(items)} posts  ·  sorted by engagement  ·  not sentiment-scored")
        for p in items:
            render_social_card(p)

    with sub1: _render_social_tab(all_posts)
    with sub2: _render_social_tab(all_posts, "Twitter/X")
    with sub3: _render_social_tab(all_posts, "Instagram")
    with sub4: _render_social_tab(all_posts, "Facebook")

# ── Tab 3: Mentions ───────────────────────────────────────────────────────────
with tab_mentions:
    # Use mentions stored in the report if available; otherwise fetch live.
    _report_mentions = report["minister"].get("mentions", [])
    _report_mm       = report["minister"].get("mentions_sentiment", {})

    if _report_mentions:
        all_mentions = _report_mentions
        _live_mm     = _report_mm
        _live_tw_n   = sum(1 for m in all_mentions if m["platform"] == "Twitter/X")
        _live_li_n   = sum(1 for m in all_mentions if m["platform"] == "LinkedIn")
        _fetched_live = False
    else:
        # Report pre-dates the mentions feature — fetch live with caching.
        _btn_col, _ = st.columns([2, 5])
        with _btn_col:
            if st.button("🔄 Refresh Mentions", key="btn_refresh_mentions"):
                _fetch_live_mentions.clear()   # bust the 1-hour cache on demand
        with st.spinner("Fetching Twitter/X and LinkedIn mentions..."):
            all_mentions, _live_mm, _live_tw_n, _live_li_n = _fetch_live_mentions()
        _fetched_live = True

    tw_ments = [m for m in all_mentions if m["platform"] == "Twitter/X"]
    li_ments = [m for m in all_mentions if m["platform"] == "LinkedIn"]

    if not all_mentions:
        st.info(
            "No mentions found in the last 48 hours on Twitter/X or LinkedIn.  "
            "This is normal when nitter mirrors are offline (Twitter/X) or when "
            "there are no recent LinkedIn articles indexed by Google.  "
            "Try **Refresh Mentions** above or **Generate Report Now** in the sidebar."
        )
    else:
        if _fetched_live:
            st.caption("Live fetch — not stored in this report. Run **Generate Report Now** to persist.")

        # Sentiment summary card
        if _live_mm.get("total", 0):
            _sc3    = _live_mm["score"]
            _color3 = POS if _sc3 >= 0.05 else (NEG if _sc3 <= -0.05 else NEU)
            st.markdown(f"""
            <div class="card" style="border-left:4px solid {_color3};margin-bottom:16px">
              <div style="font-size:10px;font-weight:700;color:#6d6a66;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px">Public Mentions Sentiment</div>
              <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:2px">
                <span style="font-size:28px;font-weight:700;color:{_color3};font-family:Georgia,serif;letter-spacing:-0.03em">{_sc3:+.3f}</span>
                <span style="font-size:15px;font-weight:600;color:{_color3}">{score_label(_sc3)}</span>
              </div>
              {sentiment_bar_html(_live_mm['positive'], _live_mm['neutral'], _live_mm['negative'])}
              <div class="meta-text">{_live_mm['total']} mentions &nbsp;·&nbsp; 🐦 {_live_tw_n} Twitter/X &nbsp;·&nbsp; 💼 {_live_li_n} LinkedIn</div>
            </div>
            """, unsafe_allow_html=True)

        msub1, msub2, msub3 = st.tabs(["All", "🐦 Twitter/X", "💼 LinkedIn"])

        def _render_mentions(items, label="mentions"):
            if not items:
                st.info(f"No {label} found.")
                return
            neg_c = sum(1 for m in items if m["sentiment_label"] == "negative")
            pos_c = sum(1 for m in items if m["sentiment_label"] == "positive")
            st.caption(
                f"{len(items)} {label}  ·  🔴 {neg_c} negative  🟢 {pos_c} positive  ·  "
                "sorted most negative first"
            )
            for m in sorted(items, key=lambda x: x["sentiment_score"]):
                render_mention_card(m)

        with msub1: _render_mentions(all_mentions, "mentions")
        with msub2: _render_mentions(tw_ments, "Twitter/X mentions")
        with msub3: _render_mentions(li_ments, "LinkedIn mentions")

# ── Tab 4: Ministry news ───────────────────────────────────────────────────────
with tab_ministry:
    ministry_news = report["ministry"]["news"]
    grouped_min = _cached_group_similar(
        report.get("generated_at", "") + ":ministry", ministry_news
    )

    st.markdown(f"""
    <div class="card" style="margin-bottom:20px;border-left:4px solid {color2}">
      <div style="font-size:10px;font-weight:700;color:#6d6a66;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px">Ministry of Commerce & Industry — Sentiment</div>
      <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:2px">
        <span style="font-size:30px;font-weight:700;color:{color2};font-family:'Georgia',serif;letter-spacing:-0.03em">{m2['score']:+.3f}</span>
        <span style="font-size:15px;font-weight:600;color:{color2}">{score_label(m2['score'])}</span>
      </div>
      {sentiment_bar_html(m2['positive'], m2['neutral'], m2['negative'])}
    </div>
    """, unsafe_allow_html=True)

    if not grouped_min:
        st.info("No ministry news in the selected period.")
    for item in grouped_min:
        render_news_card(item, show_summary)

# ── Methodology ────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
with st.expander("📐 Methodology — How Scores Are Calculated"):
    st.markdown(f"""
### Score Scale

All sentiment scores are displayed on a scale of **−100 to +100**, where:

- **+100** = maximally positive
- **0** = perfectly neutral
- **−100** = maximally negative

Internally, VADER produces a compound score in the range −1.0 to +1.0.
The display value is simply that compound score multiplied by 100
(e.g. a raw score of +0.149 appears as **+14.9**).

Items are classified into three labels based on the raw compound score:

| Label | Raw range | Display range |
|---|---|---|
| Positive | ≥ +0.05 | ≥ +5 |
| Neutral | −0.05 to +0.05 | −5 to +5 |
| Negative | ≤ −0.05 | ≤ −5 |

The VADER lexicon has been extended with India-specific political terms
(e.g. *atmanirbhar* +2.0, *scam* −2.5, *viksit* +1.5).

---

### Three Sentiment Scores

The dashboard shows three independent sentiment metrics side by side:

| Score | Source | What it measures |
|---|---|---|
| **Minister — Piyush Goyal** | News articles tagged to the Minister | How news media covers the Minister |
| **Minister — Social Mentions** | Third-party Twitter/X & LinkedIn posts that mention him | Public and professional opinion about the Minister |
| **Ministry of Commerce & Industry** | News articles tagged to the Ministry | How news media covers the Ministry's policy work |

All three use the same formula: **unweighted arithmetic mean** of compound scores
across all items in the respective source set.

```
Score = Σ(compound_score_i) / N × 100
```

---

### Score 1 — Minister, News-Based

**Inputs:** news articles collected from Google News, Economic Times, NDTV,
Business Standard, Hindustan Times, and LiveMint that mention Piyush Goyal.

**Why news only:** Minister's own social media posts reflect his outbound
messaging, not how the public or media perceives him. News coverage by independent
outlets is a more objective external signal.

---

### Score 2 — Minister, Social Media Mentions

**Inputs:** posts by **third-party accounts** that mention Piyush Goyal, collected
from two sources:

- **Twitter/X** — searched via Nitter (open-source Twitter mirror, RSS feed).
  When all Nitter instances are offline, falls back to Google News results for
  `site:x.com "Piyush Goyal"`, which indexes many public tweets.
- **LinkedIn** — searched via Google News results for
  `site:linkedin.com "Piyush Goyal"`, returning LinkedIn Pulse articles and public
  posts that Google has indexed.

The Minister's own Twitter/X, Instagram, and Facebook posts are **excluded** from
this score — only third-party mentions count.

Mentions are scored with standard VADER on the post/article text. No
subject-aware adjustment is applied (the Minister is the subject being discussed,
so negative text genuinely reflects negative public opinion).

---

### Score 3 — Ministry, News-Based

**Inputs:** news articles tagged to the Ministry of Commerce & Industry (DPIIT,
Commerce Ministry, trade policy, exports). Same formula as Score 1.

---

### Subject-Aware Adjustment (News Scores Only)

Raw VADER often misclassifies headlines where the Minister is the **critic**,
not the target of criticism. Example:

> *"Goyal criticises Congress for avoiding Lok Sabha debate"* → raw VADER −0.57

The analyser checks whether the Minister is the grammatical subject performing
a critical action (pattern: `[Minister name] + ≤50 chars + [critical verb]`).
If so, and the Minister is **not** simultaneously a target or in trouble, the
compound score is dampened:

```
adjusted = max(raw × 0.15, −0.04)  →  label = Neutral
```

Applied **only to news articles**; never to social mentions.
Items that are reclassified show a ⚙ marker on their card.

---

### Minister's Own Social Media Posts — Listed, Not Scored

Posts from the Minister's own handles (Twitter/X @PiyushGoyal,
@PiyushGoyalOffc; Instagram; Facebook) are displayed in the **Social Media** tab
for reference and ranked by engagement (likes + comments + shares).
They are **not assigned a sentiment score** and do not affect any of the three
scores above.

**Why not scored:** These posts reflect what the Minister chooses to say, not
how the public reacts. Scoring them would bias the "Minister" metric toward his
own preferred framing.

**Why not scored on comment text:** Fetching comment text requires authenticated
API access that this system does not currently have:
Twitter API v2 (developer account) for replies;
Instagram Graph API for comment text;
Facebook Graph API (Page token) for comment text.

---

### Article Grouping

News articles covering the same story are automatically clustered using
**topic-focused Jaccard similarity**. Before comparison, subject-context words
(*Piyush Goyal*, *minister*, *commerce*, etc.) and standard stop-words are
stripped so that similarity reflects the actual story topic, not shared byline
boilerplate. Two-character words (UK, EU, US) are included since they often
carry the key topic signal. The grouping threshold is 0.38 (38 % word overlap).

Within each group the **most negative article** is shown as the representative;
the others appear under an expandable "+ N more sources" link.

---

### Deduplication

Articles seen in any report in the **last 5 days** are automatically excluded from
subsequent reports. This prevents high-frequency RSS feeds from inflating scores
by re-serving the same story day after day. The history can be cleared from the
sidebar Controls panel.

---

### Word Cloud

The word cloud is built from news headlines, summaries, and the text of the
top-10 most-engaged social posts. Before generating the cloud:

1. HTML tags, URLs, and non-Latin characters are stripped.
2. **NLTK part-of-speech tagging** removes all verb forms (VB, VBD, VBG, VBN,
   VBP, VBZ) so only nouns, adjectives, and proper names remain.
3. A custom stop-list filters subject boilerplate (*Piyush Goyal*, *minister*,
   *ministry*, etc.) and generic low-signal nouns and adjectives.

Top-10 social posts are repeated in the corpus proportionally to their
engagement level (up to 5×) so viral posts carry more weight in the cloud.

---

### Report Refresh Frequency

The background scheduler generates a fresh report every **1 hour**,
collecting the latest 24 hours of news and mentions on each run.
The dashboard automatically loads the most recent saved report on page open.
Click **Generate Report Now** in the sidebar to trigger an immediate refresh.

---

*Sources: {period_str} · VADER v3.3.2 · NLTK POS tagging · Subject-aware regex*
""")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(f"""
<div class="meta-text" style="text-align:center;padding:8px 0;color:#6d6a66">
  Sources: Google News · Economic Times · NDTV · Business Standard · Hindustan Times · LiveMint · Twitter/X · Instagram · Facebook
  &nbsp;·&nbsp; Sentiment: VADER with subject-awareness &nbsp;·&nbsp; Period: {period_str}
</div>
""", unsafe_allow_html=True)

# ── PDF downloads — absolute bottom ───────────────────────────────────────────
try:
    _r_ts_bot = report.get("generated_at", "")
    _pdf_exec_bot, _pdf_full_bot = _generate_pdfs_cached(_r_ts_bot, report)
    _ts_bot = datetime.now(IST).strftime("%Y%m%d_%H%M")
    _b1, _b2, _b3 = st.columns([4, 1, 1])
    with _b2:
        st.download_button(
            label="⬇ Exec Summary",
            data=_pdf_exec_bot,
            file_name=f"executive_summary_{_ts_bot}.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Sentiment scores, word cloud, top engagement, key negative & positive items",
        )
    with _b3:
        st.download_button(
            label="⬇ Full Report",
            data=_pdf_full_bot,
            file_name=f"full_report_{_ts_bot}.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Everything — complete news feed, social posts, mentions, ministry news",
        )
except Exception:
    pass
