"""
Fetch international news coverage of Minister Piyush Goyal's visits abroad.

For a given country and date range, queries Google News (country-locale) for:
  - Articles mentioning "Piyush Goyal"
  - Articles about India–{country} trade / FTA

Results are filtered to the top 10 publications of that country and
scored for sentiment using the same pipeline as the daily feed.
"""
import concurrent.futures
import feedparser
import hashlib
import requests
import urllib.parse
from datetime import datetime, timezone
from dateutil import parser as dateparser
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sentiment.analyzer import score

MINISTER_NAME = "Piyush Goyal"

# Name / title variants used in per-publication site: queries
MINISTER_QUERIES = [
    "Piyush Goyal",
    "India's Commerce Minister",
    "India Commerce Minister",
    "Indian Commerce Minister",
    "Commerce Minister Goyal",
]

# ── Country configuration ──────────────────────────────────────────────────
# Each entry: GL = ISO country code, HL = locale, CEID = Google News ceid,
# trade_terms = additional search phrases, publications = (display name, domain)
COUNTRY_CONFIG = {
    "United Kingdom": {
        "gl": "GB", "hl": "en-GB", "ceid": "GB:en",
        "query_name": "UK",
        "trade_terms": [
            '"India UK trade deal"',
            '"India-UK trade deal"',
            '"UK-India trade deal"',
            '"UK-India trade"',
            '"India UK FTA"',
            '"UK India FTA"',
            '"India UK free trade agreement"',
            '"UK India free trade"',
            '"India UK trade"',
            '"UK India trade"',
            '"India Britain trade deal"',
            '"India Britain FTA"',
            '"India UK bilateral"',
            '"UK India CETA"',
            '"India UK CETA"',
        ],
        "publications": [
            ("Financial Times",  "ft.com"),
            ("The Guardian",     "theguardian.com"),
            ("The Telegraph",    "telegraph.co.uk"),
            ("BBC",              "bbc.co.uk"),
            ("BBC",              "bbc.com"),
            ("The Economist",    "economist.com"),
            ("Reuters",          "reuters.com"),
            ("Bloomberg",        "bloomberg.com"),
            ("The Times",        "thetimes.co.uk"),
            ("The Independent",  "independent.co.uk"),
            ("Sky News",         "news.sky.com"),
            ("Politico Europe",  "politico.eu"),
        ],
    },
    "United States": {
        "gl": "US", "hl": "en-US", "ceid": "US:en",
        "query_name": "US",
        "trade_terms": [
            '"India US trade deal"',
            '"India-US trade deal"',
            '"India USA trade deal"',
            '"India US FTA"',
            '"India America trade deal"',
            '"US India trade deal"',
            '"India US bilateral trade"',
            '"India US trade agreement"',
        ],
        "publications": [
            ("The New York Times",    "nytimes.com"),
            ("The Washington Post",   "washingtonpost.com"),
            ("Wall Street Journal",   "wsj.com"),
            ("Bloomberg",             "bloomberg.com"),
            ("Reuters",               "reuters.com"),
            ("CNBC",                  "cnbc.com"),
            ("Forbes",                "forbes.com"),
            ("Financial Times",       "ft.com"),
            ("CNN",                   "cnn.com"),
            ("Associated Press",      "apnews.com"),
        ],
    },
    "European Union": {
        "gl": "DE", "hl": "en-DE", "ceid": "DE:en",
        "query_name": "EU",
        "trade_terms": [
            '"India EU trade deal"',
            '"India-EU trade deal"',
            '"India EU FTA"',
            '"India Europe FTA"',
            '"India European Union trade"',
            '"EU India trade deal"',
            '"India EU free trade agreement"',
            '"India EU bilateral trade"',
        ],
        "publications": [
            ("Financial Times",   "ft.com"),
            ("Reuters",           "reuters.com"),
            ("Bloomberg",         "bloomberg.com"),
            ("Deutsche Welle",    "dw.com"),
            ("Euronews",          "euronews.com"),
            ("Politico Europe",   "politico.eu"),
            ("Der Spiegel",       "spiegel.de"),
            ("The Guardian",      "theguardian.com"),
            ("BBC",               "bbc.com"),
            ("Le Monde",          "lemonde.fr"),
        ],
    },
    "Australia": {
        "gl": "AU", "hl": "en-AU", "ceid": "AU:en",
        "query_name": "Australia",
        "trade_terms": [
            '"India Australia trade deal"',
            '"India-Australia trade deal"',
            '"India Australia FTA"',
            '"India Australia ECTA"',
            '"Australia India trade deal"',
            '"India Australia free trade"',
            '"India Australia bilateral trade"',
        ],
        "publications": [
            ("Sydney Morning Herald",     "smh.com.au"),
            ("The Australian",            "theaustralian.com.au"),
            ("ABC News",                  "abc.net.au"),
            ("Australian Financial Review","afr.com"),
            ("Reuters",                   "reuters.com"),
            ("The Age",                   "theage.com.au"),
            ("Guardian Australia",        "theguardian.com"),
            ("Bloomberg",                 "bloomberg.com"),
            ("SBS News",                  "sbs.com.au"),
            ("Herald Sun",                "heraldsun.com.au"),
        ],
    },
    "Canada": {
        "gl": "CA", "hl": "en-CA", "ceid": "CA:en",
        "query_name": "Canada",
        "trade_terms": [
            '"India Canada trade deal"',
            '"India-Canada trade deal"',
            '"India Canada FTA"',
            '"India Canada CEPA"',
            '"Canada India trade deal"',
            '"India Canada free trade"',
            '"India Canada bilateral trade"',
        ],
        "publications": [
            ("The Globe and Mail",  "theglobeandmail.com"),
            ("Toronto Star",        "thestar.com"),
            ("National Post",       "nationalpost.com"),
            ("CBC News",            "cbc.ca"),
            ("CTV News",            "ctvnews.ca"),
            ("Bloomberg",           "bloomberg.com"),
            ("Reuters",             "reuters.com"),
            ("Financial Post",      "financialpost.com"),
            ("Ottawa Citizen",      "ottawacitizen.com"),
            ("Vancouver Sun",       "vancouversun.com"),
        ],
    },
    "Japan": {
        "gl": "JP", "hl": "en-JP", "ceid": "JP:en",
        "query_name": "Japan",
        "trade_terms": [
            '"India Japan trade deal"',
            '"India-Japan trade deal"',
            '"India Japan FTA"',
            '"India Japan CEPA"',
            '"India Japan economic partnership"',
            '"Japan India trade deal"',
            '"India Japan bilateral trade"',
        ],
        "publications": [
            ("Japan Times",     "japantimes.co.jp"),
            ("Nikkei Asia",     "asia.nikkei.com"),
            ("NHK World",       "nhk.or.jp"),
            ("Kyodo News",      "kyodonews.net"),
            ("Reuters",         "reuters.com"),
            ("Bloomberg",       "bloomberg.com"),
            ("Financial Times", "ft.com"),
            ("The Mainichi",    "mainichi.jp"),
            ("Yomiuri Shimbun", "yomiuri.co.jp"),
            ("Asahi Shimbun",   "asahi.com"),
        ],
    },
    "Singapore": {
        "gl": "SG", "hl": "en-SG", "ceid": "SG:en",
        "query_name": "Singapore",
        "trade_terms": [
            '"India Singapore trade deal"',
            '"India-Singapore trade deal"',
            '"India Singapore FTA"',
            '"India Singapore CECA"',
            '"Singapore India trade deal"',
            '"India Singapore free trade"',
            '"India Singapore bilateral trade"',
        ],
        "publications": [
            ("The Straits Times",       "straitstimes.com"),
            ("Channel NewsAsia",        "channelnewsasia.com"),
            ("The Business Times",      "businesstimes.com.sg"),
            ("TODAY",                   "todayonline.com"),
            ("Reuters",                 "reuters.com"),
            ("Bloomberg",               "bloomberg.com"),
            ("Financial Times",         "ft.com"),
            ("The Edge Singapore",      "theedgesingapore.com"),
            ("Yahoo Finance SG",        "sg.finance.yahoo.com"),
            ("The Independent SG",      "theindependent.sg"),
        ],
    },
    "UAE": {
        "gl": "AE", "hl": "en-AE", "ceid": "AE:en",
        "query_name": "UAE",
        "trade_terms": [
            '"India UAE trade deal"',
            '"India-UAE trade deal"',
            '"India UAE CEPA"',
            '"India UAE FTA"',
            '"UAE India trade deal"',
            '"India UAE free trade"',
            '"India Emirates trade deal"',
            '"India UAE bilateral trade"',
        ],
        "publications": [
            ("Khaleej Times",    "khaleejtimes.com"),
            ("Gulf News",        "gulfnews.com"),
            ("The National",     "thenationalnews.com"),
            ("Arabian Business", "arabianbusiness.com"),
            ("ZAWYA",            "zawya.com"),
            ("Reuters",          "reuters.com"),
            ("Bloomberg",        "bloomberg.com"),
            ("Al Arabiya",       "alarabiya.net"),
            ("Arab News",        "arabnews.com"),
            ("Gulf Business",    "gulfbusiness.com"),
        ],
    },
    "Germany": {
        "gl": "DE", "hl": "de", "ceid": "DE:de",
        "query_name": "Germany",
        "trade_terms": [
            '"India Germany trade deal"',
            '"India-Germany trade deal"',
            '"India Germany FTA"',
            '"Germany India trade deal"',
            '"India Deutschland trade"',
            '"Indien Deutschland Handel"',
            '"India Germany bilateral trade"',
        ],
        "publications": [
            ("Reuters",          "reuters.com"),
            ("Bloomberg",        "bloomberg.com"),
            ("Financial Times",  "ft.com"),
            ("Deutsche Welle",   "dw.com"),
            ("Der Spiegel",      "spiegel.de"),
            ("Handelsblatt",     "handelsblatt.com"),
            ("FAZ",              "faz.net"),
            ("Die Zeit",         "zeit.de"),
            ("Süddeutsche",      "sueddeutsche.de"),
            ("The Guardian",     "theguardian.com"),
        ],
    },
    "France": {
        "gl": "FR", "hl": "fr", "ceid": "FR:fr",
        "query_name": "France",
        "trade_terms": [
            '"India France trade deal"',
            '"India-France trade deal"',
            '"India France FTA"',
            '"France India trade deal"',
            '"Inde France commerce"',
            '"India France bilateral trade"',
            '"India France agreement"',
        ],
        "publications": [
            ("Reuters",         "reuters.com"),
            ("Bloomberg",       "bloomberg.com"),
            ("Financial Times", "ft.com"),
            ("Le Monde",        "lemonde.fr"),
            ("France 24",       "france24.com"),
            ("AFP",             "afp.com"),
            ("Les Echos",       "lesechos.fr"),
            ("Politico Europe", "politico.eu"),
            ("Euronews",        "euronews.com"),
            ("The Guardian",    "theguardian.com"),
        ],
    },
    "Greece": {
        "gl": "GR", "hl": "en-GR", "ceid": "GR:en",
        "query_name": "Greece",
        "trade_terms": [
            '"India Greece trade"',
            '"India-Greece trade"',
            '"India Greece bilateral"',
            '"India Greece economic"',
            '"India Greece deal"',
            '"India Greece cooperation"',
            '"Greece India trade"',
        ],
        "publications": [
            ("Ekathimerini",      "ekathimerini.com"),
            ("Reuters",           "reuters.com"),
            ("Bloomberg",         "bloomberg.com"),
            ("Greek Reporter",    "greekreporter.com"),
            ("Athens News Agency","amna.gr"),
            ("Keep Talking Greece","keeptalkingreece.com"),
            ("Proto Thema",       "protothema.gr"),
            ("Financial Times",   "ft.com"),
            ("Euronews",          "euronews.com"),
            ("Naftemporiki",      "naftemporiki.gr"),
        ],
    },
}


def list_countries() -> list[str]:
    return sorted(COUNTRY_CONFIG.keys())


def get_publications(country: str) -> list[tuple[str, str]]:
    """Return [(display_name, domain), ...] for a country."""
    cfg = COUNTRY_CONFIG.get(country, {})
    seen_names: set[str] = set()
    out = []
    for name, domain in cfg.get("publications", []):
        if name not in seen_names:
            seen_names.add(name)
            out.append((name, domain))
    return out


# ── RSS / Google News fetching ─────────────────────────────────────────────

def _gnews_url(query: str, gl: str, hl: str, ceid: str) -> str:
    return (
        "https://news.google.com/rss/search?"
        + urllib.parse.urlencode({"q": query, "hl": hl, "gl": gl, "ceid": ceid})
    )


_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = (
    "Mozilla/5.0 (compatible; NewsBot/1.0)"
)

def _fetch_feed(url: str, timeout: int = 8):
    """Fetch and parse a Google News RSS URL with a hard timeout."""
    try:
        resp = _SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
        return feedparser.parse(resp.text)
    except Exception:
        return feedparser.FeedParserDict(entries=[])


def _parse_dt(entry) -> datetime:
    for attr in ("published", "updated", "created"):
        raw = getattr(entry, attr, None) or entry.get(attr)
        if raw:
            try:
                dt = dateparser.parse(str(raw))
                if dt:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def _domain_of(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
        return host.lstrip("www.")
    except Exception:
        return ""


def _matches_country_pubs(entry, pub_domains: set[str]) -> bool:
    """True if the article's source domain is one of the target publications."""
    # Google News puts the real publisher in entry.source.href
    src_url  = getattr(entry, "source", {}).get("href", "")
    art_url  = entry.get("link", "")
    for url in (src_url, art_url):
        d = _domain_of(url)
        if any(d == pd or d.endswith("." + pd) for pd in pub_domains):
            return True
    return False


def _make_article(entry, pub_name: str) -> dict:
    raw_title = entry.get("title", "").strip()
    # Strip " - Publisher Name" suffix that Google News appends
    src_struct = getattr(entry, "source", {})
    actual_src = src_struct.get("title", "").strip() or pub_name
    if actual_src and raw_title.endswith(" - " + actual_src):
        clean_title = raw_title[: -(len(actual_src) + 3)].strip()
    else:
        clean_title = raw_title

    summary   = entry.get("summary", "").strip()
    url       = entry.get("link", "")
    published = _parse_dt(entry)
    text_for_score = clean_title + ". " + summary[:300]
    sent = score(text_for_score, apply_subject_adjustment=True)

    return {
        "id": hashlib.md5((clean_title + url).encode()).hexdigest()[:12],
        "title": clean_title,
        "summary": summary[:400] if summary else "",
        "source": actual_src,
        "url": url,
        "published": published.isoformat(),
        "published_display": published.strftime("%d %b %Y, %I:%M %p UTC"),
        "sentiment_score": sent["compound"],
        "sentiment_label": sent["label"],
    }


def fetch_visit_coverage(
    country: str,
    start_date: datetime,
    end_date: datetime,
) -> list[dict]:
    """
    Fetch news articles about Piyush Goyal / India trade from the top 10
    publications of `country`, published between start_date and end_date.

    Returns a deduplicated list of article dicts, sorted by published date (newest first).
    """
    cfg = COUNTRY_CONFIG.get(country)
    if not cfg:
        return []

    gl, hl, ceid = cfg["gl"], cfg["hl"], cfg["ceid"]
    qname = cfg.get("query_name", country)
    pub_domains: set[str] = {domain for _, domain in cfg["publications"]}

    # Ensure UTC-aware datetimes for comparison
    start_utc = start_date.replace(tzinfo=timezone.utc) if start_date.tzinfo is None else start_date.astimezone(timezone.utc)
    end_utc   = end_date.replace(tzinfo=timezone.utc)   if end_date.tzinfo is None   else end_date.astimezone(timezone.utc)

    # "when:Nd" window hint so Google News restricts the feed to the right window
    days_ago  = (datetime.now(timezone.utc) - start_utc).days
    when_hint = f" when:{min(max(days_ago + 1, 1), 365)}d"

    # ── Build query list ───────────────────────────────────────────────────
    # Strategy 1: 2 targeted site: queries per publication (Google News
    # defaults to Indian sources for India-related topics, so we bypass that
    # by querying each publication domain directly).
    pub_queries: list[tuple[str, str, str, str, str]] = []  # (url, pub_name)
    for pub_name, domain in cfg["publications"]:
        q1 = f'"Piyush Goyal" site:{domain}{when_hint}'
        q2 = f'"India {qname} trade" site:{domain}{when_hint}'
        pub_queries.append((_gnews_url(q1, "US", "en-US", "US:en"), pub_name))
        pub_queries.append((_gnews_url(q2, "US", "en-US", "US:en"), pub_name))

    # Strategy 2: a handful of broad country-locale queries as a catch-all
    CORE_MINISTER_QUERIES = ["Piyush Goyal", "India's Commerce Minister"]
    CORE_TRADE_TERMS = cfg["trade_terms"][:4]  # top 4 trade variants
    general_queries: list[tuple[str, str]] = []
    for mq in CORE_MINISTER_QUERIES:
        general_queries.append((_gnews_url(mq + when_hint, gl, hl, ceid), ""))
    for term in CORE_TRADE_TERMS:
        general_queries.append((_gnews_url(term + when_hint, gl, hl, ceid), ""))

    all_queries = pub_queries + general_queries

    seen_ids: set[str] = set()
    articles: list[dict] = []

    # Country aliases for relevance matching (e.g. UK → uk, britain, united kingdom)
    _COUNTRY_ALIASES = {
        "United Kingdom": ["uk", "britain", "british", "united kingdom"],
        "United States": ["us", "usa", "america", "american", "united states"],
        "European Union": ["eu", "europe", "european union", "eurozone"],
        "Australia": ["australia", "australian"],
        "Canada": ["canada", "canadian"],
        "Japan": ["japan", "japanese"],
        "Singapore": ["singapore", "singaporean"],
        "UAE": ["uae", "emirates", "dubai", "abu dhabi"],
        "Germany": ["germany", "german", "deutschland"],
        "France": ["france", "french"],
        "Greece": ["greece", "greek", "athens", "hellenic"],
    }
    country_aliases = _COUNTRY_ALIASES.get(country, [qname.lower(), country.lower()])

    def _is_relevant(entry) -> bool:
        text = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
        mentions_goyal = "goyal" in text or "piyush" in text
        mentions_india = "india" in text or "indian" in text
        mentions_country_here = any(alias in text for alias in country_aliases)
        mentions_trade = any(kw in text for kw in [
            "trade", "fta", "free trade", "tariff", "export", "import",
            "commerce", "deal", "agreement", "bilateral",
        ])
        # Article must: mention Goyal directly, OR mention India + this country + trade
        return mentions_goyal or (mentions_india and mentions_country_here and mentions_trade)

    def _process_entry(entry, fallback_pub_name: str, require_domain_match: bool):
        pub_dt = _parse_dt(entry)
        if not (start_utc <= pub_dt <= end_utc):
            return

        # Always require relevance keywords (guards against site: operator being ignored)
        if not _is_relevant(entry):
            return

        # For general queries, still require the article to be from a target pub
        if require_domain_match and not _matches_country_pubs(entry, pub_domains):
            return

        # Determine publication name
        src_url  = getattr(entry, "source", {}).get("href", "")
        art_url  = entry.get("link", "")
        pub_name = fallback_pub_name
        if not pub_name or require_domain_match:
            for name, domain in cfg["publications"]:
                for test_url in (src_url, art_url):
                    d = _domain_of(test_url)
                    if d == domain or d.endswith("." + domain):
                        pub_name = name
                        break
                if pub_name:
                    break

        # For per-pub queries, still verify the article is from a target domain
        if not require_domain_match and not _matches_country_pubs(entry, pub_domains):
            return

        art = _make_article(entry, pub_name)
        if art["id"] not in seen_ids and art["title"]:
            seen_ids.add(art["id"])
            articles.append(art)

    # Fire all queries concurrently (8-second per-request timeout)
    def _fetch_and_process(url_pub: tuple):
        url, pub_name = url_pub
        is_general = pub_name == ""
        feed = _fetch_feed(url)
        return [(entry, pub_name, is_general) for entry in feed.entries]

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(_fetch_and_process, q) for q in all_queries]
        for fut in concurrent.futures.as_completed(futures):
            try:
                for entry, pub_name, is_general in fut.result():
                    _process_entry(entry, pub_name, require_domain_match=is_general)
            except Exception:
                pass

    # Sort newest first
    articles.sort(key=lambda a: a["published"], reverse=True)
    return articles
