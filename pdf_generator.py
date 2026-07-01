"""
Generate PDF versions of the daily media brief.

Two variants:
  generate_executive_summary(report) -> bytes
  generate_full_report(report)       -> bytes
  generate_pdf(report)               -> bytes  (alias for full report)
"""
from fpdf import FPDF
from datetime import timezone
import io
import re

# ── Colour palette (R, G, B) ───────────────────────────────────────────────
C_ACCENT  = (30,  58,  95)
C_POS     = (26,  127, 55)
C_NEG     = (185, 28,  28)
C_NEU     = (120, 113, 108)
C_BORDER  = (220, 218, 212)
C_TEXT    = (26,  25,  23)
C_MUTED   = (160, 162, 158)
C_SUB     = (92,  89,  86)
C_WHITE   = (255, 255, 255)
C_AMBER   = (146, 64,  14)

PAGE_W = 210
MARGIN = 15
BODY_W = PAGE_W - 2 * MARGIN   # 180 mm

# Three equal sentiment cards
CARD3_W   = (BODY_W - 6) / 3   # ~58 mm each, 3 mm gap
CARD3_H   = 32                  # mm


# ── Text helpers ───────────────────────────────────────────────────────────
_UNICODE_MAP = str.maketrans({
    "—": "-", "–": "-",
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "…": "...", "·": ".",
    "•": "*", "▲": "^", "▼": "v", "●": "*",
})


def _safe(text) -> str:
    s = str(text or "").translate(_UNICODE_MAP)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def _score_label(s: float) -> str:
    if s >= 0.3:
        return "Strongly Positive"
    if s >= 0.05:
        return "Positive"
    if s <= -0.3:
        return "Strongly Negative"
    if s <= -0.05:
        return "Negative"
    return "Neutral"


def _sent_color(s: float):
    if s >= 0.05:
        return C_POS
    if s <= -0.05:
        return C_NEG
    return C_NEU


def _sign100(s: float) -> str:
    """Format a raw VADER score as ×100 display value."""
    v = s * 100
    return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"


def _sign2(s: float) -> str:
    return f"+{s:.2f}" if s >= 0 else f"{s:.2f}"


# ── Engagement helper ──────────────────────────────────────────────────────
def _total_eng(post: dict) -> int:
    return (
        (post.get("likes", 0) or 0)
        + (post.get("comments", 0) or 0)
        + (post.get("retweets") or post.get("shares") or 0)
    )


def _fmt_eng(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _fmt_count(n) -> str:
    if n is None:
        return "-"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


# ── Word-cloud generator (mirrors dashboard logic, no Streamlit import) ────
def _strip_html_for_wc(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"&[#a-zA-Z0-9]+;", " ", text)
    text = re.sub(r'\b\w*="[^"]*"', " ", text)
    text = re.sub(r"[^a-zA-Z\s'\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _make_wordcloud_bytes(report: dict):
    try:
        from wordcloud import WordCloud, STOPWORDS
    except ImportError:
        return None

    news   = report["minister"]["news"] + report["ministry"]["news"]
    social = (report["minister"]["tweets"]
              + report["minister"]["instagram"]
              + report["minister"]["facebook"])
    top_s  = sorted(social, key=_total_eng, reverse=True)[:10]

    parts = []
    for item in news:
        parts.append(_strip_html_for_wc(item.get("title", "")))
        parts.append(_strip_html_for_wc(item.get("summary") or ""))
    for post in top_s:
        text   = _strip_html_for_wc(post.get("text", ""))
        weight = 1 + min(_total_eng(post) // 3000, 4)
        parts.extend([text] * weight)

    corpus = " ".join(p for p in parts if p.strip())
    if len(corpus.split()) < 10:
        return None

    stop_words = set(STOPWORDS)
    stop_words.update({
        "piyush","goyal","minister","union","commerce","ministry","india","indian",
        "government","bjp","amazing","global","strong","stronger","key","major",
        "important","significant","great","good","better","best","high","higher",
        "big","large","broad","robust","dynamic","strategic","comprehensive",
        "bilateral","mutual","critical","crucial","vital","landmark","historic",
        "pact","post","progress","growth","move","step","way","place","part","role",
        "work","focus","level","number","area","side","effort","efforts","goal",
        "goals","plan","plans","result","results","impact","outcome","development",
        "opportunity","opportunities","issue","issues","sector","sectors","market",
        "markets","need","needs","availability","situation","condition","conditions",
        "context","position","approach","process","framework","stage","phase","point",
        "basis","course","view","terms","term","form","forms","aspect","aspects",
        "statement","statements","report","reports","session","conference","summit",
        "event","programme","program","initiative","initiatives","measure","measures",
        "action","actions","response","responses","new","one","two","three","four",
        "five","year","years","today","day","week","time","mr","ms","per","rs",
        "lakh","crore","cent","news","article","recently","soon","early","late",
        "next","last","first","decade","decades","still","yet","already","always",
        "ever","never","just","even","well","back","around","across","among",
        "between","within","without","whether","while","though","although","however",
        "therefore","thus","s","t","u","re","ve","ll","href","src","alt","class",
    })

    # Remove verbs with NLTK POS tagging
    try:
        import nltk
        try:
            nltk.pos_tag(["test"])
        except LookupError:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        words  = re.findall(r"[a-zA-Z']{2,}", corpus)
        tagged = nltk.pos_tag(words)
        freq: dict = {}
        for w, tag in tagged:
            lw = w.lower()
            if not tag.startswith("VB") and lw not in stop_words and len(lw) >= 2:
                freq[lw] = freq.get(lw, 0) + 1
        if freq:
            wc = WordCloud(width=1400, height=460, background_color="white",
                           colormap="Blues", max_words=120, collocations=False,
                           min_font_size=11, max_font_size=130,
                           prefer_horizontal=0.80).generate_from_frequencies(freq)
        else:
            raise ValueError("empty freq")
    except Exception:
        wc = WordCloud(width=1400, height=460, background_color="white",
                       colormap="Blues", stopwords=stop_words, max_words=120,
                       collocations=False).generate(corpus)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ── PDF class ──────────────────────────────────────────────────────────────
class BriefPDF(FPDF):
    def header(self):
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.8)
        self.line(MARGIN, 9, PAGE_W - MARGIN, 9)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*C_MUTED)
        self.cell(0, 5, f"Piyush Goyal  |  Daily Media Brief  |  Page {self.page_no()}",
                  align="C")

    def section_label(self, text: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*C_TEXT)
        self.cell(0, 5, _safe(text), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.3)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)

    def sentiment_bar(self, pos: int, neu: int, neg: int, width: float = BODY_W):
        total = pos + neu + neg or 1
        y, x, h = self.get_y(), self.get_x(), 3
        for color, w in [(C_POS, pos/total*width),
                         (C_NEU, neu/total*width),
                         (C_NEG, neg/total*width)]:
            self.set_fill_color(*color)
            if w > 0:
                self.rect(x, y, w, h, style="F")
            x += w
        self.ln(h + 2)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*C_POS)
        self.cell(55, 3.5, f"{pos} positive  ({pos/total*100:.0f}%)")
        self.set_text_color(*C_NEU)
        self.cell(55, 3.5, f"{neu} neutral  ({neu/total*100:.0f}%)")
        self.set_text_color(*C_NEG)
        self.cell(55, 3.5, f"{neg} negative  ({neg/total*100:.0f}%)")
        self.ln(5)

    def sentiment_card(self, x: float, y: float, width: float,
                       title: str, sent: dict, note: str = ""):
        """Draw a single sentiment card (×100 scale)."""
        sc    = sent.get("score", 0.0)
        color = _sent_color(sc)
        total = sent.get("total", 0)

        # Left accent bar
        self.set_fill_color(*color)
        self.rect(x, y, 2, CARD3_H, style="F")

        # Card background
        self.set_fill_color(*C_WHITE)
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.2)
        self.rect(x + 2, y, width - 2, CARD3_H, style="FD")

        # Title
        self.set_xy(x + 4, y + 2.5)
        self.set_font("Helvetica", "B", 6)
        self.set_text_color(*C_MUTED)
        self.cell(width - 6, 3, _safe(title).upper())

        # Big score (×100)
        self.set_xy(x + 4, y + 7)
        self.set_font("Times", "B", 17)
        self.set_text_color(*color)
        self.cell(22, 7, _sign100(sc))
        self.set_font("Helvetica", "B", 7.5)
        self.cell(width - 27, 7, _score_label(sc))

        # Scale note
        self.set_xy(x + 4, y + 15)
        self.set_font("Helvetica", "", 5.5)
        self.set_text_color(*C_MUTED)
        self.cell(width - 8, 3, "Scale: -100 to +100")

        # Mini bar
        bar_x, bar_y, bar_w = x + 4, y + 19, width - 8
        pos_c = sent.get("positive", 0)
        neu_c = sent.get("neutral", 0)
        neg_c = sent.get("negative", 0)
        t     = pos_c + neu_c + neg_c or 1
        bx    = bar_x
        for col, cnt in [(C_POS, pos_c), (C_NEU, neu_c), (C_NEG, neg_c)]:
            w = cnt / t * bar_w
            if w > 0:
                self.set_fill_color(*col)
                self.rect(bx, bar_y, w, 2.5, style="F")
            bx += w

        # Note (source / third-party marker)
        self.set_xy(x + 4, bar_y + 4)
        self.set_font("Helvetica", "I", 5.5)
        self.set_text_color(*C_AMBER)
        self.cell(width - 8, 3, _safe(note) if note else f"{total} items")

    def follower_table(self, followers: list):
        """Compact horizontal follower count table."""
        # Header
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*C_MUTED)
        self.cell(70, 4, "PLATFORM")
        self.cell(40, 4, "FOLLOWERS", align="R")
        self.cell(35, 4, "TODAY", align="R")
        self.cell(35, 4, "LIVE", align="C")
        self.ln(4)
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.2)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(2)

        for f in followers:
            d = f.get("delta_daily")
            if d is not None:
                sign = "+" if d >= 0 else ""
                n = abs(d)
                dt = f"{n/1e6:.1f}M" if n>=1e6 else f"{n/1e3:.1f}K" if n>=1000 else str(n)
                delta_str = f"{sign}{dt}"
                d_col = C_POS if d >= 0 else C_NEG
            else:
                delta_str, d_col = "-", C_MUTED

            self.set_font("Helvetica", "B", 7.5)
            self.set_text_color(*C_TEXT)
            self.cell(70, 5, _safe(f.get("platform", "")))
            self.cell(40, 5, _safe(f.get("followers_display", "-")), align="R")
            self.set_text_color(*d_col)
            self.set_font("Helvetica", "B", 7.5)
            self.cell(35, 5, _safe(delta_str), align="R")
            self.set_text_color(C_POS[0], C_POS[1], C_POS[2] if f.get("is_live")
                                else C_MUTED[0])
            self.set_font("Helvetica", "", 7)
            self.cell(35, 5, "Live" if f.get("is_live") else "Fallback", align="C")
            self.ln(5)
        self.ln(2)

    def news_item(self, item: dict, is_grouped_child: bool = False):
        sc      = item.get("sentiment_score", 0.0)
        color   = _sent_color(sc)
        indent  = 6 if is_grouped_child else 0
        tw      = BODY_W - indent - 32
        x_base  = MARGIN + indent
        self.set_x(x_base)
        title = _safe(item.get("title", ""))
        if len(title) > 100:
            title = title[:98] + "..."
        self.set_font("Helvetica", "" if is_grouped_child else "B",
                      7 if is_grouped_child else 8)
        self.set_text_color(*C_TEXT)
        self.cell(tw, 4.5, title, new_x="RIGHT")
        # Badge shows ×100 score
        self.set_font("Helvetica", "B", 6.5)
        self.set_text_color(*color)
        self.cell(32, 4.5,
                  f"{_sign100(sc)}  {item.get('sentiment_label','').capitalize()}",
                  align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_x(x_base)
        self.set_font("Helvetica", "I", 6.5)
        self.set_text_color(*C_SUB)
        meta = f"{_safe(item.get('source',''))}  -  {_safe(item.get('published_display',''))}"
        if item.get("adjustment"):
            meta += "  - reclassified"
        self.cell(0, 3.5, meta, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def social_item(self, post: dict):
        """Social post card — engagement metrics, no sentiment badge."""
        likes = post.get("likes", 0) or 0
        cmt   = post.get("comments", 0) or 0
        rt    = post.get("retweets") or post.get("shares") or 0
        total = likes + cmt + rt
        self.set_x(MARGIN)
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*C_TEXT)
        hdr = (f"{_safe(post.get('platform',''))}  "
               f"{_safe(post.get('handle',''))}  -  "
               f"{_safe(post.get('published_display',''))}")
        self.cell(BODY_W - 28, 4.5, hdr, new_x="RIGHT")
        self.set_font("Helvetica", "B", 6.5)
        self.set_text_color(*C_ACCENT)
        self.cell(28, 4.5, f"{_fmt_eng(total)} eng",
                  align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_x(MARGIN)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*C_SUB)
        self.multi_cell(BODY_W, 3.5,
                        _safe(post.get("text", ""))[:220],
                        new_x="LMARGIN", new_y="NEXT")
        self.set_x(MARGIN)
        self.set_font("Helvetica", "I", 6.5)
        self.set_text_color(*C_MUTED)
        self.cell(0, 3.5,
                  f"Likes {likes:,}   Comments {cmt:,}   Shares {rt:,}",
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def mention_item(self, m: dict):
        """Single mention card with sentiment badge."""
        sc     = m.get("sentiment_score", 0.0)
        color  = _sent_color(sc)
        plat   = _safe(m.get("platform", ""))
        handle = _safe(m.get("handle", ""))
        date   = _safe(m.get("published_display", ""))
        text   = _safe(m.get("text", ""))[:200]
        hdr    = f"{plat}  {handle}  -  {date}" if handle else f"{plat}  -  {date}"
        self.set_x(MARGIN)
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*C_TEXT)
        self.cell(BODY_W - 32, 4.5, hdr, new_x="RIGHT")
        self.set_font("Helvetica", "B", 6.5)
        self.set_text_color(*color)
        self.cell(32, 4.5,
                  f"{_sign100(sc)}  {m.get('sentiment_label','').capitalize()}",
                  align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_x(MARGIN)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*C_SUB)
        self.cell(0, 4, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def engagement_card(self, post: dict, rank: int):
        likes = post.get("likes", 0) or 0
        cmt   = post.get("comments", 0) or 0
        rt    = post.get("retweets") or post.get("shares") or 0
        total = likes + cmt + rt
        card_h, x, y = 28, MARGIN, self.get_y()
        self.set_fill_color(*C_WHITE)
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.3)
        self.rect(x, y, BODY_W, card_h, style="FD")
        self.set_fill_color(*C_ACCENT)
        self.rect(x, y, 2, card_h, style="F")
        self.set_xy(x + 5, y + 2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*C_ACCENT)
        self.cell(12, 5, f"#{rank}")
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*C_SUB)
        hdr = (f"{_safe(post.get('platform',''))}  "
               f"{_safe(post.get('handle',''))}  -  "
               f"{_safe(post.get('published_display',''))}")
        self.cell(0, 5, hdr, new_x="LMARGIN", new_y="NEXT")
        self.set_xy(x + 5, y + 9)
        self.set_font("Times", "B", 14)
        self.set_text_color(*C_TEXT)
        self.cell(26, 6, f"{_fmt_eng(total)}")
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*C_MUTED)
        self.cell(0, 6,
                  f" total engagement  (Likes {likes:,}  Comments {cmt:,}  Shares {rt:,})",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_xy(x + 5, y + 17)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*C_SUB)
        self.cell(BODY_W - 10, 4,
                  _safe(post.get("text", ""))[:160], new_x="LMARGIN", new_y="NEXT")
        self.set_y(y + card_h + 2)


# ── Section helpers ────────────────────────────────────────────────────────

def _parse_meta(report: dict):
    """Return (date_str, period_str, gen_str) in IST."""
    try:
        from dateutil import parser as dp
        from datetime import timedelta
        import pytz
        IST    = pytz.timezone("Asia/Kolkata")
        gen_dt = dp.parse(report.get("generated_at", ""))
        if gen_dt.tzinfo is None:
            gen_dt = gen_dt.replace(tzinfo=timezone.utc)
        gen_ist    = gen_dt.astimezone(IST)
        report_9am = gen_ist.replace(hour=9, minute=0, second=0, microsecond=0)
        start_9am  = report_9am - timedelta(hours=report.get("period_hours", 24))
        date_str   = _safe(gen_ist.strftime("%d %B %Y"))
        period_str = _safe(
            f"{start_9am.strftime('%d %b, %I:%M %p')} - "
            f"{gen_ist.strftime('%d %b, %I:%M %p')} IST"
        )
        gen_str = _safe(gen_ist.strftime("%d %b %Y, %I:%M %p IST"))
        return date_str, period_str, gen_str
    except Exception:
        return "-", "-", "-"


def _resolve_mentions(report: dict):
    """
    Return (sm_sent, tw_n, li_n, all_mentions) using only data stored in the report.
    No live network calls — PDFs are a snapshot of the report data.
    """
    sm    = report["minister"].get("mentions_sentiment", {})
    ments = report["minister"].get("mentions", [])
    tw_n  = sum(1 for m in ments if m["platform"] == "Twitter/X")
    li_n  = sum(1 for m in ments if m["platform"] == "LinkedIn")
    return sm, tw_n, li_n, ments


def _sec_masthead(pdf: BriefPDF, report: dict, subtitle: str):
    date_str, period_str, gen_str = _parse_meta(report)
    stats = report.get("stats", {})

    pdf.ln(4)
    pdf.set_font("Times", "B", 22)
    pdf.set_text_color(*C_TEXT)
    pdf.cell(0, 9, "Piyush Goyal", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    pdf.set_text_color(*C_SUB)
    pdf.cell(0, 6, f"Daily Media Brief  |  {subtitle}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(0, 5,
             f"{date_str}   |   Generated: {gen_str}   |   Coverage: {period_str}",
             new_x="LMARGIN", new_y="NEXT")

    # stat counts
    parts = [
        f"{stats.get('total_news',0)} news",
        f"{stats.get('total_tweets',0)} tweets",
        f"{stats.get('total_instagram',0)} IG",
        f"{stats.get('total_facebook',0)} FB",
    ]
    if stats.get("total_tw_mentions"):
        parts.append(f"{stats['total_tw_mentions']} Twitter mentions")
    if stats.get("total_li_mentions"):
        parts.append(f"{stats['total_li_mentions']} LinkedIn mentions")
    pdf.set_font("Helvetica", "", 7)
    pdf.cell(0, 4, "  |  ".join(parts), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_draw_color(*C_BORDER)
    pdf.set_line_width(0.3)
    pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
    pdf.ln(5)


def _sec_three_sentiment_cards(pdf: BriefPDF, report: dict,
                               sm_sent: dict = None,
                               sm_tw_n: int = 0,
                               sm_li_n: int = 0):
    """Draw three sentiment cards side-by-side (Minister news | Mentions | Ministry)."""
    pdf.section_label("Sentiment Overview  (Scale: -100 to +100)")

    ms     = report["minister"]["sentiment"]
    m2     = report["ministry"]["sentiment"]
    sm     = sm_sent or {}
    card_y = pdf.get_y()
    gap    = 3

    # Card A: Minister (news-based)
    pdf.sentiment_card(
        MARGIN, card_y, CARD3_W,
        "Minister - Piyush Goyal",
        ms, "Based on news articles only"
    )

    # Card B: Minister (social mentions)
    if sm.get("total", 0) > 0:
        pdf.sentiment_card(
            MARGIN + CARD3_W + gap, card_y, CARD3_W,
            "Minister - Social Mentions",
            sm, f"Twitter/X ({sm_tw_n}) + LinkedIn ({sm_li_n}), third-party"
        )
    else:
        pdf.set_fill_color(*C_WHITE)
        pdf.set_draw_color(*C_BORDER)
        pdf.rect(MARGIN + CARD3_W + gap, card_y, CARD3_W, CARD3_H, style="FD")
        pdf.set_xy(MARGIN + CARD3_W + gap + 4, card_y + 2.5)
        pdf.set_font("Helvetica", "B", 6)
        pdf.set_text_color(*C_MUTED)
        pdf.cell(CARD3_W - 6, 3, "MINISTER - SOCIAL MENTIONS")
        pdf.set_xy(MARGIN + CARD3_W + gap + 4, card_y + 10)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(CARD3_W - 8, 5, "No mention data available")

    # Card C: Ministry (news-based)
    pdf.sentiment_card(
        MARGIN + 2*(CARD3_W + gap), card_y, CARD3_W,
        "Ministry of Commerce & Industry",
        m2, "Based on news articles only"
    )
    pdf.set_y(card_y + CARD3_H + 6)


def _sec_follower_counts(pdf: BriefPDF, report: dict):
    followers = report.get("followers", [])
    if not followers:
        return
    pdf.section_label("Official Handles - Follower Count")
    pdf.follower_table(followers)


def _sec_wordcloud(pdf: BriefPDF, report: dict):
    wc_bytes = _make_wordcloud_bytes(report)
    if not wc_bytes:
        return
    pdf.section_label("Topics in Focus - Word Cloud")
    try:
        pdf.image(io.BytesIO(wc_bytes), x=MARGIN, w=BODY_W)
        pdf.ln(4)
    except Exception:
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*C_MUTED)
        pdf.cell(0, 4, "(Word cloud unavailable)", new_x="LMARGIN", new_y="NEXT")


def _sec_negative_news(pdf: BriefPDF, report: dict, limit: int = 8):
    all_news = report["minister"]["news"] + report["ministry"]["news"]
    neg = sorted([n for n in all_news if n["sentiment_label"] == "negative"],
                 key=lambda x: x["sentiment_score"])
    if not neg:
        return
    pdf.section_label(f"News Items Needing Attention  ({len(neg)} negative articles)")
    for item in neg[:limit]:
        pdf.news_item(item)


def _sec_negative_mentions(pdf: BriefPDF, all_mentions: list, limit: int = 8):
    """Render negative-sentiment mentions. Accepts a pre-resolved mentions list."""
    neg = sorted([m for m in all_mentions if m["sentiment_label"] == "negative"],
                 key=lambda x: x["sentiment_score"])
    if not neg:
        return
    pdf.section_label(
        f"Negative Social Media Mentions - Twitter/X & LinkedIn  ({len(neg)} items)"
    )
    for m in neg[:limit]:
        pdf.mention_item(m)


def _sec_top_positive_news(pdf: BriefPDF, report: dict, limit: int = 5):
    all_news = report["minister"]["news"] + report["ministry"]["news"]
    pos = sorted([n for n in all_news if n["sentiment_label"] == "positive"],
                 key=lambda x: -x["sentiment_score"])
    if not pos:
        return
    pdf.section_label(f"Top {min(limit, len(pos))} Positive News")
    for item in pos[:limit]:
        pdf.news_item(item)


def _sec_full_news_feed(pdf: BriefPDF, report: dict):
    from collectors.news_collector import group_similar
    min_news = report["minister"]["news"]
    grouped  = sorted(group_similar(min_news), key=lambda x: x["sentiment_score"])
    pdf.section_label(f"Full News Feed - Minister  ({len(min_news)} articles)")
    for item in grouped:
        pdf.news_item(item)
        for child in item.get("group_items", []):
            pdf.news_item(child, is_grouped_child=True)


def _sec_top_engagement(pdf: BriefPDF, report: dict):
    all_social = (report["minister"]["tweets"]
                  + report["minister"]["instagram"]
                  + report["minister"]["facebook"])
    top = sorted(all_social, key=_total_eng, reverse=True)
    top = [p for p in top if _total_eng(p) > 0][:5]
    if not top:
        return
    pdf.section_label(f"Top Posts by Engagement  (top {len(top)} of {len(all_social)})")
    for i, post in enumerate(top, 1):
        pdf.engagement_card(post, i)


def _sec_social_media(pdf: BriefPDF, report: dict):
    all_social = (report["minister"]["tweets"]
                  + report["minister"]["instagram"]
                  + report["minister"]["facebook"])
    if not all_social:
        return
    posts = sorted(all_social, key=_total_eng, reverse=True)
    pdf.section_label(f"All Social Media Posts  ({len(all_social)} posts, sorted by engagement)")
    for post in posts[:25]:
        pdf.social_item(post)


def _sec_mentions(pdf: BriefPDF, report: dict):
    mentions = report["minister"].get("mentions", [])
    if not mentions:
        return
    tw_n = sum(1 for m in mentions if m["platform"] == "Twitter/X")
    li_n = sum(1 for m in mentions if m["platform"] == "LinkedIn")
    pdf.section_label(
        f"Public Mentions - Twitter/X & LinkedIn  "
        f"({tw_n} tweets  |  {li_n} LinkedIn)"
    )
    sm = report["minister"].get("mentions_sentiment", {})
    if sm.get("total", 0):
        sc3    = sm["score"]
        color3 = _sent_color(sc3)
        pdf.set_font("Times", "B", 13)
        pdf.set_text_color(*color3)
        pdf.cell(0, 6, f"{_sign100(sc3)}   {_score_label(sc3)}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.sentiment_bar(sm.get("positive", 0),
                          sm.get("neutral",  0),
                          sm.get("negative", 0))
        pdf.ln(2)
    for m in sorted(mentions, key=lambda x: x["sentiment_score"])[:40]:
        pdf.mention_item(m)


def _sec_ministry_news(pdf: BriefPDF, report: dict):
    from collectors.news_collector import group_similar
    mnt_news = report["ministry"]["news"]
    if not mnt_news:
        return
    m2     = report["ministry"]["sentiment"]
    color2 = _sent_color(m2["score"])
    pdf.section_label(f"Ministry of Commerce & Industry  ({len(mnt_news)} articles)")
    pdf.set_font("Times", "B", 13)
    pdf.set_text_color(*color2)
    pdf.cell(0, 6,
             f"{_sign100(m2['score'])}   {_score_label(m2['score'])}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.sentiment_bar(m2["positive"], m2["neutral"], m2["negative"])
    pdf.ln(2)
    grouped = sorted(group_similar(mnt_news), key=lambda x: x["sentiment_score"])
    for item in grouped:
        pdf.news_item(item)
        for child in item.get("group_items", []):
            pdf.news_item(child, is_grouped_child=True)


# ── Entry points ───────────────────────────────────────────────────────────

def generate_executive_summary(report: dict) -> bytes:
    """
    Compact executive summary:
      - 3 sentiment scores (with live mentions fallback)
      - Word cloud
      - Top social media posts by engagement
      - Negative news (news-based)
      - Negative social media mentions (Twitter/X & LinkedIn)
      - Top 5 positive news
    """
    sm_sent, sm_tw_n, sm_li_n, all_mentions = _resolve_mentions(report)

    pdf = BriefPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    _sec_masthead(pdf, report, "Executive Summary")
    _sec_three_sentiment_cards(pdf, report, sm_sent, sm_tw_n, sm_li_n)
    _sec_wordcloud(pdf, report)

    pdf.add_page()
    _sec_negative_news(pdf, report, limit=5)
    _sec_negative_mentions(pdf, all_mentions, limit=5)
    _sec_top_positive_news(pdf, report, limit=5)
    _sec_top_engagement(pdf, report)   # top 5 posts, at the end

    return bytes(pdf.output())


def generate_full_report(report: dict) -> bytes:
    """
    Complete report mirroring all dashboard sections:
      - Executive summary content (with live mentions fallback)
      - Follower counts
      - Full news feed
      - Top engagement posts
      - All social media posts
      - All mentions
      - Ministry news
    """
    sm_sent, sm_tw_n, sm_li_n, all_mentions = _resolve_mentions(report)

    pdf = BriefPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    _sec_masthead(pdf, report, "Full Report")
    _sec_three_sentiment_cards(pdf, report, sm_sent, sm_tw_n, sm_li_n)
    _sec_follower_counts(pdf, report)
    _sec_wordcloud(pdf, report)
    _sec_negative_news(pdf, report, limit=8)
    _sec_negative_mentions(pdf, all_mentions, limit=8)
    _sec_top_positive_news(pdf, report, limit=5)

    pdf.add_page()
    _sec_full_news_feed(pdf, report)

    if report["minister"].get("tweets") or report["minister"].get("instagram"):
        pdf.add_page()
        _sec_social_media(pdf, report)

    if all_mentions:
        pdf.add_page()
        _sec_mentions(pdf, report)

    if report["ministry"]["news"]:
        pdf.add_page()
        _sec_ministry_news(pdf, report)

    _sec_top_engagement(pdf, report)   # top 5 posts, at the very end

    return bytes(pdf.output())


# backward-compatible alias
def generate_pdf(report: dict) -> bytes:
    return generate_full_report(report)


# ── Visit Coverage PDF ─────────────────────────────────────────────────────

class _VisitPDF(FPDF):
    def __init__(self, country: str, date_from: str, date_to: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._country   = country
        self._date_from = date_from
        self._date_to   = date_to

    def header(self):
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.8)
        self.line(MARGIN, 9, PAGE_W - MARGIN, 9)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*C_MUTED)
        self.cell(
            0, 5,
            f"Piyush Goyal  |  {_safe(self._country)} Visit Coverage  |  Page {self.page_no()}",
            align="C",
        )

    def _masthead(self, n_articles: int, pub_names: list[str]):
        self.ln(4)
        self.set_font("Times", "B", 22)
        self.set_text_color(*C_TEXT)
        self.cell(0, 9, "Piyush Goyal", new_x="LMARGIN", new_y="NEXT")

        self.set_font("Times", "", 13)
        self.set_text_color(*C_SUB)
        self.cell(
            0, 6,
            _safe(f"{self._country} - International Press Coverage"),
            new_x="LMARGIN", new_y="NEXT",
        )

        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*C_MUTED)
        self.cell(
            0, 5,
            f"Period: {_safe(self._date_from)}  to  {_safe(self._date_to)}   |   "
            f"{n_articles} articles retrieved",
            new_x="LMARGIN", new_y="NEXT",
        )

        # Scope note
        self.ln(1)
        self.set_font("Helvetica", "I", 7)
        self.cell(
            0, 4,
            "Covers: Minister Piyush Goyal  |  India "
            + _safe(self._country)
            + " trade deal / FTA",
            new_x="LMARGIN", new_y="NEXT",
        )

        # Publication pills
        if pub_names:
            self.ln(2)
            self.set_font("Helvetica", "B", 6)
            self.set_text_color(*C_MUTED)
            self.cell(0, 4, "SOURCES MONITORED:", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "", 6.5)
            self.set_text_color(*C_SUB)
            self.multi_cell(
                BODY_W, 4,
                "  |  ".join(_safe(p) for p in pub_names),
                new_x="LMARGIN", new_y="NEXT",
            )

        self.ln(3)
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.3)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(5)

    def _sentiment_summary(self, articles: list):
        if not articles:
            return
        pos = sum(1 for a in articles if a["sentiment_label"] == "positive")
        neg = sum(1 for a in articles if a["sentiment_label"] == "negative")
        neu = sum(1 for a in articles if a["sentiment_label"] == "neutral")
        total = len(articles)
        avg   = sum(a["sentiment_score"] for a in articles) / total
        color = _sent_color(avg)

        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*C_MUTED)
        self.cell(0, 4, "SENTIMENT OVERVIEW", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

        self.set_font("Times", "B", 18)
        self.set_text_color(*color)
        self.cell(28, 7, f"{avg*100:+.1f}")
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 7, _score_label(avg), new_x="LMARGIN", new_y="NEXT")

        # Stacked bar
        bar_w = BODY_W
        t = total or 1
        bx, by, bh = MARGIN, self.get_y(), 3
        for col, cnt in [(C_POS, pos), (C_NEU, neu), (C_NEG, neg)]:
            w = cnt / t * bar_w
            if w > 0:
                self.set_fill_color(*col)
                self.rect(bx, by, w, bh, style="F")
            bx += w
        self.ln(bh + 2)

        self.set_font("Helvetica", "", 7)
        self.set_text_color(*C_POS)
        self.cell(55, 3.5, f"{pos} positive  ({pos/t*100:.0f}%)")
        self.set_text_color(*C_NEU)
        self.cell(55, 3.5, f"{neu} neutral  ({neu/t*100:.0f}%)")
        self.set_text_color(*C_NEG)
        self.cell(55, 3.5, f"{neg} negative  ({neg/t*100:.0f}%)")
        self.ln(8)

    def _pub_section(self, pub_name: str, articles: list):
        """Render a single publication's articles with a sub-header."""
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*C_ACCENT)
        self.cell(0, 5, _safe(pub_name), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.2)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(2)

        for art in articles:
            sc    = art.get("sentiment_score", 0.0)
            color = _sent_color(sc)
            title = _safe(art.get("title", ""))
            if len(title) > 110:
                title = title[:108] + "..."

            self.set_x(MARGIN)
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*C_TEXT)
            self.cell(BODY_W - 32, 4.5, title, new_x="RIGHT")
            self.set_font("Helvetica", "B", 6.5)
            self.set_text_color(*color)
            self.cell(
                32, 4.5,
                f"{sc*100:+.1f}  {art.get('sentiment_label','').capitalize()}",
                align="R", new_x="LMARGIN", new_y="NEXT",
            )

            self.set_x(MARGIN)
            self.set_font("Helvetica", "I", 6.5)
            self.set_text_color(*C_SUB)
            self.cell(
                0, 3.5,
                _safe(art.get("published_display", "")),
                new_x="LMARGIN", new_y="NEXT",
            )

            summary = _safe(art.get("summary", ""))
            if summary:
                # Strip HTML tags before rendering
                summary = re.sub(r"<[^>]+>", " ", summary)
                summary = re.sub(r"\s+", " ", summary).strip()[:300]
                self.set_x(MARGIN)
                self.set_font("Helvetica", "", 7)
                self.set_text_color(*C_TEXT)
                self.multi_cell(
                    BODY_W, 3.5, summary,
                    new_x="LMARGIN", new_y="NEXT",
                )

            self.ln(2)

        self.ln(2)


def generate_visit_coverage_pdf(
    country: str,
    date_from: str,
    date_to: str,
    articles: list,
) -> bytes:
    """
    Generate a PDF of international press coverage for a ministerial visit.

    Parameters
    ----------
    country   : Display name of the country visited (e.g. "United Kingdom")
    date_from : Display string for start of coverage window
    date_to   : Display string for end of coverage window
    articles  : List of article dicts from visit_coverage_collector.fetch_visit_coverage()
    """
    from collectors.visit_coverage_collector import get_publications

    pub_map: dict[str, list] = {}
    for art in articles:
        pub_map.setdefault(art["source"], []).append(art)

    pub_names = [name for name, _ in get_publications(country)]

    pdf = _VisitPDF(country, date_from, date_to)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf._masthead(len(articles), pub_names)

    if not articles:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*C_MUTED)
        pdf.cell(
            0, 8,
            "No articles found for the selected country and date range. "
            "Try broadening the date window or check your internet connection.",
            new_x="LMARGIN", new_y="NEXT",
        )
        return bytes(pdf.output())

    pdf._sentiment_summary(articles)

    # Group by publication in priority order
    seen_pubs: set[str] = set()
    ordered: list[tuple[str, list]] = []
    for name, _ in get_publications(country):
        for src_name, arts in pub_map.items():
            if src_name not in seen_pubs and (
                name.lower() in src_name.lower()
                or src_name.lower() in name.lower()
            ):
                seen_pubs.add(src_name)
                ordered.append((src_name, arts))
                break

    # Append any remaining sources not matched to a named publication
    for src_name, arts in pub_map.items():
        if src_name not in seen_pubs:
            ordered.append((src_name, arts))

    for pub_name, pub_articles in ordered:
        # Sort each pub's articles newest first
        pub_articles_sorted = sorted(
            pub_articles, key=lambda a: a["published"], reverse=True
        )
        pdf._pub_section(pub_name, pub_articles_sorted)

    return bytes(pdf.output())
