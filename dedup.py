"""
Article deduplication.

Keeps a persistent record of article IDs seen across report runs.
Any article seen within the last RETENTION_DAYS is excluded from
subsequent reports so the same story never surfaces twice.

Storage: data/seen_articles.json  →  { article_id: iso_utc_timestamp }
"""
import json
import os
from datetime import datetime, timezone, timedelta

_DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
_SEEN_FILE     = os.path.join(_DATA_DIR, "seen_articles.json")
RETENTION_DAYS = 5


# ── Internal helpers ───────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(iso: str) -> datetime:
    try:
        dt = datetime.fromisoformat(iso)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _cutoff() -> datetime:
    return _now() - timedelta(days=RETENTION_DAYS)


def _load() -> dict:
    try:
        with open(_SEEN_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data: dict):
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_SEEN_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Public API ─────────────────────────────────────────────────────────────────

def filter_seen(items: list) -> tuple:
    """
    Split *items* into (fresh_items, skipped_count).

    An item is considered "seen" if its ``id`` field appears in the
    seen-file with a timestamp within the last RETENTION_DAYS days.
    Items without an ``id`` are always treated as fresh.
    """
    seen   = _load()
    cutoff = _cutoff()
    fresh, skipped = [], 0
    for item in items:
        aid = item.get("id", "")
        if aid and aid in seen and _parse(seen[aid]) >= cutoff:
            skipped += 1
        else:
            fresh.append(item)
    return fresh, skipped


def mark_seen(items: list) -> int:
    """
    Record *items* as seen at the current UTC time.
    Prunes entries older than RETENTION_DAYS before saving.
    Returns the total number of tracked IDs after the update.
    """
    seen   = _load()
    cutoff = _cutoff()

    # Drop stale entries
    seen = {k: v for k, v in seen.items() if _parse(v) >= cutoff}

    now = _now().isoformat()
    for item in items:
        aid = item.get("id", "")
        if aid and aid not in seen:
            seen[aid] = now

    _save(seen)
    return len(seen)


def tracked_count() -> int:
    """Number of article IDs currently in the seen-file."""
    seen   = _load()
    cutoff = _cutoff()
    return sum(1 for v in seen.values() if _parse(v) >= cutoff)


def clear():
    """Wipe the entire seen-file (use from the dashboard's 'clear history' button)."""
    _save({})
