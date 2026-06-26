"""
Background scheduler — generates a fresh report every hour so the
dashboard always shows near-real-time data.

Run alongside the dashboard:
    python scheduler.py
"""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import pytz
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

IST = pytz.timezone("Asia/Kolkata")

REFRESH_HOURS = 6   # how often to regenerate the report


def run_report():
    print(f"[scheduler] Triggered at {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}")
    from report_generator import generate_report
    try:
        report = generate_report(hours=24)
        print(f"[scheduler] Done — {report['stats']['total_news']} news, "
              f"{report['stats']['total_tweets']} tweets, "
              f"{report['stats'].get('total_tw_mentions',0)} X mentions, "
              f"{report['stats'].get('total_li_mentions',0)} LI mentions")
    except Exception as e:
        print(f"[scheduler] ERROR: {e}")


if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone=IST)
    scheduler.add_job(
        run_report,
        IntervalTrigger(hours=REFRESH_HOURS),
        id="realtime_report",
        name=f"Sentiment Report every {REFRESH_HOURS}h",
        replace_existing=True,
    )
    print(f"[scheduler] Started. Refreshing every {REFRESH_HOURS} hour(s).")
    print("[scheduler] Running initial report now…")
    run_report()          # generate immediately on startup
    print("[scheduler] Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("[scheduler] Stopped.")
