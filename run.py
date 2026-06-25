"""
Convenience launcher — starts both the dashboard and the background scheduler
in separate processes.

Usage:
    python run.py
"""
import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    python = sys.executable
    streamlit_bin = os.path.join(os.path.dirname(python), "streamlit")
    if not os.path.exists(streamlit_bin):
        # Try user bin path (pip install --user)
        streamlit_bin = os.path.expanduser("~/Library/Python/3.9/bin/streamlit")

    print("Starting Sentiment Analysis Dashboard...")
    print("Dashboard → http://localhost:8502")
    print("Press Ctrl+C to stop both processes.\n")

    dashboard_proc = subprocess.Popen(
        [streamlit_bin, "run", os.path.join(BASE, "dashboard.py"),
         "--server.port", "8502", "--server.headless", "true"],
        cwd=BASE,
    )
    scheduler_proc = subprocess.Popen(
        [python, os.path.join(BASE, "scheduler.py")],
        cwd=BASE,
    )

    try:
        dashboard_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        dashboard_proc.terminate()
        scheduler_proc.terminate()


if __name__ == "__main__":
    main()
