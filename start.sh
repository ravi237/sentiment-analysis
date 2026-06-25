#!/bin/bash
cd "$(dirname "$0")"
echo "Starting scheduler..."
python3 scheduler.py &
SCHED_PID=$!
echo "Starting dashboard at http://localhost:8502 ..."
python3 -m streamlit run dashboard.py --server.port 8502
kill $SCHED_PID 2>/dev/null
