#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
START_TOL=25
END_TOL=26
LOG_DIR="logs"
MODE="parallel"  # Set to "parallel" to run all at once, "sequential" to run one by one

# ==========================================
# SCRIPT START
# ==========================================

# 1. Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Starting Batch Run: Tolerance $START_TOL to $END_TOL"
echo "Mode: $MODE"
echo "========================================"

for ((i=START_TOL; i<=END_TOL; i++))
do
    TIMESTAMP=$(date +%Y%m%d_%H)
    LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}_tol${i}.log"
    
    echo "[$(date +%H:%M:%S)] Launching optimization for Tolerance: $i"
    echo "   └─ Logging to: $LOG_FILE"

    if [ "$MODE" = "parallel" ]; then
        # Run in background (&) and immediately move to next loop
        nohup python3 main.py -t $i > "$LOG_FILE" 2>&1 &
    else
        # Run in foreground and WAIT for it to finish before moving to next
        python3 main.py -t $i > "$LOG_FILE" 2>&1
    fi
done

if [ "$MODE" = "parallel" ]; then
    echo "========================================"
    echo "All jobs launched in background."
    echo "Use command 'htop' to monitor progress."
else
    echo "========================================"
    echo "All jobs completed successfully."
fi