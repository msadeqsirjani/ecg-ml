#!/bin/bash

# Set up logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/preprocessing_$(date +%Y%m%d_%H%M%S).log"

# Function for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function for error handling
handle_error() {
    log "ERROR: $1"
    exit 1
}

# Run preprocessing script
log "Starting ECG preprocessing..."
python3 src/preprocess_ecg.py 2>&1 | tee -a "$LOG_FILE" || handle_error "Preprocessing failed"

# Verify output
log "Verifying preprocessing output..."
if [ ! -d "data/processed/ptb-xl-1.0.3" ] || [ ! "$(ls -A data/processed/ptb-xl-1.0.3)" ]; then
    handle_error "Preprocessing output directory is empty"
fi

# Check for each classification type
for classification in binary super sub; do
    if [ ! -d "data/processed/ptb-xl-1.0.3/$classification" ]; then
        handle_error "Missing processed data for $classification classification"
    fi
done

log "Preprocessing completed successfully!"
log "Processed data is available in data/processed/ptb-xl-1.0.3"
log "Log file: $LOG_FILE"