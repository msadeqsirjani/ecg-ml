#!/bin/bash

# Script to run ECG model training

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Main execution
main() {
    log "Starting ECG model training process..."
    
    # Create timestamp for this run
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/training_${TIMESTAMP}.log"
    
    # Run the training script
    log "Running training script..."
    log "Log file: $LOG_FILE"
    
    if python src/train_ecg.py 2>&1 | tee -a "$LOG_FILE"; then
        log "Training completed successfully!"
        log "Results saved in:"
        log "- Models: ./models/"
        log "- Logs: $LOG_FILE"
        log "- Results: ./results/"
    else
        error "Training failed! Check the log file for details: $LOG_FILE"
        exit 1
    fi
}

# Execute main function
main 