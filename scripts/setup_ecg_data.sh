#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if a command succeeded
check_error() {
    if [ $? -ne 0 ]; then
        print_error "$1"
        exit 1
    fi
}

# Check if data/raw directory already exists
if [ -d "data/raw" ]; then
    print_warning "Data/raw directory already exists. Skipping directory creation."
else
    print_status "Creating data/raw directories..."
    mkdir -p data/raw
    check_error "Failed to create directories"
fi

# Check if PTBXL dataset is already downloaded and extracted
if [ -d "data/raw/ptb-xl-1.0.3" ]; then
    print_warning "PTBXL dataset already exists. Skipping download and extraction."
else
    print_status "Downloading PTBXL database..."
    cd data/raw
    wget -q https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
    check_error "Failed to download PTBXL dataset"

    print_status "Extracting PTBXL dataset..."
    unzip -q ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
    check_error "Failed to extract PTBXL dataset"

    print_status "Organizing PTBXL files..."
    mv ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 ptb-xl-1.0.3
    check_error "Failed to organize PTBXL files"
    cd ../..
fi

print_success "All datasets have been successfully downloaded and processed!"