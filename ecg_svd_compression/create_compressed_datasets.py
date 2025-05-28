#!/usr/bin/env python3
"""
Create compressed ECG datasets using SVD compression.
This script creates 3 new datasets containing 25%, 50%, and 75% of the original signal
using SVD compression, maintaining the PTB-XL folder structure.
"""

import os
import wfdb
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.signal import correlate
from tqdm import tqdm
import shutil
from pathlib import Path

# Configuration
ORIGINAL_DATA_DIR = '/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/categorized'
OUTPUT_BASE_DIR = '/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/compressed'

# Compression levels (percentage of original signal to retain)
# Note: These ranks are based on the number of training signals, not signal length
COMPRESSION_LEVELS = {
    '25': {'rank': 100, 'name': '25_percent'},   # Use 10% of available components
    '50': {'rank': 50, 'name': '50_percent'},    # Use 5% of available components  
    '75': {'rank': 25, 'name': '75_percent'}     # Use 2.5% of available components
}

def align_signals(signals):
    """Align signals using cross-correlation"""
    if signals.shape[1] == 0:
        return signals
    
    ref = signals[:, 0]  # Use first signal as reference
    aligned = []
    for s in signals.T:
        corr = correlate(ref, s, mode='same')
        shift = np.argmax(corr) - len(ref)//2
        aligned.append(np.roll(s, -shift))
    return np.column_stack(aligned)

def load_signals_for_training(data_dir, num_records, sampling_rate=500, superclass='NORM'):
    """Load signals from all numbered directories for a superclass"""
    signals = []
    record_prefix = 'hr' if sampling_rate == 500 else 'lr'
    base_path = os.path.join(data_dir, f"records{sampling_rate}")
    
    if not os.path.exists(base_path):
        print(f"Warning: {base_path} does not exist")
        return None, None
    
    # Get all numbered directories (00000-21000)
    numbered_dirs = sorted([d for d in os.listdir(base_path)
                          if len(d) == 5 and d.isdigit()])
    
    print(f"Loading {num_records} {superclass} records from {len(numbered_dirs)} directories...")
    
    # First pass: determine the most common signal length
    signal_lengths = []
    temp_signals = []
    
    for dir_num in numbered_dirs:
        superclass_path = os.path.join(base_path, dir_num, 'super', superclass)
        if not os.path.exists(superclass_path):
            continue
        
        record_files = [f for f in os.listdir(superclass_path)
                      if f.endswith('.hea')][:min(50, num_records)]  # Sample first 50 to determine length
        
        for record_file in record_files:
            record_id = record_file.split('_')[0]
            record_path = os.path.join(superclass_path, f"{record_id}_{record_prefix}")
            try:
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal[:, 1]  # Lead II
                signal_lengths.append(len(signal))
                temp_signals.append((signal, record_path))
                if len(temp_signals) >= 50:
                    break
            except Exception as e:
                continue
        if len(temp_signals) >= 50:
            break
    
    if not signal_lengths:
        print(f"No valid signals found for {superclass}")
        return None, None
    
    # Use the most common signal length
    target_length = max(set(signal_lengths), key=signal_lengths.count)
    print(f"Target signal length for {superclass}: {target_length} samples")
    
    # Second pass: load signals with standardized length
    for dir_num in tqdm(numbered_dirs, desc=f"Loading {superclass} signals"):
        superclass_path = os.path.join(base_path, dir_num, 'super', superclass)
        if not os.path.exists(superclass_path):
            continue
        
        # Get available records in this directory
        record_files = [f for f in os.listdir(superclass_path)
                      if f.endswith('.hea')][:num_records - len(signals)]
        
        for record_file in record_files:
            record_id = record_file.split('_')[0]
            record_path = os.path.join(superclass_path, f"{record_id}_{record_prefix}")
            try:
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal[:, 1]  # Lead II
                
                # Standardize signal length
                if len(signal) >= target_length:
                    signal = signal[:target_length]  # Truncate if longer
                else:
                    # Pad with zeros if shorter
                    signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
                
                signals.append(signal / np.max(np.abs(signal)))  # Normalized
                if len(signals) >= num_records:
                    break
            except Exception as e:
                print(f"Skipping {record_path}: {str(e)}")
                continue
        if len(signals) >= num_records:
            break
    
    if not signals:
        return None, None
    
    X = np.column_stack(signals)
    return align_signals(X), target_length

def train_svd_model(X, rank=50):
    """Train SVD model with regularization"""
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    
    # Ensure rank doesn't exceed the number of available components
    max_rank = min(X.shape[0], X.shape[1])
    effective_rank = min(rank, max_rank)
    
    if effective_rank != rank:
        print(f"Warning: Requested rank {rank} reduced to {effective_rank} due to data constraints")
    
    U_r = U[:, :effective_rank]
    Q, R, pivots = qr(U_r.T, pivoting=True)
    C = np.zeros((effective_rank, X.shape[0]))
    C[np.arange(effective_rank), pivots[:effective_rank]] = 1
    
    return U_r, C, pivots[:effective_rank]

def reconstruct_signal(test_signal, U_r, C, target_length):
    """Reconstruct signal with regularization"""
    # Standardize test signal length to match training data
    original_length = len(test_signal)
    if len(test_signal) >= target_length:
        standardized_signal = test_signal[:target_length]  # Truncate if longer
    else:
        # Pad with zeros if shorter
        standardized_signal = np.pad(test_signal, (0, target_length - len(test_signal)), mode='constant')
    
    # Reconstruct using SVD
    y = C @ standardized_signal
    a = np.linalg.lstsq(C @ U_r + 1e-6*np.eye(U_r.shape[1]), y, rcond=None)[0]
    reconstructed = U_r @ a
    
    # Restore original signal length
    if original_length >= target_length:
        return reconstructed  # Already correct length
    else:
        return reconstructed[:original_length]  # Truncate to original length

def create_output_directory_structure(output_dir):
    """Create the output directory structure mirroring PTB-XL"""
    # Create records100 and records500 directories
    for rate in [100, 500]:
        records_dir = os.path.join(output_dir, f"records{rate}")
        os.makedirs(records_dir, exist_ok=True)
        
        # Create numbered directories (00000-21000)
        for dir_num in range(22):
            folder_name = str(dir_num * 1000).zfill(5)
            target_dir = os.path.join(records_dir, folder_name)
            os.makedirs(target_dir, exist_ok=True)

def compress_and_save_records(data_dir, output_dir, compression_config):
    """Compress and save all records using SVD for both 100Hz and 500Hz"""
    rank = compression_config['rank']
    
    # Create output directory structure for both sampling rates
    create_output_directory_structure(output_dir)
    
    # Process each sampling rate
    for sampling_rate in [100, 500]:
        record_prefix = 'lr' if sampling_rate == 100 else 'hr'
        print(f"\nProcessing {sampling_rate}Hz records")
        
        # Get all superclasses
        base_path = os.path.join(data_dir, f"records{sampling_rate}")
        if not os.path.exists(base_path):
            print(f"Warning: {base_path} does not exist")
            continue
        
        superclasses = set()
        numbered_dirs = [d for d in os.listdir(base_path) if len(d) == 5 and d.isdigit()]
        
        for dir_num in numbered_dirs:
            super_path = os.path.join(base_path, dir_num, 'super')
            if os.path.exists(super_path):
                superclasses.update([d for d in os.listdir(super_path) 
                                   if os.path.isdir(os.path.join(super_path, d))])
        
        print(f"Found superclasses: {sorted(superclasses)}")
        
        # Train SVD models for each superclass
        svd_models = {}
        for superclass in sorted(superclasses):
            print(f"\nTraining SVD model for superclass: {superclass}")
            
            # Load training data (use 1000 records for training)
            X, target_length = load_signals_for_training(data_dir, 1000, sampling_rate, superclass)
            if X is None:
                print(f"No training data found for {superclass}")
                continue
            
            # Train SVD model
            U_r, C, pivots = train_svd_model(X, rank)
            svd_models[superclass] = (U_r, C, pivots, target_length)
            print(f"Trained SVD model for {superclass} with rank {rank}, signal length {target_length}")
        
        # Process all records
        total_processed = 0
        
        for dir_num in tqdm(numbered_dirs, desc=f"Processing {sampling_rate}Hz directories"):
            super_path = os.path.join(base_path, dir_num, 'super')
            if not os.path.exists(super_path):
                continue
            
            # Create output directory for this numbered folder
            output_dir_path = os.path.join(output_dir, f"records{sampling_rate}", dir_num)
            
            for superclass in os.listdir(super_path):
                superclass_path = os.path.join(super_path, superclass)
                if not os.path.isdir(superclass_path) or superclass not in svd_models:
                    continue
                
                U_r, C, pivots, target_length = svd_models[superclass]
                
                # Process each record in this superclass
                record_files = [f for f in os.listdir(superclass_path) if f.endswith('.hea')]
                
                for record_file in record_files:
                    record_id = record_file.split('_')[0]
                    record_path = os.path.join(superclass_path, f"{record_id}_{record_prefix}")
                    
                    try:
                        # Load original record
                        record = wfdb.rdrecord(record_path)
                        original_signal = record.p_signal[:, 1]  # Lead II
                        normalized_signal = original_signal / np.max(np.abs(original_signal))
                        
                        # Reconstruct using SVD
                        reconstructed_signal = reconstruct_signal(normalized_signal, U_r, C, target_length)
                        
                        # Denormalize
                        reconstructed_signal = reconstructed_signal * np.max(np.abs(original_signal))
                        
                        # Create new signal matrix (keep all leads, but compress Lead II)
                        new_signal = record.p_signal.copy()
                        new_signal[:, 1] = reconstructed_signal
                        
                        # Save compressed record directly in the numbered folder
                        output_record_name = f"{record_id}_{record_prefix}"
                        
                        # Create new record with compressed signal
                        # Save current directory and change to output directory
                        current_dir = os.getcwd()
                        try:
                            os.chdir(output_dir_path)
                            wfdb.wrsamp(record_name=output_record_name,
                                      fs=record.fs,
                                      units=record.units,
                                      sig_name=record.sig_name,
                                      p_signal=new_signal,
                                      fmt=record.fmt)
                        finally:
                            # Always change back to original directory
                            os.chdir(current_dir)
                        
                        total_processed += 1
                        
                    except Exception as e:
                        print(f"Error processing {record_path}: {str(e)}")
                        continue
        
        print(f"Total {sampling_rate}Hz records processed: {total_processed}")

def copy_metadata_files(output_dir):
    """Copy metadata files from original dataset"""
    metadata_files = ['ptbxl_database.csv', 'scp_statements.csv']
    
    for filename in metadata_files:
        src_path = os.path.join(ORIGINAL_DATA_DIR, filename)
        dst_path = os.path.join(output_dir, filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {filename}")
        else:
            print(f"Warning: {filename} not found in original dataset")

def create_compression_info_file(output_dir, compression_config):
    """Create a file with compression information"""
    info = {
        'compression_level': compression_config['name'],
        'rank': compression_config['rank'],
        'sampling_rates': '100Hz and 500Hz',
        'description': f"SVD compressed dataset retaining {compression_config['name'].replace('_', ' ')} of original signal",
        'original_dataset': ORIGINAL_DATA_DIR
    }
    
    info_path = os.path.join(output_dir, 'compression_info.txt')
    with open(info_path, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Created compression info file: {info_path}")

def main():
    # Hardcoded configuration
    compression_to_create = 'all'  # Options: '25', '50', '75', 'all'
    data_dir = './data/categorized'  # Path to categorized PTB-XL dataset (with super subdirectories)
    output_dir = './data/compressed'  # Base output directory for compressed datasets
    
    # Update global variables
    global ORIGINAL_DATA_DIR, OUTPUT_BASE_DIR
    ORIGINAL_DATA_DIR = data_dir
    OUTPUT_BASE_DIR = output_dir
    
    # Check if original dataset exists
    if not os.path.exists(ORIGINAL_DATA_DIR):
        print(f"Error: Original dataset not found at {ORIGINAL_DATA_DIR}")
        print("Please update the 'data_dir' variable in main() to point to your PTB-XL dataset")
        return
    
    # Determine which compression levels to create
    if compression_to_create == 'all':
        levels_to_create = COMPRESSION_LEVELS.keys()
    else:
        levels_to_create = [compression_to_create]
    
    print("ECG Dataset SVD Compression")
    print("=" * 50)
    print(f"Creating compressed datasets for levels: {list(levels_to_create)}")
    print(f"Original dataset: {ORIGINAL_DATA_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print("=" * 50)
    
    for level in levels_to_create:
        compression_config = COMPRESSION_LEVELS[level]
        output_dataset_dir = os.path.join(OUTPUT_BASE_DIR, f"ptb-xl-{compression_config['name']}")
        
        print(f"\n{'='*60}")
        print(f"Creating {compression_config['name']} compressed dataset")
        print(f"Rank: {compression_config['rank']}")
        print(f"Output directory: {output_dataset_dir}")
        print(f"{'='*60}")
        
        # Create output directory
        os.makedirs(output_dataset_dir, exist_ok=True)
        
        # Compress and save records for both sampling rates
        compress_and_save_records(ORIGINAL_DATA_DIR, output_dataset_dir, compression_config)
        
        # Copy metadata files
        copy_metadata_files(output_dataset_dir)
        
        # Create compression info file
        create_compression_info_file(output_dataset_dir, compression_config)
        
        print(f"Completed {compression_config['name']} compressed dataset")
    
    print(f"\n{'='*60}")
    print("All compressed datasets created successfully!")
    print(f"Output location: {OUTPUT_BASE_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main() 