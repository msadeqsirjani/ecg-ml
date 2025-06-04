#!/usr/bin/env python3
"""
Create compressed ECG datasets using sampling-based compression.
This script creates 3 new datasets containing 25%, 50%, and 75% of the original points,
storing both sampled and reconstructed signals.
"""

import os
import wfdb
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
import shutil
from pathlib import Path
import json
from colorama import init, Fore, Style
import time
from datetime import datetime, timedelta
from scipy.linalg import qr

# Initialize colorama for Windows support
init()

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}  {text}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

def print_success(text):
    """Print a success message."""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

def print_warning(text):
    """Print a warning message."""
    print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")

def print_error(text):
    """Print an error message."""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

def print_info(text):
    """Print an info message."""
    print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")

def format_time(seconds):
    """Format time in a human-readable way."""
    return str(timedelta(seconds=int(seconds)))

class ProcessingStats:
    """Class to track processing statistics."""
    def __init__(self):
        self.total_processed = 0
        self.total_errors = 0
        self.start_time = time.time()
        self.sampling_stats = {100: {'processed': 0, 'errors': 0},
                             500: {'processed': 0, 'errors': 0}}

    def update(self, sampling_rate, success=True):
        if success:
            self.total_processed += 1
            self.sampling_stats[sampling_rate]['processed'] += 1
        else:
            self.total_errors += 1
            self.sampling_stats[sampling_rate]['errors'] += 1

    def print_summary(self):
        elapsed_time = time.time() - self.start_time
        print_header("Processing Summary")
        print_info(f"Total time elapsed: {format_time(elapsed_time)}")
        print_info(f"Total records processed: {self.total_processed}")
        if self.total_errors > 0:
            print_warning(f"Total errors encountered: {self.total_errors}")
        
        for rate in [100, 500]:
            stats = self.sampling_stats[rate]
            print(f"\n{rate}Hz Records:")
            print_success(f"  Processed: {stats['processed']}")
            if stats['errors'] > 0:
                print_warning(f"  Errors: {stats['errors']}")
            success_rate = (stats['processed'] / (stats['processed'] + stats['errors'])) * 100 if stats['processed'] + stats['errors'] > 0 else 0
            print_info(f"  Success rate: {success_rate:.1f}%")

# Configuration
ORIGINAL_DATA_DIR = r'/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/categorized'
OUTPUT_BASE_DIR = r'/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/compressed'

# SVD Compression Parameters
SVD_PARAMS = {
    'test_record': 'record_001',  # specific record ID to reconstruct
    'sampling_rate': 500,         # 100Hz or 500Hz
    'num_train_records': 100,     # amount of training data for SVD model
    'rank': 25,                   # number of patterns/timestamps to keep
    'superclass': 'HYP'          # classification type
}

# Compression levels (percentage of points to keep)
COMPRESSION_LEVELS = {
    '25': {'percent': 25, 'name': '25_percent'},
    '50': {'percent': 50, 'name': '50_percent'},
    '75': {'percent': 75, 'name': '75_percent'}
}

# Processing configuration
PROCESSING_CONFIG = {
    'compression_to_create': 'all',  # Options: '25', '50', '75', 'all'
    'data_dir': ORIGINAL_DATA_DIR,
    'output_dir': OUTPUT_BASE_DIR
}

def sample_and_reconstruct_signal(signal, percent_to_keep):
    """
    Sample signal points using SVD-based compression.
    
    Args:
        signal: Original signal array
        percent_to_keep: Percentage of points to keep (25, 50, or 75)
    
    Returns:
        tuple: (sampled_signal, reconstructed_signal)
        - sampled_signal: Signal with only selected points (others set to 0)
        - reconstructed_signal: Signal reconstructed using SVD
    """
    signal_length = len(signal)
    
    # Calculate rank based on percent_to_keep
    if percent_to_keep == 25:
        rank = 25  # 99.5% reduction
    elif percent_to_keep == 50:
        rank = 2500  # 50% reduction
    else:  # 75%
        rank = 3750  # 25% reduction
    
    # Ensure rank doesn't exceed signal length
    rank = min(rank, signal_length - 1)
    
    # Reshape signal for SVD (add batch dimension)
    signal_matrix = signal.reshape(1, -1)
    
    # Perform SVD
    U, s, Vt = np.linalg.svd(signal_matrix, full_matrices=False)
    
    # Keep only top 'rank' components
    U_reduced = U[:, :rank]
    s_reduced = s[:rank]
    Vt_reduced = Vt[:rank, :]
    
    # Create sampled signal (using sampling matrix)
    sampling_matrix = np.zeros((rank, signal_length))
    indices = np.linspace(0, signal_length-1, rank, dtype=int)
    for i, idx in enumerate(indices):
        sampling_matrix[i, idx] = 1.0
    
    sampled_signal = np.zeros_like(signal)
    sampled_signal[indices] = signal[indices]
    
    # Reconstruct signal using SVD components
    reconstructed = (U_reduced @ np.diag(s_reduced) @ Vt_reduced)
    reconstructed_signal = reconstructed.flatten()
    
    # Ensure output has the same length as input
    if len(reconstructed_signal) != signal_length:
        reconstructed_signal = np.interp(
            np.linspace(0, len(reconstructed_signal)-1, signal_length),
            np.arange(len(reconstructed_signal)),
            reconstructed_signal
        )
    
    return sampled_signal, reconstructed_signal

def compress_and_save_records(data_dir, output_dir, compression_config):
    """Compress and save all records using sampling-based compression."""
    stats = ProcessingStats()
    percent_to_keep = compression_config['percent']
    
    print_header(f"Starting Compression Process - {percent_to_keep}% compression")
    print_info(f"Input directory: {data_dir}")
    print_info(f"Output directory: {output_dir}")
    
    # Create base output directories for sampled and reconstructed data
    output_sampled_dir = os.path.join(output_dir, 'sampled')
    output_reconstructed_dir = os.path.join(output_dir, 'reconstructed')
    
    # Process each sampling rate
    for sampling_rate in [100, 500]:
        record_prefix = 'lr' if sampling_rate == 100 else 'hr'
        print_header(f"Processing {sampling_rate}Hz Records")
        
        base_path = os.path.join(data_dir, f"records{sampling_rate}")
        if not os.path.exists(base_path):
            print_warning(f"Directory not found: {base_path}")
            continue
        
        # Get all numbered directories
        numbered_dirs = [d for d in os.listdir(base_path) if len(d) == 5 and d.isdigit()]
        
        # Process each numbered directory with progress bar
        for dir_num in tqdm(numbered_dirs, 
                           desc=f"{Fore.BLUE}Processing directories{Style.RESET_ALL}",
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            super_path = os.path.join(base_path, dir_num, 'super')
            if not os.path.exists(super_path):
                continue
            
            # Create output directories for both sampled and reconstructed data
            sampled_records_dir = os.path.join(output_sampled_dir, f"records{sampling_rate}", dir_num)
            reconstructed_records_dir = os.path.join(output_reconstructed_dir, f"records{sampling_rate}", dir_num)
            os.makedirs(sampled_records_dir, exist_ok=True)
            os.makedirs(reconstructed_records_dir, exist_ok=True)
            
            # Process each superclass directory
            for superclass in os.listdir(super_path):
                superclass_path = os.path.join(super_path, superclass)
                if not os.path.isdir(superclass_path):
                    continue
                
                # Process each record
                record_files = [f for f in os.listdir(superclass_path) if f.endswith('.hea')]
                
                for record_file in record_files:
                    record_id = record_file.split('_')[0]
                    record_path = os.path.join(superclass_path, f"{record_id}_{record_prefix}")
                    
                    try:
                        # Load original record
                        record = wfdb.rdrecord(record_path)
                        
                        # Process each lead
                        sampled_signal = np.zeros_like(record.p_signal)
                        reconstructed_signal = np.zeros_like(record.p_signal)
                        
                        for lead in range(record.p_signal.shape[1]):
                            original_signal = record.p_signal[:, lead]
                            sampled, reconstructed = sample_and_reconstruct_signal(
                                original_signal, percent_to_keep
                            )
                            sampled_signal[:, lead] = sampled
                            reconstructed_signal[:, lead] = reconstructed
                        
                        # Save sampled signal
                        current_dir = os.getcwd()
                        try:
                            os.chdir(sampled_records_dir)
                            # Use the exact same filename format as the original
                            wfdb.wrsamp(
                                record_name=f"{record_id}_{record_prefix}",
                                fs=record.fs,
                                units=record.units,
                                sig_name=record.sig_name,
                                p_signal=sampled_signal,
                                fmt=record.fmt
                            )
                            # Copy the header file with exact same name
                            shutil.copy2(
                                os.path.join(superclass_path, record_file),
                                os.path.join(sampled_records_dir, record_file)
                            )
                        finally:
                            os.chdir(current_dir)
                        
                        # Save reconstructed signal
                        try:
                            os.chdir(reconstructed_records_dir)
                            # Use the exact same filename format as the original
                            wfdb.wrsamp(
                                record_name=f"{record_id}_{record_prefix}",
                                fs=record.fs,
                                units=record.units,
                                sig_name=record.sig_name,
                                p_signal=reconstructed_signal,
                                fmt=record.fmt
                            )
                            # Copy the header file with exact same name
                            shutil.copy2(
                                os.path.join(superclass_path, record_file),
                                os.path.join(reconstructed_records_dir, record_file)
                            )
                        finally:
                            os.chdir(current_dir)
                        
                        stats.update(sampling_rate, success=True)
                        
                    except Exception as e:
                        print_error(f"Error processing {record_path}: {str(e)}")
                        stats.update(sampling_rate, success=False)
                        continue
    
    # Print final statistics
    stats.print_summary()

def create_output_directory_structure(output_dir):
    """Create the output directory structure for both sampled and reconstructed data."""
    # Create base directories for sampled and reconstructed data
    sampled_dir = os.path.join(output_dir, 'sampled')
    reconstructed_dir = os.path.join(output_dir, 'reconstructed')
    
    # Create records100 and records500 directories for both versions
    for base_dir in [sampled_dir, reconstructed_dir]:
        for rate in [100, 500]:
            records_dir = os.path.join(base_dir, f"records{rate}")
            os.makedirs(records_dir, exist_ok=True)
            
            # Create numbered directories (00000-21000)
            for dir_num in range(22):
                folder_name = str(dir_num * 1000).zfill(5)
                target_dir = os.path.join(records_dir, folder_name)
                os.makedirs(target_dir, exist_ok=True)

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
        'percent_kept': compression_config['percent'],
        'sampling_rates': '100Hz and 500Hz',
        'description': f"SVD-based compression with {compression_config['percent']}% data retention",
        'original_dataset': ORIGINAL_DATA_DIR,
        'compression_method': 'SVD with QR decomposition',
        'svd_parameters': SVD_PARAMS,
        'versions': {
            'sampled': 'Original signal with SVD-based sampling',
            'reconstructed': 'Signal reconstructed using SVD components'
        }
    }
    
    info_path = os.path.join(output_dir, 'compression_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"Created compression info file: {info_path}")

def main():
    """Main function to process all compression levels."""
    print_header("ECG Dataset Compression Tool")
    
    start_time = time.time()
    
    try:
        # Check if original dataset exists
        if not os.path.exists(PROCESSING_CONFIG['data_dir']):
            print_error(f"Original dataset not found at {PROCESSING_CONFIG['data_dir']}")
            return
        
        # Determine which compression levels to create
        if PROCESSING_CONFIG['compression_to_create'] == 'all':
            levels_to_create = COMPRESSION_LEVELS.keys()
        else:
            levels_to_create = [PROCESSING_CONFIG['compression_to_create']]
        
        print_info("Configuration:")
        print_info(f"  Data directory: {PROCESSING_CONFIG['data_dir']}")
        print_info(f"  Output directory: {PROCESSING_CONFIG['output_dir']}")
        print_info(f"  Compression levels: {list(levels_to_create)}")
        print_info(f"  SVD Parameters:")
        for key, value in SVD_PARAMS.items():
            print_info(f"    {key}: {value}")
        
        for level in levels_to_create:
            compression_config = COMPRESSION_LEVELS[level]
            output_dataset_dir = os.path.join(PROCESSING_CONFIG['output_dir'], 
                                            f"ptb-xl-{compression_config['name']}")
            
            print_header(f"Processing {compression_config['name']} Dataset")
            print_info(f"Keeping {compression_config['percent']}% of points")
            print_info(f"Output directory: {output_dataset_dir}")
            
            # Create output directory
            os.makedirs(output_dataset_dir, exist_ok=True)
            
            # Compress and save records
            compress_and_save_records(PROCESSING_CONFIG['data_dir'], 
                                    output_dataset_dir, compression_config)
            
            # Copy metadata files
            copy_metadata_files(output_dataset_dir)
            
            # Create compression info file
            create_compression_info_file(output_dataset_dir, compression_config)
            
            print_success(f"Completed {compression_config['name']} compressed dataset")
        
        print_header("Processing Complete")
        print_success("All compressed datasets created successfully!")
        print_info(f"Output location: {PROCESSING_CONFIG['output_dir']}")
        
    except Exception as e:
        print_error(f"An error occurred during processing: {str(e)}")
    finally:
        elapsed_time = time.time() - start_time
        print_info(f"\nTotal execution time: {format_time(elapsed_time)}")

if __name__ == "__main__":
    main() 