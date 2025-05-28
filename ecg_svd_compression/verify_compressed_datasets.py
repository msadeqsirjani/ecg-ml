#!/usr/bin/env python3
"""
Verify compressed ECG datasets by sampling and plotting signals.
This script compares original and compressed signals to verify the compression quality.
"""

import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from pathlib import Path

# Configuration
ORIGINAL_DATA_DIR = '/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/categorized'
COMPRESSED_BASE_DIR = '/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/compressed'
NUM_SAMPLES = 5  # Number of records to sample for verification
SAMPLING_RATES = [100, 500]  # Both sampling rates
COMPRESSION_LEVELS = ['25_percent', '50_percent', '75_percent']  # Compression levels to verify

def find_available_records(data_dir, sampling_rate=500, max_records=10):
    """Find available records in the dataset"""
    records = []
    record_prefix = 'hr' if sampling_rate == 500 else 'lr'
    base_path = os.path.join(data_dir, f"records{sampling_rate}")
    
    if not os.path.exists(base_path):
        print(f"Warning: {base_path} does not exist")
        return records
    
    numbered_dirs = [d for d in os.listdir(base_path) if len(d) == 5 and d.isdigit()]
    
    for dir_num in numbered_dirs:
        super_path = os.path.join(base_path, dir_num, 'super')
        if not os.path.exists(super_path):
            continue
        
        for superclass in os.listdir(super_path):
            superclass_path = os.path.join(super_path, superclass)
            if not os.path.isdir(superclass_path):
                continue
            
            record_files = [f for f in os.listdir(superclass_path) if f.endswith('.hea')]
            
            for record_file in record_files[:min(2, len(record_files))]:  # Limit per superclass
                record_id = record_file.split('_')[0]
                record_path = os.path.join(superclass_path, f"{record_id}_{record_prefix}")
                
                records.append({
                    'record_id': record_id,
                    'record_path': record_path,
                    'superclass': superclass,
                    'dir_num': dir_num
                })
                
                if len(records) >= max_records:
                    return records
    
    return records

def find_compressed_record_path(original_record_info, compressed_dir, sampling_rate):
    """Find the corresponding compressed record path"""
    record_prefix = 'hr' if sampling_rate == 500 else 'lr'
    record_id = original_record_info['record_id']
    dir_num = original_record_info['dir_num']
    
    # In the new structure, files are stored directly in numbered folders
    compressed_record_path = os.path.join(compressed_dir, f"records{sampling_rate}", 
                                        dir_num, f"{record_id}_{record_prefix}")
    return compressed_record_path

def load_signal(record_path):
    """Load ECG signal from record"""
    try:
        record = wfdb.rdrecord(record_path)
        return record.p_signal[:, 1]  # Lead II
    except Exception as e:
        print(f"Error loading {record_path}: {str(e)}")
        return None

def calculate_mse(original, compressed):
    """Calculate Mean Squared Error between original and compressed signals"""
    return np.mean((original - compressed)**2)

def calculate_compression_ratio(original_signal, rank):
    """Calculate theoretical compression ratio"""
    signal_length = len(original_signal)
    original_size = signal_length  # Number of samples
    compressed_size = rank  # Number of coefficients stored
    return (1 - compressed_size / original_size) * 100

def plot_signal_comparison(original, compressed, record_info, compression_level, sampling_rate, output_dir):
    """Plot comparison between original and compressed signals"""
    mse = calculate_mse(original, compressed)
    
    plt.figure(figsize=(15, 10))
    
    # Plot full signals
    plt.subplot(3, 1, 1)
    plt.plot(original, 'b-', label='Original', alpha=0.7)
    plt.plot(compressed, 'r-', label=f'Compressed ({compression_level})', alpha=0.7)
    plt.title(f'ECG Signal Comparison - Record {record_info["record_id"]} ({record_info["superclass"]}) - {sampling_rate}Hz\nMSE: {mse:.6f}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot zoomed section (first 1000 samples)
    plt.subplot(3, 1, 2)
    zoom_samples = min(1000, len(original))
    plt.plot(original[:zoom_samples], 'b-', label='Original', linewidth=1.5)
    plt.plot(compressed[:zoom_samples], 'r--', label=f'Compressed ({compression_level})', linewidth=1.5)
    plt.title('Zoomed View (First 1000 samples)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot difference
    plt.subplot(3, 1, 3)
    difference = original - compressed
    plt.plot(difference, 'g-', label='Difference (Original - Compressed)', alpha=0.7)
    plt.title(f'Signal Difference\nMax Absolute Difference: {np.max(np.abs(difference)):.6f}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"comparison_{record_info['record_id']}_{compression_level}_{sampling_rate}Hz.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mse, plot_path

def verify_dataset(compression_level, sampling_rates=[100, 500], num_samples=5):
    """Verify a compressed dataset by comparing with original for both sampling rates"""
    print(f"\nVerifying {compression_level} compressed dataset...")
    
    # Paths
    original_dir = ORIGINAL_DATA_DIR
    compressed_dir = os.path.join(COMPRESSED_BASE_DIR, f"ptb-xl-{compression_level}")
    
    if not os.path.exists(compressed_dir):
        print(f"Error: Compressed dataset not found at {compressed_dir}")
        return None
    
    # Create output directory for plots
    output_dir = os.path.join("verification_plots", compression_level)
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # Process each sampling rate
    for sampling_rate in sampling_rates:
        print(f"\nProcessing {sampling_rate}Hz records...")
        
        # Find available records
        records = find_available_records(original_dir, sampling_rate, num_samples * 3)
        if not records:
            print(f"No {sampling_rate}Hz records found for verification")
            continue
        
        # Randomly sample records
        sample_records = random.sample(records, min(num_samples, len(records)))
        
        results = []
        
        for record_info in sample_records:
            print(f"Processing record {record_info['record_id']} ({record_info['superclass']}) - {sampling_rate}Hz...")
            
            # Load original signal
            original_signal = load_signal(record_info['record_path'])
            if original_signal is None:
                continue
            
            # Load compressed signal
            compressed_record_path = find_compressed_record_path(record_info, compressed_dir, sampling_rate)
            compressed_signal = load_signal(compressed_record_path)
            if compressed_signal is None:
                print(f"Compressed signal not found for {record_info['record_id']} at {compressed_record_path}")
                continue
            
            # Ensure signals have same length
            min_length = min(len(original_signal), len(compressed_signal))
            original_signal = original_signal[:min_length]
            compressed_signal = compressed_signal[:min_length]
            
            # Calculate metrics and create plot
            mse, plot_path = plot_signal_comparison(original_signal, compressed_signal, 
                                                  record_info, compression_level, sampling_rate, output_dir)
            
            # Get compression info
            compression_info_path = os.path.join(compressed_dir, 'compression_info.txt')
            rank = None
            if os.path.exists(compression_info_path):
                with open(compression_info_path, 'r') as f:
                    for line in f:
                        if line.startswith('rank:'):
                            rank = int(line.split(':')[1].strip())
                            break
            
            compression_ratio = calculate_compression_ratio(original_signal, rank) if rank else None
            
            results.append({
                'record_id': record_info['record_id'],
                'superclass': record_info['superclass'],
                'sampling_rate': sampling_rate,
                'mse': mse,
                'max_abs_diff': np.max(np.abs(original_signal - compressed_signal)),
                'compression_ratio': compression_ratio,
                'rank': rank,
                'plot_path': plot_path
            })
            
            print(f"  MSE: {mse:.6f}")
            print(f"  Max Absolute Difference: {np.max(np.abs(original_signal - compressed_signal)):.6f}")
            if compression_ratio:
                print(f"  Compression Ratio: {compression_ratio:.1f}%")
        
        all_results[f"{sampling_rate}Hz"] = results
    
    return all_results

def create_summary_report(all_results, output_file="verification_report.txt"):
    """Create a summary report of verification results"""
    with open(output_file, 'w') as f:
        f.write("ECG Dataset Compression Verification Report\n")
        f.write("=" * 50 + "\n\n")
        
        for compression_level, sampling_rate_results in all_results.items():
            f.write(f"Compression Level: {compression_level}\n")
            f.write("-" * 30 + "\n")
            
            for sampling_rate, results in sampling_rate_results.items():
                if not results:
                    continue
                    
                f.write(f"\nSampling Rate: {sampling_rate}\n")
                f.write("~" * 20 + "\n")
                
                mse_values = [r['mse'] for r in results]
                max_diff_values = [r['max_abs_diff'] for r in results]
                
                f.write(f"Number of samples: {len(results)}\n")
                f.write(f"Average MSE: {np.mean(mse_values):.6f}\n")
                f.write(f"MSE Range: {np.min(mse_values):.6f} - {np.max(mse_values):.6f}\n")
                f.write(f"Average Max Absolute Difference: {np.mean(max_diff_values):.6f}\n")
                
                if results[0]['compression_ratio']:
                    f.write(f"Compression Ratio: {results[0]['compression_ratio']:.1f}%\n")
                    f.write(f"Rank: {results[0]['rank']}\n")
                
                f.write("\nIndividual Results:\n")
                for result in results:
                    f.write(f"  {result['record_id']} ({result['superclass']}): ")
                    f.write(f"MSE={result['mse']:.6f}, Max_Diff={result['max_abs_diff']:.6f}\n")
                
                f.write("\n")
            
            f.write("\n")
    
    print(f"Summary report saved to: {output_file}")

def main():
    """Run verification with hardcoded configuration"""
    # Set random seed for reproducible sampling
    random.seed(42)
    
    print("ECG Dataset Compression Verification")
    print("=" * 40)
    print(f"Original dataset: {ORIGINAL_DATA_DIR}")
    print(f"Compressed datasets: {COMPRESSED_BASE_DIR}")
    print(f"Sampling rates: {SAMPLING_RATES}")
    print(f"Number of samples per compression level: {NUM_SAMPLES}")
    print(f"Compression levels to verify: {COMPRESSION_LEVELS}")
    print("=" * 40)
    
    all_results = {}
    
    for level in COMPRESSION_LEVELS:
        results = verify_dataset(level, SAMPLING_RATES, NUM_SAMPLES)
        all_results[level] = results
    
    # Create summary report
    create_summary_report(all_results)
    
    print(f"\nVerification completed!")
    print(f"Plots saved in: ./verification_plots/")
    print(f"Summary report: verification_report.txt")

if __name__ == "__main__":
    main() 