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
from colorama import init, Fore, Back, Style

# Initialize colorama
init()

# Configuration
ORIGINAL_DATA_DIR = '/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/raw/ptb-xl-1.0.3'
COMPRESSED_BASE_DIR = '/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/compressed'
NUM_SAMPLES = 5  # Number of records to sample for verification
SAMPLING_RATES = [100, 500]  # Both sampling rates
COMPRESSION_LEVELS = ['25_percent', '50_percent', '75_percent']  # Compression levels to verify

def print_path_info(path, description, exists=None):
    """Print path information with colors"""
    if exists is None:
        exists = os.path.exists(path + '.hea') and os.path.exists(path + '.dat')
    
    status_color = Fore.GREEN if exists else Fore.RED
    status = "✓" if exists else "✗"
    
    print(f"{Fore.CYAN}{description}:")
    print(f"{status_color}{status} {Style.RESET_ALL}{path}")
    if not exists:
        print(f"{Fore.YELLOW}Note: Looking for {path}.hea and {path}.dat{Style.RESET_ALL}")
    print()

def print_section_header(title):
    """Print a section header with decoration"""
    print(f"\n{Back.BLUE}{Fore.WHITE} {title} {Style.RESET_ALL}")
    print("=" * (len(title) + 4))

def find_available_records(data_dir, sampling_rate=500, max_records=10):
    """Find available records in the dataset"""
    print_section_header(f"Searching for {sampling_rate}Hz Records")
    
    records = []
    base_path = os.path.join(data_dir, f"records{sampling_rate}")
    
    print_path_info(base_path, "Base records directory", os.path.exists(base_path))
    
    if not os.path.exists(base_path):
        return records
    
    numbered_dirs = [d for d in os.listdir(base_path) if len(d) == 5 and d.isdigit()]
    print(f"{Fore.CYAN}Found {Fore.WHITE}{len(numbered_dirs)}{Fore.CYAN} numbered directories{Style.RESET_ALL}\n")
    
    for dir_num in numbered_dirs:
        dir_path = os.path.join(base_path, dir_num)
        # Look for .hea files directly in the numbered directory
        record_files = [f for f in os.listdir(dir_path) if f.endswith('.hea')]
        
        for record_file in record_files[:min(2, len(record_files))]:  # Limit per directory
            record_id = record_file.split('.')[0]  # Remove .hea extension
            record_path = os.path.join(dir_path, record_id)
            
            # Check if both .dat and .hea files exist
            if os.path.exists(record_path + '.hea') and os.path.exists(record_path + '.dat'):
                records.append({
                    'record_id': record_id,
                    'record_path': record_path,
                    'superclass': 'unknown',  # We don't have superclass info in this structure
                    'dir_num': dir_num
                })
                print(f"{Fore.GREEN}Found record: {Style.RESET_ALL}{record_id} in directory {dir_num}")
                
                if len(records) >= max_records:
                    print(f"{Fore.YELLOW}Reached maximum number of records ({max_records}){Style.RESET_ALL}")
                    return records
    
    print(f"{Fore.GREEN}Found {len(records)} valid records{Style.RESET_ALL}")
    return records

def find_compressed_record_path(original_record_info, compressed_dir, sampling_rate, signal_type):
    """Find the corresponding compressed record path"""
    record_id = original_record_info['record_id']
    dir_num = original_record_info['dir_num']
    
    # Updated path structure to match the actual layout - don't add prefix again
    compressed_record_path = os.path.join(compressed_dir, signal_type, 
                                        f"records{sampling_rate}", 
                                        dir_num, 
                                        record_id)  # record_id already has the prefix
    
    print_path_info(compressed_record_path, f"Looking for {signal_type} signal")
    return compressed_record_path

def load_signal(record_path):
    """Load ECG signal from record"""
    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, :]  # all leads
        # Normalize the signal
        signal = signal / np.max(np.abs(signal))
        print(f"{Fore.GREEN}Successfully loaded and normalized: {Style.RESET_ALL}{record_path}")
        return signal
    except Exception as e:
        print(f"{Fore.RED}Error loading {record_path}: {str(e)}{Style.RESET_ALL}")
        return None

def calculate_mse(original, compressed):
    """Calculate Mean Squared Error between original and compressed signals using all leads"""
    # Ensure both signals are normalized
    original_norm = original / np.max(np.abs(original))
    compressed_norm = compressed / np.max(np.abs(compressed))
    # Calculate MSE across all leads
    return np.mean((original_norm - compressed_norm)**2)

def calculate_compression_ratio(original_signal, rank):
    """Calculate theoretical compression ratio"""
    signal_length = len(original_signal)
    original_size = signal_length  # Number of samples
    compressed_size = rank  # Number of coefficients stored
    return (1 - compressed_size / original_size) * 100

def plot_signal_comparison(original, sampled, reconstructed, record_info, compression_level, sampling_rate, output_dir):
    """Plot comparison between original, sampled, and reconstructed signals with separate and combined views"""
    # Calculate MSE using all leads
    mse_reconstructed = calculate_mse(original, reconstructed)
    mse_sampled = calculate_mse(original, sampled)
    
    # Extract Lead II (index 1) for plotting
    original_lead2 = original[:, 1]
    sampled_lead2 = sampled[:, 1]
    reconstructed_lead2 = reconstructed[:, 1]
    
    # Create a figure with 6 subplots
    plt.figure(figsize=(20, 30))
    
    # 1. Original Signal (Lead II)
    plt.subplot(6, 1, 1)
    plt.plot(original_lead2, 'b-', label='Original', linewidth=1.5)
    plt.title(f'Original ECG Signal - Lead II - Record {record_info["record_id"]} ({record_info["superclass"]}) - {sampling_rate}Hz')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Sampled Signal (Lead II)
    plt.subplot(6, 1, 2)
    plt.plot(sampled_lead2, 'g-', label=f'Sampled ({compression_level})', linewidth=1.5)
    plt.title(f'Sampled Signal (Lead II)\nMSE (all leads): {mse_sampled:.6f}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Reconstructed Signal (Lead II)
    plt.subplot(6, 1, 3)
    plt.plot(reconstructed_lead2, 'r-', label=f'Reconstructed ({compression_level})', linewidth=1.5)
    plt.title(f'Reconstructed Signal (Lead II)\nMSE (all leads): {mse_reconstructed:.6f}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Combined View (Lead II)
    plt.subplot(6, 1, 4)
    plt.plot(original_lead2, 'b-', label='Original', alpha=0.7)
    plt.plot(sampled_lead2, 'g-', label=f'Sampled ({compression_level})', alpha=0.7)
    plt.plot(reconstructed_lead2, 'r-', label=f'Reconstructed ({compression_level})', alpha=0.7)
    plt.title('Combined Signal Comparison (Lead II)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Zoomed Section (Lead II - first 1000 samples)
    plt.subplot(6, 1, 5)
    zoom_samples = min(1000, len(original_lead2))
    plt.plot(original_lead2[:zoom_samples], 'b-', label='Original', linewidth=1.5)
    plt.plot(sampled_lead2[:zoom_samples], 'g--', label=f'Sampled ({compression_level})', linewidth=1.5)
    plt.plot(reconstructed_lead2[:zoom_samples], 'r--', label=f'Reconstructed ({compression_level})', linewidth=1.5)
    plt.title('Zoomed View - Lead II (First 1000 samples)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Difference Signals (Lead II)
    plt.subplot(6, 1, 6)
    diff_sampled = original_lead2 - sampled_lead2
    diff_reconstructed = original_lead2 - reconstructed_lead2
    plt.plot(diff_sampled, 'g-', label='Difference (Original - Sampled)', alpha=0.7)
    plt.plot(diff_reconstructed, 'r-', label='Difference (Original - Reconstructed)', alpha=0.7)
    plt.title(f'Signal Differences (Lead II)\nMax Abs Diff (Sampled): {np.max(np.abs(diff_sampled)):.6f}\nMax Abs Diff (Reconstructed): {np.max(np.abs(diff_reconstructed)):.6f}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)  # Add more padding between subplots
    
    # Save plot
    plot_filename = f"comparison_{record_info['record_id']}_{compression_level}_{sampling_rate}Hz.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mse_reconstructed, mse_sampled, plot_path

def verify_dataset(compression_level, sampling_rates=[100, 500], num_samples=5):
    """Verify a compressed dataset by comparing with original for both sampling rates"""
    print_section_header(f"Verifying {compression_level} compressed dataset")
    
    # Paths
    original_dir = ORIGINAL_DATA_DIR
    compressed_dir = os.path.join(COMPRESSED_BASE_DIR, f"ptb-xl-{compression_level}")
    
    print_path_info(original_dir, "Original data directory")
    print_path_info(compressed_dir, "Compressed data directory")
    
    if not os.path.exists(compressed_dir):
        print(f"{Fore.RED}Error: Compressed dataset not found at {compressed_dir}{Style.RESET_ALL}")
        return None
    
    # Create output directory for plots
    output_dir = os.path.join("verification_plots", compression_level)
    os.makedirs(output_dir, exist_ok=True)
    print_path_info(output_dir, "Output directory", True)
    
    all_results = {}
    
    # Process each sampling rate
    for sampling_rate in sampling_rates:
        print_section_header(f"Processing {sampling_rate}Hz Records")
        
        # Find available records
        records = find_available_records(original_dir, sampling_rate, num_samples * 3)
        if not records:
            print(f"{Fore.YELLOW}No {sampling_rate}Hz records found for verification{Style.RESET_ALL}")
            continue
        
        # Randomly sample records
        sample_records = random.sample(records, min(num_samples, len(records)))
        print(f"{Fore.CYAN}Selected {len(sample_records)} records for verification{Style.RESET_ALL}\n")
        
        results = []
        
        for record_info in sample_records:
            print_section_header(f"Processing Record {record_info['record_id']}")
            print(f"{Fore.CYAN}Superclass: {Style.RESET_ALL}{record_info['superclass']}")
            print(f"{Fore.CYAN}Sampling Rate: {Style.RESET_ALL}{sampling_rate}Hz\n")
            
            # Load original signal
            print_path_info(record_info['record_path'], "Original signal path")
            original_signal = load_signal(record_info['record_path'])
            if original_signal is None:
                continue
            
            # Load sampled signal
            sampled_record_path = find_compressed_record_path(record_info, compressed_dir, sampling_rate, 'sampled')
            sampled_signal = load_signal(sampled_record_path)
            if sampled_signal is None:
                continue
            
            # Load reconstructed signal
            reconstructed_record_path = find_compressed_record_path(record_info, compressed_dir, sampling_rate, 'reconstructed')
            reconstructed_signal = load_signal(reconstructed_record_path)
            if reconstructed_signal is None:
                continue
            
            # Process signals and create visualizations
            print(f"\n{Fore.CYAN}Processing signals...{Style.RESET_ALL}")
            
            # Ensure signals have same length
            min_length = min(len(original_signal), len(sampled_signal), len(reconstructed_signal))
            original_signal = original_signal[:min_length]
            sampled_signal = sampled_signal[:min_length]
            reconstructed_signal = reconstructed_signal[:min_length]
            
            # Calculate metrics and create plot
            mse_reconstructed, mse_sampled, plot_path = plot_signal_comparison(
                original_signal, sampled_signal, reconstructed_signal,
                record_info, compression_level, sampling_rate, output_dir
            )
            
            print(f"\n{Fore.GREEN}Results:{Style.RESET_ALL}")
            print(f"  MSE (Reconstructed): {Fore.YELLOW}{mse_reconstructed:.6f}{Style.RESET_ALL}")
            print(f"  MSE (Sampled): {Fore.YELLOW}{mse_sampled:.6f}{Style.RESET_ALL}")
            print(f"  Max Diff (Reconstructed): {Fore.YELLOW}{np.max(np.abs(original_signal - reconstructed_signal)):.6f}{Style.RESET_ALL}")
            print(f"  Max Diff (Sampled): {Fore.YELLOW}{np.max(np.abs(original_signal - sampled_signal)):.6f}{Style.RESET_ALL}")
            
            # Get compression info
            compression_info_path = os.path.join(compressed_dir, 'compression_info.txt')
            print_path_info(compression_info_path, "Compression info file")
            
            rank = None
            if os.path.exists(compression_info_path):
                with open(compression_info_path, 'r') as f:
                    for line in f:
                        if line.startswith('rank:'):
                            rank = int(line.split(':')[1].strip())
                            break
            
            compression_ratio = calculate_compression_ratio(original_signal, rank) if rank else None
            if compression_ratio:
                print(f"  Compression Ratio: {Fore.YELLOW}{compression_ratio:.1f}%{Style.RESET_ALL}")
            
            results.append({
                'record_id': record_info['record_id'],
                'superclass': record_info['superclass'],
                'sampling_rate': sampling_rate,
                'mse_reconstructed': mse_reconstructed,
                'mse_sampled': mse_sampled,
                'max_abs_diff_reconstructed': np.max(np.abs(original_signal - reconstructed_signal)),
                'max_abs_diff_sampled': np.max(np.abs(original_signal - sampled_signal)),
                'compression_ratio': compression_ratio,
                'rank': rank,
                'plot_path': plot_path
            })
            
            print(f"\n{Fore.GREEN}Plot saved: {Style.RESET_ALL}{plot_path}")
            print("-" * 80 + "\n")
        
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
                
                mse_values = [r['mse_reconstructed'] for r in results]
                max_diff_values = [r['max_abs_diff_reconstructed'] for r in results]
                
                f.write(f"Number of samples: {len(results)}\n")
                f.write(f"Average MSE (Reconstructed): {np.mean(mse_values):.6f}\n")
                f.write(f"MSE Range (Reconstructed): {np.min(mse_values):.6f} - {np.max(mse_values):.6f}\n")
                f.write(f"Average Max Absolute Difference (Reconstructed): {np.mean(max_diff_values):.6f}\n")
                
                if results[0]['compression_ratio']:
                    f.write(f"Compression Ratio: {results[0]['compression_ratio']:.1f}%\n")
                    f.write(f"Rank: {results[0]['rank']}\n")
                
                f.write("\nIndividual Results:\n")
                for result in results:
                    f.write(f"  {result['record_id']} ({result['superclass']}): ")
                    f.write(f"MSE (Reconstructed)={result['mse_reconstructed']:.6f}, Max_Diff (Reconstructed)={result['max_abs_diff_reconstructed']:.6f}\n")
                
                f.write("\n")
            
            f.write("\n")
    
    print(f"Summary report saved to: {output_file}")

def main():
    """Run verification with hardcoded configuration"""
    # Set random seed for reproducible sampling
    random.seed(42)
    
    print(f"\n{Back.GREEN}{Fore.BLACK} ECG Dataset Compression Verification {Style.RESET_ALL}")
    print("=" * 40)
    print(f"{Fore.CYAN}Original dataset: {Style.RESET_ALL}{ORIGINAL_DATA_DIR}")
    print(f"{Fore.CYAN}Compressed datasets: {Style.RESET_ALL}{COMPRESSED_BASE_DIR}")
    print(f"{Fore.CYAN}Sampling rates: {Style.RESET_ALL}{SAMPLING_RATES}")
    print(f"{Fore.CYAN}Number of samples per compression level: {Style.RESET_ALL}{NUM_SAMPLES}")
    print(f"{Fore.CYAN}Compression levels to verify: {Style.RESET_ALL}{COMPRESSION_LEVELS}")
    print("=" * 40)
    
    all_results = {}
    
    for level in COMPRESSION_LEVELS:
        results = verify_dataset(level, SAMPLING_RATES, NUM_SAMPLES)
        all_results[level] = results
    
    # Create summary report
    print_section_header("Creating Summary Report")
    create_summary_report(all_results)
    
    print(f"\n{Back.GREEN}{Fore.BLACK} Verification Completed! {Style.RESET_ALL}")
    print(f"{Fore.CYAN}Plots saved in: {Style.RESET_ALL}./verification_plots/")
    print(f"{Fore.CYAN}Summary report: {Style.RESET_ALL}verification_report.txt\n")

if __name__ == "__main__":
    main() 