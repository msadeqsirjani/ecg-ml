#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wfdb
import os
import random
from pathlib import Path
import argparse


def load_missing_info(data_path):
    """Load the missing value information CSV file."""
    missing_info_path = Path(data_path) / "missing_value_info.csv"
    if not missing_info_path.exists():
        raise FileNotFoundError(f"Missing value info not found at {missing_info_path}")
    
    return pd.read_csv(missing_info_path)


def get_random_ecg(data_path, missing_info_df, sampling_rate=500):
    """Get a random ECG signal with missing values."""
    # Filter by sampling rate
    filtered_info = missing_info_df[missing_info_df['sampling_rate'] == sampling_rate]
    
    if filtered_info.empty:
        raise ValueError(f"No records found with sampling rate {sampling_rate}")
    
    # Get a random ECG ID from the filtered dataframe
    random_row = filtered_info.drop_duplicates(subset=['ecg_id']).sample(1).iloc[0]
    ecg_id = int(random_row['ecg_id'])  # Convert to integer
    
    # Construct path to the ECG file
    record_num = f"{ecg_id:05d}"
    folder_num = f"{(ecg_id // 1000) * 1000:05d}"
    suffix = "_lr" if sampling_rate == 100 else "_hr"
    ecg_path = Path(data_path) / f"records{sampling_rate}" / folder_num / f"{record_num}{suffix}"
    
    # Load the signal
    signal, header = wfdb.rdsamp(str(ecg_path))
    
    # Get all missing periods for this ECG
    ecg_missing_info = filtered_info[filtered_info['ecg_id'] == ecg_id]
    
    return signal, header, ecg_missing_info


def plot_ecg_with_missing_values(signal, header, missing_info, save_path=None):
    """Plot ECG signal with highlighted missing value regions."""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Calculate time vector
    fs = header['fs']  # sampling frequency
    time = np.arange(signal.shape[0]) / fs
    
    # Plot each channel
    num_channels = signal.shape[1]
    colors = plt.cm.tab10.colors
    channel_names = header['sig_name']
    
    # Get missing values regions
    missing_regions = []
    for _, row in missing_info.iterrows():
        start_time = row['start_time']
        duration = row['duration']
        missing_regions.append((start_time, start_time + duration))
    
    # Plot channels
    for i in range(num_channels):
        # Add offset to separate channels visually
        offset = i * max(2, 1.5 * np.max(np.abs(signal)))
        ax.plot(time, signal[:, i] + offset, color=colors[i % len(colors)], 
                label=channel_names[i], linewidth=0.8)
    
    # Highlight missing value regions
    for start, end in missing_regions:
        ax.axvspan(start, end, color='red', alpha=0.2)
    
    # Add legend and labels
    ax.legend(loc='upper right')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title('ECG Signal with Missing Values (Highlighted in Red)')
    
    # Add text explaining the missing regions
    if missing_regions:
        missing_text = f"Missing Values: {len(missing_regions)} periods"
        if len(missing_regions) > 0:
            period = round(missing_regions[1][0] - missing_regions[0][0], 2) if len(missing_regions) > 1 else 0
            duration = round(missing_regions[0][1] - missing_regions[0][0], 2)
            missing_text += f"\nInterval: {period}s, Duration: {duration}s per period"
        ax.text(0.02, 0.02, missing_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return save_path
    else:
        plt.show()
        return None


def main():
    """Main function to plot a random ECG with missing values."""
    parser = argparse.ArgumentParser(description='Plot ECG signals with missing values')
    parser.add_argument('--data_path', type=str, default='data/raw/ptb-xl-missing-values',
                        help='Path to the processed dataset with missing values')
    parser.add_argument('--sampling_rate', type=int, default=500, choices=[100, 500],
                        help='Sampling rate to use (100 or 500 Hz)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the plot (if not provided, plot will be displayed)')
    parser.add_argument('--num_plots', type=int, default=1,
                        help='Number of random plots to generate')
    
    args = parser.parse_args()
    
    try:
        # Load missing value information
        missing_info_df = load_missing_info(args.data_path)
        
        for i in range(args.num_plots):
            # Get a random ECG signal
            signal, header, ecg_missing_info = get_random_ecg(
                args.data_path, missing_info_df, args.sampling_rate
            )
            
            # Generate save path if needed
            save_path = None
            if args.save_path:
                base, ext = os.path.splitext(args.save_path)
                if args.num_plots > 1:
                    save_path = f"{base}_{i+1}{ext}"
                else:
                    save_path = args.save_path
            
            # Plot the ECG with missing values highlighted
            output_path = plot_ecg_with_missing_values(signal, header, ecg_missing_info, save_path)
            
            if output_path:
                print(f"Plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 