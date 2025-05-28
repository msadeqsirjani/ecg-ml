import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from pathlib import Path
import seaborn as sns
from itertools import cycle

# Set the style for better visualization
plt.style.use('seaborn-v0_8')  # Using a specific seaborn style version
sns.set_theme()  # Set seaborn theme

def load_metadata(data_dir):
    """Load the PTB-XL metadata and SCP statements."""
    metadata_path = os.path.join(data_dir, 'ptbxl_database.csv')
    scp_path = os.path.join(data_dir, 'scp_statements.csv')
    
    df = pd.read_csv(metadata_path)
    scp_df = pd.read_csv(scp_path, index_col=0)
    
    # Filter diagnostic statements
    scp_df = scp_df[scp_df.diagnostic == 1]
    
    # Define classification categories
    categories = {
        'binary': ['normal', 'abnormal'],
        'super': ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
        'sub': list(scp_df.index)  # All diagnostic codes
    }
    
    return df, scp_df, categories

def get_sampling_frequency(filename):
    """Extract sampling frequency from filename."""
    if 'hr' in filename:
        return '500'  # High resolution files are 500 Hz
    return '100'  # Low resolution files are 100 Hz

def read_ecg_signal(record_path):
    """Read ECG signal using wfdb."""
    try:
        record = wfdb.rdrecord(record_path)
        return record.p_signal, record.sig_name
    except Exception as e:
        print(f"Error reading signal from {record_path}: {str(e)}")
        return None, None

def plot_ecg_signals(signals, labels, title, save_path=None, figsize=(20, 10)):
    """Plot ECG signals with proper labels."""
    if not signals:
        print(f"No signals to plot for {title}")
        return
        
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=figsize)
    if n_signals == 1:
        axes = [axes]
    
    colors = cycle(plt.cm.tab10.colors)
    
    for idx, (signal, label) in enumerate(zip(signals, labels)):
        if signal is None:
            continue
        ax = axes[idx]
        for lead in range(signal.shape[1]):
            ax.plot(signal[:, lead], label=f'Lead {lead+1}', color=next(colors))
        ax.set_title(f'ECG Signal - {label}', fontsize=14)
        ax.set_xlabel('Samples', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
    
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_comparison(signals_dict, title, save_path, figsize=(20, 15)):
    """Plot comparison of different classes in a single figure."""
    n_classes = len(signals_dict)
    fig, axes = plt.subplots(n_classes, 1, figsize=figsize)
    if n_classes == 1:
        axes = [axes]
    
    colors = cycle(plt.cm.tab10.colors)
    
    for idx, (class_name, signal) in enumerate(signals_dict.items()):
        if signal is None:
            continue
        ax = axes[idx]
        for lead in range(signal.shape[1]):
            ax.plot(signal[:, lead], label=f'Lead {lead+1}', color=next(colors))
        ax.set_title(f'Class: {class_name}', fontsize=14)
        ax.set_xlabel('Samples', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
    
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def safe_sample(df, n_samples):
    """Safely sample n_samples from dataframe, handling cases where n_samples > len(df)."""
    n_available = len(df)
    if n_available == 0:
        return pd.DataFrame()
    n_to_sample = min(n_samples, n_available)
    return df.sample(n=n_to_sample)

def visualize_binary_classification(data_dir, df, n_samples=3):
    """Visualize normal vs abnormal ECG signals."""
    print("Generating binary classification visualizations...")
    
    # Get normal and abnormal samples
    normal_samples = df[df['scp_codes'].str.contains('NORM', na=False)]
    abnormal_samples = df[~df['scp_codes'].str.contains('NORM', na=False)]
    
    # Read signals for individual plots
    for class_type, samples in [('normal', normal_samples), ('abnormal', abnormal_samples)]:
        signals = []
        labels = []
        
        sampled_data = safe_sample(samples, n_samples)
        for i, (_, row) in enumerate(sampled_data.iterrows()):
            record_path = os.path.join(data_dir, row['filename_hr'])
            signal, _ = read_ecg_signal(record_path)
            if signal is not None:
                signals.append(signal)
                labels.append(f'Sample {i+1}')
        
        if signals:  # Only plot if we have signals
            plot_ecg_signals(signals, labels, 
                            f'Binary Classification: {class_type.capitalize()} ECG Signals',
                            save_path=f'reports/figures/binary/binary_{class_type}.png',
                            figsize=(20, 5*len(signals)))
    
    # Create comparison plot
    comparison_signals = {}
    for class_type, samples in [('Normal', normal_samples), ('Abnormal', abnormal_samples)]:
        sampled_data = safe_sample(samples, 1)  # Just need one sample for comparison
        for _, row in sampled_data.iterrows():
            record_path = os.path.join(data_dir, row['filename_hr'])
            signal, _ = read_ecg_signal(record_path)
            if signal is not None:
                comparison_signals[class_type] = signal
                break
    
    if comparison_signals:  # Only plot if we have signals
        plot_comparison(comparison_signals, 
                       'Binary Classification: Normal vs Abnormal ECG Signals Comparison',
                       'reports/figures/binary/binary_comparison.png')

def visualize_superclasses(data_dir, df, categories, n_samples=2):
    """Visualize ECG signals for each superclass."""
    print("Generating superclass visualizations...")
    
    comparison_signals = {}
    
    for superclass in categories['super']:
        samples = df[df['scp_codes'].str.contains(superclass, na=False)]
        signals = []
        labels = []
        
        sampled_data = safe_sample(samples, n_samples)
        for i, (_, row) in enumerate(sampled_data.iterrows()):
            record_path = os.path.join(data_dir, row['filename_hr'])
            signal, _ = read_ecg_signal(record_path)
            if signal is not None:
                signals.append(signal)
                labels.append(f'Sample {i+1}')
                if superclass not in comparison_signals:
                    comparison_signals[superclass] = signal
        
        if signals:  # Only plot if we have signals
            plot_ecg_signals(signals, labels, 
                            f'Superclass: {superclass}',
                            save_path=f'reports/figures/superclass/superclass_{superclass}.png',
                            figsize=(20, 5*len(signals)))
    
    # Create comparison plot
    if comparison_signals:  # Only plot if we have signals
        plot_comparison(comparison_signals,
                       'Superclass Comparison: All Classes',
                       'reports/figures/superclass/superclass_comparison.png',
                       figsize=(20, 5*len(comparison_signals)))

def visualize_subclasses(data_dir, df, categories, n_samples=2):
    """Visualize ECG signals for each subclass."""
    print("Generating subclass visualizations...")
    
    comparison_signals = {}
    
    for subclass in categories['sub']:
        if pd.isna(subclass):
            continue
            
        samples = df[df['scp_codes'].str.contains(subclass, na=False)]
        signals = []
        labels = []
        
        sampled_data = safe_sample(samples, n_samples)
        for i, (_, row) in enumerate(sampled_data.iterrows()):
            record_path = os.path.join(data_dir, row['filename_hr'])
            signal, _ = read_ecg_signal(record_path)
            if signal is not None:
                signals.append(signal)
                labels.append(f'Sample {i+1}')
                if subclass not in comparison_signals:
                    comparison_signals[subclass] = signal
        
        if signals:  # Only plot if we have signals
            plot_ecg_signals(signals, labels,
                            f'Subclass: {subclass}',
                            save_path=f'reports/figures/subclass/subclass_{subclass}.png',
                            figsize=(20, 5*len(signals)))
    
    # Create comparison plot
    if comparison_signals:  # Only plot if we have signals
        plot_comparison(comparison_signals,
                       'Subclass Comparison: All Classes',
                       'reports/figures/subclass/subclass_comparison.png',
                       figsize=(20, 5*len(comparison_signals)))

def main():
    # Set up paths
    data_dir = 'data/raw/ptb-xl-1.0.3'
    os.makedirs('reports/figures/binary', exist_ok=True)
    os.makedirs('reports/figures/superclass', exist_ok=True)
    os.makedirs('reports/figures/subclass', exist_ok=True)
    
    # Load metadata
    print("Loading metadata...")
    df, scp_df, categories = load_metadata(data_dir)
    
    # Generate visualizations
    visualize_binary_classification(data_dir, df)
    visualize_superclasses(data_dir, df, categories)
    visualize_subclasses(data_dir, df, categories)
    
    print("All visualizations have been generated in the reports/figures directory.")

if __name__ == "__main__":
    main() 