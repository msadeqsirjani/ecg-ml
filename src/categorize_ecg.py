import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import wfdb
import os
import math
import json
from tqdm import tqdm

def get_signal_files(record_name):
    """Get all files associated with a record."""
    return [f"{record_name}.hea", f"{record_name}.dat"]

def create_structured_directory(base_dir, record_id):
    """Create structured directory path (e.g., 00000 for 0-999, 01000 for 1000-1999)."""
    folder_num = math.floor(record_id / 1000) * 1000
    folder_name = f"{folder_num:05d}"
    dir_path = base_dir / folder_name
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def setup_directory_structure(base_path, categories):
    """Create numbered subdirectories from 00000 to 21000 with category subdirs."""
    print("\nCreating directory structure...")
    for i in tqdm(range(0, 22000, 1000), desc="Creating folders"):
        folder_name = f"{i:05d}"
        folder_path = base_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Create category subdirectories
        for category_type in categories.keys():
            category_path = folder_path / category_type
            category_path.mkdir(parents=True, exist_ok=True)
            
            # For binary and superclass categories, create their subdirectories
            if category_type in ['binary', 'super', 'sub']:
                for cat in categories[category_type]:
                    (category_path / cat).mkdir(parents=True, exist_ok=True)

def process_signals(data_path, output_base_path, sampling_rate, categories=None):
    """Process signals for a specific sampling rate."""
    print(f"\nProcessing {sampling_rate}Hz signals...")
    
    # Create output directory
    records_dir = Path(output_base_path) / f"records{sampling_rate}"
    records_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PTB-XL data
    try:
        df = pd.read_csv(data_path / 'ptbxl_database.csv', index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(eval)
        print(f"Loaded {len(df)} records from database")
    except FileNotFoundError:
        print(f"Error: Could not find ptbxl_database.csv in {data_path}")
        return
    
    # Load diagnostic classes
    try:
        agg_df = pd.read_csv(data_path / 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
    except FileNotFoundError:
        print(f"Error: Could not find scp_statements.csv in {data_path}")
        return
    
    # Set default categories if none provided
    if categories is None:
        categories = {
            'binary': ['normal', 'abnormal'],
            'super': ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
            'sub': list(agg_df.index)  # Include all diagnostic codes
        }
    
    # Create directory structure
    setup_directory_structure(records_dir, categories)
    
    # Function to get diagnostic superclass
    def get_diagnostic_superclass(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    # Process and store signals for each category type
    df['diagnostic_superclass'] = df.scp_codes.apply(get_diagnostic_superclass)
    
    # Initialize signal counts
    signal_counts = {
        'binary': {'normal': 0, 'abnormal': 0},
        'super': {cat: 0 for cat in categories['super']},
        'sub': {cat: 0 for cat in categories['sub']}  # Count for each diagnostic code
    }
    
    # Create a copy of the DataFrame to store updated paths
    updated_df = df.copy()
    
    # Process each record
    print("\nProcessing records and copying files...")
    for record_id, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {sampling_rate}Hz records"):
        # Get record name and folder
        full_filename = row['filename_lr'] if sampling_rate == 100 else row['filename_hr']
        folder_num = math.floor(record_id / 1000) * 1000
        folder_name = f"{folder_num:05d}"
        
        # Get source file path
        src_path = data_path / full_filename
        
        # Process binary classification - Check if NORM is in superclass
        is_normal = 'NORM' in row['diagnostic_superclass']
        binary_category = 'normal' if is_normal else 'abnormal'
        
        # Copy files to binary category folder
        binary_target_dir = records_dir / folder_name / 'binary' / binary_category
        binary_target_dir.mkdir(parents=True, exist_ok=True)
        
        # Update signal counts
        signal_counts['binary'][binary_category] += 1
        
        # Copy files to superclass folders
        for superclass in row['diagnostic_superclass']:
            if superclass in categories['super']:
                super_target_dir = records_dir / folder_name / 'super' / superclass
                super_target_dir.mkdir(parents=True, exist_ok=True)
                signal_counts['super'][superclass] += 1
        
        # Copy files to subclass folders
        for subclass, value in row['scp_codes'].items():
            if subclass in categories['sub']:
                sub_target_dir = records_dir / folder_name / 'sub' / subclass
                sub_target_dir.mkdir(parents=True, exist_ok=True)
                signal_counts['sub'][subclass] += 1
                
                # Copy the files
                for ext in ['.hea', '.dat']:
                    src_file = data_path / f"{full_filename}{ext}"
                    if src_file.exists():
                        shutil.copy2(src_file, sub_target_dir)
        
        # Copy the files for binary and superclass
        for ext in ['.hea', '.dat']:
            src_file = data_path / f"{full_filename}{ext}"
            if src_file.exists():
                # Copy to binary category
                shutil.copy2(src_file, binary_target_dir)
                
                # Copy to superclass categories
                for superclass in row['diagnostic_superclass']:
                    if superclass in categories['super']:
                        super_target_dir = records_dir / folder_name / 'super' / superclass
                        shutil.copy2(src_file, super_target_dir)
        
        # Update paths in the DataFrame
        if sampling_rate == 100:
            updated_df.at[record_id, 'filename_lr'] = str(binary_target_dir / os.path.basename(full_filename))
        else:
            updated_df.at[record_id, 'filename_hr'] = str(binary_target_dir / os.path.basename(full_filename))
    
    # Save category information for each folder
    print("\nSaving category information...")
    for i in tqdm(range(0, 22000, 1000), desc="Saving category info"):
        folder_name = f"{i:05d}"
        folder_path = records_dir / folder_name
        
        for category_type in categories.keys():
            category_info = {
                'description': f"{category_type} classification ({sampling_rate}Hz)",
                'categories': categories[category_type],
                'signal_counts': signal_counts[category_type]
            }
            
            with open(folder_path / category_type / 'category_info.json', 'w') as f:
                json.dump(category_info, f, indent=4)
    
    print(f"\nClassification summary for {sampling_rate}Hz:")
    for category_type, counts in signal_counts.items():
        print(f"\n{category_type.upper()} Classification:")
        for cat, count in counts.items():
            print(f"{cat}: {count} signals")
    
    return updated_df

def categorize_and_store_signals(data_path, output_path, categories=None):
    """
    Categorize and store ECG signals for both 100Hz and 500Hz sampling rates.
    """
    # Convert paths to absolute paths
    data_path = Path(data_path).resolve()
    output_path = Path(output_path).resolve()
    
    # Process both sampling rates and get updated DataFrames
    df_100hz = process_signals(data_path, output_path, 100, categories)
    df_500hz = process_signals(data_path, output_path, 500, categories)
    
    # Merge the updated paths
    final_df = df_100hz.copy()
    final_df['filename_hr'] = df_500hz['filename_hr']
    
    # Save the updated database CSV
    output_csv_path = output_path / 'ptbxl_database.csv'
    final_df.to_csv(output_csv_path)
    print(f"\nUpdated database saved to: {output_csv_path}")
    
    print("\nProcessing completed for both sampling rates!")

if __name__ == "__main__":
    # Example usage
    data_path = "data/raw/ptb-xl-1.0.3"  # Path to the PTB-XL dataset
    output_path = "data/categorized"  # Path where categorized data will be stored
    
    # Process both sampling rates with default categories
    categorize_and_store_signals(data_path, output_path)