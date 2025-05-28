#!/usr/bin/env python3
"""
Preprocess compressed ECG datasets.
This script is adapted from preprocess_ecg.py to work with compressed datasets.
"""

import pandas as pd
import numpy as np
import wfdb
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Union
import json
import logging
from datetime import datetime
import sys
import os
from sklearn.preprocessing import MultiLabelBinarizer


class CompressedECGPreprocessor:
    """Class for preprocessing compressed PTB-XL ECG datasets with multiple lead configurations."""

    LEAD_CONFIGS = {
        "lead-I": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "bipolar-limb": [3, 4, 5, 6, 7, 8, 9, 10, 11],
        "unipolar-limb": [0, 1, 2, 6, 7, 8, 9, 10, 11],
        "limb-leads": [6, 7, 8, 9, 10, 11],
        "precordial-leads": [0, 1, 2, 3, 4, 5],
        "all-leads": [],  # Empty list means use all leads
    }

    def __init__(
        self,
        data_path: str,
        output_path: str,
        sampling_rate: int = 500,
        classification_type: str = "super",
        test_fold: int = 10,
        compression_level: str = "25_percent"
    ):
        """Initialize ECG preprocessor with logging configuration."""
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.sampling_rate = sampling_rate
        self.classification_type = classification_type
        self.test_fold = test_fold
        self.compression_level = compression_level
        self.label_binarizer = MultiLabelBinarizer()

        # Setup logging
        self._setup_logging()

        # Create output directory structure
        self._setup_output_directories()

        # Validate inputs
        self._validate_inputs()

        self.logger.info(
            f"Initialized CompressedECGPreprocessor with: "
            f"classification_type={classification_type}, "
            f"sampling_rate={sampling_rate}, "
            f"test_fold={test_fold}, "
            f"compression_level={compression_level}"
        )

    def _setup_logging(self):
        """Configure logging system."""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Create unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"preprocessing_compressed_{self.compression_level}_{self.classification_type}_{timestamp}.log"

        # Configure logger
        self.logger = logging.getLogger(f"CompressedECGPreprocessor_{self.compression_level}_{self.classification_type}")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _setup_output_directories(self):
        """Create output directory structure for each lead configuration."""
        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            for lead_config in self.LEAD_CONFIGS.keys():
                (self.output_path / lead_config).mkdir(exist_ok=True)
            self.logger.info(f"Created output directories at {self.output_path}")
        except Exception as e:
            self.logger.error(f"Failed to create output directories: {str(e)}")
            raise

    def _validate_inputs(self):
        """Validate input parameters."""
        self.logger.debug("Validating input parameters...")

        if not self.data_path.exists():
            self.logger.error(f"Data path {self.data_path} does not exist")
            raise ValueError(f"Data path {self.data_path} does not exist")

        if self.sampling_rate not in [100, 500]:
            self.logger.error(f"Invalid sampling rate: {self.sampling_rate}")
            raise ValueError("Sampling rate must be either 100 or 500 Hz")

        if self.classification_type not in ["binary", "super", "sub"]:
            self.logger.error(f"Invalid classification type: {self.classification_type}")
            raise ValueError("Invalid classification type")

        if self.compression_level not in ["25_percent", "50_percent", "75_percent"]:
            self.logger.error(f"Invalid compression level: {self.compression_level}")
            raise ValueError("Invalid compression level")

        self.logger.debug("Input validation successful")

    def find_compressed_records(self) -> list:
        """Find all compressed records in the flattened directory structure."""
        self.logger.info("Finding compressed records in flattened structure...")
        
        records = []
        record_prefix = "lr" if self.sampling_rate == 100 else "hr"
        
        # Path to compressed dataset
        compressed_path = self.data_path / f"ptb-xl-{self.compression_level}"
        records_dir = compressed_path / f"records{self.sampling_rate}"
        
        if not records_dir.exists():
            self.logger.error(f"Records directory not found: {records_dir}")
            return records
        
        # Iterate through numbered directories (00000-21000)
        numbered_dirs = sorted([d for d in records_dir.iterdir() 
                               if d.is_dir() and len(d.name) == 5 and d.name.isdigit()])
        
        for dir_path in numbered_dirs:
            # Find all record files in this directory
            record_files = list(dir_path.glob(f"*_{record_prefix}.hea"))
            
            for record_file in record_files:
                record_id = record_file.stem.replace(f"_{record_prefix}", "")
                record_path = str(record_file.parent / f"{record_id}_{record_prefix}")
                
                records.append({
                    'ecg_id': int(record_id),
                    'filename': f"records{self.sampling_rate}/{dir_path.name}/{record_id}_{record_prefix}",
                    'record_path': record_path
                })
        
        self.logger.info(f"Found {len(records)} compressed records")
        return records

    def load_raw_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Load compressed ECG signal data from flattened structure.

        Returns:
            Numpy array of shape (n_samples, sequence_length, n_channels)
        """
        self.logger.info("Loading compressed ECG signal data...")
        
        try:
            data = []
            total_files = len(df)
            expected_shape = None

            for idx, row in enumerate(df.itertuples(), 1):
                if idx % 1000 == 0:
                    self.logger.info(f"Loading file {idx}/{total_files}")
                
                try:
                    # Use the record_path from the dataframe
                    record_path = getattr(row, 'record_path', None)
                    if record_path is None:
                        self.logger.warning(f"No record_path found for row {idx}")
                        continue
                    
                    signal, _ = wfdb.rdsamp(record_path)

                    # Check and store expected shape
                    if expected_shape is None:
                        expected_shape = signal.shape
                        self.logger.info(f"Expected signal shape: {expected_shape}")
                    elif signal.shape != expected_shape:
                        self.logger.warning(
                            f"Inconsistent signal shape for {record_path}: expected {expected_shape}, got {signal.shape}"
                        )
                        # Pad or truncate to match expected shape
                        if signal.shape[0] < expected_shape[0]:
                            # Pad with zeros
                            padding = np.zeros((expected_shape[0] - signal.shape[0], signal.shape[1]))
                            signal = np.vstack([signal, padding])
                        elif signal.shape[0] > expected_shape[0]:
                            # Truncate
                            signal = signal[:expected_shape[0], :]

                    data.append(signal)

                except Exception as e:
                    self.logger.error(f"Failed to load file {record_path}: {str(e)}")
                    continue

            if not data:
                raise ValueError("No data was successfully loaded")

            # Convert to numpy array and ensure 3D shape
            data_array = np.array(data)
            if len(data_array.shape) != 3:
                self.logger.error(f"Unexpected data shape: {data_array.shape}")
                raise ValueError(f"Unexpected data shape: {data_array.shape}")

            self.logger.info(
                f"Successfully loaded {len(data)} compressed ECG recordings with shape {data_array.shape}"
            )
            return data_array

        except Exception as e:
            self.logger.error(f"Failed to load compressed data: {str(e)}")
            raise

    def load_annotations(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process annotation data."""
        self.logger.info("Loading annotation data...")

        try:
            # Load database from compressed dataset
            compressed_path = self.data_path / f"ptb-xl-{self.compression_level}"
            Y = pd.read_csv(compressed_path / "ptbxl_database.csv", index_col="ecg_id")
            Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
            self.logger.debug(f"Loaded {len(Y)} annotations from database")

            # Load SCP statements from original dataset (not available in compressed)
            original_data_path = Path("/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/raw/ptb-xl-1.0.3")
            agg_df = pd.read_csv(original_data_path / "scp_statements.csv", index_col=0)
            agg_df = agg_df[agg_df.diagnostic == 1]
            self.logger.debug(f"Loaded {len(agg_df)} diagnostic statements from original dataset")

            # Find compressed records and merge with annotations
            compressed_records = self.find_compressed_records()
            if not compressed_records:
                raise ValueError("No compressed records found")
            
            # Create dataframe from compressed records
            records_df = pd.DataFrame(compressed_records)
            records_df.set_index('ecg_id', inplace=True)
            
            # Merge annotations with available compressed records
            Y_filtered = Y.loc[Y.index.intersection(records_df.index)].copy()
            Y_filtered = Y_filtered.merge(records_df, left_index=True, right_index=True, how='inner')
            
            self.logger.info(f"Matched {len(Y_filtered)} annotations with compressed records")

            return Y_filtered, agg_df

        except Exception as e:
            self.logger.error(f"Failed to load annotations: {str(e)}")
            raise

    def set_agg_df(self, agg_df: pd.DataFrame):
        """Set the aggregation dataframe for diagnostic statements."""
        self.agg_df = agg_df

    def aggregate_diagnostic(self, y_dic: dict) -> List:
        """Aggregate diagnostic classes based on classification type."""
        if self.classification_type == "sub":
            return self._aggregate_subclass_diagnostic(y_dic)
        return self._aggregate_superclass_diagnostic(y_dic)

    def _aggregate_superclass_diagnostic(self, y_dic: dict) -> List:
        """Aggregate superclass diagnostics."""
        tmp = [
            self.agg_df.loc[key].diagnostic_class
            for key in y_dic.keys()
            if key in self.agg_df.index
        ]
        return list(set(tmp))

    def _aggregate_subclass_diagnostic(self, y_dic: dict) -> List:
        """Aggregate subclass diagnostics."""
        tmp = [
            self.agg_df.loc[key].diagnostic_subclass
            for key in y_dic.keys()
            if key in self.agg_df.index
        ]
        return list(set(tmp))

    def process_labels(self, labels: List) -> np.ndarray:
        """
        Process labels using MultiLabelBinarizer.

        Args:
            labels: List of label lists

        Returns:
            Binary matrix of shape (n_samples, n_classes)
        """
        if self.classification_type == "binary":
            # For binary classification, convert to one-hot encoding (2 classes)
            processed_labels = []
            for label in labels:
                if "NORM" in label:
                    processed_labels.append(1)  # Normal class
                elif any(c in label for c in ["MI", "HYP", "CD", "STTC"]):
                    processed_labels.append(0)  # Abnormal class
                else:
                    processed_labels.append(0)  # Default to abnormal

            # Convert to one-hot encoding (shape: n_samples, 2)
            return np.eye(2)[np.array(processed_labels)]
        else:
            # For multi-label classification (super: 5 classes, sub: 23 classes)
            # MultiLabelBinarizer will automatically create the correct number of columns
            return self.label_binarizer.fit_transform(labels)

    def select_leads(self, data: np.ndarray, lead_indices: List[int]) -> np.ndarray:
        """
        Select specific leads from the ECG data.

        Args:
            data: Input data of shape (n_samples, sequence_length, n_channels)
            lead_indices: Indices of leads to remove

        Returns:
            Processed data with selected leads
        """
        try:
            self.logger.debug(f"Input data shape: {data.shape}")

            if not lead_indices:
                return data

            # Ensure data is 3D
            if len(data.shape) != 3:
                self.logger.error(f"Expected 3D data but got shape {data.shape}")
                raise ValueError(f"Expected 3D data but got shape {data.shape}")

            # Keep only the desired leads
            all_leads = list(range(data.shape[2]))
            keep_leads = [i for i in all_leads if i not in lead_indices]

            processed_data = data[:, :, keep_leads]
            self.logger.debug(f"Output data shape: {processed_data.shape}")

            return processed_data

        except Exception as e:
            self.logger.error(f"Failed to select leads: {str(e)}")
            raise

    def save_metadata(self, lead_config: str, data_shape: tuple):
        """Save metadata about the processed dataset."""
        metadata = {
            "lead_configuration": lead_config,
            "compression_level": self.compression_level,
            "sampling_rate": self.sampling_rate,
            "classification_type": self.classification_type,
            "data_shape": {
                "n_samples": data_shape[0],
                "sequence_length": data_shape[1],
                "n_channels": data_shape[2] if len(data_shape) > 2 else 1,
            },
            "test_fold": self.test_fold,
            "source_path": str(self.data_path),
            "processing_timestamp": datetime.now().isoformat()
        }

        with open(self.output_path / lead_config / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def process_lead_configuration(
        self, X: np.ndarray, Y: pd.DataFrame, lead_config: str
    ):
        """Process data for a specific lead configuration."""
        self.logger.info(f"Processing {lead_config} configuration...")

        try:
            # Split data
            train_mask = Y.strat_fold != self.test_fold
            test_mask = Y.strat_fold == self.test_fold

            # Select leads and prepare features
            self.logger.debug(f"Selecting leads for {lead_config}")
            X_train = self.select_leads(X[train_mask], self.LEAD_CONFIGS[lead_config])
            X_test = self.select_leads(X[test_mask], self.LEAD_CONFIGS[lead_config])

            # Prepare labels
            y_train = Y[train_mask].diagnostic_superclass.tolist()
            y_test = Y[test_mask].diagnostic_superclass.tolist()

            # Process labels
            y_train = self.process_labels(y_train)
            y_test = self.process_labels(y_test)

            # Save data
            output_dir = self.output_path / lead_config
            self.logger.info(f"Saving processed data to {output_dir}")

            np.save(output_dir / "x_train.npy", X_train)
            np.save(output_dir / "x_test.npy", X_test)
            np.save(output_dir / "y_train.npy", y_train)
            np.save(output_dir / "y_test.npy", y_test)

            # Save label classes for reference
            if self.classification_type != "binary":
                classes = self.label_binarizer.classes_
                np.save(output_dir / "label_classes.npy", classes)

            # Save metadata
            self.save_metadata(lead_config, X_train.shape)

            # Log summary statistics
            self.logger.info(f"Training data shape: {X_train.shape}")
            self.logger.info(f"Test data shape: {X_test.shape}")
            self.logger.info(f"Training labels shape: {y_train.shape}")
            self.logger.info(f"Test labels shape: {y_test.shape}")

        except Exception as e:
            self.logger.error(f"Failed to process {lead_config}: {str(e)}")
            raise

    def preprocess(self, X=None, Y=None):
        """
        Main preprocessing function that processes all lead configurations.

        Args:
            X: Optional pre-loaded raw data
            Y: Optional pre-loaded annotations
        """
        self.logger.info(
            f"Starting preprocessing pipeline for {self.compression_level} compressed dataset with {self.classification_type} classification"
        )

        try:
            # Load data only if not provided
            if Y is None:
                Y, agg_df = self.load_annotations()
                self.logger.info("Loaded annotations successfully")
                self.set_agg_df(agg_df)

            if X is None:
                X = self.load_raw_data(Y)
                self.logger.info("Loaded compressed data successfully")

            # Process diagnostics
            self.logger.info("Processing diagnostics")
            Y = Y.copy()  # Create a copy to avoid modifying the original
            Y["diagnostic_superclass"] = Y.scp_codes.apply(self.aggregate_diagnostic)

            # Process each lead configuration
            for lead_config in self.LEAD_CONFIGS.keys():
                self.process_lead_configuration(X, Y, lead_config)

            self.logger.info("Preprocessing completed for all lead configurations!")

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise


def main():
    """Main function to process all compression levels, sampling rates, and classification types."""
    try:
        # Configuration with absolute paths
        base_path = Path("/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/compressed")
        output_base_path = Path("/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/processed/ptb-xl-compressed")
        sampling_rates = [100]  # Only process 100Hz
        classification_types = ["binary", "super", "sub"]
        compression_levels = ["25_percent", "50_percent", "75_percent"]

        print("Starting preprocessing of compressed datasets...")
        print(f"Base path: {base_path}")
        print(f"Output path: {output_base_path}")
        print(f"Sampling rates: {sampling_rates}")
        print(f"Classification types: {classification_types}")
        print(f"Compression levels: {compression_levels}")

        # Process each combination
        for level in compression_levels:
            for sampling_rate in sampling_rates:
                # Load data once for all classification types
                print(f"\nLoading data for {level} at {sampling_rate}Hz...")
                
                # Initialize base preprocessor to load data
                base_preprocessor = CompressedECGPreprocessor(
                    data_path=base_path,
                    output_path=output_base_path / level / "base",
                    sampling_rate=sampling_rate,
                    classification_type="super",  # Doesn't matter for loading
                    compression_level=level,
                )

                try:
                    # Load raw data and annotations once
                    Y, agg_df = base_preprocessor.load_annotations()
                    X = base_preprocessor.load_raw_data(Y)
                    base_preprocessor.logger.info(
                        f"Data loading completed for {level} at {sampling_rate}Hz. Starting processing for each classification type..."
                    )

                    # Process for each classification type using the loaded data
                    for classification_type in classification_types:
                        try:
                            print(f"Processing {level} at {sampling_rate}Hz with {classification_type} classification...")

                            preprocessor = CompressedECGPreprocessor(
                                data_path=base_path,
                                output_path=output_base_path / level / classification_type,
                                sampling_rate=sampling_rate,
                                classification_type=classification_type,
                                compression_level=level,
                            )

                            # Set the aggregation dataframe
                            preprocessor.set_agg_df(agg_df)

                            # Skip data loading and use the already loaded data
                            preprocessor.preprocess(X=X, Y=Y)

                        except Exception as e:
                            print(f"Failed to process {level} at {sampling_rate}Hz with {classification_type}: {str(e)}")
                            continue

                except Exception as e:
                    print(f"Failed to load data for {level} at {sampling_rate}Hz: {str(e)}")
                    continue

        print("\nAll preprocessing completed!")

    except Exception as e:
        print(f"Failed to initialize preprocessing: {str(e)}")


if __name__ == "__main__":
    main() 