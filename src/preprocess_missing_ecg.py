import numpy as np
import wfdb
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import ast
import sys
import shutil
import os


class MissingValueProcessor:
    """Class for adding random missing values to ECG signals."""

    def __init__(self, data_path: str, output_path: str):
        """Initialize missing value processor with logging configuration."""
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)

        # Setup logging
        self._setup_logging()

        # Create output directory structure
        self._setup_output_directory()

        # Validate inputs
        self._validate_inputs()

        self.logger.info(f"Initialized MissingValueProcessor")

    def _setup_logging(self):
        """Configure logging system."""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Create unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"missing_value_{timestamp}.log"

        # Configure logger
        self.logger = logging.getLogger("MissingValueProcessor")
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

    def _setup_output_directory(self):
        """Create output directory structure matching PTB-XL."""
        try:
            # Create main output directory
            self.output_path.mkdir(parents=True, exist_ok=True)

            # Create records100 and records500 directories
            for rate in [100, 500]:
                records_dir = self.output_path / f"records{rate}"
                records_dir.mkdir(exist_ok=True)

                # Create subdirectories 00000, 01000, 02000, ..., 21000
                for i in range(0, 22000, 1000):
                    subdir = records_dir / f"{i:05d}"
                    subdir.mkdir(exist_ok=True)

            # Copy database files
            shutil.copy2(self.data_path / "ptbxl_database.csv", self.output_path)
            shutil.copy2(self.data_path / "scp_statements.csv", self.output_path)

            self.logger.info(
                f"Created output directory structure at {self.output_path}"
            )

        except Exception as e:
            self.logger.error(f"Failed to create output directory structure: {str(e)}")
            raise

    def _validate_inputs(self):
        """Validate input parameters."""
        self.logger.debug("Validating input parameters...")

        if not self.data_path.exists():
            self.logger.error(f"Data path {self.data_path} does not exist")
            raise ValueError(f"Data path {self.data_path} does not exist")

        self.logger.debug("Input validation successful")

    def add_missing_values(self, signal: np.ndarray, sampling_rate: int) -> tuple:
        """
        Add random missing values to ECG signal.

        Args:
            signal: Input signal of shape (sequence_length, n_channels)
            sampling_rate: Sampling rate of the signal in Hz

        Returns:
            Tuple of (processed_signal, start_time, duration)
        """
        # Random duration between 1 and 9 seconds
        missing_duration = np.random.randint(1, 10)

        # Calculate total samples and samples to remove
        total_samples = signal.shape[0]
        samples_to_remove = missing_duration * sampling_rate

        # Ensure we don't try to remove more samples than we have
        if samples_to_remove >= total_samples:
            self.logger.warning(
                f"Missing duration ({missing_duration}s) is too long for signal length ({total_samples/sampling_rate}s)"
            )
            return signal, None, None

        # Random start point for missing sequence
        max_start = total_samples - samples_to_remove
        start_idx = np.random.randint(0, max_start)
        end_idx = start_idx + samples_to_remove

        try:
            # Create copy and insert missing values
            processed_signal = signal.copy()
            # Use a specific value instead of NaN (e.g., 0 or the minimum value - 1)
            missing_value = (
                0  # or you could use 'np.min(signal) - 1' or another specific value
            )
            processed_signal[start_idx:end_idx, :] = missing_value

            # Calculate start time in seconds
            start_time = start_idx / sampling_rate

            return processed_signal, start_time, missing_duration

        except Exception as e:
            self.logger.error(f"Error in modifying signal: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return signal, None, None

    def save_signal(
        self,
        signal: np.ndarray,
        start_time: float,
        duration: int,
        output_path: Path,
        sampling_rate: int,
    ):
        """Save signal and header in WFDB format."""
        # Convert output_path to Path object if it's a string
        if isinstance(output_path, str):
            output_path = Path(output_path)

        # Extract record name and directory path
        record_name = output_path.name
        write_dir = str(output_path.parent)

        # Save signal data
        wfdb.wrsamp(
            record_name=record_name,
            write_dir=write_dir,
            fs=sampling_rate,
            units=["mV"] * signal.shape[1],
            sig_name=[
                "I",
                "II",
                "III",
                "aVR",
                "aVL",
                "aVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ][: signal.shape[1]],
            p_signal=signal,
            comments=[
                f"Missing Values: segment from {start_time:.2f}s for {duration}s",
                f"Sampling Rate: {sampling_rate}Hz",
            ],
        )

    def process_dataset(self):
        """Process PTB-XL dataset by adding random missing values to each signal."""
        try:
            # Load database
            self.logger.info("Loading database...")
            Y = pd.read_csv(self.data_path / "ptbxl_database.csv", index_col="ecg_id")
            Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

            # Process each signal
            total_files = len(Y)
            missing_info = []

            sampling_rates = [100, 500]

            self.logger.info(f"Processing {total_files} signals...")
            for idx, (ecg_id, row) in enumerate(Y.iterrows(), 1):
                try:
                    # Process both sampling rates
                    for sampling_rate in sampling_rates:
                        # Get filename column based on sampling rate
                        filename_col = (
                            "filename_lr" if sampling_rate == 100 else "filename_hr"
                        )

                        # Load signal and header
                        signal, header = wfdb.rdsamp(
                            str(self.data_path / row[filename_col])
                        )

                        # Add missing values
                        noisy_signal, start_time, duration = self.add_missing_values(
                            signal, sampling_rate
                        )

                        # Prepare output path maintaining PTB-XL structure
                        record_num = f"{ecg_id:05d}"
                        # Calculate the folder number (00000, 01000, etc.)
                        folder_num = f"{(ecg_id // 1000) * 1000:05d}"
                        output_dir = (
                            self.output_path / f"records{sampling_rate}" / folder_num
                        )

                        # Create the output directory if it doesn't exist
                        output_dir.mkdir(parents=True, exist_ok=True)

                        # Set the output path with the correct suffix
                        suffix = "_lr" if sampling_rate == 100 else "_hr"
                        output_path = output_dir / f"{record_num}{suffix}"

                        # Save signal and header
                        self.save_signal(
                            noisy_signal,
                            start_time,
                            duration,
                            output_path,
                            sampling_rate,
                        )

                        # Store missing value information
                        if start_time is not None:
                            missing_info.append(
                                {
                                    "ecg_id": ecg_id,
                                    "sampling_rate": sampling_rate,
                                    "start_time": start_time,
                                    "duration": duration,
                                }
                            )

                    if idx % 100 == 0:
                        self.logger.info(f"Processed {idx}/{total_files} signals")

                except Exception as e:
                    self.logger.error(f"Error processing signal {ecg_id}: {str(e)}")
                    continue

            # Save missing value information
            missing_df = pd.DataFrame(missing_info)
            missing_df.to_csv(self.output_path / "missing_value_info.csv", index=False)

            self.logger.info("Processing completed!")
            self.logger.info(f"Processed signals saved to: {self.output_path}")
            self.logger.info(
                f"Missing value information saved to: {self.output_path}/missing_value_info.csv"
            )

        except Exception as e:
            self.logger.error(f"Failed to process dataset: {str(e)}")
            raise


def main():
    """Main function to add missing values to PTB-XL dataset."""
    try:
        # Initialize processor
        processor = MissingValueProcessor(
            data_path="data/raw/ptb-xl-1.0.3",
            output_path=f"data/raw/ptb-xl-missing-values",
        )
        processor.process_dataset()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
