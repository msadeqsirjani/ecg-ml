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
        Add missing values to ECG signal at regular intervals within a randomly selected range.
        
        This method:
        1. Randomly selects start and end points in the signal
        2. Ensures the range can accommodate at least 12 periods of 0.5s missing values
        3. Applies 0.5-second missing periods regularly within that range
        
        Args:
            signal: Input signal of shape (sequence_length, n_channels)
            sampling_rate: Sampling rate of the signal in Hz

        Returns:
            Tuple of (processed_signal, start_times, durations, period_values, range_start_time, range_duration)
        """
        # Create copy of signal
        processed_signal = signal.copy()
        
        # Calculate total number of samples
        total_samples = signal.shape[0]
        
        # Duration of each missing segment
        missing_duration = 0.5  # 500ms missing duration
        missing_samples = int(missing_duration * sampling_rate)
        
        # Period between missing segments (fixed at 0.5 seconds)
        period_seconds = 0.5  # 0.5-second period between missing values
        period_samples = int(period_seconds * sampling_rate)
        
        # Each complete cycle is: missing segment + period between
        cycle_samples = missing_samples + period_samples
        
        # Calculate required samples for 12 missing periods
        required_samples = 12 * cycle_samples
        
        # Ensure the signal is long enough for at least 12 periods
        if total_samples < required_samples:
            self.logger.warning(
                f"Signal too short ({total_samples} samples) for 12 missing periods (need {required_samples})"
            )
            return signal, [], [], [], None, None
        
        # Randomly select range start and end, ensuring it's at least 20% of the signal
        min_range_percent = 0.2
        min_range_samples = max(required_samples, int(total_samples * min_range_percent))
        max_range_samples = total_samples  # Could be limited to a percentage if needed
        
        # Generate random range size between min and max
        range_samples = np.random.randint(min_range_samples, max_range_samples)
        
        # Select random start point, ensuring the range fits within signal
        max_start = total_samples - range_samples
        start_sample = np.random.randint(0, max_start) if max_start > 0 else 0
        
        # Calculate end point
        end_sample = start_sample + range_samples
        
        # Calculate how many missing periods we can fit in this range
        possible_periods = range_samples // cycle_samples
        
        # Ensure we have at least 12 periods
        if possible_periods < 12:
            self.logger.warning(
                f"Selected range ({range_samples} samples) can only fit {possible_periods} periods"
            )
            # Adjust end point to ensure 12 periods
            end_sample = start_sample + (12 * cycle_samples)
            possible_periods = 12
        
        self.logger.info(
            f"Adding {possible_periods} missing periods within range: {start_sample/sampling_rate:.2f}s - {end_sample/sampling_rate:.2f}s"
        )
        
        # Track start times and durations
        start_times = []
        durations = []
        period_values = []
        
        try:
            # Apply missing values at regular intervals within the selected range
            current_pos = start_sample
            
            for i in range(possible_periods):
                # Set values to zero for this period
                processed_signal[current_pos:current_pos + missing_samples, :] = 0
                
                # Record start time, duration and period
                start_times.append(current_pos / sampling_rate)
                durations.append(missing_duration)
                period_values.append(period_seconds)
                
                # Move to next position
                current_pos += cycle_samples
                
                # Stop if we exceed the end point
                if current_pos + missing_samples > end_sample:
                    break
            
            # Store the overall range information
            range_start_time = start_sample / sampling_rate
            range_end_time = end_sample / sampling_rate
            range_duration = range_end_time - range_start_time
            
            return processed_signal, start_times, durations, period_values, range_start_time, range_duration

        except Exception as e:
            self.logger.error(f"Error in modifying signal: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return signal, [], [], [], None, None

    def save_signal(
        self,
        signal: np.ndarray,
        start_time,
        duration,
        output_path: Path,
        sampling_rate: int,
    ):
        """
        Save signal and header in WFDB format.
        
        Args:
            signal: Processed ECG signal
            start_time: Start time of missing values (can be string for multiple periods)
            duration: Duration of missing values
            output_path: Path to save the signal
            sampling_rate: Sampling rate of the signal in Hz
        """
        # Convert output_path to Path object if it's a string
        if isinstance(output_path, str):
            output_path = Path(output_path)

        # Extract record name and directory path
        record_name = output_path.name
        write_dir = str(output_path.parent)

        # Create comments about missing values
        if isinstance(start_time, str):
            # If start_time is a string, it contains information about multiple periods
            comments = [
                f"Missing Values: {start_time}",
                f"Sampling Rate: {sampling_rate}Hz",
                f"Each missing period: {duration}s duration"
            ]
        else:
            # For backward compatibility with single missing period
            comments = [
                f"Missing Values: segment from {start_time:.2f}s for {duration}s",
                f"Sampling Rate: {sampling_rate}Hz",
            ]

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
            comments=comments,
        )

    def process_dataset(self):
        """Process PTB-XL dataset by adding periodic missing values to each signal."""
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
                        result = self.add_missing_values(signal, sampling_rate)
                        noisy_signal, start_times, durations, period_values, range_start_time, range_duration = result

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

                        # Save signal and header with range information
                        if range_start_time is not None:
                            missing_desc = f"multiple ({len(start_times)} periods) within range {range_start_time:.2f}s - {range_start_time + range_duration:.2f}s"
                        else:
                            missing_desc = f"multiple ({len(start_times)} periods)"
                            
                        self.save_signal(
                            noisy_signal,
                            missing_desc,
                            durations[0] if durations else 0,
                            output_path,
                            sampling_rate,
                        )

                        # Store all missing value periods in metadata
                        if start_times and len(start_times) > 0:
                            for i, (start_time, duration) in enumerate(zip(start_times, durations)):
                                missing_info.append(
                                    {
                                        "ecg_id": ecg_id,
                                        "sampling_rate": sampling_rate,
                                        "start_time": start_time,
                                        "duration": duration,
                                        "period_index": i,
                                        "period_interval": period_values[i] if i < len(period_values) else 0.5,
                                        "range_start": range_start_time,
                                        "range_duration": range_duration
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
