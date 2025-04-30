#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wfdb
from pathlib import Path


class ECGPlotter:
    """Class for plotting ECG signals with and without missing values."""

    def __init__(self, raw_data_path: str, missing_data_path: str):
        """Initialize plotter with paths to raw and missing value datasets."""
        self.raw_data_path = Path(raw_data_path)
        self.missing_data_path = Path(missing_data_path)

    def load_signals(self, record_id: str, sampling_rate: int = 500):
        """Load both raw and missing value ECG signals."""
        # Construct paths
        folder_num = f"{(int(record_id) // 1000) * 1000:05d}"
        suffix = "_lr" if sampling_rate == 100 else "_hr"

        # Load raw signal
        raw_path = (
            self.raw_data_path
            / f"records{sampling_rate}"
            / folder_num
            / f"{record_id}{suffix}"
        )
        raw_signal, raw_header = wfdb.rdsamp(str(raw_path))

        # Load missing value signal
        missing_path = (
            self.missing_data_path
            / f"records{sampling_rate}"
            / folder_num
            / f"{record_id}{suffix}"
        )
        missing_signal, missing_header = wfdb.rdsamp(str(missing_path))

        return raw_signal, raw_header, missing_signal, missing_header

    def plot_comparison(
        self,
        raw_signal: np.ndarray,
        raw_header: dict,
        missing_signal: np.ndarray,
        missing_header: dict,
        save_path: str = None,
        show_plot: bool = True,
    ):
        """
        Plot comparison between raw and missing value ECG signals.

        Args:
            raw_signal: Raw ECG signal data
            raw_header: Raw signal header information
            missing_signal: Missing value ECG signal data
            missing_header: Missing value signal header information
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Calculate time vector
        fs = raw_header["fs"]
        time = np.arange(raw_signal.shape[0]) / fs

        # Plot settings
        num_channels = raw_signal.shape[1]
        colors = plt.cm.tab10.colors
        channel_names = raw_header["sig_name"]

        # Plot raw signal
        for i in range(num_channels):
            offset = i * max(2, 1.5 * np.max(np.abs(raw_signal)))
            ax1.plot(
                time,
                raw_signal[:, i] + offset,
                color=colors[i % len(colors)],
                label=channel_names[i],
                linewidth=0.8,
            )

        # Plot missing value signal
        for i in range(num_channels):
            offset = i * max(2, 1.5 * np.max(np.abs(missing_signal)))
            ax2.plot(
                time,
                missing_signal[:, i] + offset,
                color=colors[i % len(colors)],
                label=channel_names[i],
                linewidth=0.8,
            )

        # Highlight missing regions (where signal is zero)
        for i in range(num_channels):
            missing_regions = np.where(missing_signal[:, i] == 0)[0]
            if len(missing_regions) > 0:
                for start, end in zip(missing_regions[:-1], missing_regions[1:]):
                    if end - start > 1:  # Only highlight regions with multiple zeros
                        ax2.axvspan(time[start], time[end], color="lightgray", alpha=0.2)

        # Add labels and titles
        ax1.set_title("Raw ECG Signal")
        ax2.set_title("ECG Signal with Missing Values")
        ax2.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (mV)")
        ax2.set_ylabel("Amplitude (mV)")

        # Add legends
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")

        # Add missing value statistics
        total_samples = len(missing_signal)
        missing_samples = np.sum(missing_signal == 0)
        missing_percentage = (missing_samples / total_samples) * 100

        stats_text = f"Missing Values: {missing_percentage:.2f}% of signal"
        ax2.text(
            0.02,
            0.02,
            stats_text,
            transform=ax2.transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        plt.tight_layout()

        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        if show_plot:
            plt.show()

        plt.close()


# Example usage
if __name__ == "__main__":
    # Initialize plotter with paths to both datasets
    plotter = ECGPlotter(
        raw_data_path="data/raw/ptb-xl-1.0.3",
        missing_data_path="data/raw/ptb-xl-missing-values",
    )

    # Load both signals
    raw_signal, raw_header, missing_signal, missing_header = plotter.load_signals(
        record_id="00001", sampling_rate=500
    )

    # Plot comparison
    plotter.plot_comparison(
        raw_signal, raw_header, missing_signal, missing_header, show_plot=True
    )
