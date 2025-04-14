import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ECGEvaluator:
    """Evaluator for ECG classification models."""

    def __init__(self, num_classes: int, class_names: list):
        """
        Initialize the evaluator.

        Args:
            num_classes: Number of output classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute evaluation metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred_binary),
            "precision": precision_score(
                y_true, y_pred_binary, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred_binary, average="weighted", zero_division=0
            ),
            "f1": f1_score(y_true, y_pred_binary, average="weighted", zero_division=0),
            "auc": roc_auc_score(y_true, y_pred, average="weighted"),
        }

        # Class-wise metrics
        class_precision = precision_score(
            y_true, y_pred_binary, average=None, zero_division=0
        )
        class_recall = recall_score(
            y_true, y_pred_binary, average=None, zero_division=0
        )
        class_f1 = f1_score(y_true, y_pred_binary, average=None, zero_division=0)
        class_auc = []

        for i in range(self.num_classes):
            try:
                class_auc.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            except:
                class_auc.append(0.0)

        metrics["class_precision"] = class_precision.tolist()
        metrics["class_recall"] = class_recall.tolist()
        metrics["class_f1"] = class_f1.tolist()
        metrics["class_auc"] = class_auc

        return metrics

    def save_metrics(self, metrics: dict, save_path: Path) -> None:
        """
        Save metrics to CSV file.

        Args:
            metrics: Dictionary of metrics
            save_path: Path to save the metrics CSV
        """
        # Convert to DataFrame
        metrics_df = pd.DataFrame(
            {
                "Metric": list(metrics.keys()),
                "Value": [str(v) for v in metrics.values()],
            }
        )

        # Save to CSV
        metrics_df.to_csv(save_path, index=False)

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Path = None
    ) -> None:
        """
        Plot and optionally save the confusion matrix.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            save_path: Path to save the confusion matrix plot
        """
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int)

        # For multi-label classification, we create one confusion matrix per class
        for i in range(self.num_classes):
            cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="g",
                cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
            )

            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix for {self.class_names[i]}")

            # Save if path is provided
            if save_path:
                # Create directory if it doesn't exist
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)

                # Generate class-specific filename
                filename = Path(save_path).stem
                extension = Path(save_path).suffix
                class_filename = (
                    f"{filename}_{self.class_names[i].replace(' ', '_')}{extension}"
                )
                class_save_path = save_dir / class_filename

                plt.savefig(class_save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        # Also create an aggregated confusion matrix for an overall view
        # This is simplified and just shows the frequency of correct/incorrect predictions
        aggregated_cm = np.zeros((2, 2))
        for i in range(self.num_classes):
            cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
            aggregated_cm += cm

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            aggregated_cm,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Aggregated Confusion Matrix (All Classes)")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
