import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics, callbacks
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt

from model.ecg_model import create_ecg_model
from model.micro_ecg_model import (
    create_micro_ecg_model,
)
from evaluation.metrics import ECGEvaluator

# Configuration
CLASSIFICATION_TYPES = {"binary": 2, "super": 5, "sub": 23}

LEAD_CONFIGS = {
    "lead-I": 1,
    "bipolar-limb": 3,
    "unipolar-limb": 3,
    "limb-leads": 6,
    "precordial-leads": 6,
    "all-leads": 12,
}

def setup_logging(classification_type: str, lead_config: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_missing_{classification_type}_{lead_config}_{timestamp}.log"

    logger = logging.getLogger(f"ECGMissingTraining_{classification_type}_{lead_config}")
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def load_data(data_path: str, classification_type: str, lead_config: str) -> tuple:
    """Load and preprocess data with missing leads."""
    data_path = Path(data_path)
    processed_path = (
        data_path / "processed" / "ptb-xl-missing-values" / classification_type / lead_config
    )

    # Load data
    x_train = np.load(processed_path / "x_train.npy", allow_pickle=True)
    x_test = np.load(processed_path / "x_test.npy", allow_pickle=True)
    y_train = np.load(processed_path / "y_train.npy", allow_pickle=True)
    y_test = np.load(processed_path / "y_test.npy", allow_pickle=True)

    # Reshape data
    x_train = x_train.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)

    # Add channel dimension
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # Simulate missing leads by randomly masking some leads
    def create_missing_data(x, missing_rate=0.2):
        mask = np.random.random(x.shape) > missing_rate
        return x * mask

    x_train_missing = create_missing_data(x_train)
    x_test_missing = create_missing_data(x_test)

    return x_train_missing, x_test_missing, y_train, y_test

def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classification_type: str,
    lead_config: str,
    model_type: str,
    logger: logging.Logger,
) -> tuple:
    """Train the model with missing leads and return the trained model and history."""
    # Create model
    logger.info(f"Input data shape: {x_train.shape}")
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    logger.info(f"Model input shape: {input_shape}")
    num_classes = CLASSIFICATION_TYPES[classification_type]
    logger.info(f"Number of output classes: {num_classes}")

    # Create appropriate model based on type
    if model_type == "micro":
        model = create_micro_ecg_model(input_shape, num_classes)
    else:
        model = create_ecg_model(input_shape, num_classes)

    # Compile model with modified loss to handle missing data
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy(), metrics.AUC(curve="ROC", multi_label=True)],
    )

    # Callbacks with modified patience for missing data
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4),
    ]

    # Train model
    logger.info("Starting model training with missing leads...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.12,
        epochs=100,
        batch_size=32,
        callbacks=callbacks_list,
    )

    return model, history

def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classification_type: str,
    lead_config: str,
    model_type: str,
    logger: logging.Logger,
    results_dir: Path,
) -> dict:
    """Evaluate the model with missing leads and save results."""
    # Create evaluator
    evaluator = ECGEvaluator(
        num_classes=CLASSIFICATION_TYPES[classification_type],
        class_names=[
            f"Class {i}" for i in range(CLASSIFICATION_TYPES[classification_type])
        ],
    )

    # Get predictions
    y_pred = model.predict(x_test)

    # Compute metrics
    metrics = evaluator.compute_metrics(y_test, y_pred)

    # Log metrics
    logger.info("Test Metrics (with missing leads):")
    for metric_name, value in metrics.items():
        if isinstance(value, list):
            logger.info(f"{metric_name}: {[f'{v:.2f}' for v in value]}")
        else:
            logger.info(f"{metric_name}: {value:.2f}")

    # Generate and save confusion matrix
    logger.info("Generating confusion matrix...")
    # Save in the results directory
    evaluator.plot_confusion_matrix(
        y_test, y_pred, save_path=results_dir / "images" / "confusion_matrix_missing.png"
    )
    logger.info(f"Confusion matrix saved to {results_dir / 'images' / 'confusion_matrix_missing.png'}")
    
    # Save in the images directory
    images_dir = Path(results_dir / "images")
    images_dir.mkdir(exist_ok=True)
    model_config = f"{classification_type}_{lead_config}_{model_type}_missing"
    confusion_matrix_path = images_dir / f"{model_config}_confusion_matrix.png"
    evaluator.plot_confusion_matrix(
        y_test, y_pred, save_path=confusion_matrix_path
    )
    logger.info(f"Confusion matrix also saved to {confusion_matrix_path}")

    return metrics

def save_model_info(model: tf.keras.Model, model_dir: Path, logger: logging.Logger):
    """Save model information."""
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model architecture
    architecture_file = model_dir / "model_architecture_missing.json"
    with open(architecture_file, "w") as f:
        f.write(model.to_json())
    logger.info(f"Model architecture saved to {architecture_file}")

    # Save model summary
    summary_file = model_dir / "model_summary_missing.txt"
    with open(summary_file, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    logger.info(f"Model summary saved to {summary_file}")

def save_training_history(history, results_dir: Path):
    """Save training history and plots."""
    history_dir = results_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    # Save history as JSON
    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    with open(history_dir / "training_history_missing.json", "w") as f:
        json.dump(history_dict, f, indent=4)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss (with Missing Leads)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["binary_accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_binary_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy (with Missing Leads)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(history_dir / "training_curves_missing.png")
    plt.close()

def main():
    """Main function to run the training pipeline."""
    # Setup paths
    data_path = "data"
    results_base_dir = Path("results", "missing-values")
    results_base_dir.mkdir(exist_ok=True)

    # Training configurations
    configs = [
        ("binary", "all-leads", "standard"),
        ("super", "all-leads", "standard"),
        ("sub", "all-leads", "standard"),
    ]

    for classification_type, lead_config, model_type in configs:
        print(f"\nTraining {classification_type} classification with {lead_config}")
        
        # Setup results directory
        results_dir = results_base_dir / f"{classification_type}_{lead_config}_missing"
        results_dir.mkdir(exist_ok=True)
        (results_dir / "images").mkdir(exist_ok=True)
        
        # Setup logging
        logger = setup_logging(classification_type, lead_config)
        
        try:
            # Load data
            x_train, x_test, y_train, y_test = load_data(
                data_path, classification_type, lead_config
            )
            
            # Train model
            model, history = train_model(
                x_train, y_train, x_test, y_test, 
                classification_type, lead_config, model_type, logger
            )
            
            # Evaluate model
            metrics = evaluate_model(
                model, x_test, y_test, 
                classification_type, lead_config, model_type,
                logger, results_dir
            )
            
            # Save model information
            save_model_info(model, results_dir / "model", logger)
            
            # Save training history
            save_training_history(history, results_dir)
            
            logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            continue

if __name__ == "__main__":
    main() 