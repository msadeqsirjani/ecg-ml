#!/usr/bin/env python3
"""
Train ECG models on compressed datasets.
This script trains models on SVD-compressed ECG data for different compression levels.
"""

# Set matplotlib backend to non-interactive 'Agg' before any other imports
import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics, callbacks
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import sys
from colorama import init, Fore, Back, Style
from tqdm import tqdm

# Add parent directory to path to import model and evaluation modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the actual model and evaluation modules
from src.model.ecg_model import create_ecg_model
from src.evaluation.metrics import ECGEvaluator

# Initialize colorama for cross-platform colored output
init()

# GPU Configuration
print(f"{Back.BLUE}{Fore.WHITE} Checking for GPU availability... {Style.RESET_ALL}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{Fore.GREEN}✓ Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs{Style.RESET_ALL}")
        print(f"{Fore.CYAN}GPU devices: {gpus}{Style.RESET_ALL}")
    except RuntimeError as e:
        print(f"{Fore.RED}✗ Error configuring GPU: {e}{Style.RESET_ALL}")
else:
    print(f"{Fore.YELLOW}⚠ No GPU devices found. Training will run on CPU.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Note: If you have an NVIDIA GPU, please ensure:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}1. NVIDIA GPU drivers are properly installed{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}2. CUDA Toolkit is installed{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}3. cuDNN is installed{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}4. TensorFlow-GPU is installed correctly{Style.RESET_ALL}")

# Set mixed precision policy for better performance
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Configuration
CLASSIFICATION_TYPES = {"binary": 2, "super": 5, "sub": 23}
COMPRESSION_LEVELS = ["25_percent", "50_percent", "75_percent"]
SIGNAL_TYPES = ["sampled", "reconstructed"]
LEAD_CONFIG = "lead-II"  # Focus only on Lead-II


def setup_logging(compression_level: str, classification_type: str, signal_type: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_compressed_{compression_level}_{classification_type}_{signal_type}_{timestamp}.log"

    logger = logging.getLogger(f"CompressedECGTraining_{compression_level}_{classification_type}_{signal_type}")
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


def load_compressed_data(data_path: str, compression_level: str, classification_type: str, signal_type: str) -> tuple:
    """Load and preprocess compressed data."""
    data_path = Path(data_path)
    processed_path = data_path / compression_level / classification_type / signal_type / LEAD_CONFIG

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found at {processed_path}")

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

    return x_train, x_test, y_train, y_test


def train_compressed_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    compression_level: str,
    classification_type: str,
    signal_type: str,
    logger: logging.Logger,
) -> tuple:
    """Train the model on compressed data and return the trained model and history."""
    # Create model
    logger.info(f"{Fore.CYAN}Input data shape: {x_train.shape}{Style.RESET_ALL}")
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    logger.info(f"{Fore.CYAN}Model input shape: {input_shape}{Style.RESET_ALL}")
    num_classes = CLASSIFICATION_TYPES[classification_type]
    logger.info(f"{Fore.CYAN}Number of output classes: {num_classes}{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}Compression level: {compression_level}{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}Signal type: {signal_type}{Style.RESET_ALL}")

    # Create model with GPU strategy
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_ecg_model(input_shape, num_classes)
        
        # Compile model with mixed precision
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0005),
            loss=losses.BinaryFocalCrossentropy(),
            metrics=[metrics.BinaryAccuracy(), metrics.AUC(curve="ROC", multi_label=True)],
        )

    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    # Train model with increased batch size for GPU
    logger.info(f"{Back.GREEN}{Fore.BLACK} Starting model training... {Style.RESET_ALL}")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.12,
        epochs=100,
        batch_size=64,  # Increased batch size for GPU
        callbacks=callbacks_list,
        verbose=1
    )

    return model, history


def evaluate_compressed_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    compression_level: str,
    classification_type: str,
    signal_type: str,
    logger: logging.Logger,
    results_dir: Path,
) -> dict:
    """Evaluate the model and save results."""
    # Define proper class names based on classification type
    if classification_type == "binary":
        class_names = ["Abnormal", "Normal"]
    elif classification_type == "super":
        class_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    else:  # sub
        class_names = [f"Sub_Class_{i}" for i in range(CLASSIFICATION_TYPES[classification_type])]
    
    # Create evaluator
    evaluator = ECGEvaluator(
        num_classes=CLASSIFICATION_TYPES[classification_type],
        class_names=class_names,
    )

    # Get predictions with progress bar
    logger.info(f"{Back.BLUE}{Fore.WHITE} Making predictions... {Style.RESET_ALL}")
    y_pred = model.predict(x_test, verbose=0)

    # Compute metrics
    metrics = evaluator.compute_metrics(y_test, y_pred)

    # Log metrics with colors
    logger.info(f"{Back.CYAN}{Fore.BLACK} Test Metrics: {Style.RESET_ALL}")
    for metric_name, value in metrics.items():
        if isinstance(value, list):
            logger.info(f"{Fore.CYAN}{metric_name}: {[f'{v:.2f}' for v in value]}{Style.RESET_ALL}")
        else:
            logger.info(f"{Fore.CYAN}{metric_name}: {value:.2f}{Style.RESET_ALL}")

    # Generate and save confusion matrix
    logger.info(f"{Back.MAGENTA}{Fore.WHITE} Generating confusion matrix... {Style.RESET_ALL}")
    images_dir = Path(results_dir / "images")
    images_dir.mkdir(exist_ok=True)
    
    model_config = f"compressed_{compression_level}_{classification_type}_{signal_type}"
    confusion_matrix_path = images_dir / f"{model_config}_confusion_matrix.png"
    evaluator.plot_confusion_matrix(
        y_test, y_pred, save_path=confusion_matrix_path
    )
    logger.info(f"{Fore.GREEN}✓ Confusion matrix saved to {confusion_matrix_path}{Style.RESET_ALL}")
    
    # Generate example predictions
    logger.info(f"{Back.MAGENTA}{Fore.WHITE} Generating example predictions... {Style.RESET_ALL}")
    try:
        # Select a few random samples for visualization
        sample_indices = np.random.choice(len(x_test), min(5, len(x_test)), replace=False)
        samples = x_test[sample_indices]
        true_labels = y_test[sample_indices]
        
        for i, (sample, true_label) in enumerate(zip(samples, true_labels)):
            # Make prediction
            sample_batch = np.expand_dims(sample, axis=0)
            pred = model.predict(sample_batch, verbose=0)
            
            # Plot the ECG signal and prediction
            plt.figure(figsize=(15, 5))
            
            # Plot the ECG signal
            plt.subplot(1, 2, 1)
            plt.plot(sample[:, 0, 0])
            plt.title(f'Compressed ECG Signal ({compression_level}, {signal_type})')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            
            # Plot prediction probabilities
            plt.subplot(1, 2, 2)
            plt.bar(class_names, pred[0])
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.title('Prediction Probabilities')
            
            # Add true label information
            true_class = np.argmax(true_label) if len(true_label.shape) > 0 else int(true_label)
            pred_class = np.argmax(pred[0])
            plt.suptitle(f'True: {class_names[true_class]}, Predicted: {class_names[pred_class]}')
            
            # Save figure
            example_path = images_dir / f"{model_config}_example_{i}.png"
            plt.tight_layout()
            plt.savefig(example_path)
            plt.close()
            
        logger.info(f"{Fore.GREEN}✓ Example predictions saved to {images_dir}{Style.RESET_ALL}")
    except Exception as e:
        logger.warning(f"{Fore.YELLOW}⚠ Could not generate example predictions: {str(e)}{Style.RESET_ALL}")

    return metrics


def save_model_info(model: tf.keras.Model, model_dir: Path, compression_level: str, signal_type: str, logger: logging.Logger):
    """Save model architecture, size and weights information."""
    try:
        # Save model architecture as JSON
        logger.info(f"{Back.BLUE}{Fore.WHITE} Saving model architecture... {Style.RESET_ALL}")
        try:
            model_json = model.to_json()
            with open(model_dir / "model_architecture.json", "w") as f:
                f.write(model_json)
            logger.info(f"{Fore.GREEN}✓ Saved model architecture to JSON{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}✗ Failed to save model architecture: {str(e)}{Style.RESET_ALL}")

        # Save compression level info
        compression_info = {
            "compression_level": compression_level,
            "signal_type": signal_type,
            "lead_config": LEAD_CONFIG,
            "model_type": "compressed_ecg",
            "description": f"Model trained on {compression_level} compressed {signal_type} ECG data"
        }
        
        with open(model_dir / "model_info.json", "w") as f:
            json.dump(compression_info, f, indent=4)

        logger.info(f"{Fore.GREEN}✓ Model information saved successfully{Style.RESET_ALL}")
        logger.info(f"{Back.WHITE}{Fore.BLACK} ======================== {Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"{Fore.RED}✗ Failed to save model info: {str(e)}{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}Continuing despite model info save failure{Style.RESET_ALL}")


def save_training_history(history, results_dir: Path, compression_level: str, signal_type: str):
    """Save training history to JSON file and plot training curves."""
    history_dict = {
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "binary_accuracy": [float(x) for x in history.history["binary_accuracy"]],
        "val_binary_accuracy": [
            float(x) for x in history.history["val_binary_accuracy"]
        ],
        "auc": [float(x) for x in history.history["auc"]],
        "val_auc": [float(x) for x in history.history["val_auc"]],
        "compression_level": compression_level,
        "signal_type": signal_type
    }

    # Save history as JSON
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=4)
    
    # Create images directory if it doesn't exist
    images_dir = Path(results_dir / "images")
    images_dir.mkdir(exist_ok=True)
    
    # Get model configuration from results_dir path
    model_config = f"compressed_{compression_level}_{signal_type}"
    
    # Plot and save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history_dict["loss"], label='Training Loss')
    plt.plot(history_dict["val_loss"], label='Validation Loss')
    plt.title(f'Training and Validation Loss ({compression_level}, {signal_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = images_dir / f"{model_config}_loss.png"
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Training and validation accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_dict["binary_accuracy"], label='Training Accuracy')
    plt.plot(history_dict["val_binary_accuracy"], label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy ({compression_level}, {signal_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    acc_plot_path = images_dir / f"{model_config}_accuracy.png"
    plt.savefig(acc_plot_path)
    plt.close()
    
    # Training and validation AUC plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_dict["auc"], label='Training AUC')
    plt.plot(history_dict["val_auc"], label='Validation AUC')
    plt.title(f'Training and Validation AUC ({compression_level}, {signal_type})')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    auc_plot_path = images_dir / f"{model_config}_auc.png"
    plt.savefig(auc_plot_path)
    plt.close()


def main():
    """Main training function for compressed ECG models."""
    # Configuration
    data_path = "/Users/sadegh/Documents/UTSA/Spring 2025/Independent Study/Lab3/data/processed/ptb-xl-compressed"
    classification_types = ["super"]  # Focus on super classification
    compression_levels = COMPRESSION_LEVELS
    signal_types = SIGNAL_TYPES

    print(f"\n{Back.GREEN}{Fore.BLACK} Starting Training Pipeline for Compressed ECG Models {Style.RESET_ALL}")
    print("=" * 60)
    print(f"{Fore.CYAN}Data path: {Style.RESET_ALL}{data_path}")
    print(f"{Fore.CYAN}Classification types: {Style.RESET_ALL}{classification_types}")
    print(f"{Fore.CYAN}Lead configuration: {Style.RESET_ALL}{LEAD_CONFIG}")
    print(f"{Fore.CYAN}Compression levels: {Style.RESET_ALL}{compression_levels}")
    print(f"{Fore.CYAN}Signal types: {Style.RESET_ALL}{signal_types}")
    print("=" * 60)

    # Train for each combination
    for compression_level in compression_levels:
        print(f"\n{Back.BLUE}{Fore.WHITE} Processing Compression Level: {compression_level} {Style.RESET_ALL}")
        
        for signal_type in signal_types:
            print(f"\n{Back.MAGENTA}{Fore.WHITE} Processing {signal_type} signals {Style.RESET_ALL}")
            
            for classification_type in classification_types:
                try:
                    # Setup logging
                    logger = setup_logging(compression_level, classification_type, signal_type)
                    logger.info(
                        f"{Back.CYAN}{Fore.BLACK} Starting training for {compression_level} compression with {classification_type} classification using {signal_type} signals {Style.RESET_ALL}"
                    )
                    
                    # Create models directory
                    models_dir = (
                        Path("models") / "compressed" / compression_level / classification_type / signal_type / LEAD_CONFIG
                    )
                    models_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Load data
                    logger.info(f"{Back.BLUE}{Fore.WHITE} Loading compressed data... {Style.RESET_ALL}")
                    try:
                        x_train, x_test, y_train, y_test = load_compressed_data(
                            data_path, compression_level, classification_type, signal_type
                        )
                        
                        logger.info(f"{Fore.CYAN}Training data shape: {x_train.shape}{Style.RESET_ALL}")
                        logger.info(f"{Fore.CYAN}Test data shape: {x_test.shape}{Style.RESET_ALL}")
                        logger.info(f"{Fore.CYAN}Training labels shape: {y_train.shape}{Style.RESET_ALL}")
                        logger.info(f"{Fore.CYAN}Test labels shape: {y_test.shape}{Style.RESET_ALL}")
                    except FileNotFoundError as e:
                        logger.error(f"{Fore.RED}✗ {str(e)}{Style.RESET_ALL}")
                        logger.error(f"{Fore.YELLOW}Skipping {compression_level} - {classification_type} - {signal_type}{Style.RESET_ALL}")
                        continue
                    
                    # Train model
                    logger.info(f"{Back.GREEN}{Fore.BLACK} Training compressed model... {Style.RESET_ALL}")
                    model, history = train_compressed_model(
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        compression_level,
                        classification_type,
                        signal_type,
                        logger,
                    )
                    
                    # Save model first - this is the most important step
                    logger.info(f"{Back.BLUE}{Fore.WHITE} Saving model... {Style.RESET_ALL}")
                    try:
                        model.save(models_dir / "model.h5")
                        logger.info(f"{Fore.GREEN}✓ Model saved to {models_dir / 'model.h5'}{Style.RESET_ALL}")
                    except Exception as e:
                        logger.error(f"{Fore.RED}✗ Failed to save model: {str(e)}{Style.RESET_ALL}")
                        raise
                    
                    # Save model architecture and information
                    logger.info(f"{Back.BLUE}{Fore.WHITE} Saving model info... {Style.RESET_ALL}")
                    try:
                        save_model_info(model, models_dir, compression_level, signal_type, logger)
                    except Exception as e:
                        logger.error(f"{Fore.RED}✗ Failed to save model info: {str(e)}{Style.RESET_ALL}")
                    
                    # Evaluate model
                    logger.info(f"{Back.BLUE}{Fore.WHITE} Evaluating model... {Style.RESET_ALL}")
                    try:
                        metrics = evaluate_compressed_model(
                            model,
                            x_test,
                            y_test,
                            compression_level,
                            classification_type,
                            signal_type,
                            logger,
                            models_dir,
                        )
                        # Save metrics as JSON
                        with open(models_dir / "metrics.json", "w") as f:
                            json.dump(metrics, f, indent=4)
                        logger.info(f"{Fore.GREEN}✓ Metrics saved to {models_dir / 'metrics.json'}{Style.RESET_ALL}")
                    except Exception as e:
                        logger.error(f"{Fore.RED}✗ Error during evaluation: {str(e)}{Style.RESET_ALL}")
                    
                    # Save training history
                    logger.info(f"{Back.BLUE}{Fore.WHITE} Saving training history... {Style.RESET_ALL}")
                    try:
                        save_training_history(history, models_dir, compression_level, signal_type)
                        logger.info(f"{Fore.GREEN}✓ Training history saved to {models_dir}{Style.RESET_ALL}")
                    except Exception as e:
                        logger.error(f"{Fore.RED}✗ Failed to save training history: {str(e)}{Style.RESET_ALL}")
                    
                    logger.info(f"{Back.GREEN}{Fore.BLACK} Training completed successfully for {compression_level} {signal_type} {Style.RESET_ALL}")
                    logger.info("=" * 60)
                    
                except Exception as e:
                    print(f"{Back.RED}{Fore.WHITE} Error during training {compression_level} - {classification_type} - {signal_type}: {str(e)} {Style.RESET_ALL}")
                    continue

    print(f"\n{Back.GREEN}{Fore.BLACK} All compressed model training completed! {Style.RESET_ALL}")
    print(f"{Fore.CYAN}Models saved in: ./models/compressed/{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main() 