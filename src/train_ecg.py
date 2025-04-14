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
    log_file = log_dir / f"training_{classification_type}_{lead_config}_{timestamp}.log"

    logger = logging.getLogger(f"ECGTraining_{classification_type}_{lead_config}")
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
    """Load and preprocess data."""
    data_path = Path(data_path)
    processed_path = (
        data_path / "processed" / "ptb-xl-1.0.3" / classification_type / lead_config
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

    return x_train, x_test, y_train, y_test


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
    """Train the model and return the trained model and history."""
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

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy(), metrics.AUC(curve="ROC", multi_label=True)],
    )

    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3),
    ]

    # Train model
    logger.info("Starting model training...")
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
    """Evaluate the model and save results."""
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
    logger.info("Test Metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, list):
            logger.info(f"{metric_name}: {[f'{v:.2f}' for v in value]}")
        else:
            logger.info(f"{metric_name}: {value:.2f}")

    # Generate and save confusion matrix
    logger.info("Generating confusion matrix...")
    # Save in the results directory
    evaluator.plot_confusion_matrix(
        y_test, y_pred, save_path=results_dir / "images" / "confusion_matrix.png"
    )
    logger.info(f"Confusion matrix saved to {results_dir / 'images' / 'confusion_matrix.png'}")
    
    # Save in the images directory
    images_dir = Path(results_dir / "images")
    images_dir.mkdir(exist_ok=True)
    model_config = f"{classification_type}_{lead_config}_{model_type}"
    confusion_matrix_path = images_dir / f"{model_config}_confusion_matrix.png"
    evaluator.plot_confusion_matrix(
        y_test, y_pred, save_path=confusion_matrix_path
    )
    logger.info(f"Confusion matrix also saved to {confusion_matrix_path}")
    
    # Generate example predictions
    logger.info("Generating example predictions...")
    try:
        # Select a few random samples for visualization
        sample_indices = np.random.choice(len(x_test), min(5, len(x_test)), replace=False)
        samples = x_test[sample_indices]
        true_labels = y_test[sample_indices]
        
        for i, (sample, true_label) in enumerate(zip(samples, true_labels)):
            # Make prediction
            sample_batch = np.expand_dims(sample, axis=0)
            pred = model.predict(sample_batch)
            
            # Plot the ECG signal and prediction
            plt.figure(figsize=(15, 5))
            
            # Plot the ECG signal
            plt.subplot(1, 2, 1)
            # If sample has shape (time_steps, leads), plot first lead
            if len(sample.shape) > 1:
                plt.plot(sample[:, 0])
            else:
                plt.plot(sample)
            plt.title('ECG Signal')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            
            # Plot prediction probabilities
            plt.subplot(1, 2, 2)
            class_names = [f"Class {j}" for j in range(CLASSIFICATION_TYPES[classification_type])]
            plt.bar(class_names, pred[0])
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.title('Prediction Probabilities')
            
            # Add true label information
            true_class = np.argmax(true_label) if len(true_label.shape) > 0 else int(true_label)
            pred_class = np.argmax(pred[0])
            plt.suptitle(f'True: Class {true_class}, Predicted: Class {pred_class}')
            
            # Save figure
            example_path = images_dir / f"{model_config}_example_{i}.png"
            plt.tight_layout()
            plt.savefig(example_path)
            plt.close()
            
        logger.info(f"Example predictions saved to {images_dir}")
    except Exception as e:
        logger.warning(f"Could not generate example predictions: {str(e)}")

    return metrics


def save_model_info(model: tf.keras.Model, model_dir: Path, logger: logging.Logger):
    """Save model architecture, size and weights information."""
    try:
        # Save model architecture as JSON
        logger.info("Saving model architecture...")
        try:
            model_json = model.to_json()
            with open(model_dir / "model_architecture.json", "w") as f:
                f.write(model_json)
            logger.info("Saved model architecture to JSON")
        except Exception as e:
            logger.error(f"Failed to save model architecture: {str(e)}")

        # Get model summary
        logger.info("Saving model summary...")
        try:
            string_list = []
            model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
            with open(model_dir / "model_summary.txt", "w") as f:
                f.write("\n".join(string_list))
            logger.info("Saved model summary")
        except Exception as e:
            logger.error(f"Failed to save model summary: {str(e)}")

        # Save weights and size info
        logger.info("Calculating model size information...")
        try:
            weights_info = []
            total_params = 0
            trainable_params = 0
            non_trainable_params = 0
            layer_sizes = {}

            for layer in model.layers:
                layer_info = {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "trainable": layer.trainable,
                    "weights": [],
                }

                layer_params = 0
                layer_size = 0

                for weight in layer.weights:
                    # Make sure weight is actually a weight tensor and not a string
                    if isinstance(weight, str):
                        logger.warning(f"Skipping invalid weight object (string) in layer {layer.name}")
                        continue
                        
                    try:
                        param_count = np.prod(weight.shape.as_list())
                        size_bytes = param_count * 4  # Assuming float32 (4 bytes)

                        weight_info = {
                            "name": weight.name if hasattr(weight, 'name') else "unknown",
                            "shape": weight.shape.as_list() if hasattr(weight, 'shape') else [],
                            "dtype": weight.dtype.name if hasattr(weight, 'dtype') else "unknown",
                            "parameters": int(param_count),
                            "size_bytes": size_bytes,
                            "size_mb": size_bytes / (1024 * 1024),
                        }

                        layer_params += param_count
                        layer_size += size_bytes

                        if hasattr(weight, 'name') and "trainable" in weight.name or layer.trainable:
                            trainable_params += param_count
                        else:
                            non_trainable_params += param_count

                        layer_info["weights"].append(weight_info)
                    except Exception as e:
                        logger.warning(f"Error processing weight in layer {layer.name}: {str(e)}")
                        continue

                layer_info["total_parameters"] = int(layer_params)
                layer_info["size_mb"] = layer_size / (1024 * 1024)
                weights_info.append(layer_info)

                # Track layer sizes
                layer_sizes[layer.name] = {
                    "parameters": int(layer_params),
                    "size_mb": layer_size / (1024 * 1024),
                }

                total_params += layer_params

            # Get model file size
            logger.info("Calculating model file size...")
            try:
                temp_model_path = model_dir / "temp_model.h5"
                model.save(temp_model_path)
                model_file_size = temp_model_path.stat().st_size
                temp_model_path.unlink()  # Remove temporary file
            except Exception as e:
                logger.warning(f"Failed to calculate model file size: {str(e)}")
                model_file_size = 0

            # Prepare size summary
            size_summary = {
                "model_statistics": {
                    "total_parameters": int(total_params),
                    "trainable_parameters": int(trainable_params),
                    "non_trainable_parameters": int(non_trainable_params),
                    "model_file_size_mb": model_file_size / (1024 * 1024),
                    "weights_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
                },
                "layer_sizes": layer_sizes,
                "detailed_layer_info": weights_info,
            }

            # Save size information as JSON
            logger.info("Saving model size info as JSON...")
            try:
                with open(model_dir / "model_size_info.json", "w") as f:
                    json.dump(size_summary, f, indent=4)
                logger.info(f"Size information saved to {model_dir / 'model_size_info.json'}")
            except Exception as e:
                logger.error(f"Failed to save model_size_info.json: {str(e)}")

            # Create a human-readable report
            logger.info("Creating MODEL_SIZE_REPORT.txt...")
            try:
                with open(model_dir / "MODEL_SIZE_REPORT.txt", "w") as f:
                    f.write("=" * 80 + "\n")
                    f.write("MODEL SIZE REPORT\n")
                    f.write("=" * 80 + "\n\n")

                    # Overall Statistics
                    f.write("OVERALL STATISTICS\n")
                    f.write("-" * 50 + "\n")
                    f.write(
                        f"Total Parameters:        {size_summary['model_statistics']['total_parameters']:,}\n"
                    )
                    f.write(
                        f"Trainable Parameters:    {size_summary['model_statistics']['trainable_parameters']:,}\n"
                    )
                    f.write(
                        f"Non-trainable Parameters:{size_summary['model_statistics']['non_trainable_parameters']:,}\n"
                    )
                    f.write(
                        f"Model File Size:         {size_summary['model_statistics']['model_file_size_mb']:.2f} MB\n"
                    )
                    f.write(
                        f"Weights Size:            {size_summary['model_statistics']['weights_size_mb']:.2f} MB\n\n"
                    )

                    # Layer-wise Statistics
                    f.write("LAYER-WISE STATISTICS\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"{'Layer Name':<30} {'Parameters':>12} {'Size (MB)':>12}\n")
                    f.write("-" * 80 + "\n")
                    for layer_name, info in layer_sizes.items():
                        f.write(
                            f"{layer_name:<30} {info['parameters']:>12,} {info['size_mb']:>12.2f}\n"
                        )
                    f.write("\n")

                    # Detailed Layer Information
                    f.write("DETAILED LAYER INFORMATION\n")
                    f.write("-" * 50 + "\n")
                    for layer in weights_info:
                        f.write(f"\nLayer: {layer['name']} ({layer['type']})\n")
                        f.write(f"Trainable: {layer['trainable']}\n")
                        f.write(f"Total Parameters: {layer['total_parameters']:,}\n")
                        f.write(f"Size: {layer['size_mb']:.2f} MB\n")
                        if layer["weights"]:
                            f.write("Weights:\n")
                            for w in layer["weights"]:
                                f.write(f"  - {w['name']}\n")
                                f.write(f"    Shape: {w['shape']}\n")
                                f.write(f"    Parameters: {w['parameters']:,}\n")
                                f.write(f"    Size: {w['size_mb']:.2f} MB\n")
                        f.write("-" * 30 + "\n")
                logger.info(f"Model size report saved to {model_dir / 'MODEL_SIZE_REPORT.txt'}")
            except Exception as e:
                logger.error(f"Failed to create MODEL_SIZE_REPORT.txt: {str(e)}")

            # Log size summary
            logger.info("\n=== Model Size Summary ===")
            logger.info(
                f"Total Parameters: {size_summary['model_statistics']['total_parameters']:,}"
            )
            logger.info(
                f"Trainable Parameters: {size_summary['model_statistics']['trainable_parameters']:,}"
            )
            logger.info(
                f"Non-trainable Parameters: {size_summary['model_statistics']['non_trainable_parameters']:,}"
            )
            logger.info(
                f"Model File Size: {size_summary['model_statistics']['model_file_size_mb']:.2f} MB"
            )
            logger.info(
                f"Weights Size: {size_summary['model_statistics']['weights_size_mb']:.2f} MB"
            )
        except Exception as e:
            logger.error(f"Failed to calculate and save model size information: {str(e)}")
            
        logger.info("Model information saved successfully")
        logger.info("========================")

    except Exception as e:
        logger.error(f"Failed to save model information: {str(e)}")
        # Don't raise here, continue with rest of the process
        logger.info("Continuing despite model info save failure")


def save_training_history(history, results_dir: Path):
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
    }

    # Save history as JSON
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=4)
    
    # Create images directory if it doesn't exist
    images_dir = Path(results_dir / "images")
    images_dir.mkdir(exist_ok=True)
    
    # Get model configuration from results_dir path
    model_config = str(results_dir).replace('/', '_')
    
    # Plot and save loss curves
    # Training and validation loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_dict["loss"], label='Training Loss')
    plt.plot(history_dict["val_loss"], label='Validation Loss')
    plt.title('Training and Validation Loss')
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
    plt.title('Training and Validation Accuracy')
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
    plt.title('Training and Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    auc_plot_path = images_dir / f"{model_config}_auc.png"
    plt.savefig(auc_plot_path)
    plt.close()


def main():
    """Main training function."""
    # Configuration
    data_path = "data"
    classification_types = ["binary", "super", "sub"]  # "binary", "super", "sub"
    model_types = [
        "micro",
    ]  # "standard", "micro"
    lead_configs = [
        "all-leads"
    ]  # "lead-I", "bipolar-limb", "unipolar-limb", "limb-leads", "precordial-leads", "all-leads"

    # Train for each combination
    for classification_type in classification_types:
        for lead_config in lead_configs:
            for model_type in model_types:
                try:
                    # Setup logging
                    logger = setup_logging(classification_type, lead_config)
                    logger.info(
                        f"Starting training for {classification_type} classification with {lead_config} using {model_type} model"
                    )

                    models_dir = (
                        Path("models") / classification_type / lead_config / model_type
                    )
                    models_dir.mkdir(parents=True, exist_ok=True)

                    # Load data
                    logger.info("Loading data...")
                    x_train, x_test, y_train, y_test = load_data(
                        data_path, classification_type, lead_config
                    )

                    # Train model
                    logger.info("Training model...")
                    model, history = train_model(
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        classification_type,
                        lead_config,
                        model_type,
                        logger,
                    )

                    # Save model first - this is the most important step
                    logger.info("Saving model...")
                    try:
                        model.save(models_dir / "model.h5")
                        logger.info(f"Model saved to {models_dir / 'model.h5'}")
                    except Exception as e:
                        logger.error(f"Failed to save model: {str(e)}")
                        raise

                    # Save model architecture and information
                    logger.info("Saving model info...")
                    try:
                        save_model_info(model, models_dir, logger)
                    except Exception as e:
                        logger.error(f"Failed to save model info: {str(e)}")
                        # Don't raise here, continue with evaluation

                    # Evaluate model
                    logger.info("Evaluating model...")
                    try:
                        metrics = evaluate_model(
                            model,
                            x_test,
                            y_test,
                            classification_type,
                            lead_config,
                            model_type,
                            logger,
                            models_dir,
                        )

                        # Save metrics
                        with open(models_dir / "metrics.json", "w") as f:
                            json.dump(metrics, f, indent=4)
                        logger.info(f"Metrics saved to {models_dir / 'metrics.json'}")
                    except Exception as e:
                        logger.error(f"Error during evaluation: {str(e)}")
                        # Don't raise here, continue with history saving

                    # Save training history
                    logger.info("Saving training history...")
                    try:
                        save_training_history(history, models_dir)
                        logger.info(f"Training history saved to {models_dir}")
                    except Exception as e:
                        logger.error(f"Failed to save training history: {str(e)}")

                    logger.info("Training completed successfully")

                except Exception as e:
                    logger.error(f"Error during training: {str(e)}")
                    continue


if __name__ == "__main__":
    main()
