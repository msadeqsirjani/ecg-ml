import os
import numpy as np
from pathlib import Path
import tensorflow as tf
import tf_keras as keras
from tf_keras import optimizers, losses, metrics, callbacks
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tensorflow_model_optimization.sparsity import keras as sparsity

# Import everything from micro_ecg_model instead
from model.micro_ecg_model import (
    apply_pruning, 
    create_ecg_model,
    apply_quantization,
    convert_to_tflite
)

from evaluation.metrics import ECGEvaluator

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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


def setup_logging(
    classification_type: str, lead_config: str, optimization: str
) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path("logs") / optimization # Log directory based on optimization type
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = (
        log_dir
        / f"{optimization}_{classification_type}_{lead_config}_{timestamp}.log"
    )

    logger = logging.getLogger(
        f"ECG_{optimization}_{classification_type}_{lead_config}"
    )
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # Avoid adding handlers multiple times if the function is called repeatedly
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def load_data(data_path: str, classification_type: str, lead_config: str) -> tuple:
    """Load and preprocess data."""
    data_path = Path(data_path)
    # Assuming processed data is stored here
    processed_path = (
        data_path
        / "processed"
        / "ptb-xl-1.0.3" # Adjust if your processed data path differs
        / classification_type
        / lead_config
    )

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found at {processed_path}. Please ensure data is preprocessed.")

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


def create_representative_dataset_generator(x_train, sample_size=100):
    """
    Create a representative dataset generator for quantization calibration.
    
    Args:
        x_train: Training dataset
        sample_size: Number of samples to use for calibration
        
    Returns:
        Generator function for representative dataset
    """
    # Select a subset of training data for calibration
    if len(x_train) > sample_size:
        indices = np.random.choice(len(x_train), sample_size, replace=False)
        calibration_data = x_train[indices]
    else:
        calibration_data = x_train
    
    def representative_dataset_gen():
        for data in calibration_data:
            # Add batch dimension and ensure data type
            data = np.expand_dims(data, axis=0).astype(np.float32)
            yield [data]
    
    return representative_dataset_gen


def train_pruned_quantized_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classification_type: str,
    lead_config: str,
    pretrained_weights_path: str,
    logger: logging.Logger,
    fine_tune_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    final_sparsity: float = 0.8 # Higher sparsity for more aggressive pruning
) -> tuple:
    """Load a pre-trained model, apply pruning, fine-tune, and then quantize it."""
    logger.info(f"Input data shape: {x_train.shape}")
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    num_classes = CLASSIFICATION_TYPES[classification_type]

    # 1. Create the base model structure
    base_model = create_ecg_model(input_shape, num_classes)
    logger.info("Base model structure created.")

    # 2. Load pre-trained weights
    weights_path = Path(pretrained_weights_path)
    if weights_path.exists():
        logger.info(f"Loading pre-trained weights from {weights_path}...")
        base_model.load_weights(str(weights_path))
        logger.info("Pre-trained weights loaded.")
    else:
        logger.warning(f"Pre-trained weights not found at {weights_path}. Starting with random weights.")

    # 3. Set pruning parameters
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=final_sparsity,  # Increased for more aggressive pruning
            begin_step=0,
            end_step=fine_tune_epochs * (len(x_train) // batch_size)
        )
    }
    
    # 4. Apply pruning wrapper
    logger.info("Applying pruning wrapper...")
    model_to_prune = apply_pruning(base_model, pruning_params)
    logger.info("Pruning wrapper applied.")

    # 5. Compile the pruned model for fine-tuning
    model_to_prune.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy(), metrics.AUC(curve="ROC", multi_label=True)],
    )
    logger.info(f"Pruned model compiled with Adam optimizer (LR={learning_rate}).")

    # 6. Define Callbacks for fine-tuning (including pruning)
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
        # Pruning callbacks are essential during fine-tuning
        sparsity.UpdatePruningStep(),
        sparsity.PruningSummaries(
            log_dir=f"logs/pruning/summaries_{classification_type}_{lead_config}",
            update_freq='epoch'
        ),
    ]

    # 7. Fine-tune the pruned model
    logger.info(f"Starting model fine-tuning for {fine_tune_epochs} epochs...")
    history = model_to_prune.fit(
        x_train,
        y_train,
        validation_split=0.12,
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
    )
    logger.info("Model fine-tuning completed.")

    # 8. Strip pruning wrappers for evaluation/saving final model
    final_pruned_model = sparsity.strip_pruning(model_to_prune)
    logger.info("Pruning wrappers stripped for final model.")
    
    # 9. Create representative dataset generator for quantization
    logger.info("Creating representative dataset for quantization calibration...")
    representative_dataset_gen = create_representative_dataset_generator(x_train)
    
    # 10. Set quantization parameters for full integer quantization
    quantization_params = {
        'optimizations': [tf.lite.Optimize.DEFAULT],
        'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
        'inference_input_type': tf.int8,
        'inference_output_type': tf.int8
    }
    
    # 11. Apply quantization by converting to TFLite
    logger.info("Converting to TFLite with quantization...")
    tflite_quantized_model = convert_to_tflite(
        final_pruned_model,
        quantization_params,
        representative_dataset_gen
    )
    logger.info(f"Model quantized. TFLite model size: {len(tflite_quantized_model) / (1024 * 1024):.2f} MB")

    return final_pruned_model, tflite_quantized_model, history


def save_tflite_model(tflite_model: bytes, output_path: str, logger: logging.Logger):
    """Save TFLite model to file."""
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Log model size
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"TFLite model saved to {output_path}")
    logger.info(f"Model size: {model_size_mb:.2f} MB")


def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classification_type: str,
    lead_config: str,
    logger: logging.Logger,
    results_dir: Path,
) -> dict:
    """Evaluate the model and save results."""
    num_classes = CLASSIFICATION_TYPES[classification_type]
    evaluator = ECGEvaluator(
        num_classes=num_classes,
        class_names=[f"Class {i}" for i in range(num_classes)],
    )

    # Re-compile the stripped model before evaluation if needed (depends on how you save/load)
    # model.compile(...) # Usually not needed if just predicting

    y_pred = model.predict(x_test)
    metrics_dict = evaluator.compute_metrics(y_test, y_pred)

    # Log metrics
    logger.info("Test Metrics:")
    for metric_name, value in metrics_dict.items():
        if isinstance(value, (list, np.ndarray)): # Handle lists and numpy arrays
             logger.info(f"{metric_name}: {[f'{v:.4f}' for v in value]}")
        elif isinstance(value, (int, float)):
             logger.info(f"{metric_name}: {value:.4f}")
        else:
             logger.info(f"{metric_name}: {value}")


    # Save confusion matrix
    images_dir = results_dir / "images"
    images_dir.mkdir(exist_ok=True)

    model_config = f"{classification_type}_{lead_config}_optimized" # Indicate optimized model
    confusion_matrix_path = images_dir / f"{model_config}_confusion_matrix.png"
    evaluator.plot_confusion_matrix(y_test, y_pred, save_path=confusion_matrix_path)

    # Generate example predictions (optional)
    try:
        sample_indices = np.random.choice(
            len(x_test), min(5, len(x_test)), replace=False
        )
        for i, idx in enumerate(sample_indices):
            plt.figure(figsize=(15, 5))
            plt.plot(x_test[idx, :, :, 0].squeeze()) # Adjust plotting based on data shape
            plt.title(f"ECG Signal Example {i+1} (Lead 0)")
            plt.xlabel("Time Step")
            plt.ylabel("Amplitude")
            example_path = images_dir / f"{model_config}_signal_example_{i}.png"
            plt.savefig(example_path)
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.bar(range(num_classes), y_pred[idx])
            plt.title(f"Predictions for Example {i+1}")
            plt.xlabel("Class")
            plt.ylabel("Probability")
            plt.xticks(range(num_classes))
            example_path = images_dir / f"{model_config}_prediction_example_{i}.png"
            plt.savefig(example_path)
            plt.close()

    except Exception as e:
        logger.warning(f"Could not generate example predictions: {str(e)}")

    return metrics_dict


def main():
    """Main training function for pruning and quantization."""
    data_path = "data"
    # --- Configuration ---
    classification_types = ["binary", "super", "sub"]  # e.g., ["binary", "super", "sub"]
    lead_configs = ["all-leads"] # e.g., ["all-leads", "limb-leads"]
    fine_tune_epochs = 20
    batch_size = 32
    learning_rate = 0.0001
    final_sparsity = 0.8  # 80% sparsity for aggressive pruning
    # --- End Configuration ---

    for classification_type in classification_types:
        for lead_config in lead_configs:
            try:
                logger = setup_logging(
                    classification_type, lead_config, "pruning_quantization"
                )
                logger.info(
                    f"Starting pruning and quantization for {classification_type} classification with {lead_config}"
                )

                # --- Define Paths ---
                # Path to original model weights (non-pruned)
                pretrained_weights_path = (
                    Path("models")
                    / "standard"
                    / "original"
                    / classification_type
                    / lead_config
                    / "model.h5"
                )

                # Directory to save the optimized models
                optimized_models_dir = (
                    Path("models")
                    / "optimized"  # General directory for all optimized models
                    / "original"
                    / classification_type
                    / lead_config
                )
                optimized_models_dir.mkdir(parents=True, exist_ok=True)
                # --- End Define Paths ---

                # Load data
                logger.info("Loading data...")
                x_train, x_test, y_train, y_test = load_data(
                    data_path, classification_type, lead_config
                )
                logger.info("Data loaded.")

                # Apply pruning and quantization pipeline
                final_pruned_model, tflite_model, history = train_pruned_quantized_model(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    classification_type,
                    lead_config,
                    str(pretrained_weights_path),
                    logger,
                    fine_tune_epochs=fine_tune_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    final_sparsity=final_sparsity
                )

                # Save the pruned Keras model (H5 format)
                h5_model_path = optimized_models_dir / "pruned_model.h5"
                final_pruned_model.save(h5_model_path)
                logger.info(f"Pruned Keras model saved to {h5_model_path}")
                
                # Save the quantized TFLite model
                tflite_model_path = optimized_models_dir / "quantized_model.tflite"
                save_tflite_model(tflite_model, str(tflite_model_path), logger)

                # Evaluate the final model before quantization
                logger.info("Evaluating pruned model (before quantization)...")
                metrics_results = evaluate_model(
                    final_pruned_model,
                    x_test,
                    y_test,
                    classification_type,
                    lead_config,
                    logger,
                    optimized_models_dir,
                )

                # Save metrics
                metrics_path = optimized_models_dir / "metrics.json"
                with open(metrics_path, "w") as f:
                    # Convert numpy types for JSON serialization
                    serializable_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics_results.items()}
                    json.dump(serializable_metrics, f, indent=4)
                logger.info(f"Evaluation metrics saved to {metrics_path}")

                # Save training history
                history_path = optimized_models_dir / "fine_tuning_history.json"
                history_dict = history.history
                serializable_history = {k: [float(val) for val in v] for k, v in history_dict.items()}
                with open(history_path, "w") as f:
                    json.dump(serializable_history, f, indent=4)
                logger.info(f"Fine-tuning history saved to {history_path}")

                logger.info(
                    f"Pruning and quantization process completed successfully for {classification_type} / {lead_config}."
                )

            except FileNotFoundError as e:
                 logger.error(f"File not found error: {e}. Please check paths.")
                 continue # Continue with the next configuration
            except Exception as e:
                logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
                continue # Continue with the next configuration


if __name__ == "__main__":
    main()
