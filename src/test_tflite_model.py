import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import random
import json

class ECGClassifier:
    def __init__(self, model_path):
        """Initialize the ECG classifier with a TFLite model."""
        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get quantization parameters
        self.input_scale, self.input_zero_point = self.input_details[0]["quantization"]
        self.output_scale, self.output_zero_point = self.output_details[0]["quantization"]
        
        # Log model details
        print("\nModel Details:")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Input type: {self.input_details[0]['dtype']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        print(f"Output type: {self.output_details[0]['dtype']}")
        print(f"Input scale: {self.input_scale}, zero point: {self.input_zero_point}")
        print(f"Output scale: {self.output_scale}, zero point: {self.output_zero_point}")
    
    def preprocess_input(self, ecg_signal):
        """Preprocess ECG signal for the model."""
        # Add batch dimension if needed
        if len(ecg_signal.shape) == 3:
            ecg_signal = np.expand_dims(ecg_signal, axis=0)
        
        # Scale input for quantized model
        if self.input_scale != 0:
            ecg_signal = ecg_signal / self.input_scale + self.input_zero_point
            ecg_signal = ecg_signal.astype(np.int8)
        
        return ecg_signal
    
    def predict(self, ecg_signal):
        """Make prediction with the quantized model."""
        # Preprocess input
        input_data = self.preprocess_input(ecg_signal)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Scale output if quantized
        if self.output_scale != 0:
            output_data = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
        
        return output_data

def load_random_ecg_signal(data_path, classification_type, lead_config):
    """Load a random ECG signal from the processed data directory."""
    data_path = Path(data_path)
    processed_path = (
        data_path
        / "processed"
        / "ptb-xl-1.0.3"
        / classification_type
        / lead_config
    )
    
    # Load test data
    x_test = np.load(processed_path / "x_test.npy", allow_pickle=True)
    y_test = np.load(processed_path / "y_test.npy", allow_pickle=True)
    
    # Select a random sample
    idx = random.randint(0, len(x_test) - 1)
    signal = x_test[idx]
    label = y_test[idx]
    
    # Reshape signal to match model input
    signal = signal.transpose(1, 0)  # Transpose to (time_steps, leads)
    signal = signal.reshape(signal.shape[0], signal.shape[1], 1)  # Add channel dimension
    
    return signal, label, idx

def plot_ecg_signal(signal, prediction, label, idx, save_path=None):
    """Plot the ECG signal and prediction results."""
    plt.figure(figsize=(15, 10))
    
    # Plot ECG signal (first lead)
    plt.subplot(2, 1, 1)
    plt.plot(signal[:, 0, 0])
    plt.title(f'ECG Signal (Sample {idx})')
    plt.xlabel('Time Step')
    plt.ylabel('Amplitude')
    
    # Plot prediction probabilities
    plt.subplot(2, 1, 2)
    classes = range(len(prediction[0]))
    plt.bar(classes, prediction[0])
    plt.title('Prediction Probabilities')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(classes)
    
    # Add true label information
    true_class = np.argmax(label) if len(label.shape) > 0 else int(label)
    pred_class = np.argmax(prediction[0])
    plt.suptitle(f'True Class: {true_class}, Predicted Class: {pred_class}')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def main():
    # Configuration
    data_path = "data"
    classification_type = "sub"  # or "super" or "sub"
    lead_config = "all-leads"
    
    # Path to the quantized model
    model_path = (
        Path("models")
        / "optimized"
        / "original"
        / classification_type
        / lead_config
        / "quantized_model.tflite"
    )
    
    # Create output directory for plots
    output_dir = Path("output") / "tflite_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load random ECG signal
    print("\nLoading random ECG signal...")
    signal, label, idx = load_random_ecg_signal(data_path, classification_type, lead_config)
    print(f"Loaded signal shape: {signal.shape}")
    print(f"True label: {label}")
    
    # Initialize classifier
    print("\nLoading TFLite model...")
    classifier = ECGClassifier(str(model_path))
    
    # Make prediction
    print("\nMaking prediction...")
    prediction = classifier.predict(signal)
    print(f"Raw prediction: {prediction}")
    print(f"Predicted class: {np.argmax(prediction[0])}")
    
    # Plot results
    plot_path = output_dir / f"prediction_sample_{idx}.png"
    plot_ecg_signal(signal, prediction, label, idx, save_path=plot_path)
    print(f"\nPlot saved to: {plot_path}")
    
    # Save prediction details
    results = {
        "sample_index": int(idx),
        "true_label": label.tolist() if isinstance(label, np.ndarray) else float(label),
        "prediction": prediction[0].tolist(),
        "predicted_class": int(np.argmax(prediction[0])),
        "model_path": str(model_path)
    }
    
    results_path = output_dir / f"prediction_details_{idx}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Prediction details saved to: {results_path}")

if __name__ == "__main__":
    main() 