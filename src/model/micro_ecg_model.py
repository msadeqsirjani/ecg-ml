import tensorflow as tf
import tf_keras as keras
from tf_keras.models import Model, clone_model
from tf_keras.layers import (
    Conv2D, 
    Dense, 
    DepthwiseConv2D,
    Add,
    Activation,
    Dropout,
    Flatten,
    Input,
    BatchNormalization,
    ReLU,
    MaxPooling2D,
    Concatenate,
    GlobalAveragePooling2D,
)
import tensorflow_model_optimization as tfmot


def apply_pruning(model: Model, pruning_params: dict = None) -> Model:
    """
    Apply pruning to the model using TensorFlow Model Optimization toolkit.
    This implementation attempts to directly apply pruning to the entire model
    using the TFMOT pruning API.
    
    Args:
        model: The Keras model to be pruned
        pruning_params: Dictionary of pruning parameters. If None, default parameters will be used.
        
    Returns:
        Pruned model ready for training
    """
    # Set default pruning parameters if none provided
    if pruning_params is None:
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=1000
        )
    else:
        pruning_schedule = pruning_params.get('pruning_schedule', 
            tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.5,
                begin_step=0,
                end_step=1000
            )
        )

    # The proper way to prune a model with tf_keras and TFMOT
    # This uses a direct approach without attempting to handle individual layers
    pruning_config = {
        'pruning_schedule': pruning_schedule
    }
    
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        model, **pruning_config
    )
    
    return model_for_pruning


def apply_quantization(model: Model, quantization_params: dict = None) -> Model:
    """
    Apply quantization to the model using TensorFlow Model Optimization toolkit.
    This function can be applied to a model after pruning and fine-tuning.
    
    Args:
        model: The Keras model to be quantized (can be a pruned model)
        quantization_params: Dictionary of quantization parameters. If None, default parameters will be used.
        
    Returns:
        Quantized model
    """
    # There are two main approaches for quantization in TF:
    # 1. Quantization-aware training (QAT) - during training
    # 2. Post-training quantization (PTQ) - after training
    
    # For simplicity and since we're applying this after pruning & training,
    # we'll use post-training quantization (PTQ)
    
    # Default to 8-bit quantization for all operations
    if quantization_params is None:
        quantization_params = {
            'optimizations': [tf.lite.Optimize.DEFAULT],
            'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
            'inference_input_type': tf.int8,
            'inference_output_type': tf.int8,
            'representative_dataset': None  # This must be provided
        }
    
    # Warning if representative dataset is not provided
    if quantization_params.get('representative_dataset') is None:
        print("WARNING: A representative dataset must be provided for full integer quantization.")
        print("Using float fallback, which won't achieve maximum size reduction.")
    
    return model  # Return the original model for the converter function to handle
    

def convert_to_tflite(model: Model, quantization_params: dict = None, 
                     representative_dataset_gen=None) -> bytes:
    """
    Convert a Keras model to TFLite format with optional quantization.
    
    Args:
        model: The Keras model to convert
        quantization_params: Dictionary of quantization parameters
        representative_dataset_gen: Generator function for representative dataset
                                   for quantization calibration
    
    Returns:
        TFLite model as bytes
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization if parameters are provided
    if quantization_params:
        # Set optimization flags
        converter.optimizations = quantization_params.get(
            'optimizations', [tf.lite.Optimize.DEFAULT]
        )
        
        # Set supported ops
        converter.target_spec.supported_ops = quantization_params.get(
            'supported_ops', [tf.lite.OpsSet.TFLITE_BUILTINS]
        )
        
        # Set input/output types if specified
        if 'inference_input_type' in quantization_params:
            converter.inference_input_type = quantization_params['inference_input_type']
        
        if 'inference_output_type' in quantization_params:
            converter.inference_output_type = quantization_params['inference_output_type']
        
        # Set representative dataset if provided
        if representative_dataset_gen:
            converter.representative_dataset = representative_dataset_gen
    
    # Convert the model
    tflite_model = converter.convert()
    return tflite_model


def process_model_pipeline(original_model: Model, 
                         pruning_params: dict = None,
                         quantization_params: dict = None,
                         representative_dataset_gen=None) -> tuple:
    """
    Complete pipeline for model optimization: pruning → strip pruning → quantization.
    
    Args:
        original_model: The original Keras model
        pruning_params: Parameters for pruning
        quantization_params: Parameters for quantization
        representative_dataset_gen: Generator function for representative dataset
        
    Returns:
        Tuple of (pruned_model, tflite_quantized_model_bytes)
    """
    # 1. Apply pruning
    pruned_model = apply_pruning(original_model, pruning_params)
    
    # Note: At this point you would train/fine-tune the pruned model
    # This is not done in this function - it should be done externally
    
    # 2. Strip pruning (after training)
    final_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    
    # 3. Convert to TFLite with quantization
    tflite_model = convert_to_tflite(
        final_pruned_model, 
        quantization_params, 
        representative_dataset_gen
    )
    
    return final_pruned_model, tflite_model


def create_ecg_model(input_shape: tuple, num_classes: int) -> Model:
    """
    Create the ECG classification model architecture with enhanced regularization.

    Args:
        input_shape: Tuple of (num_leads, sequence_length, channels)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    input_layer = Input(shape=input_shape)

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1))(input_layer)
    batch1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch1)

    # Second block
    conv2 = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 1))(relu1)
    batch2 = BatchNormalization()(conv2)
    relu2 = ReLU()(batch2)
    drop2 = Dropout(rate=0.1)(relu2)
    conv2 = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 2))(drop2)

    max1 = MaxPooling2D(pool_size=(1, 9), strides=(1, 2))(relu1)
    conv_ = Conv2D(64, (1, 1))(max1)
    conc1 = Add()([conv2, conv_])

    # Third block
    batch3 = BatchNormalization()(conc1)
    relu3 = ReLU()(batch3)
    drop3 = Dropout(rate=0.1)(relu3)
    conv3 = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 1))(drop3)
    batch3 = BatchNormalization()(conv3)
    relu3 = ReLU()(batch3)
    drop3 = Dropout(rate=0.1)(relu3)
    conv3 = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 2))(drop3)

    max2 = MaxPooling2D(pool_size=(1, 9), strides=(1, 2))(conc1)
    conc2 = Add()([conv3, max2])

    # Fourth block
    batch4 = BatchNormalization()(conc2)
    relu4 = ReLU()(batch4)
    drop4 = Dropout(rate=0.1)(relu4)
    conv4 = Conv2D(filters=128, kernel_size=(1, 5), strides=(1, 1))(drop4)
    batch4 = BatchNormalization()(conv4)
    relu4 = ReLU()(batch4)
    drop4 = Dropout(rate=0.1)(relu4)
    conv4 = Conv2D(filters=128, kernel_size=(1, 5), strides=(1, 2))(drop4)

    max3 = MaxPooling2D(pool_size=(1, 9), strides=(1, 2))(conc2)
    conv_ = Conv2D(128, (1, 1))(max3)
    conc3 = Add()([conv4, conv_])

    # Fifth block
    batch5 = BatchNormalization()(conc3)
    relu5 = ReLU()(batch5)
    drop5 = Dropout(rate=0.1)(relu5)
    conv5 = Conv2D(filters=128, kernel_size=(1, 5), strides=(1, 1))(drop5)
    batch5 = BatchNormalization()(conv5)
    relu5 = ReLU()(batch5)
    drop5 = Dropout(rate=0.1)(relu5)
    conv5 = Conv2D(filters=128, kernel_size=(1, 5), strides=(1, 2))(drop5)

    max4 = MaxPooling2D(pool_size=(1, 9), strides=(1, 2))(conc3)
    conc4 = Add()([conv5, max4])

    # Final layers
    conv_final = Conv2D(filters=128, kernel_size=(input_shape[0], 1))(conc4)
    X = BatchNormalization()(conv_final)
    X = ReLU()(X)
    X = GlobalAveragePooling2D()(X)
    X = Dropout(rate=0.2)(X)

    X = Flatten()(X)

    # Dense layers with stronger regularization
    X = Dense(units=128, kernel_regularizer=keras.regularizers.L2(0.01))(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = Dropout(rate=0.3)(X)

    X = Dense(units=64, kernel_regularizer=keras.regularizers.L2(0.015))(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = Dropout(rate=0.3)(X)

    output = Dense(num_classes, activation="sigmoid")(X)

    model = Model(inputs=input_layer, outputs=output)

    return model
