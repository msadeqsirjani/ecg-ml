from tensorflow.keras.layers import (
    Conv2D,
    Add,
    Activation,
    Dropout,
    Dense,
    Flatten,
    Input,
    BatchNormalization,
    ReLU,
    MaxPooling2D,
    Concatenate,
    GlobalAveragePooling2D,
    DepthwiseConv2D,
)
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer
from tensorflow_model_optimization.quantization.keras import quantize_apply

def create_micro_ecg_model(
    input_shape: tuple,
    num_classes: int,
    enable_pruning: bool = True,
    enable_quantization: bool = True,
    enable_distillation: bool = True,
    teacher_model: Model = None,
) -> Model:
    """
    Create a compressed version of the ECG model with optional pruning, quantization,
    and knowledge distillation techniques.

    Args:
        input_shape: Tuple of (num_leads, sequence_length, channels)
        num_classes: Number of output classes
        enable_pruning: Whether to enable weight pruning
        enable_quantization: Whether to enable quantization
        enable_distillation: Whether to enable knowledge distillation
        teacher_model: The teacher model for knowledge distillation

    Returns:
        Compiled Keras model
    """
    # Define pruning parameters
    pruning_params = {
        'initial_sparsity': 0.0,
        'final_sparsity': 0.5,
        'begin_step': 0,
        'end_step': 1000,
        'frequency': 100
    }

    # Create the model architecture
    input_layer = Input(shape=input_shape)

    # First block
    conv1 = Conv2D(filters=16, kernel_size=(1, 5), strides=(1, 1))(input_layer)
    if enable_pruning:
        conv1 = sparsity.prune_low_magnitude(conv1, **pruning_params)
    if enable_quantization:
        conv1 = quantize_annotate_layer(conv1)
    batch1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch1)

    # Second block
    conv2 = DepthwiseConv2D(kernel_size=(1, 5), strides=(1, 1))(relu1)
    if enable_pruning:
        conv2 = sparsity.prune_low_magnitude(conv2, **pruning_params)
    if enable_quantization:
        conv2 = quantize_annotate_layer(conv2)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1))(conv2)
    if enable_pruning:
        conv2 = sparsity.prune_low_magnitude(conv2, **pruning_params)
    if enable_quantization:
        conv2 = quantize_annotate_layer(conv2)
    batch2 = BatchNormalization()(conv2)
    relu2 = ReLU()(batch2)
    drop2 = Dropout(rate=0.1)(relu2)
    conv2 = Conv2D(filters=32, kernel_size=(1, 5), strides=(1, 2))(drop2)
    if enable_pruning:
        conv2 = sparsity.prune_low_magnitude(conv2, **pruning_params)
    if enable_quantization:
        conv2 = quantize_annotate_layer(conv2)

    max1 = MaxPooling2D(pool_size=(1, 7), strides=(1, 2))(relu1)
    conv_ = Conv2D(32, (1, 1))(max1)
    if enable_pruning:
        conv_ = sparsity.prune_low_magnitude(conv_, **pruning_params)
    if enable_quantization:
        conv_ = quantize_annotate_layer(conv_)
    conc1 = Add()([conv2, conv_])

    # Third block
    batch3 = BatchNormalization()(conc1)
    relu3 = ReLU()(batch3)
    drop3 = Dropout(rate=0.1)(relu3)
    conv3 = Conv2D(filters=32, kernel_size=(1, 5), strides=(1, 1))(drop3)
    if enable_pruning:
        conv3 = sparsity.prune_low_magnitude(conv3, **pruning_params)
    if enable_quantization:
        conv3 = quantize_annotate_layer(conv3)
    batch3 = BatchNormalization()(conv3)
    relu3 = ReLU()(batch3)
    drop3 = Dropout(rate=0.1)(relu3)
    conv3 = Conv2D(filters=32, kernel_size=(1, 5), strides=(1, 2))(drop3)
    if enable_pruning:
        conv3 = sparsity.prune_low_magnitude(conv3, **pruning_params)
    if enable_quantization:
        conv3 = quantize_annotate_layer(conv3)

    max2 = MaxPooling2D(pool_size=(1, 7), strides=(1, 2))(conc1)
    conc2 = Add()([conv3, max2])

    # Fourth block
    batch4 = BatchNormalization()(conc2)
    relu4 = ReLU()(batch4)
    drop4 = Dropout(rate=0.1)(relu4)
    conv4 = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 1))(drop4)
    if enable_pruning:
        conv4 = sparsity.prune_low_magnitude(conv4, **pruning_params)
    if enable_quantization:
        conv4 = quantize_annotate_layer(conv4)
    batch4 = BatchNormalization()(conv4)
    relu4 = ReLU()(relu4)
    drop4 = Dropout(rate=0.1)(drop4)
    conv4 = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 2))(drop4)
    if enable_pruning:
        conv4 = sparsity.prune_low_magnitude(conv4, **pruning_params)
    if enable_quantization:
        conv4 = quantize_annotate_layer(conv4)

    max3 = MaxPooling2D(pool_size=(1, 7), strides=(1, 2))(conc2)
    conv_ = Conv2D(64, (1, 1))(max3)
    if enable_pruning:
        conv_ = sparsity.prune_low_magnitude(conv_, **pruning_params)
    if enable_quantization:
        conv_ = quantize_annotate_layer(conv_)
    conc3 = Add()([conv4, conv_])

    # Final layers
    conv_final = Conv2D(filters=64, kernel_size=(input_shape[0], 1))(conc3)
    if enable_pruning:
        conv_final = sparsity.prune_low_magnitude(conv_final, **pruning_params)
    if enable_quantization:
        conv_final = quantize_annotate_layer(conv_final)
    X = BatchNormalization()(conv_final)
    X = ReLU()(X)
    X = GlobalAveragePooling2D()(X)
    X = Dropout(rate=0.2)(X)

    X = Flatten()(X)

    # Dense layers
    X = Dense(units=64, kernel_regularizer=tf.keras.regularizers.L2(0.02))(X)
    if enable_pruning:
        X = sparsity.prune_low_magnitude(X, **pruning_params)
    if enable_quantization:
        X = quantize_annotate_layer(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = Dropout(rate=0.3)(X)

    X = Dense(units=32, kernel_regularizer=tf.keras.regularizers.L2(0.02))(X)
    if enable_pruning:
        X = sparsity.prune_low_magnitude(X, **pruning_params)
    if enable_quantization:
        X = quantize_annotate_layer(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = Dropout(rate=0.3)(X)

    # Output layer
    output = Dense(num_classes, activation="sigmoid")(X)
    if enable_pruning:
        output = sparsity.prune_low_magnitude(output, **pruning_params)
    if enable_quantization:
        output = quantize_annotate_layer(output)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    # Apply quantization if enabled
    if enable_quantization:
        model = quantize_apply(model)

    # Add knowledge distillation if enabled
    if enable_distillation and teacher_model is not None:
        # Create a custom loss that combines the original loss with distillation
        def distillation_loss(y_true, y_pred):
            # Original loss
            original_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            
            # Get teacher predictions
            teacher_pred = teacher_model(input_layer, training=False)
            
            # Distillation loss (KL divergence)
            distillation_loss = tf.keras.losses.kl_divergence(
                tf.nn.softmax(teacher_pred / 2.0),
                tf.nn.softmax(y_pred / 2.0)
            ) * (2.0 * 2.0)  # Scale factor
            
            return original_loss + 0.5 * distillation_loss

        model.compile(
            optimizer='adam',
            loss=distillation_loss,
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    return model
