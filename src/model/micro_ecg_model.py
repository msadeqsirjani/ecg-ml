from tensorflow.keras.layers import (
    Dense,
    Input,
    BatchNormalization,
    ReLU,
    MaxPooling2D,
    GlobalAveragePooling2D,
    DepthwiseConv2D,
    Conv2D,
)
from tensorflow.keras.models import Model
import tensorflow as tf

def create_micro_ecg_model(input_shape: tuple, num_classes: int) -> Model:
    """
    Create an extremely tiny ECG model targeting ~50-100KB file size.
    Uses aggressive compression techniques like weight quantization in the design.

    Args:
        input_shape: Tuple of (num_leads, sequence_length, channels)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    input_layer = Input(shape=input_shape)

    # Initial feature extraction with strong dimensionality reduction
    # Use a single standard conv with minimal filters
    x = Conv2D(filters=4, kernel_size=(1, 9), strides=(1, 4))(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(1, 4), strides=(1, 4))(x)

    # Single depthwise separable block
    x = DepthwiseConv2D(kernel_size=(1, 5), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=8, kernel_size=(1, 1))(x)  # Pointwise conv
    x = MaxPooling2D(pool_size=(1, 4), strides=(1, 4))(x)

    # Final feature aggregation
    x = DepthwiseConv2D(kernel_size=(input_shape[0], 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Global pooling to avoid dense layers
    x = GlobalAveragePooling2D()(x)

    # Minimal classification head
    x = Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.02))(
        x
    )
    output = Dense(num_classes, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output)

    return model
