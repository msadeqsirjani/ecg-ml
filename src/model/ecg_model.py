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
)
from tensorflow.keras.models import Model
import tensorflow as tf


def create_ecg_model(input_shape: tuple, num_classes: int) -> Model:
    """
    Create the ECG classification model architecture.

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

    X = Flatten()(X)

    # Dense layers
    X = Dense(units=128, kernel_regularizer=tf.keras.regularizers.L2(0.005))(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = Dropout(rate=0.1)(X)

    X = Dense(units=64, kernel_regularizer=tf.keras.regularizers.L2(0.009))(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = Dropout(rate=0.15)(X)

    output = Dense(num_classes, activation="sigmoid")(X)

    model = Model(inputs=input_layer, outputs=output)

    return model
