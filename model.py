"""
Shared model architecture and configuration for leaf disease classification.

This module contains the CNN model architecture and constants used across
training, prediction, and evaluation scripts.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Activation,
    Dropout,
    BatchNormalization
)
import tensorflow as tf


# Model configuration constants
CLASS_NAMES = [
    'Apple_Black_rot',
    'Apple_healthy',
    'Apple_rust',
    'Apple_scab',
    'Grape_Black_rot',
    'Grape_Esca',
    'Grape_healthy',
    'Grape_spot'
]

IMG_SIZE = (256, 256)
INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = len(CLASS_NAMES)
RESCALE_FACTOR = 1./255


def create_model(num_classes=NUM_CLASSES):
    """
    Create the CNN model architecture for leaf disease classification.

    Architecture:
    - Input: 256x256x3
    - 4 Convolutional blocks with increasing filters (32, 64, 128, 256)
    - Each block: 2x Conv2D + BatchNorm + ReLU + MaxPooling + Dropout
    - Global Average Pooling
    - Dense layer (256 units) + BatchNorm + ReLU + Dropout
    - Output layer with softmax activation

    :param num_classes: Number of output classes (default: 8)
    :return: Sequential model (not compiled)
    """
    model = Sequential()
    model.add(Input(shape=INPUT_SHAPE))

    # Block 0
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 1
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten
    model.add(GlobalAveragePooling2D())

    # Fully Connected Layers
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    return model


def load_model_weights(model, model_path):
    """
    Load model weights or full model from file.

    Handles both .weights.h5 files (weights only) and full model files.

    :param model: Model instance to load weights into (ignored for full models)
    :param model_path: Path to the model file
    :return: Loaded model
    """
    if model_path.endswith('.weights.h5'):
        # If it's a weights file, load into provided model
        model.load_weights(model_path)
        return model
    else:
        # If it's a full model file, load the entire model
        return tf.keras.models.load_model(model_path)
