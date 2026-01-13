import argparse
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.layers import (Input,
                                     Conv2D,
                                     MaxPooling2D,
                                     GlobalAveragePooling2D,
                                     Dense,
                                     Activation,
                                     Dropout,
                                     BatchNormalization)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        prog='predict',
        description='Predict leaf disease from an image using a trained model'
    )

    parser.add_argument('model_path',
                        help='Path to the .h5 model file')
    parser.add_argument('image_path',
                        help='Path to the image to predict')

    args = parser.parse_args()
    return args.model_path, args.image_path


def create_model(num_classes: int = 8):
    """
    Create the same model architecture as in train.py

    :param num_classes: Number of output classes
    :return: Compiled model
    """
    model = Sequential()
    model.add(Input(shape=(256, 256, 3)))

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


def load_and_preprocess_image(image_path: str,
                              target_size: tuple = (256, 256)):
    """
    Load and preprocess an image for prediction

    :param image_path: Path to the image file
    :param target_size: Target size for the image (width, height)
    :return: Preprocessed image array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    img = image.load_img(image_path, target_size=target_size)

    # Convert to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize pixel values (same as training: rescale=1./255)
    img_array = img_array / 255.0

    return img_array


def predict(model_path: str, image_path: str, class_names: list = CLASS_NAMES):
    """
    Make a prediction on an image using a trained model

    :param model_path: Path to the .h5 model file
    :param image_path: Path to the image to predict
    :param class_names: List of class names
    """
    print(f"\nLoading model from: {model_path}")

    # Create model architecture
    model = create_model(num_classes=len(class_names))

    # Load weights
    if model_path.endswith('.weights.h5'):
        model.load_weights(model_path)
    else:
        # If it's a full model file
        model = tf.keras.models.load_model(model_path)

    print("Model loaded successfully")
    print(f"\nLoading and preprocessing image: {image_path}")

    # Load and preprocess image
    img_array = load_and_preprocess_image(image_path)

    print("Making prediction...")

    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]

    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nPredicted Class: {class_names[predicted_class_idx]}")
    print(f"Confidence: {confidence * 100:.2f}%")

    print("\nAll class probabilities:")
    print("-" * 60)

    # Sort predictions by confidence
    sorted_indices = np.argsort(predictions[0])[::-1]

    for idx in sorted_indices:
        class_name = class_names[idx]
        prob = predictions[0][idx]
        print(f"{class_name:25s} : {prob * 100:6.2f}%")

    print("=" * 60 + "\n")

    return class_names[predicted_class_idx], confidence


def main():
    """Main function"""
    model_path, image_path = parse_args()

    try:
        predicted_class, confidence = predict(model_path, image_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        exit(1)
    except Exception as e:
        print(f"\nError during prediction: {e}")
        exit(1)


if __name__ == "__main__":
    main()
