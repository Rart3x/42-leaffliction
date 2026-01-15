import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.preprocessing import image

from Transformation import Transformation
from utils.model import CLASS_NAMES, create_model, load_model_weights


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


def load_and_preprocess_image(image_path: str,
                              target_size: tuple[int, int] = (256, 256)):
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


def create_transformed_image(image_path: str):
    """
    Create a transformed image using the mask from Transformation class
    
    :param image_path: Path to the image file
    :return: Tuple of (original_img, transformed_img) as numpy arrays in RGB
    """
    # Load original image with cv2
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB for display
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Apply transformation using Transformation class
    transformer = Transformation(image_path, p_visual=False)
    
    # Extract HSV saturation channel and apply threshold
    hsv = cv2.cvtColor(transformer.img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask_binary = cv2.threshold(saturation, 60, 255, cv2.THRESH_BINARY)
    
    # Apply mask with white background
    mask_3channel = cv2.merge([mask_binary, mask_binary, mask_binary]) / 255.0
    background = np.ones_like(transformer.img) * 255
    transformed_img = (transformer.img * mask_3channel + 
                      background * (1 - mask_3channel)).astype(np.uint8)
    
    # Convert BGR to RGB for display
    transformed_img_rgb = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
    
    return original_img_rgb, transformed_img_rgb


def display_prediction_with_images(original_img, transformed_img, 
                                   predicted_class, confidence, 
                                   all_predictions, class_names):
    """
    Display original and transformed images with prediction results
    
    :param original_img: Original image as numpy array (RGB)
    :param transformed_img: Transformed image as numpy array (RGB)
    :param predicted_class: Predicted class name
    :param confidence: Confidence percentage
    :param all_predictions: Array of all class probabilities
    :param class_names: List of class names
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Display original image
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Display transformed image
    axes[1].imshow(transformed_img)
    axes[1].set_title("Transformed Image (Mask Applied)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add main title with prediction
    fig.suptitle(f"Predicted: {predicted_class} (Confidence: {confidence * 100:.2f}%)",
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def predict(model_path: str, image_path: str, class_names: list = CLASS_NAMES):
    """
    Make a prediction on an image using a trained model

    :param model_path: Path to the .h5 model file
    :param image_path: Path to the image to predict
    :param class_names: List of class names
    """
    print(f"\nLoading model from: {model_path}")

    # Create model architecture and load weights
    model = create_model(num_classes=len(class_names))
    model = load_model_weights(model, model_path)

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
    
    # Create and display visualization
    print("Generating visualization...")
    try:
        original_img, transformed_img = create_transformed_image(image_path)
        display_prediction_with_images(
            original_img, 
            transformed_img,
            class_names[predicted_class_idx],
            confidence,
            predictions[0],
            class_names
        )
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")


def main():
    """Main function"""
    model_path, image_path = parse_args()

    try:
        predict(model_path, image_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        exit(1)
    except Exception as e:
        print(f"\nError during prediction: {e}")
        exit(1)


if __name__ == "__main__":
    main()
