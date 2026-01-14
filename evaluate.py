import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import create_model, load_model_weights


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        prog='evaluate',
        description='Evaluate leaf disease '
                    'classification model on test dataset'
    )

    parser.add_argument('model_path',
                        help='Path to the .h5 model weights file')
    parser.add_argument('test_dir',
                        help='Path to test dataset '
                             'directory (should contain subdirectories '
                             'for each class)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    parser.add_argument('--output-dir', default='evaluation_results',
                        help='Directory to save '
                             'evaluation results '
                             '(default: evaluation_results)')

    args = parser.parse_args()
    return args


def load_test_data(test_dir: str, batch_size: int = 32):
    """
    Load test data from directory structure

    :param test_dir: Directory containing test images organized by class
    :param batch_size: Batch size for data loading
    :return: Test data generator and number of samples
    """
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Create data generator (only rescaling, no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load test data
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Important: don't shuffle for evaluation
    )

    return test_generator


def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot and save confusion matrix

    :param cm: Confusion matrix array
    :param class_names: List of class names
    :param output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Number of samples'})
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def save_misclassified_examples(y_true,
                                y_pred,
                                filenames,
                                class_names,
                                output_path,
                                max_examples=20):
    """
    Save list of misclassified examples

    :param y_true: True labels (indices)
    :param y_pred: Predicted labels (indices)
    :param filenames: List of image filenames
    :param class_names: List of class names
    :param output_path: Path to save the results
    :param max_examples: Maximum number of examples to save
    """
    misclassified = []

    for i, (true_idx, pred_idx) in enumerate(zip(y_true, y_pred)):
        if true_idx != pred_idx:
            misclassified.append({
                'filename': filenames[i],
                'true_label': class_names[true_idx],
                'predicted_label': class_names[pred_idx]
            })

    # Limit number of examples
    misclassified = misclassified[:max_examples]

    with open(output_path, 'w') as f:
        f.write("MISCLASSIFIED EXAMPLES\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"Total misclassified: {len(misclassified)} "
            f"(showing first {max_examples})\n\n"
        )

        for i, example in enumerate(misclassified, 1):
            f.write(f"{i}. {example['filename']}\n")
            f.write(f"   True: {example['true_label']}\n")
            f.write(f"   Predicted: {example['predicted_label']}\n\n")

    print(f"Misclassified examples saved to: {output_path}")


def evaluate_model(model_path: str,
                   test_dir: str,
                   batch_size: int = 32,
                   output_dir: str = 'evaluation_results'):
    """
    Evaluate model on test dataset and generate comprehensive metrics

    :param model_path: Path to model weights file
    :param test_dir: Path to test dataset directory
    :param batch_size: Batch size for evaluation
    :param output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("LEAF DISEASE CLASSIFICATION - MODEL EVALUATION")
    print("="*80 + "\n")

    # Load test data
    print(f"Loading test data from: {test_dir}")
    test_generator = load_test_data(test_dir, batch_size)

    num_samples = test_generator.samples
    num_classes = len(test_generator.class_indices)

    print(f"✓ Found {num_samples} test images")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Classes: {list(test_generator.class_indices.keys())}\n")

    if num_samples < 100:
        print(f"WARNING: Test set contains only "
              f"{num_samples} images (minimum requirement: 100)")
        print("Consider creating a larger test "
              "set for more reliable evaluation.\n")

    # Create and load model
    print(f"Loading model from: {model_path}")
    model = create_model(num_classes=num_classes)
    model = load_model_weights(model, model_path)

    print("✓ Model loaded successfully\n")

    # Make predictions
    print("Running predictions on test set...")
    predictions = model.predict(test_generator, verbose=1)

    # Get predicted classes
    y_pred = np.argmax(predictions, axis=1)

    # Get true classes
    y_true = test_generator.classes

    # Get class names in correct order
    class_indices = test_generator.class_indices
    class_names = [k for k, v in sorted(class_indices.items(),
                                        key=lambda item: item[1])]

    print("\n" + "-"*80)
    print("EVALUATION RESULTS")
    print("-"*80 + "\n")

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")

    # Check if meets requirement
    if overall_accuracy >= 0.90:
        print("PASSED: Accuracy exceeds 90% requirement")
    else:
        print(f"FAILED: Accuracy is below "
              f"90% requirement (current: {overall_accuracy*100:.2f}%)")

    print("\n")

    # Generate classification report
    report = classification_report(y_true,
                                   y_pred,
                                   target_names=class_names,
                                   digits=4)
    print("Classification Report:")
    print("-"*80)
    print(report)

    # Get classification report as dict for plotting
    report_dict = classification_report(y_true,
                                        y_pred,
                                        target_names=class_names,
                                        output_dict=True)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("-"*80)
    print(cm)
    print()

    # Plot confusion matrix
    cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    # Save misclassified examples
    misclassified_path = os.path.join(output_dir, 'misclassified.txt')
    save_misclassified_examples(y_true,
                                y_pred,
                                test_generator.filenames,
                                class_names,
                                misclassified_path)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")

    if overall_accuracy >= 0.90:
        print("\nSUCCESS: Model meets the "
              "90% accuracy requirement for Part 4")
    else:
        print("\nModel does not meet the "
              "90% accuracy requirement")
        print(f"  Current: {overall_accuracy*100:.2f}% | Required: 90.00%")

    print()

    return overall_accuracy, report_dict, cm


def main():
    """Main function"""
    args = parse_args()

    try:
        evaluate_model(
            model_path=args.model_path,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
