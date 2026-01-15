# Leaffliction - Leaf Disease Classification

A deep learning system for automated classification of plant leaf diseases using convolutional neural networks. The model identifies healthy leaves and specific disease types in apple and grape plants, achieving over 90% accuracy on test data.

## Table of Contents
- [Overview](#overview)
- [Classes Detected](#classes-detected)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)

## Overview

Leaffliction uses computer vision and deep learning to help farmers and agricultural professionals quickly identify leaf diseases in apple and grape crops. Early disease detection is critical for crop management and can significantly reduce losses.

The system is built on TensorFlow/Keras and uses a custom CNN architecture optimized for leaf disease classification. The model processes 256×256 RGB images and outputs predictions across 8 disease categories.

## Classes Detected

The model classifies leaves into 8 categories:

**Apple (4 classes):**
- Black rot
- Healthy
- Rust
- Scab

**Grape (4 classes):**
- Black rot
- Esca
- Healthy
- Leaf spot

## Model Architecture

Custom CNN designed for multi-class leaf disease classification:

**Input Layer:**
- Image size: 256×256×3 (RGB)
- Preprocessing: Pixel normalization (0-1 range)

**Convolutional Blocks:**
The architecture consists of 4 progressive convolutional blocks with increasing filter depth:

1. **Block 1** (32 filters):
   - Conv2D(32, 3×3) → BatchNormalization → ReLU
   - Conv2D(32, 3×3) → BatchNormalization → ReLU
   - MaxPooling2D(2×2)
   - Dropout(0.25)

2. **Block 2** (64 filters):
   - Conv2D(64, 3×3) → BatchNormalization → ReLU
   - Conv2D(64, 3×3) → BatchNormalization → ReLU
   - MaxPooling2D(2×2)
   - Dropout(0.25)

3. **Block 3** (128 filters):
   - Conv2D(128, 3×3) → BatchNormalization → ReLU
   - Conv2D(128, 3×3) → BatchNormalization → ReLU
   - MaxPooling2D(2×2)
   - Dropout(0.25)

4. **Block 4** (256 filters):
   - Conv2D(256, 3×3) → BatchNormalization → ReLU
   - Conv2D(256, 3×3) → BatchNormalization → ReLU
   - MaxPooling2D(2×2)
   - Dropout(0.25)

**Classification Head:**
- GlobalAveragePooling2D → Reduces spatial dimensions
- Dense(256) → BatchNormalization → ReLU → Dropout(0.5)
- Dense(8, softmax) → Output layer for 8 classes

**Training Configuration:**
- Optimizer: Adam (learning rate = 0.001)
- Loss: Categorical crossentropy
- Metrics: Accuracy
- Regularization: Batch normalization + Dropout (prevents overfitting)

**Key Features:**
- Progressive feature extraction with increasing filter depth
- Batch normalization for stable training
- Dropout layers to reduce overfitting
- Global average pooling to reduce parameters
- Softmax activation for multi-class probability distribution

## Dataset

### Structure
```
leaves/images/
├── Apple_Black_rot/
├── Apple_healthy/
├── Apple_rust/
├── Apple_scab/
├── Grape_Black_rot/
├── Grape_Esca/
├── Grape_healthy/
└── Grape_spot/
```

Each subdirectory contains JPEG images of the corresponding disease category.

### Data Split
- **Training:** 80% of data with augmentation
- **Validation:** 20% of training data (no augmentation)
- **Test:** Separate test set in `test_set/` directory

### Data Augmentation
Training images are augmented with:
- Rotation (±40 degrees)
- Width/height shift (±20%)
- Shear transformation (20%)
- Zoom (±20%)
- Horizontal flip
- Brightness adjustment
- Contrast adjustment
- Blur effects
- Perspective transformation

Augmentation is applied only to training data to prevent information leakage.

## Setup

### Prerequisites
- Python 3.8+
- pip package manager
- GPU recommended for faster training (optional)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd 42-leaffliction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- TensorFlow 2.20.0
- NumPy 2.2.6
- Matplotlib 3.10.8
- scikit-learn 1.7.2
- Pillow 12.0.0
- seaborn

3. **Prepare dataset:**
Place your leaf images in the appropriate subdirectories under `leaves/images/`.

## Usage

### Train the Model
```bash
python3 train.py <data_path> [-m MODEL]
```

Trains the CNN model on the dataset. The best model weights are saved based on validation accuracy.

**Arguments:**
- `data_path` (positional, required): Direct path to the data folder containing training images (e.g., `leaves/images/`)
- `-m, --model MODEL` (optional): Path to save/load the model weights file (default: `models/best_model.weights.h5`)

**Examples:**
```bash
python3 train.py leaves/images/
python3 train.py leaves/images/ -m models/my_model.weights.h5
```

**Training features:**
- Automatic learning rate reduction on plateau
- Early stopping to prevent overfitting
- Model checkpointing (saves best model)
- Real-time validation monitoring

### Evaluate the Model
```bash
python3 evaluate.py <model_path> <test_dir> [--batch-size BATCH_SIZE] [--output-dir OUTPUT_DIR]
```

Evaluates the trained model on the test dataset and generates comprehensive performance metrics.

**Arguments:**
- `model_path` (positional, required): Path to the `.h5` model weights file
- `test_dir` (positional, required): Path to test dataset directory (should contain subdirectories for each class)
- `--batch-size BATCH_SIZE` (optional): Batch size for evaluation (default: 32)
- `--output-dir OUTPUT_DIR` (optional): Directory to save evaluation results (default: `evaluation_results`)

**Examples:**
```bash
python3 evaluate.py models/best_model.weights.h5 test_set/
python3 evaluate.py models/best_model.weights.h5 test_set/ --batch-size 64 --output-dir my_results/
```

**Output:**
- Overall accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score per class)
- List of misclassified images saved to the output directory

### Predict Single Image
```bash
python3 predict.py <model_path> <image_path>
```

Makes a prediction on a single leaf image and outputs the predicted class with confidence scores.

**Arguments:**
- `model_path` (positional, required): Path to the `.h5` model weights file
- `image_path` (positional, required): Path to the image to predict

**Examples:**
```bash
python3 predict.py models/best_model.weights.h5 path/to/leaf/image.jpg
python3 predict.py models/best_model.weights.h5 leaves/images/Apple_healthy/image001.jpg
```

**Output:**
- Predicted class
- Confidence scores for all 8 classes

### Visualize Dataset Distribution
```bash
python3 Distribution.py <directory_path>
```

Generates pie charts and bar plots showing the distribution of images across classes.

**Arguments:**
- `directory_path` (positional, required): Path to directory containing image subdirectories organized by class

**Examples:**
```bash
python3 Distribution.py leaves/images/
```

**Output:**
- Pie chart showing class distribution
- Bar plot showing number of images per class

### Create Test Split
```bash
python3 create_test_split.py <source_dir> <test_dir> [--test-size TEST_SIZE] [--min-test-per-class MIN_TEST_PER_CLASS] [--seed SEED] [--copy]
```

Splits a portion of the dataset into a separate test set for model evaluation.

**Arguments:**
- `source_dir` (positional, required): Source directory containing training images (e.g., `leaves/images/`)
- `test_dir` (positional, required): Destination directory for test images
- `--test-size TEST_SIZE` (optional): Fraction of images to use for testing (default: 0.15)
- `--min-test-per-class MIN_TEST_PER_CLASS` (optional): Minimum number of test images per class (default: 15)
- `--seed SEED` (optional): Random seed for reproducibility (default: 42)
- `--copy` (flag): Copy files instead of moving them (keeps original data intact)

**Examples:**
```bash
python3 create_test_split.py leaves/images/ test_set/
python3 create_test_split.py leaves/images/ test_set/ --test-size 0.2 --copy
python3 create_test_split.py leaves/images/ test_set/ --test-size 0.15 --min-test-per-class 20 --seed 123
```

### Apply Data Augmentation
```bash
python3 Augmentation.py [-v] [-l LIMIT] <image_paths...>
```

Applies various augmentation techniques to expand the training dataset. Supports both single image visualization and batch processing of multiple images.

**Arguments:**
- `image_paths` (positional): One or more image file paths or glob patterns (e.g., `./images/*/*`)
- `-v, --visual` (flag): Enable visual rendering of augmented images
- `-l, --limit LIMIT` (optional): Target number of images per class (default: 1640)

**Behavior:**
- When processing a directory with glob patterns, generates augmented images until each class reaches the target limit
- Each original image can generate up to 6 augmented variants (rotated, blurred, contrasted, illuminated, scaled, projected)
- Skips already augmented images (those ending with augmentation suffixes)

**Examples:**
```bash
# Single image with visual display
python3 Augmentation.py -v ./image.jpg

# Process all images in subdirectories with custom limit
python3 Augmentation.py ./images/*/* --limit 1500

# Process specific directory pattern
python3 Augmentation.py leaves/images/Apple_healthy/*.jpg --limit 2000
```

### Apply Image Transformations
```bash
python3 Transformation.py <image_path> [-dst DESTINATION] [-v]
```

Applies various image transformations including Gaussian blur, masking, ROI detection, object analysis, pseudolandmarks, and color histogram analysis.

**Arguments:**
- `image_path` (positional, required): Path to a single image file OR directory path containing images
- `-dst, --destination DESTINATION` (required for directory mode): Destination directory for saving transformations (required when `image_path` is a directory, ignored for single image)
- `-v, --visual` (flag): Enable visual rendering of transformations

**Behavior:**
- **Single image mode**: When given a single image file, displays all transformations visually in an interactive window
  - Shows: Original, Gaussian blur, Mask applied, ROI detection, Analyzed objects, Pseudolandmarks, Color histogram
  - Clicking on any image opens it in a separate window
- **Directory mode**: When given a directory path, processes all JPG images and saves transformations to the destination directory
  - Requires `-dst` argument
  - Saves transformed images with suffixes: `_gaussian_blur`, `_mask_applied`, `_ROI_detection`, `_analyzed_objects`, `_pseudolandmarks`

**Examples:**
```bash
# Single image with visual display
python3 Transformation.py image.jpg -v

# Process directory and save transformations
python3 Transformation.py Apple/apple_healthy/ -dst dst_directory/

# Process directory with visual feedback
python3 Transformation.py leaves/images/Apple_healthy/ -dst output/ -v
```

## Training

### Training Process
The model is trained with the following configuration:

- **Batch size:** 32
- **Epochs:** 100 (with early stopping)
- **Image size:** 256×256 pixels
- **Color mode:** RGB
- **Class mode:** Categorical (one-hot encoding)

### Callbacks
1. **ReduceLROnPlateau:**
   - Monitors validation loss
   - Reduces learning rate by factor of 0.5 when plateau detected
   - Patience: 5 epochs

2. **EarlyStopping:**
   - Monitors validation loss
   - Stops training if no improvement
   - Patience: 10 epochs
   - Restores best weights

3. **ModelCheckpoint:**
   - Saves model weights after each epoch
   - Keeps only the best model based on validation accuracy

### Training Data Flow
- Images loaded from `leaves/images/` directory
- Real-time augmentation applied during training
- Validation data kept separate without augmentation
- Batch processing for memory efficiency

## Evaluation

The evaluation script provides comprehensive model performance analysis:

### Metrics Generated
- **Accuracy:** Overall percentage of correct predictions
- **Confusion Matrix:** Visual representation of predictions vs. actual classes
- **Classification Report:** Per-class metrics including:
  - Precision: Accuracy of positive predictions
  - Recall: Coverage of actual positives
  - F1-Score: Harmonic mean of precision and recall
  - Support: Number of samples per class

### Output Files
Results are saved in `evaluation_results/` with timestamps:
- Confusion matrix plot
- Classification report text file
- Misclassified images list with predicted vs. actual labels

### Performance Threshold
The model is designed to achieve **>90% accuracy** on the test set to meet production requirements.

## Results

### Model Performance
The trained model achieves over **90% accuracy** on the independent test dataset, demonstrating strong generalization to unseen leaf images.

### Key Findings
- High precision across all disease categories
- Balanced performance between apple and grape classes
- Effective distinction between healthy and diseased leaves
- Robust to variations in lighting, background, and leaf orientation

### Detailed Results
Complete evaluation metrics including confusion matrices, per-class performance, and misclassified examples are available in the `evaluation_results/` directory after running `evaluate.py`.

## Project Structure

```
42-leaffliction/
├── train.py                    # Main training script
├── evaluate.py                 # Model evaluation and metrics
├── predict.py                  # Single image prediction
├── Distribution.py             # Dataset visualization
├── Augmentation.py             # Data augmentation utilities
├── Transformation.py           # Image transformation functions
├── create_test_split.py        # Test set creation
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── utils/
│   ├── model.py               # CNN model architecture
│   └── utils.py               # Helper functions
│
├── leaves/
│   └── images/                # Training dataset
│       ├── Apple_Black_rot/
│       ├── Apple_healthy/
│       ├── Apple_rust/
│       ├── Apple_scab/
│       ├── Grape_Black_rot/
│       ├── Grape_Esca/
│       ├── Grape_healthy/
│       └── Grape_spot/
│
├── test_set/                  # Test dataset (separate)
│   ├── split_info.txt
│   └── [disease categories]
│
├── models/
│   └── best_model.weights.h5  # Trained model weights
│
└── evaluation_results/        # Evaluation outputs
    └── [timestamped results]
```

---

**Project:** 42-leaffliction  
**Task:** Leaf disease classification using deep learning  
**Framework:** TensorFlow/Keras  
**Accuracy Target:** >90%
