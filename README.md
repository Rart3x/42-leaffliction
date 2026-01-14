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
python train.py
```
Trains the CNN model on the dataset. The best model weights are saved to `models/best_model.weights.h5` based on validation accuracy.

**Training features:**
- Automatic learning rate reduction on plateau
- Early stopping to prevent overfitting
- Model checkpointing (saves best model)
- Real-time validation monitoring

### Evaluate the Model
```bash
python evaluate.py
```
Evaluates the trained model on the test dataset and generates:
- Overall accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score per class)
- List of misclassified images saved to `evaluation_results/`

### Predict Single Image
```bash
python predict.py path/to/leaf/image.jpg
```
Makes a prediction on a single leaf image and outputs:
- Predicted class
- Confidence scores for all 8 classes

### Visualize Dataset Distribution
```bash
python Distribution.py
```
Generates pie charts and bar plots showing the distribution of images across classes.

### Create Test Split
```bash
python create_test_split.py
```
Splits a portion of the dataset into a separate test set.

### Apply Data Augmentation
```bash
python Augmentation.py
```
Applies various augmentation techniques to expand the training dataset.

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
