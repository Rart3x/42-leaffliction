import argparse
from typing import Optional
import matplotlib.pyplot as plt
import os


from tensorflow.keras.callbacks import (ReduceLROnPlateau,
                                        EarlyStopping,
                                        ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.model import create_model


def parse_args() -> tuple[str, Optional[str]]:
    parser = argparse.ArgumentParser(prog='train')

    parser.add_argument('data_path',
                        help='Direct path to the data folder')
    parser.add_argument('-m',
                        '--model',
                        dest='model',
                        help='Path to save/load the model')
    v_args = parser.parse_args()
    v_path = v_args.data_path
    v_model = v_args.model
    return v_path, v_model


def create_data_generators(p_path: str,
                           batch_size: int = 16,
                           img_size: tuple[int, int] = (256, 256)):
    """
    Create ImageDataGenerators for memory-efficient data loading.

    :param p_path: Path to the directory containing class subdirectories
    :param batch_size: Number of images to load per batch
    :param img_size: Target image size (width, height)
    :return: Tuple of (train_generator, validation_generator, num_classes)
    """
    if not os.path.isdir(p_path):
        raise ValueError(f"Directory does not exist: {p_path}")

    # Create ImageDataGenerator with DATA AUGMENTATION for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        fill_mode='nearest',
        validation_split=0.2  # 80% train, 20% validation
    )

    # Validation generator - NO augmentation, only rescaling
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        p_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    # Validation data generator (no augmentation)
    validation_generator = val_datagen.flow_from_directory(
        p_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    num_classes = len(train_generator.class_indices)

    print(f"\nFound {train_generator.samples} training images")
    print(f"Found {validation_generator.samples} validation images")
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {train_generator.class_indices}\n")

    return train_generator, validation_generator, num_classes


def main():
    v_path, v_model = parse_args()

    # Use memory-efficient data generators instead of loading all data at once
    batch_size = 16

    (train_generator,
     validation_generator,
     num_classes) = create_data_generators(
        v_path, batch_size=batch_size
    )

    model = create_model(num_classes=num_classes)

    if v_model:
        model.load_weights(v_model)

    # Compile with lower learning rate
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Callbacks for better training
    callbacks = [
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # Stop early if not improving
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model during training
        ModelCheckpoint(
            'best_model.weights.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]

    try:
        print("\n" + "="*60)
        print("Starting training with improved architecture...")
        print("="*60 + "\n")

        # Use fit with generators for memory-efficient training
        history = model.fit(
            train_generator,
            epochs=50,  # More epochs but will stop early if not improving
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )

        val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
        print(f'\nValidation Accuracy: {val_acc:.4f}')

        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Leaf Disease Classification')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        print("\nTraining history saved to 'training_history.png'")

        # Save the model
        model.save('leaf_disease_model.h5')
        print("Model saved to 'leaf_disease_model.h5'")
    except KeyboardInterrupt:
        model.save_weights('checkpoint.weights.h5')
        print("Training canceled by user. "
              "Weights saved to 'checkpoint.weights.h5'")
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()
