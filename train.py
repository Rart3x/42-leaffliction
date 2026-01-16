import argparse
from typing import Optional
import matplotlib.pyplot as plt
import os
import shutil
import zipfile
from datetime import datetime
from PIL import Image

from tensorflow.keras.callbacks import (ReduceLROnPlateau,
                                        EarlyStopping,
                                        ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.model import create_model
from Augmentation import Augmentation


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

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        p_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

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


def generate_augmented_images(p_path: str,
                              output_dir: str = 'augmented_data',
                              target_count: int = 1640,
                              augment_limit: int = 6):
    """
    Generate and save augmented images for all classes in the dataset
    using Augmentation class.

    :param p_path: Path to the directory containing class subdirectories
    :param output_dir: Directory to save augmented images
    :param target_count: Target number of images per class
    :param augment_limit: Maximum number of augmentation
           methods to apply per image
    :return: Path to augmented data directory
    """
    if not os.path.isdir(p_path):
        raise ValueError(f"Directory does not exist: {p_path}")

    os.makedirs(output_dir, exist_ok=True)

    total_images = 0

    for class_name in os.listdir(p_path):
        class_path = os.path.join(p_path, class_name)

        if not os.path.isdir(class_path):
            continue

        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        print(f"\nProcessing class: {class_name}")

        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg',
                                              '.jpeg',
                                              '.png',
                                              '.gif',
                                              '.bmp'))]

        valid_images_count = len(image_files)
        needed = max(0, target_count - valid_images_count)

        if needed == 0:
            print(f"  Class '{class_name}' already has "
                  f"{valid_images_count} images. Skipping augmentation.")
            for img_file in image_files:
                src = os.path.join(class_path, img_file)
                dst = os.path.join(output_class_dir, img_file)
                shutil.copy2(src, dst)
                total_images += 1
            continue

        print(f"  Found {valid_images_count} images. "
              f"Generating {needed} more.")

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)

            if not img_file.lower().endswith(('.jpg', '.jpeg')):
                try:
                    img = Image.open(img_path).convert('RGB')
                    jpg_filename = os.path.splitext(img_file)[0] + '.jpg'
                    jpg_path = os.path.join(output_class_dir, jpg_filename)
                    img.save(jpg_path, 'JPEG')
                    total_images += 1
                except Exception as e:
                    print(f"  Error converting {img_file}: {e}")
                continue

            dst_original = os.path.join(output_class_dir, img_file)
            shutil.copy2(img_path, dst_original)
            total_images += 1

        for img_idx, img_file in enumerate(image_files):
            if needed <= 0:
                break

            img_path = os.path.join(class_path, img_file)

            if not img_file.lower().endswith(('.jpg', '.jpeg')):
                continue

            try:
                temp_img_path = os.path.join(output_class_dir,
                                             f"temp_{img_idx}_{img_file}")
                shutil.copy2(img_path, temp_img_path)

                aug = Augmentation(temp_img_path, p_visual_mode=False)
                aug_methods = [
                    aug.rotation,
                    aug.blur,
                    aug.contrast,
                    aug.scaling,
                    aug.illumination,
                    aug.projective
                ]

                for method_idx, method in enumerate(aug_methods):
                    if needed <= 0:
                        break
                    if method_idx < augment_limit:
                        method()
                        total_images += 1
                        needed -= 1

                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

                if (img_idx + 1) % 5 == 0:
                    print(f"  Processed {img_idx + 1}/{len(image_files)} "
                          f"images (needed: {needed})")

            except Exception as e:
                print(f"  Error augmenting {img_file}: {e}")
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                continue

        print(f"  Completed class {class_name}")

    print(f"\nTotal images in augmented dataset: {total_images}")
    return output_dir


def create_training_package(data_dir: str,
                            model_path: str = 'leaf_disease_model.h5',
                            weights_path: str = 'best_model.weights.h5',
                            history_path: str = 'training_history.png'):
    """
    Create a zip file containing augmented images and model files.

    :param data_dir: Directory containing augmented images
    :param model_path: Path to saved model
    :param weights_path: Path to model weights
    :param history_path: Path to training history plot
    :param augmented_data_dir: Name for augmented data in zip
    :return: Path to created zip file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f'leaf_disease_training_package_{timestamp}.zip'

    print(f"\nCreating training package: {zip_filename}")

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.isdir(data_dir):
            print(f"  Adding augmented images from {data_dir}")
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('augmented_images',
                                           os.path.relpath(file_path,
                                                           data_dir))
                    zipf.write(file_path, arcname)

        if os.path.exists(model_path):
            print(f"  Adding model: {model_path}")
            zipf.write(model_path, 'models/' + os.path.basename(model_path))

        if os.path.exists(weights_path):
            print(f"  Adding weights: {weights_path}")
            zipf.write(weights_path,
                       'models/' + os.path.basename(weights_path))

        if os.path.exists(history_path):
            print(f"  Adding training history: {history_path}")
            zipf.write(history_path, os.path.basename(history_path))

    print(f"Training package created: {zip_filename}")
    return zip_filename


def main():
    v_path, v_model = parse_args()

    batch_size = 16

    (train_generator,
     validation_generator,
     num_classes) = create_data_generators(
        v_path, batch_size=batch_size
    )

    model = create_model(num_classes=num_classes)

    if v_model:
        model.load_weights(v_model)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
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
        print("Generating augmented training data...")
        print("="*60 + "\n")

        augmented_data_dir = generate_augmented_images(
            v_path,
            output_dir='augmented_data',
            target_count=1640,
            augment_limit=6
        )

        print("\n" + "="*60)
        print("Starting training with improved architecture...")
        print("="*60 + "\n")

        history = model.fit(
            train_generator,
            epochs=50,
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

        model.save('leaf_disease_model.h5')
        print("Model saved to 'leaf_disease_model.h5'")

        print("\n" + "="*60)
        print("Creating training package...")
        print("="*60)

        zip_file = create_training_package(
            augmented_data_dir,
            model_path='leaf_disease_model.h5',
            weights_path='best_model.weights.h5',
            history_path='training_history.png',
            augmented_data_dir='augmented_data'
        )

        print("\n" + "="*60)
        print("Training complete!")
        print(f"Package saved as: {zip_file}")
        print("="*60)

    except KeyboardInterrupt:
        model.save_weights('checkpoint.weights.h5')
        print("Training canceled by user. "
              "Weights saved to 'checkpoint.weights.h5'")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
