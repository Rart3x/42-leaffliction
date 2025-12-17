import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from utils import is_jpg


def parse_args() -> str:
    parser = argparse.ArgumentParser(prog='train')

    parser.add_argument('data_path',
                        help='Direct path to the data folder')
    v_args = parser.parse_args()
    v_path = v_args.data_path
    return v_path


def load_data(p_path: str):
    """
    Load images from subdirectories, resize to 256x256, create labels,
    and split into train/test sets.

    :param p_path: Path to the directory containing class subdirectories
    :return: Tuple of ((x_train, y_train), (x_test, y_test))
             where x arrays have shape (nb_images, 256, 256, 3)
             and y arrays have shape (nb_images, 1)
    """
    if not os.path.isdir(p_path):
        raise ValueError(f"Directory does not exist: {p_path}")

    class_dirs = []
    for item in os.listdir(p_path):
        item_path = os.path.join(p_path, item)
        if os.path.isdir(item_path):
            class_dirs.append(item)
    
    class_dirs.sort()
    
    if not class_dirs:
        raise ValueError(f"No subdirectories found in {p_path}")

    class_to_label = {class_name: idx for idx, class_name in enumerate(class_dirs)}

    images = []
    labels = []

    for class_name in class_dirs:
        class_path = os.path.join(p_path, class_name)
        class_label = class_to_label[class_name]

        # Get all JPG files in this class directory
        jpg_files = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if is_jpg(os.path.join(class_path, f))
            and os.path.isfile(os.path.join(class_path, f))
        ]

        if not jpg_files:
            print(f"Warning: No JPG images found in {class_path}")
            continue

        for img_path in jpg_files:
            try:
                img = Image.open(img_path)
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize((256, 256), Image.LANCZOS)
                
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                images.append(img_array)
                labels.append(class_label)
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                continue

    if not images:
        raise ValueError(f"No images were successfully loaded from {p_path}")

    images = np.array(images)
    labels = np.array(labels, dtype=np.int32)

    labels = labels.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, shuffle=True
    )

    num_classes = len(class_dirs)
    y_train = to_categorical(y_train.flatten(), num_classes)
    y_test = to_categorical(y_test.flatten(), num_classes)

    return (x_train, y_train), (x_test, y_test)


def main():
    v_path = parse_args()
    (x_train, y_train), (x_test, y_test) = load_data(v_path)

    num_classes = y_train.shape[1]

    model = Sequential()
    model.add(Input(shape=(256, 256, 3)))

    # Layer 1
    model.add(Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    # Layer 2
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())

    # Layer 3
    model.add(Conv2D(384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # Layer 4
    model.add(Conv2D(384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # Layer 5
    model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Flatten
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully Connected Layer 2
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=15,
                    validation_split=0.2,
                    verbose=1)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test Accuracy: {test_acc:.4f}')

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('AlexNet on CIFAR-10 (GPU)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("mes morts", e)
