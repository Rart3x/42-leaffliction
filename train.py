import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import argparse
import os


def parse_args() -> str:
    parser = argparse.ArgumentParser(prog='train')

    parser.add_argument('data_path',
                        help='Direct path to the data folder')
    v_args = parser.parse_args()
    v_path = v_args.data_path
    return v_path


def create_data_generators(p_path: str, batch_size: int = 16, img_size: (int, int) = (224, 224)):
    """
    Create ImageDataGenerators for memory-efficient data loading.
    
    :param p_path: Path to the directory containing class subdirectories
    :param batch_size: Number of images to load per batch
    :param img_size: Target image size (width, height)
    :return: Tuple of (train_generator, validation_generator, num_classes)
    """
    if not os.path.isdir(p_path):
        raise ValueError(f"Directory does not exist: {p_path}")

    # Create ImageDataGenerator with normalization and validation split
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # 80% train, 20% validation
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

    # Validation data generator
    validation_generator = train_datagen.flow_from_directory(
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
    v_path = parse_args()
    
    # Use memory-efficient data generators instead of loading all data at once
    img_size = (224, 224)  # Reduced from 256 to save memory
    batch_size = 16
    
    train_generator, validation_generator, num_classes = create_data_generators(
        v_path, batch_size=batch_size, img_size=img_size
    )

    model = Sequential()
    model.add(Input(shape=(img_size[0], img_size[1], 3)))

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

    try:
        # Use fit with generators for memory-efficient training
        history = model.fit(
            train_generator,
            epochs=15,
            validation_data=validation_generator,
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
    except KeyboardInterrupt as e:
        model.save('model_WIP.keras')
        print("Canceled by user", e)
    except Exception as e:
        print("Error", e)


if __name__ == "__main__":
    main()
