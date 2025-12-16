import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from emnist import extract_training_samples, extract_test_samples

# Configuration
MODEL_NAME = 'my_super_model.keras'
BATCH_SIZE = 64
EPOCHS = 5

def fix_emnist_rotation(images):
    print("Normalizing image orientation...")
    return np.array([np.fliplr(np.rot90(img, k=-1)) for img in images])

def main():
    print("Downloading EMNIST dataset (Balanced split)...")
    # Load Data (47 classes: 0-9, A-Z, a-z)
    train_images, train_labels = extract_training_samples('balanced')
    test_images, test_labels = extract_test_samples('balanced')

    # Preprocessing
    train_images = fix_emnist_rotation(train_images)
    test_images = fix_emnist_rotation(test_images)

    # Normalize pixel values to 0-1 and reshape for CNN
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') / 255
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32') / 255

    print(f"Training Data Shape: {train_images.shape}")

    # Build CNN Architecture
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(47, activation='softmax') # 47 Classes
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    print("Starting training process...")
    model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save
    model.save(MODEL_NAME)
    print(f"Success! Model saved to: {os.path.abspath(MODEL_NAME)}")

if __name__ == "__main__":
    main()