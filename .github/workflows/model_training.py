# model_training.py
"""
Trains a CNN model on the FER2013 dataset and saves it as face_emotionModel.h5.
Dataset folder structure:
dataset/
  train/
    angry/
    disgust/
    fear/
    happy/
    sad/
    surprise/
    neutral/
  test/
    angry/
    ...
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_PATH = os.path.join(BASE_DIR, "face_emotionModel.h5")

# --- Parameters ---
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 15

# --- Step 1: Load Dataset ---
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE
)

# Normalize pixel values (0–255 -> 0–1)
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Step 2: Build the CNN Model ---
num_classes = len(train_ds.class_names)

model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# --- Step 3: Compile the Model ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Step 4: Train the Model ---
print("\nTraining started...\n")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

# --- Step 5: Evaluate and Save ---
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

model.save(MODEL_PATH)
print(f"\n🎉 Model saved successfully as {MODEL_PATH}")