# ============================================================
# AI & Big Data Powered Smart Waste Monitoring System
# Prototype for: Future of Engineers Festival - Tashkent, Uzbekistan
# Description: CNN-based image classifier to detect Waste vs Clean streets
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = 224          # Resize all images to 224x224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIRS = {
    0: "dataset/clean",   # Label 0 = Clean
    1: "dataset/waste"    # Label 1 = Waste
}
MODEL_SAVE_PATH = "waste_detection_model.h5"

# ============================================================
# STEP 1: LOAD AND PREPROCESS IMAGES
# ============================================================
def load_images(dataset_dirs, img_size):
    """
    Loads images from the given directories and assigns labels.
    Returns:
        images (np.array): Normalized image array
        labels (np.array): Corresponding label array
    """
    images = []
    labels = []

    for label, folder_path in dataset_dirs.items():
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder not found: {folder_path}. Skipping...")
            continue

        files = os.listdir(folder_path)
        print(f"[INFO] Loading {len(files)} images from '{folder_path}' (Label: {label})")

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            # Read image using OpenCV
            img = cv2.imread(file_path)

            if img is None:
                print(f"[WARNING] Could not read image: {file_path}. Skipping...")
                continue

            # Resize image to target size
            img = cv2.resize(img, (img_size, img_size))

            # Normalize pixel values to range [0, 1]
            img = img / 255.0

            images.append(img)
            labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"\n[INFO] Total images loaded: {len(images)}")
    return images, labels


# ============================================================
# STEP 2: SUMMARIZE DATASET WITH PANDAS
# ============================================================
def summarize_dataset(labels):
    """
    Uses Pandas to print a summary of the dataset distribution.
    """
    label_names = {0: "Clean", 1: "Waste"}
    df = pd.DataFrame({"Label": labels})
    df["Category"] = df["Label"].map(label_names)
    summary = df["Category"].value_counts().reset_index()
    summary.columns = ["Category", "Count"]
    print("\n[INFO] Dataset Summary:")
    print(summary.to_string(index=False))


# ============================================================
# STEP 3: BUILD THE CNN MODEL
# ============================================================
def build_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Builds and returns a simple CNN model for binary image classification.
    Architecture:
        - 3 Convolutional + MaxPooling blocks
        - Flatten layer
        - Dense layers with Dropout for regularization
        - Softmax output for classification
    """
    model = Sequential([

        # Block 1: First Convolutional Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2: Second Convolutional Layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 3: Third Convolutional Layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten feature maps into a 1D vector
        Flatten(),

        # Fully connected Dense layer
        Dense(128, activation='relu'),

        # Dropout to prevent overfitting (drop 50% of neurons randomly)
        Dropout(0.5),

        # Output layer: 2 neurons (Clean, Waste) with Softmax activation
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# STEP 4: MAIN TRAINING PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("  AI & Big Data Powered Smart Waste Monitoring System")
    print("  Future of Engineers Festival - Tashkent, Uzbekistan")
    print("=" * 60)

    # --- Load Images ---
    images, labels = load_images(DATASET_DIRS, IMG_SIZE)

    if len(images) == 0:
        print("[ERROR] No images were loaded. Please check your dataset folders.")
        return

    # --- Summarize Dataset ---
    summarize_dataset(labels)

    # --- One-Hot Encode Labels ---
    # e.g., 0 -> [1, 0] (Clean), 1 -> [0, 1] (Waste)
    labels_categorical = to_categorical(labels, num_classes=2)

    # --- Split into Training and Testing Sets (80% train, 20% test) ---
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_categorical,
        test_size=0.2,
        random_state=42
    )
    print(f"\n[INFO] Training samples : {len(X_train)}")
    print(f"[INFO] Testing  samples : {len(X_test)}")

    # --- Build Model ---
    model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=2)
    print("\n[INFO] Model Architecture:")
    model.summary()

    # --- Train the Model ---
    print("\n[INFO] Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # --- Print Final Accuracy Results ---
    final_train_acc = history.history['accuracy'][-1] * 100
    final_val_acc   = history.history['val_accuracy'][-1] * 100

    print("\n" + "=" * 60)
    print(f"  Final Training Accuracy   : {final_train_acc:.2f}%")
    print(f"  Final Validation Accuracy : {final_val_acc:.2f}%")
    print("=" * 60)

    # --- Save the Trained Model ---
    model.save(MODEL_SAVE_PATH)
    print(f"\n[INFO] Model saved successfully as '{MODEL_SAVE_PATH}'")
    print("[DONE] Training complete!")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()