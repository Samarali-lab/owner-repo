# ============================================================
# AI & Big Data Powered Smart Waste Monitoring System
# Prototype for: Future of Engineers Festival - Tashkent, Uzbekistan
# Description: CNN-based image classifier to detect Waste vs Clean streets
# Compatible with: Python 3.10, 3.11, 3.12, 3.13 + TensorFlow 2.13+
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE    = 224          # Resize all images to 224x224
BATCH_SIZE  = 32
EPOCHS      = 10
DATASET_DIRS = {
    0: "dataset/clean",    # Label 0 = Clean
    1: "dataset/waste"     # Label 1 = Waste
}
MODEL_SAVE_PATH = "waste_detection_model.h5"

# ============================================================
# STEP 1: LOAD AND PREPROCESS IMAGES
# ============================================================
def load_images(dataset_dirs, img_size):
    images = []
    labels = []

    for label, folder_path in dataset_dirs.items():
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder not found: '{folder_path}'. Skipping...")
            continue

        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(valid_extensions)]

        print(f"[INFO] Loading {len(files)} images from '{folder_path}' (Label: {label})")

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            img = cv2.imread(file_path)

            if img is None:
                print(f"[WARNING] Could not read: '{file_path}'. Skipping...")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(label)

    if len(images) == 0:
        return np.array([]), np.array([])

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"\n[INFO] Total images loaded: {len(images)}")
    return images, labels


# ============================================================
# STEP 2: SUMMARIZE DATASET WITH PANDAS
# ============================================================
def summarize_dataset(labels):
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
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ], name="WasteDetectionCNN")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ============================================================
# STEP 4: MAIN TRAINING PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("  AI & Big Data Powered Smart Waste Monitoring System")
    print("  Future of Engineers Festival - Tashkent, Uzbekistan")
    print(f"  TensorFlow version : {tf.__version__}")
    print("  Python compatible  : 3.10 / 3.11 / 3.12 / 3.13")
    print("=" * 60)

    images, labels = load_images(DATASET_DIRS, IMG_SIZE)

    if len(images) == 0:
        print("\n[ERROR] No images were loaded.")
        print("  Please add images to 'dataset/waste/' and 'dataset/clean/'")
        return

    summarize_dataset(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"\n[INFO] Training samples : {len(X_train)}")
    print(f"[INFO] Testing  samples : {len(X_test)}")

    model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=2)
    print("\n[INFO] Model Architecture:")
    model.summary()

    print("\n[INFO] Starting training ...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )

    final_train_acc = history.history["accuracy"][-1]    * 100
    final_val_acc   = history.history["val_accuracy"][-1] * 100

    print("\n" + "=" * 60)
    print(f"  Final Training   Accuracy : {final_train_acc:.2f}%")
    print(f"  Final Validation Accuracy : {final_val_acc:.2f}%")
    print("=" * 60)

    model.save(MODEL_SAVE_PATH)
    print(f"\n[INFO] Model saved as '{MODEL_SAVE_PATH}'")
    print("[DONE] Training complete!")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()