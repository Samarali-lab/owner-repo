# ============================================================
# AI & Big Data Powered Smart Waste Monitoring System
# predict.py - Test a single image with the trained model
# Future of Engineers Festival - Tashkent, Uzbekistan
# ============================================================

import cv2
import numpy as np
import tensorflow as tf
import sys
import os

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "waste_detection_model.h5"
IMG_SIZE   = 224
LABELS     = {0: "Clean ✅", 1: "Waste 🗑️"}

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file '{MODEL_PATH}' not found.")
    print("  Please run 'py -3.11 waste_detection.py' first to train and save the model.")
    sys.exit(1)

print(f"[INFO] Loading model from '{MODEL_PATH}' ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!\n")


# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_image(image_path):
    """
    Loads an image, preprocesses it, and predicts Waste or Clean.

    Args:
        image_path (str): Path to the image file
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: '{image_path}'")
        return

    # Load and preprocess
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: '{image_path}'")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype("float32") / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)  # Shape: (1, 224, 224, 3)

    # Predict
    prediction  = model.predict(img_batch, verbose=0)
    class_index = int(np.argmax(prediction))
    confidence  = float(prediction[0][class_index]) * 100
    label       = LABELS[class_index]

    # Display result
    print("=" * 50)
    print(f"  Image      : {image_path}")
    print(f"  Result     : {label}")
    print(f"  Confidence : {confidence:.2f}%")
    print("=" * 50)

    # Show image with result overlay
    color = (0, 200, 0) if class_index == 0 else (0, 0, 255)
    display_label = f"{label}  ({confidence:.1f}%)"
    cv2.putText(img, display_label, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 3)

    cv2.imshow("Waste Detection - Prediction Result", img)
    print("\n[INFO] Close the image window or press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.11 predict.py <path_to_image>")
        print("Example: py -3.11 predict.py test.jpg")
        print("Example: py -3.11 predict.py dataset/waste/img1.jpg")
    else:
        predict_image(sys.argv[1])
