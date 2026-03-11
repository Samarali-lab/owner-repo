# ============================================================
# AI & Big Data Powered Smart Waste Monitoring System
# test_model.py - Test the trained model on new images
# Future of Engineers Festival - Tashkent, Uzbekistan
#
# Requirements:
#   pip install tensorflow opencv-python matplotlib
#
# Usage:
#   py -3.11 test_model.py <path_to_image>
#   py -3.11 test_model.py test.jpg
# ============================================================

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ============================================================
# CONFIGURATION  (must match training settings)
# ============================================================
MODEL_PATH   = "waste_detection_model.h5"   # Saved model file
IMG_SIZE     = 224                           # Same size used during training
CLASS_NAMES  = {0: "Clean", 1: "Waste"}     # Label map
CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c"} # Green=Clean, Red=Waste
CLASS_ICONS  = {0: "CLEAN", 1: "WASTE"}


# ============================================================
# STEP 1 - LOAD THE SAVED MODEL
# ============================================================
def load_model(model_path):
    """
    Loads the trained Keras model from disk.
    Args:
        model_path (str): Path to the .h5 model file
    Returns:
        model (tf.keras.Model): Loaded model ready for prediction
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: '{model_path}'")
        print("  -> Please run 'py -3.11 waste_detection.py' first to train the model.")
        sys.exit(1)

    print(f"[INFO] Loading model from '{model_path}' ...")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded successfully!")
    return model


# ============================================================
# STEP 2 - LOAD AND PREPROCESS THE IMAGE
# ============================================================
def preprocess_image(image_path, img_size):
    """
    Loads an image and preprocesses it exactly the same
    way as the training data:
        1. Read with OpenCV
        2. Convert BGR -> RGB
        3. Resize to (img_size x img_size)
        4. Normalize pixels to [0.0, 1.0]
        5. Add batch dimension -> shape (1, H, W, 3)
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: '{image_path}'")
        sys.exit(1)

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        print(f"[ERROR] Could not read image: '{image_path}'")
        sys.exit(1)

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_display = img_rgb.copy()

    # Resize to 224x224
    img_resized = cv2.resize(img_rgb, (img_size, img_size))

    # Normalize [0,255] -> [0.0, 1.0]
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    print(f"[INFO] Image loaded: '{image_path}'")
    print(f"       Original size : {img_bgr.shape[1]} x {img_bgr.shape[0]} px")
    print(f"       Resized to    : {img_size} x {img_size} px")

    return img_batch, img_display


# ============================================================
# STEP 3 - PREDICT
# ============================================================
def predict(model, img_batch):
    """
    Runs the model and returns class index, confidence, probabilities.
    """
    print("[INFO] Running prediction ...")
    probabilities = model.predict(img_batch, verbose=0)[0]
    class_index   = int(np.argmax(probabilities))
    confidence    = float(probabilities[class_index]) * 100
    return class_index, confidence, probabilities


# ============================================================
# STEP 4 - PRINT RESULTS TO CONSOLE
# ============================================================
def print_results(image_path, class_index, confidence, probabilities):
    """
    Prints formatted prediction results with ASCII bar chart.
    """
    label = CLASS_NAMES[class_index]
    icon  = CLASS_ICONS[class_index]

    print()
    print("=" * 55)
    print("  PREDICTION RESULT")
    print("=" * 55)
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Prediction : [{icon}]  {label}")
    print(f"  Confidence : {confidence:.2f}%")
    print("-" * 55)
    for idx, prob in enumerate(probabilities):
        bar  = "#" * int(prob * 30)
        name = CLASS_NAMES[idx]
        print(f"  {name:<8} : {prob * 100:6.2f}%  |{bar}")
    print("=" * 55)


# ============================================================
# STEP 5 - DISPLAY IMAGE WITH MATPLOTLIB
# ============================================================
def display_result(image_path, img_display, class_index, confidence, probabilities):
    """
    Shows the image with coloured border + confidence bar chart
    side by side in a Matplotlib window.
    """
    label = CLASS_NAMES[class_index]
    color = CLASS_COLORS[class_index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")

    # Left: image
    ax_img = axes[0]
    ax_img.imshow(img_display)
    ax_img.set_facecolor("#1a1a2e")
    for spine in ax_img.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(6)
    ax_img.set_title(
        f"[{CLASS_ICONS[class_index]}]  {label}  -  {confidence:.1f}% confident",
        fontsize=15, fontweight="bold", color=color, pad=12
    )
    ax_img.axis("off")

    # Right: bar chart
    ax_bar = axes[1]
    ax_bar.set_facecolor("#16213e")
    class_labels = [CLASS_NAMES[i] for i in range(len(probabilities))]
    bar_colors   = [CLASS_COLORS[i] for i in range(len(probabilities))]
    pct_values   = [p * 100 for p in probabilities]

    bars = ax_bar.barh(class_labels, pct_values,
                       color=bar_colors, edgecolor="white",
                       linewidth=0.8, height=0.4)

    for bar, pct in zip(bars, pct_values):
        ax_bar.text(
            min(pct + 1.5, 95),
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="left",
            color="white", fontsize=13, fontweight="bold"
        )

    ax_bar.set_xlim(0, 115)
    ax_bar.set_xlabel("Confidence (%)", color="white", fontsize=12)
    ax_bar.set_title("Prediction Confidence",
                     color="white", fontsize=14, fontweight="bold", pad=12)
    ax_bar.tick_params(colors="white", labelsize=12)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444444")

    fig.suptitle(
        "AI & Big Data Powered Smart Waste Monitoring System\n"
        "Future of Engineers Festival  -  Tashkent, Uzbekistan",
        fontsize=12, color="#cccccc", y=1.02
    )

    plt.tight_layout()
    print("[INFO] Displaying result window (close window to exit) ...")
    plt.show()


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("\n  Usage   : py -3.11 test_model.py <image_path>")
        print("  Example : py -3.11 test_model.py test.jpg")
        print("  Example : py -3.11 test_model.py dataset/waste/img1.jpg\n")
        sys.exit(0)

    image_path = sys.argv[1]

    print()
    print("=" * 55)
    print("  AI Waste Detection - Model Tester")
    print(f"  TensorFlow : {tf.__version__}")
    print("=" * 55)

    model                                    = load_model(MODEL_PATH)
    img_batch, img_display                   = preprocess_image(image_path, IMG_SIZE)
    class_index, confidence, probabilities   = predict(model, img_batch)
    print_results(image_path, class_index, confidence, probabilities)
    display_result(image_path, img_display, class_index, confidence, probabilities)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()