# ============================================================
# AI & Big Data Powered Smart Waste Monitoring System
# webcam.py - Real-time waste detection using webcam
# Future of Engineers Festival - Tashkent, Uzbekistan
# ============================================================

import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "waste_detection_model.h5"
IMG_SIZE   = 224
LABELS     = {0: "Clean", 1: "Waste"}
COLORS     = {0: (0, 220, 0), 1: (0, 0, 255)}   # Green=Clean, Red=Waste
EMOJI      = {0: "✅", 1: "🗑️"}

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file '{MODEL_PATH}' not found.")
    print("  Please run 'py -3.11 waste_detection.py' first to train the model.")
    sys.exit(1)

print("[INFO] Loading model ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded! Opening webcam ...\n")
print("  Press  Q  to quit the webcam window.")
print("=" * 50)


# ============================================================
# REAL-TIME DETECTION LOOP
# ============================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    print("  Make sure your webcam is connected and not used by another app.")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Could not read frame from webcam.")
        break

    # ── Preprocess frame for model ──────────────────────────
    img_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_norm    = img_resized.astype("float32") / 255.0
    img_batch   = np.expand_dims(img_norm, axis=0)

    # ── Predict ─────────────────────────────────────────────
    prediction  = model.predict(img_batch, verbose=0)
    class_index = int(np.argmax(prediction))
    confidence  = float(prediction[0][class_index]) * 100

    label = LABELS[class_index]
    color = COLORS[class_index]

    # ── Draw overlay on frame ────────────────────────────────
    # Colored border
    cv2.rectangle(frame, (0, 0),
                  (frame.shape[1] - 1, frame.shape[0] - 1),
                  color, 5)

    # Background box for text
    cv2.rectangle(frame, (0, 0), (420, 70), color, -1)

    # Label text
    display_text = f"{label}  {confidence:.1f}%"
    cv2.putText(frame, display_text, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    # Festival watermark
    cv2.putText(frame, "AI Waste Detector | Future of Engineers Festival",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── Show frame ──────────────────────────────────────────
    cv2.imshow("AI Waste Detector - Press Q to Quit", frame)

    # ── Quit on Q key ────────────────────────────────────────
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── Cleanup ──────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Webcam closed. Goodbye! 👋")
