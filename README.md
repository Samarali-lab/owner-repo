# 🗑️ AI & Big Data Powered Smart Waste Monitoring System

> 🏆 Prototype project for the **Future of Engineers Festival** — Tashkent, Uzbekistan 🇺🇿

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This project is a **CNN-based AI image classifier** that detects whether a street or area contains **waste** or is **clean**. Built as a prototype for the **Future of Engineers Festival** in Tashkent, Uzbekistan.

The system includes:
- 🏋️ A **training pipeline** to train the CNN model
- 🌐 A **Streamlit web dashboard** for real-time image upload and prediction
- 🖼️ A **single image tester** with Matplotlib visualisation
- 🎥 A **live webcam detector** for real-time demo
- 🔍 A **quick predictor** for fast command-line testing

---

## 🧠 How It Works

```
📁 Dataset Images
       │
       ▼
🔄 Preprocess (Resize 224×224, Normalize [0,1], BGR→RGB)
       │
       ▼
🧠 CNN Model (3 Conv blocks + Dense layers)
       │
       ▼
🎯 Prediction: "Waste" or "Clean" + Confidence %
```

1. Images are loaded from `dataset/waste/` and `dataset/clean/`
2. Each image is resized to **224×224**, converted BGR→RGB, and normalized to `[0.0, 1.0]`
3. A **CNN model** with 3 convolutional blocks is trained for binary classification
4. The trained model is saved as `waste_detection_model.h5`
5. The model is used in the dashboard, webcam, or tester scripts

---

## 📁 Project Structure

```
owner-repo/
│
├── dataset/
│   ├── waste/                  ← Street images WITH waste (JPG/PNG/BMP)
│   └── clean/                  ← CLEAN street images (JPG/PNG/BMP)
│
├── waste_detection.py          ← 🏋️  STEP 1: Train the CNN model
├── dashboard.py                ← 🌐  STEP 2: Streamlit web dashboard
├── test_model.py               ← 🖼️  STEP 3: Test model + Matplotlib chart
├── predict.py                  �� 🔍  STEP 4: Quick single image prediction
├── webcam.py                   ← 🎥  STEP 5: Live real-time webcam detection
├── requirements.txt            ← 📦  All Python dependencies
└── README.md                   ← 📄  This file
```

---

## 🔧 Requirements & Installation

### ✅ Python Version

This project requires **Python 3.10, 3.11, 3.12, or 3.13**.

> ⚠️ Python 3.14 is **NOT** supported by TensorFlow yet.
> Download Python 3.11 from: https://www.python.org/downloads/release/python-3119/

### 📦 Install All Libraries

```bash
py -3.11 -m pip install -r requirements.txt
```

Then install additional tools:

```bash
py -3.11 -m pip install streamlit pillow matplotlib
```

---

## 🖼️ Dataset Setup

Before training, add your images to the dataset folders:

```
dataset/
├── waste/     ← Add 50+ images of streets WITH rubbish/waste
└── clean/     ← Add 50+ images of CLEAN streets
```

**Where to get free images:**
- https://unsplash.com → search "dirty street", "clean street"
- https://images.google.com → right-click → Save image
- Take photos yourself with your phone 📱

> 💡 **The more images you add, the higher the accuracy!**
> Minimum: 20 per folder | Recommended: 100+ per folder

---

## 🚀 Step-by-Step Implementation Guide

---

### 🏋️ STEP 1 — Train the Model

```bash
py -3.11 waste_detection.py
```

**What it does:**
- Loads all images from `dataset/waste/` and `dataset/clean/`
- Preprocesses images (resize 224×224, BGR→RGB, normalize)
- Prints a dataset summary using Pandas
- Trains the CNN for 10 epochs with 80/20 train/test split
- Saves the trained model as `waste_detection_model.h5`

**Expected output:**
```
============================================================
  AI & Big Data Powered Smart Waste Monitoring System
  TensorFlow version : 2.13.0
============================================================
[INFO] Loading 150 images from 'dataset/waste' (Label: 1)
[INFO] Loading 150 images from 'dataset/clean' (Label: 0)

  Final Training   Accuracy : 98.50%
  Final Validation Accuracy : 91.67%
============================================================
[INFO] Model saved as 'waste_detection_model.h5'
[DONE] Training complete!
```

---

### 🌐 STEP 2 — Launch the Web Dashboard

```bash
py -3.11 -m pip install streamlit pillow
py -3.11 -m streamlit run dashboard.py
```

**Opens automatically at:** http://localhost:8501

**Dashboard features:**

| Feature | Description |
|---|---|
| 📤 Image Upload | Upload any JPG, PNG, BMP photo |
| 🖼️ Image Display | Shows uploaded image on the left |
| 🔴🟢 Result Box | Big coloured — WASTE DETECTED or CLEAN AREA |
| 📊 Confidence Score | Percentage + live progress bar |
| 📈 Probability Breakdown | Confidence bar for each class |
| 📊 Session Summary | Total analyzed, waste %, clean % |
| 🔄 Reset Button | Reset all stats from the sidebar |
| 🌙 Dark Theme | Beautiful dark gradient UI |

---

### 🖼️ STEP 3 — Test the Model on Any Image

```bash
py -3.11 -m pip install matplotlib
py -3.11 test_model.py dataset/waste/yourimage.jpg
```

**Terminal output:**
```
=======================================================
  PREDICTION RESULT
=======================================================
  Image      : yourimage.jpg
  Prediction : [WASTE]  Waste
  Confidence : 94.73%
-------------------------------------------------------
  Clean    :   5.27%  |#
  Waste    :  94.73%  |############################
=======================================================
```

**Matplotlib popup window:**
- Left: Your image with 🟢 green (Clean) or 🔴 red (Waste) border
- Right: Confidence bar chart for both classes

---

### 🔍 STEP 4 — Quick Command-Line Prediction

```bash
py -3.11 predict.py dataset/clean/yourimage.jpg
```

Prints the result and opens an OpenCV image window. Fast and simple.

---

### 🎥 STEP 5 — Live Webcam Detection

```bash
py -3.11 webcam.py
```

**What it does:**
- Opens your laptop/PC webcam
- Detects waste in **real time** from the camera feed
- 🔴 Red border = Waste detected
- 🟢 Green border = Clean area
- Shows confidence % on screen continuously
- Press **Q** to quit

---

## 🏗️ CNN Model Architecture

| Layer | Details | Activation |
|---|---|---|
| Input | 224 × 224 × 3 | — |
| Conv2D Block 1 | 32 filters, 3×3, padding=same | ReLU |
| MaxPooling2D | Pool size 2×2 | — |
| Conv2D Block 2 | 64 filters, 3×3, padding=same | ReLU |
| MaxPooling2D | Pool size 2×2 | — |
| Conv2D Block 3 | 128 filters, 3×3, padding=same | ReLU |
| MaxPooling2D | Pool size 2×2 | — |
| Flatten | Convert to 1D vector | — |
| Dense | 128 neurons | ReLU |
| Dropout | 50% rate | — |
| Output Dense | 2 neurons | Softmax |

**Optimizer:** Adam | **Loss:** Sparse Categorical Crossentropy | **Epochs:** 10

---

## 📦 Libraries Used

| Library | Version | Purpose |
|---|---|---|
| TensorFlow / Keras | ≥ 2.13.0 | Build and train the CNN model |
| OpenCV | ≥ 4.8.0 | Load, resize, preprocess images |
| NumPy | ≥ 1.24.0 | Array and matrix operations |
| Pandas | ≥ 2.0.0 | Dataset summary and analysis |
| Scikit-learn | ≥ 1.3.0 | Train/test split with stratification |
| Streamlit | latest | Web dashboard UI |
| Matplotlib | latest | Image + confidence chart display |
| Pillow | latest | Image handling in Streamlit |

---

## 🎯 Tips to Improve Accuracy

| Tip | Details |
|---|---|
| ➕ Add more images | 100+ images per class gives much better accuracy |
| 🔄 More epochs | Change `EPOCHS = 10` to `EPOCHS = 20` or `30` |
| ⚖️ Balance dataset | Keep roughly equal waste and clean images |
| 📐 Data augmentation | Flip/rotate images to create more training samples |
| 🧠 Transfer learning | Use MobileNetV2 or EfficientNet for higher accuracy |

---

## ❓ Troubleshooting

| Problem | Solution |
|---|---|
| `tensorflow not found` | Use `py -3.11 -m pip install tensorflow` |
| `No images were loaded` | Add images to `dataset/waste/` and `dataset/clean/` |
| `Model file not found` | Run `py -3.11 waste_detection.py` first |
| Python 3.14 not supported | Install Python 3.11 from python.org/downloads |
| Webcam not opening | Make sure no other app is using the camera |
| Low accuracy (< 70%) | Add more images to the dataset folders |
| Streamlit not found | Run `py -3.11 -m pip install streamlit` |

---

## 🗺️ Project Roadmap

- [x] CNN model training pipeline
- [x] Image preprocessing (BGR→RGB, resize, normalize)
- [x] Streamlit web dashboard
- [x] Single image tester with Matplotlib
- [x] Live webcam real-time detection
- [x] Quick command-line predictor
- [ ] Data augmentation for better accuracy
- [ ] Transfer learning with MobileNetV2
- [ ] REST API with FastAPI
- [ ] Mobile app integration

---

## 👩‍💻 Author

**Samar Ali**
- 📧 fd25502005@nutech.edu.pk / samaraliuae5@gmail.com 
- 🏫 NUTECH University, Islamabad
- 🏆 Future of Engineers Festival 2026

---

## 📄 License

MIT License — free to use, modify, and share.

---

<p align="center">
  <b>🗑️ AI & Big Data Powered Smart Waste Monitoring System</b><br/>
  Future of Engineers Festival — Tashkent, Uzbekistan 🇺🇿<br/>
  Built with ❤️ using TensorFlow, OpenCV & Streamlit
</p>
