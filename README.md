# AI & Big Data Powered Smart Waste Monitoring System

> 🏆 Prototype project for the **Future of Engineers Festival** — Tashkent, Uzbekistan 🇺🇿

## 📌 Overview

This project is a simple AI model that detects waste in street images and classifies them as **"Waste"** or **"Clean"** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## 🧠 How It Works

1. Images are loaded from `dataset/waste/` and `dataset/clean/`
2. Images are resized to **224×224** and normalized to `[0, 1]`
3. A **CNN model** is trained to classify images into two categories
4. The trained model is saved as `waste_detection_model.h5`

## 📁 Project Structure

```
owner-repo/
│
├── dataset/
│   ├── waste/          ← Street images WITH waste (JPG/PNG)
│   └── clean/          ← CLEAN street images (JPG/PNG)
│
├── waste_detection.py  ← Main training script
├── requirements.txt    ← Python dependencies
└── README.md
```

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

1. Add your images to `dataset/waste/` and `dataset/clean/`
2. Run the training script:

```bash
python waste_detection.py
```

## 📊 Expected Output

```
============================================================
  AI & Big Data Powered Smart Waste Monitoring System
  Future of Engineers Festival - Tashkent, Uzbekistan
============================================================
[INFO] Loading 150 images from 'dataset/waste' (Label: 1)
[INFO] Loading 150 images from 'dataset/clean' (Label: 0)
...
  Final Training Accuracy   : 92.50%
  Final Validation Accuracy : 88.33%
============================================================
[INFO] Model saved successfully as 'waste_detection_model.h5'
[DONE] Training complete!
```

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| TensorFlow/Keras | Build and train the CNN model |
| OpenCV | Load and preprocess images |
| NumPy | Array operations |
| Pandas | Dataset summary and analysis |
| Scikit-learn | Train/test split |

## 🏗️ Model Architecture

| Layer | Details |
|---|---|
| Conv2D + MaxPooling | Block 1: 32 filters |
| Conv2D + MaxPooling | Block 2: 64 filters |
| Conv2D + MaxPooling | Block 3: 128 filters |
| Flatten | Convert to 1D vector |
| Dense + Dropout | 128 neurons, 50% dropout |
| Dense (Output) | 2 neurons, Softmax |

## 📄 License

MIT License
