# ============================================================
# AI & Big Data Powered Smart Waste Monitoring System
# app.py - Main entry point for Streamlit Cloud deployment
# Future of Engineers Festival - Tashkent, Uzbekistan
#
# Run locally : streamlit run app.py
# Deploy on  : https://streamlit.io/cloud
# ============================================================

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Smart Waste Monitoring System",
    page_icon="🗑️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH  = "waste_detection_model.h5"
IMG_SIZE    = 224
CLASS_NAMES = {0: "Clean", 1: "Waste"}

# ============================================================
# SESSION STATE
# ============================================================
if "total_analyzed" not in st.session_state:
    st.session_state.total_analyzed = 0
if "total_waste" not in st.session_state:
    st.session_state.total_waste    = 0
if "total_clean" not in st.session_state:
    st.session_state.total_clean    = 0

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# ============================================================
# PREPROCESS IMAGE
# ============================================================
def preprocess(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 12px 18px;
    border: 1px solid rgba(255,255,255,0.12);
}
.result-box {
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
    margin: 18px 0;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.waste-box { background:#e74c3c22; border:2px solid #e74c3c; color:#e74c3c; }
.clean-box { background:#2ecc7122; border:2px solid #2ecc71; color:#2ecc71; }
.bar-label { font-size:0.85rem; color:#aaa; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/trash.png", width=72)
    st.title("Smart Waste\nMonitor")
    st.markdown("---")
    st.markdown("""
    **About this app**

    Upload any street or area photo.
    The AI model will instantly tell you
    whether it contains **waste** or is **clean**.

    **Built with:**
    - TensorFlow / Keras CNN
    - OpenCV
    - Streamlit

    ---
    **Future of Engineers Festival**
    Tashkent, Uzbekistan 🇺🇿
    "