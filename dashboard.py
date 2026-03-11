# ============================================================
# AI & Big Data Powered Smart Waste Monitoring System
# dashboard.py - Streamlit Web Dashboard
# Future of Engineers Festival - Tashkent, Uzbekistan
#
# Requirements:
#   pip install streamlit tensorflow opencv-python numpy pillow
#
# Run:
#   py -3.11 -m streamlit run dashboard.py
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
# SESSION STATE  (persists across reruns)
# ============================================================
if "total_analyzed" not in st.session_state:
    st.session_state.total_analyzed = 0
if "total_waste" not in st.session_state:
    st.session_state.total_waste    = 0
if "total_clean" not in st.session_state:
    st.session_state.total_clean    = 0

# ============================================================
# LOAD MODEL  (cached so it only loads once)
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

    Built with:
    - TensorFlow / Keras CNN
    - OpenCV
    - Streamlit

    ---
    **Future of Engineers Festival**
    Tashkent, Uzbekistan 🇺🇿
    """)
    st.markdown("---")
    if st.button("🔄 Reset Statistics"):
        st.session_state.total_analyzed = 0
        st.session_state.total_waste    = 0
        st.session_state.total_clean    = 0
        st.success("Statistics reset!")

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 style='text-align:center;color:#fff;'>🗑️ Smart Waste Monitoring System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#aaa;font-size:1.05rem;'>AI & Big Data Powered Image Classifier | Future of Engineers Festival 🇺🇿</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# LOAD MODEL
# ============================================================
model = load_model()

if model is None:
    st.error("❌ Model file `waste_detection_model.h5` not found.\n\nPlease run `py -3.11 waste_detection.py` first to train and save the model.")
    st.stop()

# ============================================================
# IMAGE UPLOAD
# ============================================================
st.subheader("📤 Upload an Image")
uploaded_file = st.file_uploader(
    "Choose a JPG, PNG, or BMP image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

# ============================================================
# PREDICTION
# ============================================================
if uploaded_file is not None:
    pil_image = Image.open(io.BytesIO(uploaded_file.read()))
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**Uploaded Image**")
        st.image(pil_image, use_container_width=True)
        st.caption(f"File: {uploaded_file.name} | Size: {pil_image.width} x {pil_image.height} px")

    with col_result:
        st.markdown("**Prediction Result**")
        with st.spinner("Analysing image ..."):
            img_batch  = preprocess(pil_image)
            probs      = model.predict(img_batch, verbose=0)[0]
            class_idx  = int(np.argmax(probs))
            confidence = float(probs[class_idx]) * 100
            label      = CLASS_NAMES[class_idx]

        st.session_state.total_analyzed += 1
        if class_idx == 1:
            st.session_state.total_waste += 1
            st.markdown("<div class='result-box waste-box'>🗑️ &nbsp; WASTE DETECTED</div>", unsafe_allow_html=True)
        else:
            st.session_state.total_clean += 1
            st.markdown("<div class='result-box clean-box'>✅ &nbsp; CLEAN AREA</div>", unsafe_allow_html=True)

        st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
        st.progress(int(confidence))

        st.markdown("**Probability Breakdown:**")
        for i, prob in enumerate(probs):
            name = CLASS_NAMES[i]
            pct  = float(prob) * 100
            st.markdown(f"<p class='bar-label'>{name}: {pct:.1f}%</p>", unsafe_allow_html=True)
            st.progress(int(pct))

    st.markdown("---")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
st.subheader("📊 Session Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="📸 Total Images Analyzed", value=st.session_state.total_analyzed)

with col2:
    st.metric(
        label="🗑️ Waste Detections",
        value=st.session_state.total_waste,
        delta=(f"{(st.session_state.total_waste / st.session_state.total_analyzed * 100):.0f}%" if st.session_state.total_analyzed > 0 else "0%"),
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="✅ Clean Detections",
        value=st.session_state.total_clean,
        delta=(f"{(st.session_state.total_clean / st.session_state.total_analyzed * 100):.0f}%" if st.session_state.total_analyzed > 0 else "0%")
    )

if st.session_state.total_analyzed > 0:
    waste_rate = int(st.session_state.total_waste / st.session_state.total_analyzed * 100)
    st.markdown(f"**Overall Waste Rate: {waste_rate}%**")
    st.progress(waste_rate)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#666;font-size:0.85rem;'>AI & Big Data Powered Smart Waste Monitoring System | Future of Engineers Festival, Tashkent 🇺🇿</p>", unsafe_allow_html=True)