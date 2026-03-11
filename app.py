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

# PAGE CONFIG
st.set_page_config(
    page_title="Smart Waste Monitoring System",
    page_icon="🗑️",
    layout="centered",
    initial_sidebar_state="expanded"
)

MODEL_PATH  = "waste_detection_model.h5"
IMG_SIZE    = 224
CLASS_NAMES = {0: "Clean", 1: "Waste"}

if "total_analyzed" not in st.session_state:
    st.session_state.total_analyzed = 0
if "total_waste" not in st.session_state:
    st.session_state.total_waste = 0
if "total_clean" not in st.session_state:
    st.session_state.total_clean = 0

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.1); }
[data-testid="stMetric"] { background: rgba(255,255,255,0.07); border-radius: 12px; padding: 12px 18px; border: 1px solid rgba(255,255,255,0.12); }
.result-box { border-radius: 14px; padding: 22px 28px; text-align: center; margin: 18px 0; font-size: 1.4rem; font-weight: 700; }
.waste-box { background:#e74c3c22; border:2px solid #e74c3c; color:#e74c3c; }
.clean-box { background:#2ecc7122; border:2px solid #2ecc71; color:#2ecc71; }
.bar-label { font-size:0.85rem; color:#aaa; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)

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
    """)
    st.markdown("---")
    if st.button("🔄 Reset Statistics"):
        st.session_state.total_analyzed = 0
        st.session_state.total_waste = 0
        st.session_state.total_clean = 0
        st.success("Statistics reset!")

st.markdown("<h1 style='text-align:center;color:#fff;'>🗑️ Smart Waste Monitoring System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#aaa;font-size:1.05rem;'>AI & Big Data Powered | Future of Engineers Festival 🇺🇿</p>", unsafe_allow_html=True)
st.markdown("---")

model = load_model()
if model is None:
    st.error("❌ Model not found: waste_detection_model.h5\n\nPlease upload the trained model to GitHub and redeploy.")
    st.stop()

st.subheader("📤 Upload an Image")
uploaded_file = st.file_uploader("Choose a JPG, PNG, BMP, or WEBP image", type=["jpg","jpeg","png","bmp","webp"])

if uploaded_file is not None:
    pil_image = Image.open(io.BytesIO(uploaded_file.read()))
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**📸 Uploaded Image**")
        st.image(pil_image, use_container_width=True)
        st.caption(f"📄 {uploaded_file.name}  |  📐 {pil_image.width}×{pil_image.height} px")

    with col_result:
        st.markdown("**🤖 AI Prediction**")
        with st.spinner("🔍 Analysing ..."):
            img_batch  = preprocess(pil_image)
            probs      = model.predict(img_batch, verbose=0)[0]
            class_idx  = int(np.argmax(probs))
            confidence = float(probs[class_idx]) * 100

        st.session_state.total_analyzed += 1
        if class_idx == 1:
            st.session_state.total_waste += 1
            st.markdown("<div class='result-box waste-box'>🗑️ &nbsp; WASTE DETECTED</div>", unsafe_allow_html=True)
        else:
            st.session_state.total_clean += 1
            st.markdown("<div class='result-box clean-box'>✅ &nbsp; CLEAN AREA</div>", unsafe_allow_html=True)

        st.markdown(f"**🎯 Confidence:** `{{confidence:.2f}}%`")
        st.progress(int(confidence))

        st.markdown("**📊 Probability Breakdown:**")
        for i, prob in enumerate(probs):
            pct  = float(prob) * 100
            icon = "🗑️" if i == 1 else "✅"
            st.markdown(f"<p class='bar-label'>{{icon}} {{CLASS_NAMES[i]}}: {{pct:.1f}}%</p>", unsafe_allow_html=True)
            st.progress(int(pct))

    st.markdown("---")

st.subheader("📊 Session Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📸 Total Analyzed", st.session_state.total_analyzed)
with col2:
    wp = f"{{(st.session_state.total_waste/st.session_state.total_analyzed*100):.0f}}%" if st.session_state.total_analyzed > 0 else "0%"
    st.metric("🗑️ Waste Detections", st.session_state.total_waste, delta=wp, delta_color="inverse")
with col3:
    cp = f"{{(st.session_state.total_clean/st.session_state.total_analyzed*100):.0f}}%" if st.session_state.total_analyzed > 0 else "0%"
    st.metric("✅ Clean Detections", st.session_state.total_clean, delta=cp)

if st.session_state.total_analyzed > 0:
    wr = int(st.session_state.total_waste / st.session_state.total_analyzed * 100)
    st.markdown(f"**🌍 Overall Waste Rate: {{wr}}%**")
    st.progress(wr)
else:
    st.info("📂 Upload images above to start analyzing.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#555;font-size:0.85rem;'>🗑️ Smart Waste Monitoring System | Future of Engineers Festival, Tashkent 🇺🇿 | TensorFlow · OpenCV · Streamlit</p>", unsafe_allow_html=True)