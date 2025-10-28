import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="🫁 Pneumonia Detection | CNN",
    page_icon="🫀",
    layout="centered"
)

# ----------------------------
# CUSTOM STYLING
# ----------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://cdn.pixabay.com/photo/2016/12/14/23/08/lungs-1900981_1280.png");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    backdrop-filter: blur(6px);
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.7);
}

h1, h2, h3, h4, h5, h6, p {
    color: white !important;
    text-shadow: 1px 1px 2px #000;
}

div.stButton > button {
    background-color: #1F77B4;
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    border: none;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    background-color: #17A589;
    transform: scale(1.05);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model

model = load_model()

# ----------------------------
# PAGE TITLE
# ----------------------------
st.title("🫁 Pneumonia Detection from Chest X-rays")
st.markdown(
    """
    Upload a **chest X-ray image** below for AI-based detection.  
    The model will analyze the X-ray and determine if it shows signs of:
    - 🧬 **Pneumonia (Infected)**
    - 💨 **Normal (Healthy Lungs)**
    """
)

# ----------------------------
# IMAGE UPLOAD SECTION
# ----------------------------
uploaded_file = st.file_uploader("📸 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🩻 Uploaded X-ray", use_container_width=True)

    # ----------------------------
    # PREPROCESS IMAGE
    # ----------------------------
    img = np.array(image.convert("L"))  # Convert to grayscale
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1,128,128,1)

    st.markdown("### 🔍 Running Model Prediction...")
    with st.spinner("Please wait — analyzing image..."):
        pred = model.predict(img)[0][0]

    # ----------------------------
    # DISPLAY RESULT
    # ----------------------------
    label = "🧬 Pneumonia Detected" if pred > 0.5 else "💨 Normal (Healthy)"
    confidence = pred if pred > 0.5 else 1 - pred

    st.markdown("---")
    st.subheader("🔬 Prediction Result")
    if pred > 0.5:
        st.error(f"⚠️ **{label}**")
        st.write(f"Confidence: {confidence*100:.2f}%")
        st.warning("Seek medical advice promptly for further confirmation.")
    else:
        st.success(f"✅ **{label}**")
        st.write(f"Confidence: {confidence*100:.2f}%")
        st.balloons()
        st.info("🎉 Great news! The X-ray shows healthy lungs.")
    st.markdown("---")
else:
    st.info("☁️ Please upload a chest X-ray image to begin diagnosis.")
