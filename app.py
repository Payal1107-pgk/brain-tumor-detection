import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

CLASSES = ["Normal", "Benign Tumor", "Malignant Tumor"]
IMG_SIZE = 224
MODEL_PATH = "best_model_phase2.keras"
CLASS_COLORS = {
    "Normal": "#22c55e",
    "Benign Tumor": "#f59e0b",
    "Malignant Tumor": "#ef4444"
}

@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

def preprocess(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0), img

def predict(model, img_array):
    probs = model.predict(img_array, verbose=0)[0]
    idx = np.argmax(probs)
    return CLASSES[idx], probs

def make_gradcam(model, img_array, class_idx):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break

        if last_conv is None:
            return None

        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)[0]
        pooled = tf.reduce_mean(grads, axis=(0, 1))
        heatmap = (conv_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap
    except:
        return None

def overlay_heatmap(original_img, heatmap):
    h, w = np.array(original_img).shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_col = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm_col = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * np.array(original_img) + 0.4 * hm_col).astype(np.uint8)
    return overlay

st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor Detection from MRI")
st.markdown("Upload an MRI brain scan to classify it as **Normal**, **Benign Tumor**, or **Malignant Tumor**.")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, caption="Uploaded MRI", width=300)

        if st.button("🔍 Analyze", type="primary"):
            with st.spinner("Analyzing MRI scan..."):
                try:
                    model = load_model_cached()
                    img_array, original_img = preprocess(uploaded)
                    label, probs = predict(model, img_array)
                    class_idx = np.argmax(probs)

                    with col2:
                        color = CLASS_COLORS.get(label, "#6366f1")
                        st.markdown(
                            f"<div style='padding:16px;border-radius:12px;"
                            f"background:{color}22;border:2px solid {color};text-align:center'>"
                            f"<h2 style='color:{color};margin:0'>{label}</h2>"
                            f"<p style='font-size:1.2em;margin:4px 0 0'>Confidence: "
                            f"<b>{max(probs)*100:.1f}%</b></p></div>",
                            unsafe_allow_html=True
                        )

                        st.markdown("#### Probability breakdown")
                        for cls, p in zip(CLASSES, probs):
                            c = CLASS_COLORS.get(cls, "#6366f1")
                            st.markdown(
                                f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
                                f"<span style='width:140px'>{cls}</span>"
                                f"<div style='flex:1;background:#e5e7eb;border-radius:6px;height:18px'>"
                                f"<div style='width:{p*100:.1f}%;background:{c};height:100%;border-radius:6px'></div></div>"
                                f"<b style='width:48px;text-align:right'>{p*100:.1f}%</b></div>",
                                unsafe_allow_html=True
                            )

                        heatmap = make_gradcam(model, img_array, class_idx)
                        if heatmap is not None:
                            st.markdown("#### Grad-CAM — Region AI focused on")
                            overlay = overlay_heatmap(original_img, heatmap)
                            st.image(overlay, caption="Red area = where AI detected tumor", width=300)

                        st.warning("⚠️ For educational purposes only. Always consult a qualified doctor.")

                except Exception as e:
                    st.error(f"Error: {e}")

with col2:
    if not uploaded:
        st.info("Upload an MRI image on the left to get started.")
        st.markdown("""
        **How it works:**
        1. Upload a brain MRI scan
        2. MobileNetV2 AI analyzes the image
        3. Shows Normal / Benign / Malignant
        4. Grad-CAM highlights the tumor region

        **Classes:**
        - 🟢 Normal — No tumor
        - 🟡 Benign — Non-cancerous tumor
        - 🔴 Malignant — Cancerous tumor

        **Model accuracy: 92%**
        """)
