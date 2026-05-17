"""
Predict on a single MRI image + Grad-CAM visualization
Usage: python predict.py --image path/to/mri.jpg
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224
CLASSES  = ["Normal", "Benign", "Malignant"]
MODEL_PATH = "best_model_phase2.h5"

# ─────────────────────────────────────────
# Load & preprocess
# ─────────────────────────────────────────
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), img

# ─────────────────────────────────────────
# Predict
# ─────────────────────────────────────────
def predict(model, img_array):
    probs = model.predict(img_array, verbose=0)[0]
    idx   = np.argmax(probs)
    return CLASSES[idx], probs

# ─────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────
def grad_cam(model, img_array, class_idx):
    # Find last conv layer (EfficientNetB3 last block)
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        # Try inner model (base is a sub-model)
        base = model.layers[1]
        for layer in reversed(base.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[base.get_layer(last_conv).output, model.output]
        )
    else:
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv).output, model.output]
        )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def overlay_cam(original_img, heatmap):
    h, w = np.array(original_img).shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    original_np = np.array(original_img)
    overlay = (0.6 * original_np + 0.4 * heatmap_colored).astype(np.uint8)
    return overlay

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main(img_path):
    model = load_model(MODEL_PATH)
    img_array, original_img = preprocess(img_path)
    label, probs = predict(model, img_array)

    print(f"\n{'='*40}")
    print(f"Prediction : {label}")
    print(f"Confidence : {max(probs)*100:.2f}%")
    print(f"{'='*40}")
    for cls, p in zip(CLASSES, probs):
        bar = "█" * int(p * 30)
        print(f"  {cls:<12} {bar:<30} {p*100:.1f}%")

    # Grad-CAM
    class_idx = np.argmax(probs)
    heatmap = grad_cam(model, img_array, class_idx)
    overlay = overlay_cam(original_img, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(original_img);     axes[0].set_title("Original MRI");  axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    axes[2].imshow(overlay);           axes[2].set_title(f"Prediction: {label} ({max(probs)*100:.1f}%)"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("gradcam_result.png", dpi=150)
    plt.show()
    print("\nGrad-CAM visualization saved to gradcam_result.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to MRI image")
    args = parser.parse_args()
    main(args.image)
