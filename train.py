"""
Brain Tumor Detection - MobileNetV2 (lightweight, works great on CPU!)
Classification: Normal | Benign | Malignant
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_1    = 10
EPOCHS_2    = 15
LR_1        = 1e-3
LR_2        = 1e-4
NUM_CLASSES = 3
CLASSES     = ["Normal", "Benign", "Malignant"]
DATA_DIR    = "dataset"

def build_generators():
    train_aug = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_aug = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = train_aug.flow_from_directory(
        DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical",
        subset="training", shuffle=True, seed=42
    )
    val_gen = val_aug.flow_from_directory(
        DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical",
        subset="validation", shuffle=False, seed=42
    )
    return train_gen, val_gen

def build_model():
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    return model, base

def get_class_weights(gen):
    w = compute_class_weight("balanced", classes=np.unique(gen.classes), y=gen.classes)
    return dict(enumerate(w))

def train():
    print("Loading dataset...")
    train_gen, val_gen = build_generators()
    class_weights = get_class_weights(train_gen)
    print(f"Classes: {train_gen.class_indices}")
    print(f"Class weights: {class_weights}")

    model, base = build_model()

    print("\n=== Phase 1: Training classifier head ===")
    model.compile(
        optimizer=optimizers.Adam(LR_1),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    cb1 = [
        tf.keras.callbacks.ModelCheckpoint(
            "best_phase1.keras", monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4,
            restore_best_weights=True, verbose=1
        )
    ]

    h1 = model.fit(
        train_gen, epochs=EPOCHS_1,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=cb1
    )

    print("\n=== Phase 2: Fine-tuning top layers ===")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(LR_2),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb2 = [
        tf.keras.callbacks.ModelCheckpoint(
            "best_model_phase2.keras", monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        )
    ]

    h2 = model.fit(
        train_gen, epochs=EPOCHS_2,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=cb2
    )

    model.save("best_model_phase2.keras")
    print("\nModel saved to best_model_phase2.keras")
    return model, val_gen, h1, h2

def evaluate(model, val_gen):
    print("\n=== Evaluation ===")
    val_gen.reset()
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix - Brain Tumor Detection")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved!")

def plot_history(h1, h2):
    acc     = h1.history["accuracy"]     + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss    = h1.history["loss"]         + h2.history["loss"]
    val_loss= h1.history["val_loss"]     + h2.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(acc,     label="Train Accuracy")
    axes[0].plot(val_acc, label="Val Accuracy")
    axes[0].axvline(len(h1.history["accuracy"])-1, color="gray",
                    linestyle="--", label="Fine-tune starts")
    axes[0].set_title("Model Accuracy")
    axes[0].legend()

    axes[1].plot(loss,     label="Train Loss")
    axes[1].plot(val_loss, label="Val Loss")
    axes[1].axvline(len(h1.history["loss"])-1, color="gray",
                    linestyle="--", label="Fine-tune starts")
    axes[1].set_title("Model Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Training history saved!")

if __name__ == "__main__":
    model, val_gen, h1, h2 = train()
    evaluate(model, val_gen)
    plot_history(h1, h2)
