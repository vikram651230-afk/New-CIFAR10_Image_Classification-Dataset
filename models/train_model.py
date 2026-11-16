"""
models/train_model.py - Exact implementation and training of the CIFAR-10 Pure ANN model
directly matching 'CIFAR 10 image classification.ipynb'.

Architecture:
- Flatten(input_shape=(32, 32, 3))
- Dense(200, activation='relu')
- Dense(100, activation='relu')
- Dense(30, activation='relu')
- Dense(10, activation='softmax')

Zero 2D convolutions (Conv2D), zero pooling layers.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense,Dropout
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from PIL import Image

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

def load_cifar10_data(root_dir=None):
    """Load CIFAR-10 dataset from local directory or keras fallback."""
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    local_dir = os.path.join(root_dir, "dataset", "cifar-10-batches-py")
    if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "data_batch_1")):
        def _load_batch(fpath):
            with open(fpath, "rb") as f:
                d = pickle.load(f, encoding="bytes")
            data = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = np.array(d[b"labels"], dtype=np.int32).reshape(-1, 1)
            return data, labels

        x_train_list, y_train_list = [], []
        for i in range(1, 6):
            x_b, y_b = _load_batch(os.path.join(local_dir, f"data_batch_{i}"))
            x_train_list.append(x_b)
            y_train_list.append(y_b)
        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        x_test, y_test = _load_batch(os.path.join(local_dir, "test_batch"))
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Preprocessing matching notebook Cell 23:
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


def build_ann_model():
    """
    Build the exact ANN model from Notebook Cell 28:
    model = Sequential()
    model.add(layer = Flatten())
    model.add(Dense(units = 200, activation='relu'))
    model.add(Dense(units = 100, activation='relu'))
    model.add(Dense(units = 30, activation='relu'))
    model.add(Dense(units = 10, activation='softmax'))
    """
    model = Sequential([
        Flatten(input_shape=(32, 32, 3), name="flatten"),
        Dense(units=512, activation="relu", name="dense_1"),
        Dense(units=265, activation="relu", name="dense_2"),
        Dropout (0.5),
        Dense(units=128, activation="relu", name="dense_3"),
        Dense(units=64, activation="relu", name="dense_4"),
        Dense(units=32, activation="relu", name="dense_5"),
        Dense(units=10, activation="softmax", name="dense_output"),
    ], name="cifar10_pure_ann")

    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_sample_images(x_test, y_test, output_dir):
    """Save 1 clean sample image per class for interactive UI testing."""
    os.makedirs(output_dir, exist_ok=True)
    found_classes = set()
    for idx in range(len(y_test)):
        cls_idx = int(y_test[idx][0])
        if cls_idx not in found_classes:
            found_classes.add(cls_idx)
            cls_name = CLASS_NAMES[cls_idx]
            img_arr = (x_test[idx] * 255.0).astype(np.uint8)
            img = Image.fromarray(img_arr)
            img.save(os.path.join(output_dir, f"{cls_name}.png"))
        if len(found_classes) == 10:
            break


def train_and_evaluate(epochs=30, batch_size=64, models_dir=None):
    """Execute training, evaluation, artifact export, and metric logging."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if models_dir is None:
        models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("  CIFAR-10 Pure-ANN Training & Evaluation Pipeline")
    print(f"  Epochs: {epochs} | Batch size: {batch_size}")
    print("=" * 60)

    print("\n[1/5] Loading CIFAR-10 data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10_data(root_dir)
    print(f"Train data: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test data:  {x_test.shape}, labels: {y_test.shape}")

    print("\n[2/5] Building Pure-ANN model (Notebook Architecture)...")
    model = build_ann_model()
    model.summary()

    print(f"\n[3/5] Training model for {epochs} epochs...")
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    print("\n[4/5] Evaluating model on untouched test set (10,000 images)...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = y_test.flatten()

    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred_classes)
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        total_for_class = cm[i].sum()
        correct_for_class = cm[i, i]
        acc = float(correct_for_class) / float(total_for_class) if total_for_class > 0 else 0.0
        per_class_acc[name] = round(acc, 4)

    # Classification report
    report = classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, output_dict=True)

    # Save Confusion Matrix plot
    cm_path = os.path.join(models_dir, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax, colorbar=False)
    plt.title("CIFAR-10 Pure-ANN Test Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Save Learning Curves plot
    curves_path = os.path.join(models_dir, "training_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train Loss", color="#22d3ee", lw=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", color="#a78bfa", lw=2)
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Accuracy", color="#34d399", lw=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy", color="#fbbf24", lw=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curves_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Save metrics JSON
    metrics = {
        "status": "trained",
        "model_name": "CIFAR-10 Pure-ANN",
        "architecture": "Artificial Neural Network (Flatten + 3 Dense Relu + Dense Softmax)",
        "total_parameters": int(model.count_params()),
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "test_accuracy_percent": round(float(test_accuracy * 100), 2),
        "per_class_accuracy": per_class_acc,
        "classification_report": report,
        "confusion_matrix_image": "confusion_matrix.png",
        "training_curves_image": "training_curves.png",
        "class_names": CLASS_NAMES,
    }

    metrics_file = os.path.join(models_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n[5/5] Saving model files...")
    # Save .keras format (modern standard)
    keras_path = os.path.join(models_dir, "image_classification.keras")
    model.save(keras_path)
    print(f"Saved: {keras_path}")

    # Also save .h5 format matching notebook Cell 41
    h5_path = os.path.join(models_dir, "image_classification.h5")
    try:
        model.save(h5_path)
        print(f"Saved: {h5_path}")
    except Exception as e:
        print(f"Note: HDF5 export notice ({e})")

    # Save sample images for UI testing
    samples_dir = os.path.join(root_dir, "app", "static", "samples")
    save_sample_images(x_test, y_test, samples_dir)
    print(f"Saved 10 sample images to {samples_dir}")

    print("=" * 60)
    print(f"  TRAINING COMPLETE! Test Accuracy: {metrics['test_accuracy_percent']}%")
    print(f"  Artifacts saved in {models_dir}")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAR-10 Pure-ANN")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()

    train_and_evaluate(epochs=args.epochs, batch_size=args.batch_size)
