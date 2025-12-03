# =============================================================================
# Organ Router Training Script
# Purpose: Train a model that routes an input image to the correct organ
# Dataset: Multi Cancer Dataset (folder-per-class structure)
# Environment: TensorFlow/Keras, OpenCV, LIME (optional)
# Date: March 2025
# =============================================================================

import argparse
import json
import os
import random
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Train the organ routing classifier.")
    parser.add_argument(
        "--data-root",
        default="data/Multi_Cancer",
        help="Root folder containing organ-class folders (e.g., Brain Cancer, Kidney Cancer).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Maximum number of images to sample per organ folder.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=(224, 224),
        metavar=("H", "W"),
        help="Input image size (height width).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs.",
    )
    parser.add_argument(
        "--model-output",
        default="models/organ_router.h5",
        help="Where to save the router model.",
    )
    parser.add_argument(
        "--labels-output",
        default="models/organ_router_labels.json",
        help="Where to save the class label order for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def format_label(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def load_data(data_root: str, sample_size: int, img_size: Tuple[int, int]):
    raw_classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    friendly_classes = [format_label(name) for name in raw_classes]
    images = []
    labels = []
    for idx, (folder, friendly) in enumerate(zip(raw_classes, friendly_classes)):
        class_path = os.path.join(data_root, folder)
        all_img_paths = []
        for root, _, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    all_img_paths.append(os.path.join(root, file))
        if not all_img_paths:
            print(f"Warning: No images for {friendly} ({class_path})")
            continue
        chosen = random.sample(all_img_paths, min(sample_size, len(all_img_paths)))
        for img_path in chosen:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load {img_path}")
                continue
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            images.append(img)
            labels.append(idx)
    return np.array(images), np.array(labels), friendly_classes


def build_model(num_classes: int, input_shape: Tuple[int, int, int]):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    img_size = tuple(args.img_size)
    print(f"Loading data from {args.data_root}...")
    X, y, class_names = load_data(args.data_root, args.sample_size, img_size)
    if len(X) == 0:
        print("Error: No images loaded for router training.")
        return

    num_classes = len(class_names)
    if num_classes < 2:
        print("Error: Need at least two organ folders to train the router.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    model = build_model(num_classes, input_shape=(img_size[0], img_size[1], 3))

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    datagen.fit(X_train)

    model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Router Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    model.save(args.model_output)
    print(f"Router model saved to {args.model_output}")

    with open(args.labels_output, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f"Router labels saved to {args.labels_output}")


if __name__ == "__main__":
    main()
# =============================================================================
