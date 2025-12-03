# =============================================================================
# Explainable AI for Cancer Classification with Sampling and XAI (Notebook-style)
# =============================================================================

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tf_keras_vis.gradcam import Gradcam
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications import VGG16


IMAGE_SIZE = [224, 224]
LR = 0.001
EPOCHS = 20
BATCH_SIZE = 256
BASE_DATA_PATH = "data/Multi_Cancer"
OUTPUT_DIR = "models"


def initiateGenerator(path):
    base_path = path
    print("\nTotal : ", end=" ")
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(batch_size=32, directory=base_path)

    train_datagen = ImageDataGenerator(validation_split=0.3)

    print("\nFor Training : ", end=" ")
    train_generator = train_datagen.flow_from_directory(
        base_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical', subset='training')

    print("\nFor Val : ", end=" ")
    validation_generator = train_datagen.flow_from_directory(
        base_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation', shuffle=False)

    class_names = train_dataset.class_names
    noOfClasses = len(class_names)
    print("\nNo of Classes : ", noOfClasses)
    print("Classes : ", class_names)

    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(min(noOfClasses, 16)):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    for image_batch, labels_batch in train_dataset:
        print("Image Shape : ", image_batch.shape)
        break

    return noOfClasses, class_names, train_generator, validation_generator


def initiateModel(noOfClasses):
    modelInput = VGG16(
        input_shape=IMAGE_SIZE + [3],
        include_top=False,
        weights="imagenet"
    )
    for layer in modelInput.layers:
        layer.trainable = False
    x = Flatten()(modelInput.output)
    prediction = Dense(noOfClasses, activation='softmax')(x)
    model = Model(inputs=modelInput.input, outputs=prediction)
    return model


def initiateParams(className, model, lr):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
    slug = className.strip().replace(" ", "_").replace("/", "_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{slug}_VGG16.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)
    return model, annealer, checkpoint


def modelFit(model, annealer, checkpoint, train_generator, validation_generator, epochs=20, batchSize=256):
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        batch_size=batchSize,
        callbacks=[annealer, checkpoint],
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator)
    )
    return history


def plotOutput(history, className, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(3, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(f"{className.strip()}_graph.png")
    plt.close()


def saveModel(model, className):
    slug = className.strip().replace(" ", "_").replace("/", "_")
    output_path = os.path.join(OUTPUT_DIR, f"{slug} - VGG16.h5")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save(output_path)
    print(f"Model saved to {output_path}")


def main():
    class_paths = [d for d in os.listdir(BASE_DATA_PATH) if os.path.isdir(os.path.join(BASE_DATA_PATH, d))]
    print("Found organs:", class_paths)
    for className in class_paths:
        print(f"\n=== Training {className} ===")
        cpath = os.path.join(BASE_DATA_PATH, className)
        noOfClasses, class_names, train_generator, validation_generator = initiateGenerator(cpath)
        curModel = initiateModel(noOfClasses)
        curModel, annealer, checkpoint = initiateParams(className, curModel, LR)
        history = modelFit(curModel, annealer, checkpoint, train_generator, validation_generator, epochs=EPOCHS, batchSize=BATCH_SIZE)
        plotOutput(history, className, EPOCHS)
        saveModel(curModel, className)


if __name__ == "__main__":
    main()
