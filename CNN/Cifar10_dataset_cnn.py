# -*- coding: utf-8 -*-
"""
Author: Raghavvram
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Define the model architecture
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),  # Input layer with shape (32, 32, 3)
        layers.Conv2D(32, 3, padding='valid', activation='relu'),  # 32 filters, 3x3 kernel, ReLU activation
        layers.MaxPool2D(pool_size=(2, 2)),  # Max pooling with 2x2 pool size
        layers.Conv2D(64, 3, activation='relu'),  # 64 filters, 3x3 kernel, ReLU activation
        layers.MaxPool2D(),  # Default 2x2 max pooling
        layers.Conv2D(128, 3, activation='relu'),  # 128 filters, 3x3 kernel, ReLU activation
        layers.Flatten(),  # Flatten the output for fully connected layers
        layers.Dense(64, activation='relu'),  # Fully connected layer with 64 units, ReLU activation
        layers.Dense(10),  # Output layer with 10 units (for 10 classes)
    ]
)

# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"],
)

# Print model summary
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)

# Evaluate the model on test data
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
