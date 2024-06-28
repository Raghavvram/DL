# -*- coding: utf-8 -*-
"""
Author: Raghavvram
"""

import random
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train_1, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train_1.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Display a random image from the training set
idx = random.randint(0, len(x_train_1))
plt.imshow(x_train_1[idx])
plt.show()

# Print the shape of the training data
print("Training data shape:", x_train.shape)

# Define a convolutional neural network (CNN) model
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),  # Input layer with shape (28, 28, 1)
    layers.Conv2D(32, 3, padding='valid', activation='relu'),  # 32 filters, 3x3 kernel, ReLU activation
    layers.Conv2D(64, 3, activation='relu'),  # 64 filters, 3x3 kernel, ReLU activation
    layers.MaxPool2D(),  # Max pooling layer
    layers.BatchNormalization(),  # Batch normalization layer
    layers.Conv2D(128, 3, activation='relu'),  # 128 filters, 3x3 kernel, ReLU activation
    layers.MaxPool2D(),  # Max pooling layer
    layers.BatchNormalization(),  # Batch normalization layer
    layers.Flatten(),  # Flatten the output for fully connected layers
    layers.Dense(64, activation='relu'),  # Fully connected layer with 64 units, ReLU activation
    layers.Dropout(0.5),  # Dropout layer with 50% dropout rate
    layers.Dense(10, activation='softmax'),  # Output layer with 10 units (for 10 classes), softmax activation
])


# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),  # Loss function for training
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),  # Optimizer with learning rate 0.0003
    metrics=["accuracy"],  # Evaluation metric during training
)


# Print a summary of the model architecture
model.summary()

# Train the model on the training data
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)

# Evaluate the model on the test data
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
