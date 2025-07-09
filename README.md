
---

## 🔹 2. Handwritten Digit Recognition — `README.md`

```markdown
# ✍️ Handwritten Digit Recognition (MNIST)

## 🔍 Overview
This project uses a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) from the MNIST dataset.

## 📁 Dataset
- Built-in Keras dataset: `mnist`
- No download required

## 🛠 Technologies Used
- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)

## 🚀 How to Run
1. Run the script `digit_recognition.py` directly.
```bash
python digit_recognition.py

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Build model
model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)
