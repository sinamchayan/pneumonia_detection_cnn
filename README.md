# Pneumonia Detection from Chest X-Rays 🫁

This repository contains a Deep Learning solution for detecting Pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). The model is built with **TensorFlow** and **Keras** and includes data augmentation to handle class imbalance and improve generalization.

---

##  Project Overview
The goal of this project is to automate the detection of pneumonia in pediatric chest X-rays. Because the dataset has a significant imbalance and a very small default validation set, this implementation optimizes the training process by rebalancing the validation logic.

### Model Architecture
The CNN consists of:
* **3 Convolutional Layers**: 64 filters each, using ReLU activation for feature extraction.
* **Max-Pooling Layers**: To reduce spatial dimensions and computational load.
* **Dense Layers**: A 128-unit fully connected layer followed by a single-unit output layer.
* **Activation**: Sigmoid output for binary classification (Normal vs. Pneumonia).




## 📁 Dataset Structure
The project uses the standard Chest X-Ray dataset structure:

```text
chest_xray/
├── train/      # 5,216 images (Used for training with augmentation)
├── test/       # 624 images (Used for validation & final evaluation)
└── val/        # 16 images (Small subset)---

```
### Training
To train the model from scratch, run the Jupyter Notebook `Pneumonia Detection.ipynb`. The training script includes:

* **Rescaling**: Normalizing pixel values to $[0, 1]$.
* **Augmentation**: Shear, zoom, and horizontal flips.
* **Optimization**: Adam optimizer with Binary Cross-Entropy loss.

### Loading the Model
To perform inference without retraining, use the following:

```python
from tensorflow.keras.models import load_model

# Load the modern Keras format
model = load_model('Pneumonia_Detection_model.keras')

# Run evaluation on test data
results = model.evaluate(test_generator)
print(f"Test Accuracy: {results[1]*100:.2f}%")
