# MNIST Handwritten Digit Classification 🔢

A simple feedforward neural network (TensorFlow/Keras) that classifies handwritten digits (0–9) from the MNIST dataset.

## 📌 Overview

This project walks through the full deep learning workflow: loading and exploring the MNIST dataset, preprocessing images, building and training multiple neural network models, evaluating performance, visualizing training/validation curves, testing predictions on individual images, and running a controlled experiment with Dropout regularization.

## 📂 Repository Contents

| File | Description |
|---|---|
| `main.ipynb` | Main Jupyter Notebook — full code, outputs, and plots |
| `report.pdf` | Handwritten report (Introduction, Theory, Model, Results, Experiment, Conclusion) |
| `training_curves.png` | Training vs validation accuracy/loss plot |
| `predictions_sample.png` | Actual vs predicted labels on 5 test images |
| `baseline_vs_experiment.png` | Baseline vs Dropout experiment comparison plot |
| `README.md` | This file |

## 📊 Dataset

- **MNIST**: 70,000 grayscale handwritten digit images (60,000 train / 10,000 test)
- **Image size**: 28×28 pixels
- **Classes**: 10 (digits 0–9)
- Loaded directly via `keras.datasets.mnist.load_data()`

## 🧠 Models

| Model | Architecture | Output Activation | Test Accuracy |
|---|---|---|---|
| Attempt 1 | 784 → 10 | Sigmoid | 92.63% |
| Attempt 2 | 784 → 100 (ReLU) → 10 | Sigmoid | 97.67% |
| Final Baseline | 784 → 100 (ReLU) → 10 | Softmax | 97.54% |
| Experiment (+ Dropout 0.3) | 784 → 100 (ReLU) → Dropout → 10 | Softmax | **97.82%** |

- **Optimizer:** Adam
- **Loss function:** Sparse Categorical Crossentropy
- **Epochs:** 10 (final baseline & experiment), 10% validation split

## 🧪 Experiment: Dropout Regularization

Added `Dropout(0.3)` after the hidden layer, keeping every other setting identical to the baseline. Result: higher test accuracy (97.82% vs 97.54%) and a much smaller train–validation accuracy gap, showing reduced overfitting and better generalization.

## 🖼️ Sample Prediction Results

5 randomly sampled test images — all correctly classified:

| Actual | Predicted | Correct |
|---|---|---|
| 6 | 6 | ✅ |
| 8 | 8 | ✅ |
| 5 | 5 | ✅ |
| 3 | 3 | ✅ |
| 8 | 8 | ✅ |

## ▶️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Suresh-DG/MNIST_Handwritten_Digit_Classification
   cd MNIST_Handwritten_Digit_Classification
   ```
2. Open `main.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run all cells top to bottom (**Run All**). The dataset downloads automatically via Keras (internet connection required on first run).

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn

## 🚀 Future Improvements

- Replace the Dense network with a Convolutional Neural Network (CNN) to exploit the 2D spatial structure of the images — typically pushes MNIST accuracy above 99%.
- Try data augmentation (rotation, shifting) to improve robustness.
- Experiment with other regularization techniques (L2 weight decay, batch normalization).
