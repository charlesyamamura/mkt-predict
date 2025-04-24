# Neural Network Regression Model with TensorFlow/Keras

A new product forecasting method integrating industry domain knowledge and an artificial neural network model. Industry expertise and feature engineering was used to carefully craft the model, precluding the need for a large dataset.
This repository contains a neural network implementation for a regression task, built using TensorFlow/Keras and scikit-learn. The model is designed to predict a continuous target variable based on engineered features extracted from an Excel dataset. It includes data normalization, training-validation-testing split, model training, evaluation, and performance visualization.

## 🧠 Model Summary

- **Framework**: TensorFlow/Keras
- **Architecture**:
  - Input layer: 16 features
  - Hidden layer: Dense (10 units, ReLU)
  - Output layer: Dense (1 unit, ReLU)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Evaluation metrics**: MAE, R² Score, Adjusted Accuracy

## 🔧 Setup

### Dependencies

Ensure Python 3 is installed, then install required packages:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib openpyxl
