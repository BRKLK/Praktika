import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Filtering Our Data (only 1s and 0s)
filter_train = (y_train == 0) | (y_train == 1)
x_train_filtered, y_train_filtered = x_train[filter_train], y_train[filter_train]

# Normalizing and Flattening images 
x_train_filtered = x_train_filtered / 255.0
x_train_filtered = x_train_filtered.reshape(x_train_filtered.shape[0], -1)
y_train_filtered = y_train_filtered.reshape(-1, 1)

# Number of units(neurons) per layer
input_units_num = x_train_filtered.shape[0]
hidden1_units_num = 25
hidden2_units_num = 15
output_units_num = 1

# Setting the activation functions
def relu(z):
    return np.maximum(0, z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def activation_function(z, function_type):
    if function_type == "relu":
        return relu(z)
    elif function_type == 'sigmoid':
        return sigmoid(z)
    else:
        raise ValueError("Invalid activation type. Use 'relu' or 'sigmoid' functions.")
    

# Forward pass through the Neural Network
def forward_pass(x, W1, b1, W2, b2, W3, b3, activation_hidden, activation_output) -> float:
    z1 = np.dot(x, W1) + b1
    a1 = activation_function(z1, activation_hidden)
    z2 = np.dot(a1, W2)  + b2
    a2 = activation_function(z2, activation_hidden)
    z3 = np.dot(a2, W3) + b3
    a3 = activation_function(z3, activation_output)
    return a1, a2, a3


def compute_cost(y: np.ndarray, y_pred: np.ndarray):
    m = y.shape[0]
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

    cost = -(1/m) * (np.sum((y * np.log(y_pred)) + (1-y) * np.log(1 - y_pred)))
    return cost

