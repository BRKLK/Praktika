import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import time
import sys


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Filtering Our Data (only 1s and 0s)
filter_train = (y_train == 0) | (y_train == 1)
x_train_filtered, y_train_filtered = x_train[filter_train], y_train[filter_train]

# Normalizing and Flattening images 
x_train_filtered = x_train_filtered / 255.0
x_train_filtered = x_train_filtered.reshape(x_train_filtered.shape[0], -1)
y_train_filtered = y_train_filtered.reshape(-1, 1)

# Number of units(neurons) per layer
input_units_num = x_train_filtered.shape[1]
hidden1_units_num = 25
hidden2_units_num = 15
output_units_num = 1

num_images = x_train_filtered.shape[0]
num_zeros = np.sum((x_train_filtered == 0).astype(int))
num_ones = np.sum((x_train_filtered == 1).astype(int))


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


def compute_gradient(x, y, a1, a2, a3, W2, W3, activation_hidden):
    m = y.shape[0]
    error_3 = a3 - y # Output error

    # Gradients for output layer
    dW3 = (1/m) * np.dot(a2.T, error_3)
    db3 = (1/m) * np.sum(error_3, axis=0, keepdims=True)

    # Gradients for the second hidden layer
    if activation_hidden == "relu":
        error_2 = np.dot(error_3, W3.T) * (a2 > 0).astype(float)
    else:
        error_2 = np.dot(error_3, W3.T) * (a2 * (1 - a2))
    dW2 = (1/m) * np.dot(a1.T, error_2)
    db2 = (1/m) * np.sum(error_2, axis=0, keepdims=True)

    # Gradients for the first hidden layer
    if activation_hidden == "relu":
        error_1 = np.dot(error_2, W2.T) * (a1 > 0).astype(float)
    else:
        error_1 = np.dot(error_2, W2.T) * (a1 * (1 - a1))
    dW1 = (1/m) * np.dot(x.T, error_1)
    db1 = (1/m) * np.sum(error_1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3


def gradient_descent(x_train, y_train, W1, b1, W2, b2, W3, b3, activation_hidden, learning_rate=0.01, num_of_epochs=1000):
    costs = []
    start_time = time.time()
    previous_cost = None


    for i in range(num_of_epochs):
        # Forward pass
        a1, a2, a3 =  forward_pass(x_train, W1, b1, W2, b2, W3, b3, activation_hidden, "sigmoid")

        cost = compute_cost(y_train, a3)
        costs.append(cost)

        dW1, db1, dW2, db2, dW3, db3 = compute_gradient(x_train, y_train, a1, a2, a3, W2, W3, activation_hidden)

        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
        W3 -= learning_rate * dW3
        b1 -= learning_rate * db1
        b2 -= learning_rate * db2
        b3 -= learning_rate * db3

        end_time = time.time()
        elapsed_time = end_time - start_time

        
        if i % 100 == 0:
            if previous_cost is not None:
                delta_cost = cost - previous_cost
                print(f"Iteration: {i}, Cost: {cost}, D: {delta_cost}, t: {elapsed_time}")
            else:
                print(f"Iteration: {i}, Cost:{cost}, t: {elapsed_time}")
            previous_cost = cost
    
    end_time = time.time()
    training_time = end_time - start_time

    return W1, b1, W2, b2, W3, b3, training_time, costs
            

def print_sample_data(n_rows=10, n_cols=10):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))

    for i in range(n_rows * n_cols):

        if i % 10 == 0:
            print(f"Processing image {i}")

        image = x_train_filtered[i].reshape(28, 28)
        label = y_train_filtered[i][0]

        axes[i // n_cols, i % n_cols].imshow(image, cmap='gray')
        axes[i // n_cols, i % n_cols].axis('off')
        axes[i // n_cols, i % n_cols].set_title(str(label), font_size=12)

    image_path = os.path.join("Images", "Test_1s_and_0s.png")
    plt.savefig(image_path,  bbox_inches='tight', dpi=80)
    plt.close
    print("Saved images with labels in a grid.")


def model_summary(W1, b1, W2, b2, W3, b3, training_time, initial_cost, final_cost, learning_rate, num_epochs, activation_hidden, init_method, accuracy):
    
    # Calculate total parameters
    total_parameters = (W1.size + b1.size) + (W2.size + b2.size) + (W3.size + b3.size)
    cost_reduction = ((initial_cost - final_cost) / initial_cost) * 100  # Percentage reduction

    print("\nModel Summary:")
    print(f"Number of Training Images: {num_images}")
    print(f"Number of '0' Images: {num_zeros}")
    print(f"Number of '1' Images: {num_ones}")
    print(f"Input Image Shape: (784,) (Flattened from 28x28)")
    print("Normalization: Pixel values scaled to [0, 1]\n")

    print("Training Configuration:")
    print(f"- Activation Function (Hidden Layers): {activation_hidden.capitalize()}")
    print(f"- Activation Function (Output Layer): Sigmoid")
    print(f"- Weight Initialization: {init_method.capitalize()}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Number of Epochs: {num_epochs}")
    print(f"- Initial Cost Value: {initial_cost:.4f}")
    print(f"- Final Cost Value: {final_cost:.4f}")
    print(f"- Cost Reduction: {cost_reduction:.2f}%")
    print(f"- Accuracy: {accuracy:.5f}%\n")
    print(f"- Training Time: {training_time:.2f} seconds\n")

    print("Weight and Bias Parameters:")
    print(f"- W1: {W1.shape}, b1: {b1.shape}")
    print(f"- W2: {W2.shape}, b2: {b2.shape}")
    print(f"- W3: {W3.shape}, b3: {b3.shape}")
    print(f"- Total Parameters: {total_parameters}\n")

    print("Environment Details:")
    print("- Hardware: CPU")  # Update this if you use GPU in the future
    print("- Python Version:", sys.version)
    print("- NumPy Version:", np.__version__)


def compute_accuracy(x_train, y_train, W1, b1, W2, b2, W3, b3, activation_hidden):
    a1, a2, a3= forward_pass(x_train, W1, b1, W2, b2, W3, b3, activation_hidden, "sigmoid")
    return (np.sum(((a3 >= 0.5).astype(int) == y_train).astype(float)) / y_train.shape[0]) * 100

def predict(x_train, W1, b1, W2, b2, W3, b3, activation_hidden):
    _, _, a3= forward_pass(x_train, W1, b1, W2, b2, W3, b3, activation_hidden, "sigmoid")
    return (a3 >= 0.5).astype(int)

def display_predictions(sample_images, predictions, sample_labels):
    num_samples =  len(sample_images)

    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))

    #case for a single image
    if num_samples == 1:
        image = sample_images[0].reshape(28, 28)
        axes[1].imshow(image, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title(f"Pred: {predictions}, True: {sample_labels}")
    #case for multiple images
    else:
        for i in range(num_samples):
            image = sample_images[i].reshape(28, 28)
            axes[i].imshow(image,  cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"Pred: {predictions[i]}, True: {sample_labels[i]}")
    
    image_path = os.path.join("Images", "Sample_predictions.png")
    plt.savefig(image_path, bbox_inches='tight', dpi=800)
    plt.close()

    print(f"Saved sample images and their prediction at {image_path}")

# Configuration
init_method = 'r' # 'r' stands for random
activation = 'r' # 's' stands for sigmoid activation function and 'r' is for relu
mode = 't' # 't' for training, 'p' for prediction (if we  already got weights and biases)
learning_rate = 0.01
num_of_epochs = 10_000

activation_hidden = 'sigmoid' if  activation == 's' else 'relu'

if mode == 't':
    # Check if saved weights and biases exist
    files = ["model\\W1.npy", "model\\b1.npy", "model\\W2.npy", "model\\b2.npy", "model\\W3.npy", "model\\b3.npy"]
    if all(os.path.exists(path) for path in files):
        W1 = np.load("model\\W1.npy")
        b1 = np.load("model\\b1.npy")
        W2 = np.load("model\\W2.npy")
        b2 = np.load("model\\b2.npy")
        W3 = np.load("model\\W3.npy")
        b3 = np.load("model\\b3.npy")
        print("Loaded saved weights and biases.")
    else:
        print("No saved weights and biases found")
        print("Now initialising W and B")
        # if init_method == 'r':
        # Initializing weights with random variables from normal distribution
        W1 = np.random.randn(input_units_num, hidden1_units_num) * 0.01
        W2 = np.random.randn(hidden1_units_num, hidden2_units_num) * 0.01
        W3 = np.random.randn(hidden2_units_num, output_units_num) * 0.01
        # Initializing biases with zeros
        b1 = np.zeros((1, hidden1_units_num))
        b2 = np.zeros((1, hidden2_units_num))
        b3 = np.zeros((1, output_units_num))

    # Training the model
    W1, b1, W2, b2, W3, b3, training_time, costs = gradient_descent(
        x_train_filtered, 
        y_train_filtered, 
        W1, b1, W2, b2, W3, b3, 
        activation_hidden, 
        learning_rate, 
        num_of_epochs
        )
    
    iterations = range(1, len(costs) + 1)
    plt.plot(iterations, costs, label="Cost")
    plt.title("Cost v. Iterations")
    plt.xlabel('Iterations')
    plt.ylabel("Cost")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join("Images", "Cost_vs_iterations.png"), bbox_inches='tight', dpi=800)

    # Saving model parameters (Weights and biases)
    np.save("model\\W1.npy", W1)
    np.save("model\\b1.npy", b1)
    np.save("model\\W2.npy", W2)
    np.save("model\\b2.npy", b2)
    np.save("model\\W3.npy", W3)
    np.save("model\\b3.npy", b3)
    
    # Summarizing the model
    accuracy = compute_accuracy(x_train_filtered, y_train_filtered, W1, b1, W2, b2, W3, b3, activation_hidden)
    model_summary(W1, b1, W2, b2, W3, b3, training_time, costs[0], costs[-1], learning_rate, num_of_epochs, activation_hidden, init_method, accuracy)

elif mode == "p":
    print("Prediction mode: loading the model and making predictions")

    # Loading Weights and biases
    W1 = np.load("model\\W1.npy")
    b1 = np.load("model\\b1.npy")
    W2 = np.load("model\\W2.npy")
    b2 = np.load("model\\b2.npy")
    W3 = np.load("model\\W3.npy")
    b3 = np.load("model\\b3.npy")

    user_input = input("Enter the range of indices (index1-index2): ")

    start, end = map(int, user_input.split('-'))

    x_train_sample = x_train_filtered[start: end+1]
    y_train_sample = y_train_filtered[start: end+1]
    predictions = predict(x_train_sample, W1, b1, W2, b2, W3, b3, activation_hidden)

    display_predictions(x_train_sample, predictions, y_train_sample)

