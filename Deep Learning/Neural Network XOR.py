import random
import math

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(output):
    return output * (1 - output)

# Training data (XOR)
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

Y = [
    [0],
    [1],
    [1],
    [0]
]

# Architecture
input_size = 2
hidden_size = 2
output_size = 1

# Learning rate
lr = 0.5

# Random initialization
weights_input_hidden = [
    [random.uniform(-1, 1) for _ in range(hidden_size)]
    for _ in range(input_size)
]

weights_hidden_output = [
    [random.uniform(-1, 1) for _ in range(output_size)]
    for _ in range(hidden_size)
]

bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
bias_output = [random.uniform(-1, 1) for _ in range(output_size)]

# Training
epochs = 10000

for epoch in range(epochs):

    for x, y in zip(X, Y):

        # -------- Forward Propagation --------

        hidden = []

        for j in range(hidden_size):
            total = bias_hidden[j]
            for i in range(input_size):
                total += x[i] * weights_input_hidden[i][j]
            hidden.append(sigmoid(total))

        output = []

        for k in range(output_size):
            total = bias_output[k]
            for j in range(hidden_size):
                total += hidden[j] * weights_hidden_output[j][k]
            output.append(sigmoid(total))

        # -------- Backpropagation --------

        output_errors = []

        for k in range(output_size):
            error = y[k] - output[k]
            delta = error * sigmoid_derivative(output[k])
            output_errors.append(delta)

        hidden_errors = []

        for j in range(hidden_size):
            error = 0
            for k in range(output_size):
                error += output_errors[k] * weights_hidden_output[j][k]
            delta = error * sigmoid_derivative(hidden[j])
            hidden_errors.append(delta)

        # -------- Update Hidden -> Output --------

        for j in range(hidden_size):
            for k in range(output_size):
                weights_hidden_output[j][k] += (
                    lr * output_errors[k] * hidden[j]
                )

        for k in range(output_size):
            bias_output[k] += lr * output_errors[k]

        # -------- Update Input -> Hidden --------

        for i in range(input_size):
            for j in range(hidden_size):
                weights_input_hidden[i][j] += (
                    lr * hidden_errors[j] * x[i]
                )

        for j in range(hidden_size):
            bias_hidden[j] += lr * hidden_errors[j]

# Testing

print("Training Complete\n")

for x in X:

    hidden = []

    for j in range(hidden_size):
        total = bias_hidden[j]
        for i in range(input_size):
            total += x[i] * weights_input_hidden[i][j]
        hidden.append(sigmoid(total))

    output = []

    for k in range(output_size):
        total = bias_output[k]
        for j in range(hidden_size):
            total += hidden[j] * weights_hidden_output[j][k]
        output.append(sigmoid(total))

    print(f"{x} -> {round(output[0], 3)}")
