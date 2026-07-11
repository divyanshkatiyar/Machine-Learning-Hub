import numpy as np

# Training data
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 6, 8, 10], dtype=float)

# Parameters
w = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01
epochs = 1000

n = len(X)

for epoch in range(epochs):

    # Forward pass
    y_pred = w * X + b

    # Compute gradients
    dw = (-2 / n) * np.sum(X * (Y - y_pred))
    db = (-2 / n) * np.sum(Y - y_pred)

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

# Results
print("Weight:", w)
print("Bias:", b)

# Prediction
x_test = 6
prediction = w * x_test + b

print("Prediction for x = 6:", prediction)
