import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

X = np.array([[0, 0, 1], [0, 1, 1],
              [1, 0, 1], [1, 1, 1]])

y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
w0 = 2 * nr.random((3, 4)) - 1
w1 = 2 * nr.random((4, 1)) - 1

n_iter = 60000
layer2_errors = np.zeros(n_iter, dtype=float)

for j in range(n_iter):
    # Feed forward through layers 0, 1, and 2
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, w0))
    layer2 = sigmoid(np.dot(layer1, w1))

    # how much did we miss the target value?
    layer2_error = y - layer2  # L1 norm
    layer2_errors[j] = np.mean(np.abs(layer2_error))

    if (j % 10000) == 0:
        print("Error:" + str(layer2_errors[j]))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    layer2_delta = layer2_error * sigmoid_deriv(layer2)

    # how much did each l1 value contribute to the layer2 error
    # (according to the weights)?
    layer1_error = layer2_delta.dot(w1.T)

    # in what direction is the target layer1?
    # were we really sure? if so, don't change too much.
    layer1_delta = layer1_error * sigmoid_deriv(layer1)

    w1 += np.dot(layer1.T, layer2_delta)
    w0 += np.dot(layer0.T, layer1_delta)

start_idx = 1000
plt.plot(range(start_idx, n_iter), layer2_errors[start_idx:])
