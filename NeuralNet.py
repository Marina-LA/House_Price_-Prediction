import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42) # For reproducibility

class NeuralNet:
    def __init__(self, layers, epochs, learning_rate, momentum, fact, val_split):
        self.L = len(layers)  # number of layers
        self.n = layers.copy()  # number of units per layer
        self.epochs = epochs  # number of epochs for training
        self.learning_rate = learning_rate  # learning rate
        self.momentum = momentum  # momentum
        self.val_split = val_split  # Percentage of data for validation [0..1]
        self.fact = fact  # Activation function

        # Initialize arrays of arrays
        self.xi = [] # activations
        self.h = [] # unit fields
        self.delta = [] # errors
        self.theta = [] # thresholds
        self.d_theta = [] # threshold updates
        self.d_theta_prev = [] # previous threshold updates
        for lay in range(self.L):
            self.xi.append(np.zeros(layers[lay]))
            self.h.append(np.zeros(layers[lay]))
            self.delta.append(np.zeros(layers[lay]))
            self.theta.append(np.random.randn(layers[lay]) * np.sqrt(2 / layers[lay - 1] if lay > 0 else 1))
            self.d_theta.append(np.zeros(layers[lay]))
            self.d_theta_prev.append(np.zeros(layers[lay]))

        # Initialize arrays of matrices
        self.w = [] # weights
        self.d_w = [] # weight updates
        self.d_w_prev = [] # previous weight updates
        self.w.append(np.random.randn(1, 1) * np.sqrt(2))
        self.d_w.append(np.zeros((1, 1)))
        self.d_w_prev.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.w.append(np.random.randn(layers[lay], layers[lay - 1]) * np.sqrt(2 / layers[lay - 1]))
            self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))
            self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))


    def _activation_function(self, x):
        """Compute the activation and its derivative based on the chosen function."""
        if self.fact == 'sigmoid':
            act = 1 / (1 + np.exp(-x))
            return act, act * (1 - act)
        elif self.fact == 'relu':
            act = np.maximum(0, x)
            return act, np.where(x > 0, 1, 0)
        elif self.fact == 'linear':
            return x, 1
        elif self.fact == 'tanh':
            act = np.tanh(x)
            return act, 1 - act**2
        else:
            raise ValueError("Unknown activation function.")
            

    def fit(self, X, y):
        """Training of the neural network."""
        n_samples = X.shape[0]
        idx = np.arange(n_samples)
        #np.random.shuffle(idx)

        val_size = int(self.val_split * n_samples)
        train_idx, val_idx = idx[val_size:], idx[:val_size]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Store errors for each epoch
        self.train_errors = []
        self.val_errors = []

        for epoch in tqdm(range(self.epochs)):
            #print(f"Epoch {epoch + 1}/{self.epochs}")

            for pat in range(len(X_train)):
                # Choose a pattern at random
                x, target = X_train[pat], y_train[pat]
                
                # Perform a feed forward pass
                self._feed_forward(x)

                # Perform a backpropagation pass
                self._back_propagate(target)

                # Update the weights and thresholds
                self._update_weights_thresholds()

            # Calculate training and validation errors
            train_error = self._calculate_error(X_train, y_train)
            val_error = self._calculate_error(X_val, y_val)

            self.train_errors.append(train_error)
            self.val_errors.append(val_error)
            #print(f"Train Error: {train_error:.4f} | Validation Error: {val_error:.4f}")


    def _feed_forward(self, x):
        """Perform a feed forward pass."""
        # Initialize the activation for the input layer
        self.xi[0] = x

        # Compute the activations for the hidden layers
        for l in range(1, self.L):
            self.h[l] = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
            self.xi[l] = self._activation_function(self.h[l])[0]


    def _back_propagate(self, target):
        """Perform a backpropagation pass."""
        # Compute the delta for the output layer
        self.delta[-1] = self._activation_function(self.h[-1])[1] * (self.xi[-1] - target)

        # Compute the delta for the hidden layers
        for l in range(self.L - 1, 0, -1):
            self.delta[l-1] = self._activation_function(self.h[l-1])[1] * np.dot(self.w[l].T, self.delta[l])


    def _update_weights_thresholds(self):
        """Update the weights and thresholds."""
        for l in range(1, self.L):
            self.d_w[l] = -self.learning_rate * np.outer(self.delta[l], self.xi[l - 1])
            self.w[l] += self.d_w[l] + self.momentum * self.d_w_prev[l]
            self.d_w_prev[l] = self.d_w[l]

            self.d_theta[l] = self.learning_rate * self.delta[l]
            self.theta[l] += self.d_theta[l] + self.momentum * self.d_theta_prev[l]
            self.d_theta_prev[l] = self.d_theta[l]


    def predict(self, X):
        """Predict the output for a given input."""
        predictions = []
        for x in X:
            self._feed_forward(x)
            predicted = self.xi[-1]
            predictions.append(predicted)
        return np.array(predictions)


    def _calculate_error(self, X, y):
        """Calculate the quadratic error for a given dataset."""
        total_error = 0
        for x, target in zip(X, y):
            self._feed_forward(x)
            total_error += (target - self.xi[-1]) ** 2
        mean_error = total_error / len(target)
        return mean_error


    def loss_epochs(self):
        """Return arrays with training and validation errors for each epoch."""
        return np.array(self.train_errors), np.array(self.val_errors)