import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

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
            self.theta.append(np.random.randn(layers[lay]) * np.sqrt(2.0 / layers[lay - 1] if lay > 0 else 1))
            self.d_theta.append(np.zeros(layers[lay]))
            self.d_theta_prev.append(np.zeros(layers[lay]))

        # Initialize arrays of matrices
        self.w = [] # weights
        self.d_w = [] # weight updates
        self.d_w_prev = [] # previous weight updates
        self.w.append(np.random.randn(1, 1) * np.sqrt(2.0))
        self.d_w.append(np.zeros((1, 1)))
        self.d_w_prev.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.w.append(np.random.randn(layers[lay], layers[lay - 1]) * np.sqrt(2.0 / layers[lay - 1]))
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
            
    def fit(self, X, y, k_folds=4):
        """Training the neural network with Cross-Validation."""
        n_samples = X.shape[0]
        idx = np.arange(n_samples)
        np.random.shuffle(idx)  # Shuffle indices to randomize the folds

        fold_size = n_samples // k_folds  # Calculate the size of each fold

        # Store errors for each fold
        self.fold_train_errors = []
        self.fold_val_errors = []

        for fold in range(k_folds):
            print(f"\nFold {fold + 1}/{k_folds}")

            # Define validation indices for the current fold
            val_idx = idx[fold * fold_size: (fold + 1) * fold_size]
            # Define training indices as the remaining data
            train_idx = np.setdiff1d(idx, val_idx)

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Initialize errors for this fold
            self.train_errors = []
            self.val_errors = []

            for epoch in tqdm(range(self.epochs)):
                for pat in range(len(X_train)):
                    # Choose a pattern at random
                    x, target = X_train[pat], y_train[pat]
                    
                    # Perform a feed forward pass
                    self._feed_forward(x)

                    # Perform a backpropagation pass
                    self._back_propagate(target)

                    # Update the weights and thresholds
                    self._update_weights_thresholds()

                # Calculate training and validation errors for this epoch
                train_error = self._calculate_error(X_train, y_train)
                val_error = self._calculate_error(X_val, y_val)

                self.train_errors.append(train_error)
                self.val_errors.append(val_error)

            # Store the final errors of this fold
            self.fold_train_errors.append(self.train_errors)
            self.fold_val_errors.append(self.val_errors)



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
            total_error += 0.5 * np.sum((self.xi[-1] - target) ** 2)
        return total_error / len(X)


    def loss_epochs(self):
        """Return arrays with training and validation errors for each epoch."""
        return np.array(self.fold_train_errors), np.array(self.fold_val_errors)
    


def read_data(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data.append([float(value) for value in row])
    return np.array(data)

def mean_squared_error(y_pred, y_true):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def mean_absolute_error(y_pred, y_true):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def mean_absolute_percentage_error(y_pred, y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    non_zero_indices = y_true != 0
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

if __name__ == "__main__":
    X_train = read_data('./data/X_train.csv')
    y_train = read_data('./data/y_train.csv')

    X_test = read_data('./data/X_test.csv')
    y_test = read_data('./data/y_test.csv')

    nn = NeuralNet(layers=[14, 128, 64, 32, 1], epochs=190, learning_rate=0.2, momentum=0.15, fact='tanh', val_split=0.2)

    nn.fit(X_train, y_train, k_folds=10)

    predictions = nn.predict(X_test)

    print(f"Mean Squared Error: {mean_squared_error(predictions, y_test)}")
    print(f"Mean Absolute Error: {mean_absolute_error(predictions, y_test)}")
    print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(predictions, y_test):.4f}")   