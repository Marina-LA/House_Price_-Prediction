import numpy as np

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
        self.h = [None] + [np.zeros(self.n[i]) for i in range(1, self.L)]
        self.xi = [None] + [np.zeros(self.n[i]) for i in range(1, self.L)]
        self.delta = [None] + [np.zeros(self.n[i]) for i in range(1, self.L)]
        self.theta = [None] + [np.random.randn(self.n[i]) for i in range(1, self.L)]
        self.d_theta = [None] + [np.zeros(self.n[i]) for i in range(1, self.L)]
        self.d_theta_prev = [None] + [np.zeros(self.n[i]) for i in range(1, self.L)]

        # Initialize arrays of matrices
        self.w = [None] + [np.random.randn(self.n[i], self.n[i - 1]) for i in range(1, self.L)]
        self.d_w = [None] + [np.random.randn(self.n[i], self.n[i - 1]) for i in range(1, self.L)]
        self.d_w_prev = [None] + [np.zeros((self.n[i], self.n[i - 1])) for i in range(1, self.L)]


    def _activation_function(self, x):
        """Compute the activation and its derivative based on the chosen function."""
        if self.fact == 'sigmoid':
            act = 1 / (1 + np.exp(-x))
            return act, act * (1 - act)
        elif self.fact == 'relu':
            act = np.maximum(0, x)
            return act, np.where(x > 0, 1, 0)
        elif self.fact == 'linear':
            return x, np.ones_like(x)
        elif self.fact == 'tanh':
            act = np.tanh(x)
            return act, 1 - act**2
        else:
            raise ValueError("Unknown activation function.")
            

    def fit(self, X, y):
        """Training of the neural network."""
        n_samples = X.shape[0]
        idx = np.arange(n_samples)
        np.random.shuffle(idx)

        val_size = int(self.val_split * n_samples)
        train_idx, val_idx = idx[val_size:], idx[:val_size]

        X_train, y_train = X[train_idx], y[train_idx]

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")

            for x, target in zip(X_train, y_train):
                # Perform a feed forward pass
                self._feed_forward(x)

                # Perform a backpropagation pass
                self._back_propagate(target)

                # Update the weights and thresholds
                self._update_weights_thresholds()


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
        self.delta[-1] = self._activation_function(self.xi[-1])[1] * (self.xi[-1] - target)

        # Compute the delta for the hidden layers
        for l in range(self.L - 1, 0, -1):
            self.delta[l-1] = self._activation_function(self.xi[l-1])[1] * np.dot(self.w[l].T, self.delta[l])


    def _update_weights_thresholds(self):
        """Update the weights and thresholds."""
        for l in range(1, self.L):
            self.d_w[l] = -self.learning_rate * np.outer(self.delta[l], self.xi[l - 1])
            self.w[l] += self.d_w[l] + self.momentum * self.d_w_prev[l]
            self.d_w_prev[l] = self.d_w[l]

            self.d_theta[l] = self.learning_rate * self.delta[l]
            self.theta[l] += self.d_theta[l] + self.momentum * self.d_theta_prev[l]
            self.d_theta_prev[l] = self.d_theta[l]



if __name__ == "__main__":
	# example data
    X = np.array([[0.0, 1.0], [0.1, 1.0], [0.2, 1.0], [0.3, 1.0], [0.4, 1.0],
              [0.5, 1.0], [0.6, 1.0], [0.7, 1.0], [0.8, 1.0], [0.9, 1.0],
              [1.0, 1.0], [0.0, 0.9], [0.1, 0.9], [0.2, 0.9], [0.3, 0.9],
              [0.4, 0.9], [0.5, 0.9], [0.6, 0.9], [0.7, 0.9], [0.8, 0.9],
              [0.9, 0.9], [1.0, 0.9], [0.0, 0.8], [0.1, 0.8], [0.2, 0.8],
              [0.3, 0.8], [0.4, 0.8], [0.5, 0.8], [0.6, 0.8], [0.7, 0.8],
              [0.8, 0.8], [0.9, 0.8], [1.0, 0.8], [0.0, 0.7], [0.1, 0.7],
              [0.2, 0.7], [0.3, 0.7], [0.4, 0.7], [0.5, 0.7], [0.6, 0.7],
              [0.7, 0.7], [0.8, 0.7], [0.9, 0.7], [1.0, 0.7], [0.0, 0.6],
              [0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.6], [0.5, 0.6],
              [0.6, 0.6], [0.7, 0.6], [0.8, 0.6], [0.9, 0.6], [1.0, 0.6],
              [0.0, 0.5], [0.1, 0.5], [0.2, 0.5], [0.3, 0.5], [0.4, 0.5],
              [0.5, 0.5], [0.6, 0.5], [0.7, 0.5], [0.8, 0.5], [0.9, 0.5],
              [1.0, 0.5], [0.0, 0.4], [0.1, 0.4], [0.2, 0.4], [0.3, 0.4],
              [0.4, 0.4], [0.5, 0.4], [0.6, 0.4], [0.7, 0.4], [0.8, 0.4],
              [0.9, 0.4], [1.0, 0.4], [0.0, 0.3], [0.1, 0.3], [0.2, 0.3],
              [0.3, 0.3], [0.4, 0.3], [0.5, 0.3], [0.6, 0.3], [0.7, 0.3],
              [0.8, 0.3], [0.9, 0.3], [1.0, 0.3], [0.0, 0.2], [0.1, 0.2],
              [0.2, 0.2], [0.3, 0.2], [0.4, 0.2], [0.5, 0.2], [0.6, 0.2],
              [0.7, 0.2], [0.8, 0.2], [0.9, 0.2], [1.0, 0.2], [0.0, 0.1],
              [0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1], [0.5, 0.1],
              [0.6, 0.1], [0.7, 0.1], [0.8, 0.1], [0.9, 0.1], [1.0, 0.1],
              [0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0],
              [0.5, 0.0], [0.6, 0.0], [0.7, 0.0], [0.8, 0.0], [0.9, 0.0],
              [1.0, 0.0]])


    y = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
                [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0],
                [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0],
                [0], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0],
                [0], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0],
                [0], [0], [1], [1], [1], [1], [1], [0], [0], [0], [0],
                [0], [0], [1], [1], [1], [1], [1], [0], [0], [0], [0],
                [0], [0], [0], [1], [1], [1], [1], [1], [0], [0], [0],
                [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [0],
                [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [0],
                [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

    # Create the neural network
    nn = NeuralNet(layers=[2, 5, 3, 1], epochs=100, learning_rate=0.01, momentum=0.8, fact="sigmoid", val_split=0.2)

    # Train the neural network
    nn.fit(X, y)