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
        X_val, y_val = X[val_idx], y[val_idx]

        train_errors = []
        val_errors = []

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")

            for x, target in zip(X_train, y_train):
                # Perform a feed forward pass
                self._feed_forward(x, target)

                # Perform a backpropagation pass
                self._back_propagate(target)




    def _feed_forward(self, x):
        """Perform a feed forward pass."""
        # Initialize the activation for the input layer
        self.xi[0] = x

        # Compute the activations for the hidden layers
        for l in range(1, self.L):
            self.h[l] = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
            self.xi[l] = self._activation_function(self.h[l])


    def _back_propagate(self, target):
        pass


    def _update_weights_thresholds(self):
        pass



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
    nn = NeuralNet(layers=[2, 9, 5, 1], epochs=100, learning_rate=0.01, momentum=0.8, fact="sigmoid", val_split=0.2)
