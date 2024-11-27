from NeuralNet import NeuralNet
import numpy as np

def cross_validation(X, y, layers, epochs, learning_rate, momentum, fact, val_split, k):
    num_samples = X.shape[0]
    fold_size = num_samples // k
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # split data into k folds
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k)]

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Initialize the neural network
        nn = NeuralNet(layers, epochs, learning_rate, momentum, fact, val_split)

        # Train the neural network
        nn.fit(X_train, y_train)

        # Evaluate the neural network
        train_error = nn._calculate_error(X_train, y_train)
        val_error = nn._calculate_error(X_val, y_val)

        print(f"Fold {i + 1}/{k} - Train Error: {train_error:.4f}, Validation Error: {val_error:.4f}")


def load_data_csv(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    return data


if __name__ == "__main__":
    
    data = load_data_csv('./data/processed_data.csv')

