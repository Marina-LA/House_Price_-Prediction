from NeuralNet import NeuralNet
import matplotlib.pyplot as plt
import csv
import numpy as np
import tqdm
import json


def read_data(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Ignorar la primera línea (nombres de las columnas)
        for row in csv_reader:
            data.append([float(value) for value in row])
    return np.array(data)


def mean_squared_error(y_pred, y_true):
    total_error = 0
    for predicted, real in zip(y_pred, y_true):
        total_error += (real - predicted) ** 2
    mean_error = total_error / len(y_true)
    return mean_error


def mean_absolute_error(y_pred, y_true):
    total_error = 0
    for predicted, real in zip(y_pred, y_true):
        total_error += abs(real - predicted)
    mean_error = total_error / len(y_true)
    return mean_error


def mean_absolute_percentage_error(y_pred, y_true):
    total_error = 0
    for predicted, real in zip(y_pred, y_true):
        total_error += abs((real - predicted) / real)
    mean_error = total_error / len(y_true)
    return (mean_error * 100)


def scatter_plot(y_pred, y_true):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, label='Predictions')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Predictions')
    plt.title('Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.grid()
    plt.show()


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data



def main():
    X_train = read_data('./data/X_train.csv')
    y_train = read_data('./data/y_train.csv')

    X_test = read_data('./data/X_test.csv')
    y_test = read_data('./data/y_test.csv')

    """
    parameters_list = read_json('./parameters.json')['parameters']

    results = []

    for configuration in parameters_list:
        nn = NeuralNet(layers=configuration['layers'], epochs=configuration['epochs'],
                       learning_rate=configuration['learning_rate'], momentum=configuration['momentum'],
                       fact=configuration['activation'], val_split=0.2)

        nn.fit(X_train, y_train)
    
        predictions = nn.predict(X_test)

        mse = mean_squared_error(predictions, y_test)
        mae = mean_absolute_error(predictions, y_test)
        mape = mean_absolute_percentage_error(predictions, y_test)

        avg_error = (mse + mae + mape) / 3

        # Store configuration and errors in results
        results.append({
            **configuration,  # Unpack all parameters
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "avg_error": avg_error
        })

    # Sort results by average error
    sorted_results = sorted(results, key=lambda x: x["avg_error"])

    # Write sorted results to a CSV file
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = list(sorted_results[0].keys())  # Extract column names from the first result
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()  # Write column headers
        writer.writerows(sorted_results)  # Write all rows
    """



    nn = NeuralNet(layers=[14, 128, 64, 32, 1], epochs=120, learning_rate=0.01, momentum=0.8, fact="sigmoid", val_split=0.2)

    nn.fit(X_train, y_train)

    train_errors, val_errors = nn.loss_epochs()

    nn.plot_errors(train_errors, val_errors)

    predictions = nn.predict(X_test)

    mse = mean_squared_error(predictions, y_test)
    mae = mean_absolute_error(predictions, y_test)
    mape = mean_absolute_percentage_error(predictions, y_test)

    print("=============================================================")
    print('Metrics:')
    print(f'MSE: {mse}, MAE: {mae}, MAPE: {mape}')

    scatter_plot(predictions, y_test)




if __name__ == "__main__":
    main()