from NeuralNet import NeuralNet
import csv

def read_data(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data


def main():
    data = read_data('./data/processed_data.csv')
    for row in data:
        print(row)


if __name__ == "__main__":
    main()