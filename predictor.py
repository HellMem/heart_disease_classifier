# Based using this dataset https://www.kaggle.com/sulianova/cardiovascular-disease-dataset


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def get_nn_model(data, labels, layer_size, learning_rate):
    clf = MLPClassifier(solver='adam', alpha=learning_rate, hidden_layer_sizes=layer_size, random_state=1)
    return clf.fit(data, labels)


def experiment(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.20, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    parameter_space = {
        'hidden_layer_sizes': [(5, 2), (6, 2), (7, 2), (8, 2), (9, 2)],
        'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    }

    for hidden_layer_size in parameter_space['hidden_layer_sizes']:
        for alpha in parameter_space['alpha']:
            nn_model = get_nn_model(X_train, y_train, hidden_layer_size, alpha)

            print('For {} hidden layers and {} learning rate:'.format(hidden_layer_size, alpha))

            '''
            pred_train_y = nn_model.predict(X_train)
            print("Train report")
            print(classification_report(y_train, pred_train_y))

            print('-' * 100)'''

            pred_test_y = nn_model.predict(X_test)
            print("Test report")
            print(classification_report(y_test, pred_test_y))
            print('-' * 100)
            print('-' * 100)
            print('-' * 100)

        print('*' * 500)


if __name__ == "__main__":
    cardio = pd.read_csv("data/cardio.csv", delimiter=';')
    experiment(cardio.drop('cardio', 1).values, cardio['cardio'].values)
