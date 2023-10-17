"""
Created on 17 Oct2023

@author: diane
"""
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn import metrics
import numpy as np


def pipeline(dataset, column):
    cols = dataset.drop(columns=[column])
    column = dataset[column]
    # sépare les datas en training et testing set
    X_train, X_test, y_train, y_test = splitTrainTest(
        cols, column, test_size=0.2, random_state=42)

    # cree un modele de regression multiple
    model = multipleLinReg()
    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', metrics.mean_squared_error(
        y_test, y_pred, squared=False))

    # Print the coefficients
    coefficients = pd.DataFrame(
        {'Variable': cols.columns, 'Coefficient': model.coef_})
    print(coefficients)

    # Print the intercept
    print('Intercept:', model.intercept_)

    # la variance est vraiment elevée : utilisation du modele regression ridge avec scikit learn
    if metrics.mean_absolute_error(y_test, y_pred) > 0.5:
        model_ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
        model_ridge.fit(X_train, y_train)
        best_alpha = model_ridge.alpha_

        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', metrics.mean_squared_error(
            y_test, y_pred, squared=False))
        # Print the coefficients
        coefficients = pd.DataFrame(
            {'Variable': cols.columns, 'Coefficient': model.coef_})
        print(coefficients)
        # Print the intercept
        print('Intercept:', model.intercept_)
        return (coefficients, model.intercept_)


def splitTrainTest(data, column, test_size=0.2, random_state=None):
    # test_size = Proportion of the dataset to include in the test split
    # random_state: Seed for random number generation to ensure reproducibility. Peut prendre n'importe quelle valeur entiere
    if isinstance(data, pd.DataFrame):
        data = data.values
    if isinstance(column, pd.Series):
        column = column.values

    if random_state is not None:
        np.random.seed(random_state)

    # mélange les indices
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    # calcule le nombre d'echantillons pour la base de test
    test_samples = int(len(data) * test_size)

    # sépare les données
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    X_train, X_test = data[train_indices], data[test_indices]
    y_train, y_test = column[train_indices], column[test_indices]

    return X_train, X_test, y_train, y_test


class multipleLinReg:
    def __init__(self):
        # initialise les coefficients intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, data, column):
        # définit le modele de regression lineaire

        if isinstance(data, pd.DataFrame):
            data = data.values
        if isinstance(column, pd.Series):
            column = column.values

        # ajoute une colonne de 1 aux données pour contenir le terme intercept
        X_with_intercept = np.c_[np.ones(data.shape[0]), data]

        # calcule les
        coefficients = np.linalg.inv(
            X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ column

        # ajoute les coeff et remplace les valeurs d'intercept
        self.intercept_ = coefficients[0]
        self.coef_ = coefficients[1:]

    def predict(self, data):
        # effectue les prédictions utilisant un modele de regression lineaire

        if isinstance(data, pd.DataFrame):
            data = data.values

        # Add a column of ones to data for the intercept term
        data_with_intercept = np.c_[np.ones(data.shape[0]), data]

        # Compute predictions
        predictions = data_with_intercept @ np.concatenate(
            [[self.intercept_], self.coef_])

        return predictions
