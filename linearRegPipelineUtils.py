"""
Created on 17 Oct2023

@author: diane
"""
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression, Lasso
from sklearn import metrics
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import dataViz as dv


import evaluationUtils as ev

def displayResults(model, X_test, y_test, y_pred, regType, features, target, foldNumber = 10):
    print("\n\nDisplaying results for ", regType)
    getMetrics(y_test, y_pred)
    plotEvaluationGraphs(model, X_test, y_test, y_pred, regType)
    coefficients = getEvaluationInfo(model, features, target, foldNumber)
    return coefficients

def getMetrics(y_test, y_pred):
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))
    print('R²:', metrics.r2_score(y_test, y_pred))
    
def plotEvaluationGraphs(model, X_test, y_test, y_pred, regType):
    ev.plot_learning_curve(model, X_test, y_test)
    ev.error_plot(y_test, y_pred, model)
    ev.residual_plot(y_test, y_pred, regType)
    
def getEvaluationInfo(model, features, target, foldNumber = 10):
    print("Matrice de validation croisée")
    ev.cross_validation_matrix(model, features.values, target.values, foldNumber)
    
    # Print the coefficients
    coefficients = pd.DataFrame(
        {'Variable': features.columns, 'Coefficient': model.coef_})
    print(coefficients)

    # Print the intercept
    print('Intercept:', model.intercept_)
    return coefficients

def executePipelines(X, y):
    pipelineData = [
        ('Linear Regression', LinearRegression()),
        ('Linear Regression', LinearRegression()),
        ('Linear Regression', LinearRegression()),
        ('Linear Regression', LinearRegression()),
        ('Linear Regression', LinearRegression()),
        ('Linear Regression', LinearRegression()),
        ('Linear Regression', LinearRegression()),
        ('Linear Regression', LinearRegression()),
        ('Linear Regression', multipleLinReg()),
        ('Lasso', Lasso(alpha=1.0)),
        ('Ridge', RidgeCV(alphas=[0.1, 1.0, 10.0]))
    ]
    feature_names = X.columns
    selected_features = feature_names.append(pd.Index(["all"] * 3))
    combined_data = [(pipelineData[i][0], pipelineData[i][1], selected_features[i]) for i in range(len(pipelineData))]
    for model_name, model, col in combined_data:
        match col:
            case "all":
                if (model_name == 'Linear Regression'):
                    pipeline = Pipeline([
                        ('pca', PCA(n_components = 0.7, svd_solver = 'full')),
                        ('regressor', model)  # Linear regression model
                    ])
                else:
                    pipeline = Pipeline([
                        ('regressor', model)  # Linear regression model
                    ])
                    
                executeSinglePipeline(model_name, pipeline, X, y)
        
            case _:
                selected_X = X[[col]]
                pipeline = Pipeline([
                    ('regressor', model)  # Linear regression model
                ])
                executeSinglePipeline(model_name + " for " + col, pipeline, selected_X, y)
                
                
def executeSinglePipeline(model_name, pipeline, X, y):
    # Split data into training et testing set
    X_train, X_test, y_train, y_test = splitTrainTest(X, y, test_size=0.2, 
                                                      random_state=42)
    
    # Fit the model to the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = pipeline.predict(X_test)
    
    # Retrieve model
    model = pipeline['regressor']
    
    # Get metrics, graphs and other info for the results
    coefficients = displayResults(model, X_test, y_test, y_pred, 
                                  model_name, X, y)
    
    dv.analyzeReg(X, y, coefficients, model.intercept_)


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
