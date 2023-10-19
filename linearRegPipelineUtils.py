"""
Created on 17 Oct 2023

@author: diane
"""
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression, Lasso, ElasticNet
from sklearn import metrics
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import dataViz as dv
import matplotlib.pyplot as plt
import evaluationUtils as ev
import dataTreatmentUtils

def displayResults(model, X_test, y_test, y_pred, regType, features, target, foldNumber = 10):
    # affiche les réultats : type de regression, métriques,  graphes et coefficients de régression
    print("\n\nRésultat du type de regression : ", regType)
    getMetrics(y_test, y_pred)
    plotEvaluationGraphs(model, X_test, y_test, y_pred, regType)
    coefficients = getEvaluationInfo(model, features, target, foldNumber)
    return coefficients

def getMetrics(y_test, y_pred):
    # affiche la métrique du modèle
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', metrics.mean_squared_error(y_test, y_pred, squared=False))
    print('R²:', metrics.r2_score(y_test, y_pred))
    
def plotEvaluationGraphs(model, X_test, y_test, y_pred, regType):
    # affiche les graphes
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axes[0] = ev.plot_learning_curve(axes[0], model, X_test, y_test)
    axes[1] = ev.error_plot(axes[1], y_test, y_pred, model)
    axes[2] = ev.residual_plot(axes[2], y_test, y_pred)
    
    fig.suptitle('Graphe pour le modèle : ' + regType, fontsize=16)
    plt.tight_layout()
    plt.show()

    
def getEvaluationInfo(model, features, target, foldNumber = 10):
    print("Matrice de validation croisée")
    ev.cross_validation_matrix(model, features.values, target.values, foldNumber)
    
    # print les coefficients
    coefficients = pd.DataFrame(
        {'Variable': features.columns, 'Coefficient': model.coef_})
    print(coefficients)

    # Print l'interception
    print('Constante de régression:', model.intercept_)
    return coefficients

def checkColumnsLinearity(X, y):
    pca = PCA(n_components = 0.7, svd_solver = 'full')
    df_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    print(axes.shape)
    for i, feature in enumerate(df_pca.columns):
        # Extract the current feature and reshape for sklearn
        current_feature = df_pca[feature].values.reshape(-1, 1)
        
        # Create and fit a linear regression model
        model = multipleLinReg()
        model.fit(current_feature, y)
        
        # Predict using the model
        y_pred = model.predict(current_feature)
        
        # Calculate R-squared
        r_squared = metrics.r2_score(y, y_pred)
        
        # Print R-squared for the current feature
        print(f'R² pour {feature}: {r_squared}')
        
        # Optionally, you can plot the regression line and data points
        axes[i].scatter(current_feature, y, color='blue', label='Data')
        axes[i].plot(current_feature, y_pred, color='red', label='Regression Line')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Qualité')
        axes[i].set_title(f'Regression pour {feature}')
        axes[i].legend()
    
    fig.suptitle('Colinéarité entre PCA, colonnes et qualité', fontsize=16)
    plt.show()

def executePipelines(X, y):
    # Prepare la pipeline d'elements
    pipelineData = []
    for name in X.columns:
        pipelineData.append(('Regression linéaire simple pour ' + name 
                             + " caractéristique :", LinearRegression(), name))
    pipelineData.append(('Regression linéaire multiple', 
                         LinearRegression(), "all"))
    pipelineData.append(('Regression lasso', Lasso(alpha=1.0), "all"))
    pipelineData.append(('Regression Ridge', 
                         RidgeCV(alphas=[0.1, 1.0, 10.0]), "all"))
    pipelineData.append(('Regression Elastic Net', ElasticNet(), "all"))
    
    # crée une pipeline pour chaque regression qu'on souhaite appliquer puis l execute
    for model_name, model, col in pipelineData:
        match col:
            case "all":
                if (model_name == 'Regression Lineaire Multiple'):
                    pipeline = Pipeline([
                        ('pca', PCA(n_components = 0.7, svd_solver = 'full')),
                        ('regressor', model)  # Linear regression model
                    ])
                    checkColumnsLinearity(X, y)
                    # Retire les colonnes qui ne sont pas colinéaires avec la cible après le PCA
                    dataTreatmentUtils.removeNotColinearCol(X, y)
                    
                else:
                    pipeline = Pipeline([
                        ('regressor', model)  #Modele de regression linéaire
                    ])
                    
                executeSinglePipeline(model_name, pipeline, X, y)
        
            case _:
                selected_X = X[[col]]
                pipeline = Pipeline([
                    ('regressor', model)  #Modele de regression linéaire
                ])
                executeSinglePipeline(model_name, pipeline, selected_X, y)

def executeSinglePipeline(model_name, pipeline, X, y):
    # sépare le dataset en training et testing set
    X_train, X_test, y_train, y_test = splitTrainTest(X, y, test_size=0.2, 
                                                      random_state=42)
    
    # retrouve model
    model = pipeline['regressor']
    
    # applique le modele au training dataset
    pipeline.fit(X_train, y_train)

    # fait des predictions sur le testing set
    y_pred = pipeline.predict(X_test)
    
    
    # récupère les metriques, graph et autres infos sur les résultats
    coefficients = displayResults(model, X_test, y_test, y_pred, 
                                  model_name, X, y)
    
    dv.analyzeReg(X, y, coefficients, model.intercept_)


def splitTrainTest(data, column, test_size=0.2, random_state=None):
    # test_size = Proportion du dataset a inclure dans le test split
    # random_state: Seed pour le nombre random pour assurer la reproductibilité. Peut prendre n'importe quelle valeur entiere
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
        # initialise les coefficients
        self.coef_ = None
        self.intercept_ = None

    def fit(self, data, column):
        # définit le modele de regression lineaire

        if isinstance(data, pd.DataFrame):
            data = data.values
        if isinstance(column, pd.Series):
            column = column.values

        # ajoute une colonne de 1 aux données pour contenir la constante de régression
        X_with_intercept = np.c_[np.ones(data.shape[0]), data]

        # calcule les coef
        coefficients = np.linalg.inv(
            X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ column

        # ajoute les coeff et remplace les valeurs d'intercept
        self.intercept_ = coefficients[0]
        self.coef_ = coefficients[1:]

    def predict(self, data):
        # effectue les prédictions utilisant un modele de regression lineaire

        if isinstance(data, pd.DataFrame):
            data = data.values

        # ajoute une colonne contenant des '1' au dataset pour contenir ensuite les constantes de régression
        data_with_intercept = np.c_[np.ones(data.shape[0]), data]

        # affiche les prédictions
        predictions = data_with_intercept @ np.concatenate(
            [[self.intercept_], self.coef_])

        return predictions
