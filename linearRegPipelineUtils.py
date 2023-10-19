"""
Created on 17 Oct 2023

@author: diane
"""

import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import evaluationUtils as ev
from sklearn.linear_model import LinearRegression

def displayResults(model, X_test, y_test, y_pred, regType, features, target, foldNumber = 10):
    # affiche les réultats : type de regression, métriques,  graphes et 
    # coefficients de régression
    print("\n\nRésultat du type de regression : ", regType)
    metrics = getMetrics(y_test, y_pred)
    plotEvaluationGraphs(model, X_test, y_test, y_pred, regType)
    coefficients = getEvaluationInfo(model, features, target, foldNumber)
    return metrics, coefficients

def getMetrics(y_test, y_pred):
    # affiche et retourne  les métriques MAE, MSE, RMSE, et R²
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = metrics.mean_squared_error(y_test, y_pred, squared=False)
    R2 = metrics.r2_score(y_test, y_pred)
    print('MAE:', MAE)
    print('MSE:', MSE)
    print('RMSE:', RMSE)
    print('R²:', R2)
    
    return (MAE, MSE, RMSE, R2)
    
def plotEvaluationGraphs(model, X_test, y_test, y_pred, regType):
    # affiche les graphes (courbe apprentissage, résidus et erreurs)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axes[0] = ev.plot_learning_curve(axes[0], model, X_test, y_test)
    axes[1] = ev.error_plot(axes[1], y_test, y_pred, model)
    axes[2] = ev.residual_plot(axes[2], y_test, y_pred)
    
    fig.suptitle('Graphes pour le modèle : ' + regType, fontsize=16)
    plt.tight_layout()
    plt.show()

    
def getEvaluationInfo(model, features, target, foldNumber = 10):
    # Affiche les résultats pour la validation croisée
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
    # Affiche la colinéarité des colonnes dans X avec y
    pca = PCA(n_components = 0.7, svd_solver = 'full')
    df_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    print(axes.shape)
    for i, feature in enumerate(df_pca.columns):
        # Extract the current feature and reshape for sklearn
        current_feature = df_pca[feature].values.reshape(-1, 1)
        
        # Create and fit a linear regression model
        model = LinearRegression()
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