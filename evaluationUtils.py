# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:42:39 2023

@author: solene
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import metrics

def mae(y_true, y_pred):
    return ((y_pred - y_true) ** 2).mean()


def rmse(y_true, y_pred):
    return np.sqrt(mae(y_true, y_pred))


def r2(y_true, y_pred):
    sse_m1 = ((y_pred-y_true) ** 2).sum()
    sse_mb = ((y_true.mean() - y_true) ** 2).sum()
    return 1 - sse_m1 / sse_mb


#x et y sont les caractéristiques et la variable cible
def plot_learning_curve(model, X, y):
    train_errors = []
    validation_errors = []
    
    # Divise les données en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Essayez différentes tailles de l'ensemble d'entraînement
    for m in range(1, len(X_train) + 1):
        model.fit(X_train[:m], y_train[:m])  # Entraînez le modèle sur un sous-ensemble d'entraînement
    
        y_train_pred = model.predict(X_train[:m])
        train_error = metrics.mean_squared_error(y_train[:m], y_train_pred)
        train_errors.append(train_error)
    
        y_val_pred = model.predict(X_val)
        validation_error = metrics.mean_squared_error(y_val, y_val_pred)
        validation_errors.append(validation_error)
    
    plt.plot(range(1, len(X_train) + 1), train_errors, label="Erreur d'entraînement")
    plt.plot(range(1, len(X_train) + 1), validation_errors, label="Erreur de validation")
    plt.xlabel("Taille de l'ensemble d'entraînement")
    plt.ylabel("Erreur quadratique moyenne (MSE)")
    plt.legend()
    plt.title("Courbe d'apprentissage")
    plt.grid(True)
    plt.show()
    
    
def error_plot(y_true, y_pred, model):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, c='b', label='Valeurs Prédites')
    
    #y = x
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '-', label='y = x')
    
    plt.xlabel("Valeurs Réelles")
    plt.ylabel("Valeurs Prédites")
    plt.title("Comparaison des Valeurs Réelles et Prédites")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def residual_plot(y_true, y_predicted, model_name): 
    plt.plot(y_predicted, y_true -y_predicted, "*") 
    plt.plot(y_predicted, np.zeros_like(y_predicted), "-") 
    plt.legend(["Data", "Perfection"]) 
    plt.title("Residual Plot of " + model_name) 
    plt.xlabel("Predicted Value") 
    plt.ylabel("Residual") 
    plt.show()
    


def cross_validation_matrix(model, X, y, k):
    fold_size = len(X) // k
    cross_val_scores = []
    
    for i in range(k):
        # Divise les données en ensembles d'entraînement et de validation
        validation_start = i * fold_size
        validation_end = (i + 1) * fold_size
        
        X_test = X[validation_start:validation_end]
        y_test = y[validation_start:validation_end]
        
        X_train = np.concatenate([X[:validation_start], X[validation_end:]], axis=0)
        y_train = np.concatenate([y[:validation_start], y[validation_end:]], axis=0)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = r2(y_pred, y_test)
        cross_val_scores.append(score)
    df = pd.DataFrame({"Plis": range(1, len(cross_val_scores) + 1), "Score de Validation": cross_val_scores})
    # Affichez le DataFrame
    print(df)
    return cross_val_scores