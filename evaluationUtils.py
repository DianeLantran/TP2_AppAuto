# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:42:39 2023

@author: solene
"""

import numpy as np
import pandas as pd
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
def plot_learning_curve(axis, model, X, y):
    train_errors = []
    validation_errors = []
    
    # Divise les données en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Essaye différentes tailles de l'ensemble d'entraînement
    for m in range(1, len(X_train) + 1):
        model.fit(X_train[:m], y_train[:m])  # Entraîne le modèle sur un sous-ensemble d'entraînement
    
        y_train_pred = model.predict(X_train[:m])
        train_error = metrics.mean_squared_error(y_train[:m], y_train_pred)
        train_errors.append(train_error)
    
        y_val_pred = model.predict(X_val)
        validation_error = metrics.mean_squared_error(y_val, y_val_pred)
        validation_errors.append(validation_error)
    
    axis.plot(range(1, len(X_train) + 1), train_errors, label="Training error")
    axis.plot(range(1, len(X_train) + 1), validation_errors, 
              linestyle='dashed', label="Erreur de validation")
    axis.set_xlabel("Taille du data set d entrainement")
    axis.set_ylabel("MSE")
    axis.legend()
    axis.set_title("Courbe apprentissage")
    axis.grid(True)
    return axis
    

def error_plot(axis, y_true, y_pred, model):
    axis.scatter(y_true, y_pred, c='b', label='Valeurs prédites')

    # Plot y = x    
    axis.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '-', label='y = x')
    
    axis.set_xlabel("Valeurs attendues")
    axis.set_ylabel("Valeurs prédites")
    axis.set_title("Comparaison valeurs attendues vs prédites")
    axis.legend()
    axis.grid(True)
    return axis
    
    
def residual_plot(axis, y_true, y_predicted): 
    axis.plot(y_predicted, y_true -y_predicted, "*") 
    axis.plot(y_predicted, np.zeros_like(y_predicted), "-") 
    axis.legend(["Données", "Perfection"]) 
    axis.set_title("Plot résiduel") 
    axis.set_xlabel("Valeurs prédites") 
    axis.set_ylabel("Résiduelles") 
    return axis
    


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
    # Affiche le DataFrame
    print(df)
    return cross_val_scores