# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:42:39 2023

@author: solene
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold

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
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title("Courbe d'apprentissage")
    plt.xlabel("Taille de l'ensemble d'entraînement")
    plt.ylabel("Erreur quadratique moyenne (MSE)")
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entraînement")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation")
    plt.legend(loc="best")
    plt.show()
    
    
def error_plot(y_true, y_pred, model):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, c='b', label='Valeurs Prédites')
    
    #y = x
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='y = x')
    
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
    
    
def confusion_matrix(y_true, y_pred):
    confusion_matrix = [[0, 0], [0, 0]]
    for i in range(len(y_true)):
        true_label = y_true[i]
        predicted_label = y_pred[i]
        confusion_matrix[true_label][predicted_label] += 1
    
    pp = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    pn = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
    precision = (pp + pn) / 2
    
    rp = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
    rn = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
    recall = (rp + rn) / 2
    
    f_score = 2 * (precision * recall) / (precision + recall)
    
    return confusion_matrix, precision, recall, f_score


def cross_validation_matrix(model, X, y, k):
    # Créez une instance de KFold avec k plis
    kf = KFold(n_splits=k)
    
    # Initialisation de la matrice de validation croisée
    cross_val_matrix = []
    
    # Effectuez la validation croisée
    for train_indices, test_indices in kf.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
    
        # Entraînez votre modèle sur X_train et évaluez-le sur X_test
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
    
        cross_val_matrix.append(score)
        
    return cross_val_matrix