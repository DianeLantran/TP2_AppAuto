# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 01:04:09 2023

@author: Mathilde
"""

import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as prep
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Import
FILE_PATH_R = "data/winequality-red.csv"
FILE_PATH_W = "data/winequality-white.csv"
df1 = pd.read_csv(FILE_PATH_R, sep=';')
df2 = pd.read_csv(FILE_PATH_W, sep=';')
colNames = df1.columns.tolist()


def compare2Colonnes1set(col1, col2, df):
    sns.scatterplot(data=df, x=col1, y=col2)
    plt.xlabel(col1)
    plt.ylabel(col2)
    title = 'Comparaison de ' + col1 + ' avec ' + col2
    plt.title(title)
    plt.show()


def compare2Colonnes2sets(col1, col2, df1, df2):
    sns.scatterplot(data=df1, x=col1, y=col2, color='red')
    sns.scatterplot(data=df2, x=col1, y=col2, color='blue')
    plt.xlabel(col1)
    plt.ylabel(col2)
    title = 'Comparaison de ' + col1 + ' avec ' + col2
    plt.title(title)
    plt.show()


def plotMoyParQualite():
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    plt.title('Comparaison des moyennes, classée par valeur de qualité')
    moyennes_par_qualite_R = df1.groupby('quality').mean()
    moyennes_par_qualite_W = df2.groupby('quality').mean()
    for i in range(0, len(colNames)-1):
        col1 = 'quality'
        col2 = colNames[i]
        print(col2)
        if i < 3:
            sns.scatterplot(data=moyennes_par_qualite_R,
                            x=col1, y=col2, color='red', ax=axes[0, i-1])
            sns.scatterplot(data=moyennes_par_qualite_W,
                            x=col1, y=col2, color='blue', ax=axes[0, i-1])
        else:
            if (i < 6) & (i != 3):
                sns.scatterplot(data=moyennes_par_qualite_R,
                                x=col1, y=col2, color='red', ax=axes[0, i - 2])
                sns.scatterplot(data=moyennes_par_qualite_W,
                                x=col1, y=col2, color='blue', ax=axes[0, i - 2])
            else:
                if i > 3:
                    sns.scatterplot(data=moyennes_par_qualite_R,
                                    x=col1, y=col2, color='red', ax=axes[1, i-6])
                    sns.scatterplot(data=moyennes_par_qualite_W,
                                    x=col1, y=col2, color='blue', ax=axes[1, i-6])
        plt.xlabel(col1)
        plt.ylabel(col2)
        title = col1 + ' to ' + col2
        if i < 3:
            axes[0, i - 1].set_title(title)
            print('on fait les titres <5')
            print(col2)
            print(i)
        else:
            if (i < 6) & (i != 3):
                print('on fait les titres entre')
                print(col2)
                print(i)
                axes[0, i - 2].set_title(title)
            else:
                if i > 3:
                    print('on fait les titres >5')
                    print(col2)
                    print(i)
                    axes[1, i - 6].set_title(title)
    plt.tight_layout()
    plt.show()


def plotQualiteVSColDecrois():
    scaler = MinMaxScaler()
    colonnes_a_mettre_a_l_echelle = [
        col for col in df2.columns if col != "quality"]
    df1[colonnes_a_mettre_a_l_echelle] = scaler.fit_transform(
        df1[colonnes_a_mettre_a_l_echelle])
    df1['Sum'] = df1['volatile acidity'] + \
        df1['chlorides']+df1['density'] + df1['pH']
    sns.boxplot(x='quality', y='Sum', data=df1, color='red', width=0.4)
    plt.xlabel('Qualité')
    plt.ylabel(
        "Somme de l'acidité volatile, des chlorides, de la densité et du pH")
    title = "Comparaison entre qualité et la somme des colonnes d'intérêt"
    plt.title(title)
    plt.show()


def plotQualiteVSColCrois():
    scaler = MinMaxScaler()
    colonnes_a_mettre_a_l_echelle = [
        col for col in df2.columns if col != "quality"]
    df1[colonnes_a_mettre_a_l_echelle] = scaler.fit_transform(
        df1[colonnes_a_mettre_a_l_echelle])
    df1['Sum1'] = df1['citric acid']+df1['sulphates'] + \
        df1['fixed acidity'] + df1['alcohol']
    sns.boxplot(x='quality', y='Sum1', data=df1, color='red', width=0.4)
    plt.xlabel('quality')
    plt.ylabel(
        "Somme de l'acide citrique, l'acidite fixe, le taux d'alcool et les sulphates")
    title = "Comparaison de la qualité à la somme des colonnes d'intérêt"
    plt.title(title)
    plt.show()


def plotBoitesMoustQuali():
    for i in range(len(colNames)-1):
        plt.figure(figsize=(8, 6))

        sns.boxplot(x='quality', y=colNames[i], data=df1, color='red')
        sns.boxplot(x='quality', y=colNames[i], data=df2, color='blue')

        blue_patch = plt.Line2D([0], [0], color='red', label='DATASET_R')
        red_patch = plt.Line2D([0], [0], color='blue', label='DATASET_W')
        plt.legend(handles=[blue_patch, red_patch])

        plt.title(f'Diagramme boite pour {colNames[i]}')
        plt.xlabel('Qualité')
        plt.ylabel(colNames[i])
        plt.show()


def plotQualiteVSColTot():
    scaler = MinMaxScaler()
    colonnes_a_mettre_a_l_echelle = [
        col for col in df2.columns if col != "quality"]
    df1[colonnes_a_mettre_a_l_echelle] = scaler.fit_transform(
        df1[colonnes_a_mettre_a_l_echelle])
    df1['Sum1'] = df1['citric acid']+df1['sulphates'] + \
        df1['fixed acidity'] + df1['alcohol']
    df1['Sum2'] = df1['volatile acidity'] + \
        df1['chlorides']+df1['density'] + df1['total sulfur dioxide']
    df1['Tot'] = df1['Sum1'] - df1['Sum2']
    plt.figure(figsize=(10, 6))

    sns.boxplot(x='quality', y='Tot', data=df1, color='red', width=0.4)
    plt.xlabel('quality')
    plt.ylabel('Sum')
    title = "Comparaison de la qualité par rapport à la somme de \nl'acide citrique, des sulphates, de l'acidité fixe, du taux d alcool \nen soustrayant l'acidité volatile, les chlorides, la densité et le total de dioxyde de soufre"
    plt.title(title)
    plt.show()


def analyzeReg(X, y, coefs, intercept):
    df = pd.concat([X, y], axis=1)
    df['Y'] = intercept
    for i in range(len(coefs['Variable'])):
        df['Y'] = df['Y'] + coefs['Coefficient'][i]*df[coefs['Variable'][i]]
    sns.scatterplot(data=df, x='quality', y='Y')
    plt.xlabel('Qualité réelle')
    plt.ylabel('Qualité prédite')
    plt.show()

    plt.figure(figsize=(10, 6))

    # Tracer un histogramme des différences entre les valeurs réelles et prédites
    sns.histplot(df['quality'] - df['Y'], bins=20, kde=True)

    plt.xlabel('Différence entre Qualité Réelle et Prédite')
    plt.ylabel('Fréquence')
    plt.title('Distribution des Erreurs de Prédiction')

    plt.show()
