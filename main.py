import csv
import readingFileUtils
import dataTreatmentUtils
import downsizeUtils
import pandas as pd
from tabulate import tabulate
import preprocessing as prep

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif  # Vous pouvez utiliser une autre fonction de score

## Import
FILE_PATH_R = "/winequality-red.csv"
FILE_PATH_W = "winequality-white.csv"
DATASET_R = pd.read_csv(FILE_PATH_R, sep=';')
DATASET_W = pd.read_csv(FILE_PATH_W, sep=';')


## Nettoyage des données (<70% de données sur lignes et colones)
df_r = dataTreatmentUtils.removeUselessColumns(DATASET_R, 30)
df_w = dataTreatmentUtils.removeUselessColumns(DATASET_W, 30)


## Preprocessing
# on a que des colonnes avec des nombres donc pas besoin

## Standardization
df_r = prep.standardize(df_r)
df_w = prep.standardize(df_w)

## PCA
threshold = 0.1 #les valeurs propres < 10% ne sont pas prises en compte
data_PCA_r = downsizeUtils.PCA(df_r, threshold) 
data_PCA_w = downsizeUtils.PCA(df_w, threshold) 

print("red : ")
print(data_PCA_r)

## Linear Regression Pipeline avec hypothèse sur les graphes de Mathilde