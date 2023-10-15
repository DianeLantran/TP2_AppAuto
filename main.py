import csv
import readingFileUtils
import dataTreatmentUtils
import mathsUtils
import pandas as pd
from tabulate import tabulate
import preprocessing as prep

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif  # Vous pouvez utiliser une autre fonction de score

FILE_PATH_R = "winequality-red.csv"
FILE_PATH_W = "winequality-white.csv"
DATASET_R = pd.read_csv(FILE_PATH_R, sep=';')
DATASET_W = pd.read_csv(FILE_PATH_W, sep=';')


## Nettoyage des données (<70% de données sur lignes et colones)
df_r = dataTreatmentUtils.removeUselessColumns(DATASET_R, 30)
df_w = dataTreatmentUtils.removeUselessColumns(DATASET_W, 30)
#aucune donnée manquante


## Preprocessing
# on a que des colonnes avec des nombres donc pas besoin

## Standardization
df_r = prep.standardize(df_r)
df_w = prep.standardize(df_w)

## PCA
threshold = 0.1 #les valeurs propres < 10% ne sont pas prises en compte
data_PCA_r = mathsUtils.PCA(df_r, threshold) 
data_PCA_w = mathsUtils.PCA(df_w, threshold) 

print("red : ")
print(data_PCA_r)

## Comparaison avec SelectKBest
# k_best_r = SelectKBest(score_func=f_classif, k=len(df_r.columns))  # k = nombre de caractéristiques souhaitées
# reducData_SB_r = k_best_r.fit_transform(df_r, None) #étiquettes de classe cibles = None -> non supervisé
# print(reducData_SB_r)

# k_best_w = SelectKBest(score_func=f_classif, k=len(df_w.columns))  # k = nombre de caractéristiques souhaitées
# reducData_SB_w = k_best_w.fit_transform(df_w, None) #étiquettes de classe cibles = None -> non supervisé
# print(reducData_SB_w)