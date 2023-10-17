import csv
import readingFileUtils
import dataTreatmentUtils
import downsizeUtils
import linearRegPipelineUtils
import pandas as pd
from tabulate import tabulate
import preprocessing as prep

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif  # Vous pouvez utiliser une autre fonction de score

## Import
FILE_PATH_R = "winequality-red.csv"
DATASET_R = pd.read_csv(FILE_PATH_R, sep=';')

## D'après les graphes réalisés on observes que la qualité du vin ne semble pas dépendre linéairement des colonnes acidité volatile, sucre résiduel et qualité des sulfates
## Nettoyage des données (<70% de données sur lignes et colones)
df_r = dataTreatmentUtils.removeUselessColumns(DATASET_R, 30)
dataTreatmentUtils.checkLinear(df_r, 'quality')

#nouvelle base clean
FILE_PATH_R = "dataRW_regclean.csv"
DATASET_R = pd.read_csv(FILE_PATH_R, sep=';')

## Preprocessing
# on a que des colonnes avec des nombres donc pas besoin

## Standardization
df_r = prep.standardize(df_r)

## PCA
threshold = 0.1 #les valeurs propres < 10% ne sont pas prises en compte
data_PCA_r = downsizeUtils.PCA(df_r, threshold) 

## Linear Regression Pipeline
linearRegPipelineUtils.pipeline(df_r, 'quality')

