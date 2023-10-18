import dataTreatmentUtils
import linearRegPipelineUtils
import pandas as pd
import preprocessing as prep

# Import
FILE_PATH_R = "data\winequality-red.csv"
DATASET_R = pd.read_csv(FILE_PATH_R, sep=';')

# D'après les graphes réalisés on observes que la qualité du vin ne semble pas dépendre linéairement des colonnes acidité volatile, sucre résiduel et qualité des sulfates
# Nettoyage des données (<70% de données sur lignes et colones)
df_r = dataTreatmentUtils.removeUselessColumns(DATASET_R, 30)
# verifie si les colonnes sont adaptées à la regression lineaire
dataTreatmentUtils.checkLinear(df_r, 'quality')

# nouvelle base clean
FILE_PATH_R = "dataRW_regclean.csv"
DATASET_R = pd.read_csv(FILE_PATH_R, sep=';')

# Preprocessing
# on a que des colonnes avec des nombres et sans données manquantes
# donc pas besoin

# Prepare data
X = df_r.drop(columns=['quality'])
X = prep.standardize(X)
y = df_r['quality']


# Linear Regression Pipeline
linearRegPipelineUtils.executePipelines(X, y)
