import dataTreatmentUtils
import linearRegPipeline as pipeline
import pandas as pd
import preprocessing as prep

# D'après les graphes réalisés on observes que :
# - la qualité du vin rouge ne semble pas dépendre linéairement des colonnes 
#   acidité volatile, sucre résiduel et qualité des sulfates
# - la qualité du vin blanc ne semble pas dépendre linérairement d'une 
#   combinaison linéaire des autres colonnes de la base
# On ne travaille que sur le vin rouge

# Importation de la base de donnée
FILE_PATH_R = "data\winequality-red.csv"
DATASET_R = pd.read_csv(FILE_PATH_R, sep=';')

target = "quality"

# Nettoyage des données (<70% de données sur lignes et colones)
df_r = dataTreatmentUtils.removeUselessColumns(DATASET_R, 30)

# Prétraitement
# On a uniquement des colonnes avec des données quantifiables, 
# sans données manquantes

# Séparation de la base : X les caractéristiques et y la cible (colonne qualité)
X = df_r.drop(columns=[target])
y = df_r[target]

# Supprime les caractéristiques qui n'ont pas une dépendance linéaire assez 
# forte avec la qualité
X = dataTreatmentUtils.removeNotColinearCol(X, y)

# Standardise les données
X = prep.standardize(X)

# Lance la pipeline de régression linéaire
pipeline.executePipelines(X, y)
