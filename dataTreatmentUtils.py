import pandas as pd
import statsmodels.api as sm #pip install statmodels


#Traitement des colonnes
def getMissingDataPercentagePerColumn(dataset):
    # Calculer le nombre de données manquantes par colonne
    missing_data = dataset.isnull().sum()

    # Calculer le pourcentage de données manquantes par colonne
    percentage = (missing_data / len(dataset)) * 100

    # Créer un DataFrame pour afficher les résultats
    missing_data_tab = pd.DataFrame({
        'Colonne': dataset.columns,
        'Données manquantes': missing_data,
        'Pourcentage de données manquantes': percentage
    })
    return missing_data_tab
    
def removeUselessColumns(dataset, max_percentage):
    missing_data_tab = getMissingDataPercentagePerColumn(dataset)
    #Suppression des colonnes avec trop de valeurs nulles
    for i in range(len(missing_data_tab)):
        if missing_data_tab.iloc[i, 2] >= 30:
            dataset = dataset.drop(missing_data_tab.iloc[i, 0], axis=1)
    return dataset


# evalue la realtion de linearité entre chaque colomne et la colonne "qualité"
def checkLinearold(dataset, column):
    cols = dataset.drop(columns=[column])
    for feature in cols:
        # ajoute un terme lineaire constant pour creer un modele de regression lineaire
        X = sm.add_constant(dataset[feature])
        # ajuste le modele lineaire
        model = sm.OLS(dataset[column], X)
        results = model.fit()
        r_squared = results.rsquared

        thresh = 0.005 #seuil en dessous duquel la colonne n'est pas utilisée par le modèle
        if r_squared < thresh:
            dataset.drop(columns=[feature], inplace=True)
            print(feature, " is deleted")
    # Save the modified DataFrame back to a CSV file
    dataset.to_csv('dataRW_regclean.csv', index=False) 
    
    
# evalue la realtion de linearité entre chaque colomne et la colonne "qualité"
def removeNotColinearCol(features_df, y, thresh = 0.005):
    for feature in features_df:
        # ajoute un terme lineaire constant pour creer un modele de regression lineaire
        X = sm.add_constant(features_df[feature])
        # ajuste le modele lineaire
        model = sm.OLS(y, X)
        results = model.fit()
        r_squared = results.rsquared

        if r_squared < thresh:
            features_df.drop(columns=[feature], inplace=True)

#Traitement des lignes
def getMissingDataPercentageForOneRow(row):
    missing_data = row.isnull().sum()
    percentage = (missing_data / len(row)) * 100
    return percentage

def removeUselessRows(dataset, max_percentage):
    dataset = dataset[dataset.apply(getMissingDataPercentageForOneRow, axis=1) <= max_percentage]
    return dataset   

