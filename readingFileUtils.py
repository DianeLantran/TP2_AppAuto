import csv
import pandas as pd

def getVariable(i, j, file_path):
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)

        # Saute ) la ligne désirée (i-1 fois)
        for _ in range(i - 1):
            next(csv_reader)

        # Lis la ligne actuelle
        selected_row = next(csv_reader, None)

        if selected_row is not None:
            # Vérifie si la ligne a assez de colonnes
            if j <= len(selected_row):
                # retrouve la j eme colonne (indice j-1)
                selected_cell = selected_row[j - 1]
                print(f"Valeur de la ligne {i}, colonne {j}: {selected_cell}")
                return (selected_cell)
            else:
                print(f"Ligne {i} n'a pas assez de {j} colonnes")
        else:
            print(f"Ligne {i} n exite pas dans le fichier CSV.")


