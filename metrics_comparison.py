import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Compare les métriques des différents algorithmes contenues dans metrics_array
def compareMetrics(metrics_array, pipelineData):
    # Crée une palette de couleurs distinctes
    colors = sns.color_palette("Set1", n_colors=10)
    
    metrics = ['MAE', 'MSE', 'RMSE', 'R²']
    algorithms  = [item[0] for item in pipelineData]
    bar_width = 0.2
    x = np.arange(len(algorithms))
    
    # Pour chaque métrique, réalise un bar plot. Chaque barre correspond à 
    # un algorithme
    for j, metric in enumerate(metrics):
        plt.figure(figsize=(10, 10))
        for i, algorithm in enumerate(algorithms):
            plt.bar(i, metrics_array[i][j], width=bar_width, label=algorithm, color=colors[i])
        plt.xlabel('Algorithmes')
        plt.ylabel(f'{metric}')
        plt.title(f'Comparaison des valeurs pour la métrique {metric}')
        plt.xticks(x)
        plt.legend(loc='best')
        plt.show()