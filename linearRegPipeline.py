"""
Created on 17 Oct 2023

@author: diane
"""
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import dataViz as dv
import evaluationUtils as ev
import dataTreatmentUtils
import metrics_comparison as metricsCmp
from sklearn.base import BaseEstimator, TransformerMixin
import linearRegPipelineUtils as pipeUtils

def executePipelines(X, y):
    # Prepare la pipeline d'elements
    pipelineData = []
    metrics_array = []
    for name in X.columns:
        pipelineData.append(('Regression linéaire simple pour la caractéristique ' + name 
                             , LinearRegression(), name))
    pipelineData.append(('Regression linéaire multiple', 
                         LinearRegression(), "all"))
    pipelineData.append(('Regression Ridge', 
                         RidgeCV(alphas=[0.1, 1.0, 10.0]), "all"))
    
    # crée une pipeline pour chaque regression qu'on souhaite appliquer puis l execute
    for model_name, model, col in pipelineData:
        match col:
            case "all":
                if (model_name == 'Regression linéaire multiple'):
                    pipeline = Pipeline([
                        ('pca', PCA(n_components = 0.7, svd_solver = 'full')),
                        ('uncorrelatedColRemoval', 
                         DeleteUncorrelatedColsTransformer()),
                        ('regressor', model)  # Linear regression model
                    ])
                    pipeUtils.checkColumnsLinearity(X, y)
                    
                else:
                    pipeline = Pipeline([
                        ('regressor', model)  #Modele de regression linéaire
                    ])
                    
                metrics = executeSinglePipeline(model_name, pipeline, X, y)
        
            case _:
                selected_X = X[[col]]
                pipeline = Pipeline([
                    ('regressor', model)  #Modele de regression linéaire
                ])
                metrics = executeSinglePipeline(model_name, pipeline, 
                                                selected_X, y)
        metrics_array.append(metrics)
    metricsCmp.compareMetrics(metrics_array, pipelineData)
    

def executeSinglePipeline(model_name, pipeline, X, y):
    # sépare le dataset en training et testing set
    X_train, X_test, y_train, y_test = ev.splitTrainTest(X, y, test_size=0.2, 
                                                      random_state=42)
    
    # retrouve model
    model = pipeline['regressor']
    # applique le modele au training 
    if 'uncorrelatedColRemoval' in pipeline.named_steps:
        pipeline.fit(X_train, y_train, uncorrelatedColRemoval__y_test = y_test)
    else:
        pipeline.fit(X_train, y_train)

    # fait des predictions sur le testing set
    y_pred = pipeline.predict(X_test)
    
    # récupère les metriques, graph et autres infos sur les résultats
    metrics, coefficients = pipeUtils.displayResults(model, X_test, y_test, y_pred, 
                                  model_name, X, y)
    
    dv.analyzeReg(X, y, coefficients, model.intercept_)
    return metrics

class DeleteUncorrelatedColsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_transformed = False
        self.cols_to_keep = None
        
    def fit(self, X, y, y_test):
        self.y = y
        self.y_test = y_test
        return self

    def transform(self, X):
        if not self.is_transformed:
            X_df = pd.DataFrame(data=X)
            X_df = dataTreatmentUtils.removeNotColinearCol(X_df, self.y)
            self.cols_to_keep = X_df.columns
            self.is_transformed = True
        else:
            X_df = pd.DataFrame(data=X)
            X_df = X_df[self.cols_to_keep]
        return X_df.to_numpy()
