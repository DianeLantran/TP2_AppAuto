# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:26:08 2023

@author: basil
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Normalise les colonnes du dataframe
def standardize(df):
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df)
    df_standardized = pd.DataFrame(df_standardized, columns=df.columns)
    return df_standardized