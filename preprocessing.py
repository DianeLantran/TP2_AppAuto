# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:26:08 2023

@author: basil
"""
import pandas as pd
import re
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def colToOrdinal(df, colnames):
    # Prepare encoder object
    encoder = OrdinalEncoder(encoded_missing_value=-1)
    
    # Fit and transform the selected column
    df[colnames] = encoder.fit_transform(df[colnames])
    return df

def standardize(df):
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df)
    df_standardized = pd.DataFrame(df_standardized, columns=df.columns)
    return df_standardized