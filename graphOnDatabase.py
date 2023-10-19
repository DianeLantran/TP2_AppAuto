# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 01:04:09 2023

@author: Mathilde
"""

import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as prep
import numpy as np
import dataViz as dv
import seaborn as sns


# Import
FILE_PATH_R = "data/winequality-red.csv"
FILE_PATH_W = "data/winequality-white.csv"
DATASET_R = pd.read_csv(FILE_PATH_R, sep=';')
DATASET_W = pd.read_csv(FILE_PATH_W, sep=';')

description = DATASET_R.describe()
description.to_markdown('description.md')

# dv.plotMoyParQualite()
dv.plotQualiteVSColDecrois()
dv.plotQualiteVSColCrois()
dv.plotQualiteVSColTot()
# dv.plotBoitesMoustQuali()
