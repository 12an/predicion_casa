# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:58:02 2019

@author: asuazo
"""

# Multiple Linear Regression

# Importing the libraries

import pandas as pd

# Importing the dataset
dataset = pd.read_csv('informacion/organizacion.csv',)


criterios = pd.read_csv('informacion/confirmacion.csv')

criterios_compia = criterios["Id"].tolist()



for j in criterios["Id"]:
    
    for i in dataset["nombre"]:
        
        if i == j:
        
            criterios_compia.remove(j)

        
    
