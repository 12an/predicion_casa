# -*- coding: utf-8 -*-


import pandas as pd

# Importing the dataset
dataset = pd.read_csv("informacion/organizacion.csv")


criterios = pd.read_csv("informacion/confirmacion.csv")

criterios_compia = criterios["Id"].tolist()



for j in criterios["Id"]:
    
    for i in dataset["nombre"]:
        
        if i == j:
        
            criterios_compia.remove(j)
