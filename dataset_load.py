# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:20:52 2019

@author: asuazo
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class data_read():
    
    
    
    def __init__(self):
    
        # Importing the dataset
        self.organizacion = pd.read_csv('informacion/organizacion_bon_representation.csv')

        self.tipo  = self.organizacion.iloc[0:80 , 3]
        self.variable = self.organizacion.iloc[0:80 , 0] 

        self.dataset = pd.read_csv('datos/train.csv') #dtype se puede configurar para selecionar por columna el typo 


        self.X = self.dataset.iloc[:,1:80]
        self.Y = self.dataset.iloc[:, -1]

        #indexando los lebol de las colunnas para selecional esa columna y configurar su tipo de dato
        i = 0
        for x_config_label in self.variable[0:79]:

    
            self.X[x_config_label].astype(self.definiendo_tipo_dato(i))

            i += 1
        
     
    def definiendo_tipo_dato(self,tipo_index):
        # si no se deja un default el programa tomara el ultimo tipo aplicado para el nuevo caso y dara error

        
    
        if self.tipo[tipo_index] == "int":
            return np.int
    
        elif self.tipo[tipo_index] == "float":
            return np.float

        else:
            return str      
        
        
        
b = data_read()

asp = b.X