# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:20:52 2019

@author: asuazo
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels as st
from sklearn.preprocessing import StandardScaler


class data_read():
    
    
    
    def __init__(self):
        #indeces
        self.idex_variables_catagoricas = []
        self.idex_variables_int = []
        self.idex_variables_float = []
    
        # Importing the dataset
        self.organizacion = pd.read_csv('info/organizacion_bon_representation.csv',encoding = "ISO-8859-1")

        self.tipo  = self.organizacion.iloc[0:80 , 3]

        self.variable = self.organizacion.iloc[0:80 , 0] 

        self.dataset_train = pd.read_csv('datos/train.csv',encoding = "ISO-8859-1") #dtype se puede configurar para selecionar por columna el typo 
        self.dataset_test = pd.read_csv('datos/test.csv',encoding = "ISO-8859-1")

        self.X_train = self.dataset_train.iloc[:,1:80]
        self.Y_train = self.dataset_train.iloc[:, -1]
 
        self.X_test = self.dataset_test.iloc[:,1:80]
        self.Y_test = self.dataset_test.iloc[:, -1]       


    """
    con estza funcion espero reyenal los valores perdidos sin que estos afecten mucho 
    el equilibrio general
    """

        
        
 
        
    def definiendo_tipo_dato(self):
        # si no se deja un default el programa tomara el ultimo tipo aplicado para el nuevo caso y dara error
        
        #indexando los lebol de las colunnas para selecional esa columna y configurar su tipo de dato
        i = 0
        for x_config_label in self.variable[0:79]:


    
            if self.tipo[i] == "int":
                self.idex_variables_int.append(i)
                self.X[x_config_label].astype(np.int)
                

                
    
            elif self.tipo[i] == "float":
                self.idex_variables_float.append(i)
                self.X[x_config_label].astype(np.float)
                

            else:
                
                self.idex_variables_catagoricas.append(i)
                self.X[x_config_label].astype(str)
                
            
            i += 1
            
        #normalizando
        #x
        sc_X_normalise = StandardScaler()    
        self.X_train = sc_X_normalise.fit_transform(self.X_train)
        self.X_test = sc_X_normalise.transform(self.X_test)
        #y
        sc_y_normalise = StandardScaler()
        self.Y_train = sc_y_normalise.fit_transform(self.Y_train)
        self.Y_test = sc_y_normalise.fit_transform(self.Y_test)
        
    
    """
    esta funcion solo sirve para los valores float e int porque
    en categoria ya nan es una categoria pero se deve remplasar
    por otra letra
    """
    def nan_values_delete(self):
        
        
        
        for j in range(0,len(self.variable)):
            
            
            
            if(self.idex_variables_int.__contains__(j)):
                
                normalized = self.array_normalizado(self.X.iloc[:,j],j)
                
                for i in self.X.iloc[:,j]:
                    
                    if np.isnan(i):
                        
                        


            if(self.idex_variables_catagoricas.__contains__(j)):
                
                for i in self.X.iloc[:,j]:
                    
                    if np.isnan(i):
                        
                        self.X[i,j] = "DEFAUT"
   

        
    def catagorica_dummy_variables(self):
        
        labelencoder = LabelEncoder()
        
        for j in self.idex_variables_catagoricas:
            
            X[:, j] = labelencoder.fit_transform(X[:, j])
        
        
        
        
b = data_read()
b.definiendo_tipo_dato()
asp = b.X