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
import random

class data_read():
    
    
    
    def __init__(self):
        #indeces
        self.idex_variables_catagoricas = []
        self.idex_variables_int = []
        self.idex_variables_float = []
        
        #key strin 
        #para indexar y dejar en la misma posision
        self.key_variables_catagoricas = []
        self.key_variables_int = []
        self.key_variables_float = [] 
    
        # Importing the dataset
        self.organizacion = pd.read_csv('info/organizacion_bon_representation.csv',encoding = "ISO-8859-1")

        self.tipo  = self.organizacion.iloc[0:79 , 3]

        self.variable = self.organizacion.iloc[0:79 , 0]

        self.dataset_train = pd.read_csv('datos/train.csv',encoding = "ISO-8859-1") #dtype se puede configurar para selecionar por columna el typo 
        self.dataset_test = pd.read_csv('datos/test.csv',encoding = "ISO-8859-1")

        self.X_train = self.delet_inconnu_value(self.dataset_train.iloc[:,1:80])
        self.Y_train = self.delet_inconnu_value(self.dataset_train.iloc[:, -1])
 
        self.X_test = self.delet_inconnu_value(self.dataset_test.iloc[:,1:80])
        self.Y_test = self.X_test[: 1]
        
        self.M_train = len(self.X_train.iloc[:1])
        self.M_test = len(self.X_test.iloc[:1])
        
        """
        eliminando los valores NAN
        """
        self.X_train = self.nan_delete(self.X_train, self.M_train)
        self.X_test = self.nan_delete(self.X_test, self.M_test)
        self.Y_train = self.nan_delete(self.Y_train, self.M_train)
        
        #informacion
        self.valeurs_interdis = [np.nan, np.inf, -np.inf]

    """
    con estza funcion espero reyenal los valores perdidos sin que estos afecten mucho 
    el equilibrio general
    """


   
        
 
        
    def definiendo_dato(self,y_tipe):
        
        if y_tipe == np.int:
            self.Y_train.astype(np.int)
            self.Y_test.astype(np.int)
            
        elif y_tipe == np.float:
            self.Y_train.astype(np.float)
            self.Y_test.astype(np.float)                
        else:
            self.Y_train.astype(str)
            self.Y_test.astype(str)        
        
        # si no se deja un default el programa tomara el ultimo tipo aplicado para el nuevo caso y dara error
        
        #indexando los lebol de las colunnas para selecional esa columna y configurar su tipo de dato
        i = 0
        for x_config_label in self.variable[0:79]:


    
            if self.tipo[i] == "int":
                self.idex_variables_int.append(i)

                self.key_variables_int.append(x_config_label)

                
                try:
                    
                    self.X_train[x_config_label].astype(np.int)
                    
                except ValueError:
                    
                    if self.valeurs_interdis in self.X_train[x_config_label].isna():
                    
                        print("encontramos un valor inapropiado")
                    
                    
                try:
                    
                    self.X_test[x_config_label].astype(np.int)
                
                except ValueError:
                    
                    if self.valeurs_interdis in self.X_test[x_config_label].isna():
                        

                    
                        print("encontramos un valor inapropiado")
                

                
    
            if self.tipo[i] == "float":
                self.idex_variables_float.append(i)
                self.key_variables_float.append(x_config_label)
                self.X_train[x_config_label].astype(np.float)
                self.X_test[x_config_label].astype(np.float)
               

            if self.tipo[i] == "categoria":
                
                self.idex_variables_catagoricas.append(i)
                self.key_variables_catagoricas.append(x_config_label)
                self.X_train[x_config_label].astype(str)
                self.X_test[x_config_label].astype(str)
               
            
            i += 1
            


        
        
    def normalizar_datos(self):
        #normalizando
        #x


        sc_X_normalise = StandardScaler()    
        #self.X_train[( self.key_variables_float + self.key_variables_int )] = 
        sc_X_normalise.fit(self.X_train.loc[: , ( self.key_variables_float + self.key_variables_int )])
        self.X_train.loc[: , ( self.key_variables_float + self.key_variables_int )] = sc_X_normalise.transform(self.X_train.loc[:, ( self.key_variables_float + self.key_variables_int )])
        
#        sc_X_normalise.fit(self.X_test.loc[: , ( self.key_variables_float + self.key_variables_int )])
#        self.X_test.loc[( self.key_variables_float + self.key_variables_int )] = sc_X_normalise.transform(self.X_test.loc[ ( self.key_variables_float + self.key_variables_int )])
        
        
        #y
        #sc_y_normalise = StandardScaler()
        #y = np.asarray(self.Y_train)
        
        #Y = y.reshape(1,-1) #ojo con esta funcion solo le da el formato a algunas pocas 20 filas
        #sc_y_normalise.fit(Y)
        #self.Y_train = sc_y_normalise.transform(Y)
        self.Y_train = self.standarscaler_1d(self.Y_train)
        
        
    def standarscaler_1d(self, _array):
        average = np.average(_array)
        standar_desviation = np.std(_array)
        
        #normalizado valor
        
        return (_array - average) / standar_desviation
        
        
        
    """
    esta funcion solo sirve para los valores float e int porque
    en categoria ya nan es una categoria pero se deve remplasar
    por otra letra
    """
    def nan_delete(self,_array,len_):

        i = 0
        for x_config_label in self.variable[0:79]:


    
            if self.tipo[i] == "int":     
                

                mean = _array[x_config_label].mean()
                
                for n in range(0, len_):
                    
                    if _array.loc[n, x_config_label] == np.nan:
                        
                        _array.loc[n, x_config_label] = mean
                        
                    
            if self.tipo[i] == "categoria":
                count = 0
                nan_true_array = _array.loc[:,x_config_label].isna()
                
                index_nan_true_array = nan_true_array.index[nan_true_array].tolist()
                print(index_nan_true_array)

                        
                    
                
            i += 1 
            
        return _array

    def nan_delete_1d(self,_array,len_):

        mean = _array.mean()
            
        for i in range(0, len_):
            
            if _array[i] == np.nan:
                
               _array[i] = mean
                
        return _array       
  
    
    def delet_inconnu_value(self,array_):

        return array_.replace([np.inf, -np.inf], np.nan)
        
        
       




        

    

        
        
        
        
b = data_read()

y = b.Y_train
xt = b.X_test
xd = b.X_train
b.definiendo_dato(1)

organiza = b.organizacion
organiza1 = b.variable
organiza2 = b.tipo
b.normalizar_datos()
ya = b.Y_train
xa = b.X_train

