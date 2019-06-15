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

        self.X_train = self.dataset_train.iloc[:,1:80]
        self.Y_train = self.dataset_train.iloc[:, -1]
 
        self.X_test = self.dataset_test.iloc[:,1:80]
        self.Y_test = self.Y_train
    


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
                self.X_train[x_config_label].astype(np.int)
           #     self.X_test[x_config_label].astype(np.int)

                

                
    
            if self.tipo[i] == "float":
                self.idex_variables_float.append(i)
                self.key_variables_float.append(x_config_label)
                self.X_train[x_config_label].astype(np.float)
            #    self.X_test[x_config_label].astype(np.float)
               

            if self.tipo[i] == "categoria":
                
                self.idex_variables_catagoricas.append(i)
                self.key_variables_catagoricas.append(x_config_label)
                self.X_train[x_config_label].astype(str)
            #    self.X_test[x_config_label].astype(str)
               
            
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
        sc_y_normalise = StandardScaler()
        y = np.asarray(self.Y_train)
        Y = y.reshape(10,-1)
        sc_y_normalise.fit(Y)
        self.Y_train = sc_y_normalise.transform(Y)
        
    
    """
    esta funcion solo sirve para los valores float e int porque
    en categoria ya nan es una categoria pero se deve remplasar
    por otra letra
    """
    def nan_delete(self):
        promedios = []
        
        
        for j in range(0,len(self.variable)):
            
            
            
            if(self.idex_variables_int.__contains__(j)):
                
                promedios, n = self.promedio_ar_qu_me(self.X.iloc[:,j])
                random_value = self.random_(self.X.iloc[:,j], n)
                
                
                for i in self.X.iloc[:,j]:
                    
                    if np.isnan(i):
                        
                        self.X[i,j] = (self.random_(promedios , 3) + random_value) / 2
                        
                    
            if(self.idex_variables_catagoricas.__contains__(j)):
                
                for i in self.X.iloc[:,j]:
                    
                    if np.isnan(i):
                        
                        self.X[i,j] = "DEFAUT"
   

    def promedio_ar_qu_me(self, array_):
       array_, n = self.enleve_nan_values(array_)
       
       armonico = n / sum(np.reciprocal(array_))
       quadratico = np.sqrt(n / sum(np.power(array_)))
       media = sum(array_) / n
       
       return armonico, quadratico, media, n

    def random_(self, array_ ,n):
        
        return array_[random.randint(1,n)]
        
    def enleve_nan_values(self,array_):
        
        k = array_
        n = 0
        for i in array_:
            n += 1
            if np.isnan(i):
                n -= 1
                k.drop(i)
        return k,n
    
    #def catagorica_dummy_variables(self):
        
    #    labelencoder = LabelEncoder()
        
   #     for j in self.idex_variables_catagoricas:
            
  #          X[:, j] = labelencoder.fit_transform(X[:, j])
        
        
        
        
b = data_read()

y = b.Y_train
x = b.X_train
b.definiendo_dato(1)

organiza = b.organizacion
organiza1 = b.variable
organiza2 = b.tipo
b.normalizar_datos()
ya = b.Y_train
xa = b.X_train

