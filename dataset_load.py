# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:20:52 2019

@author: asuazo
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class data_read():

    def __init__(self, path_train, path_test, path_info):
        self.info_dataset =pd.read_csv(path_info, encoding = "ISO-8859-1") 
        #indeces
        self.columns_variables_catagoricas = list()
        self.columns_variables_int = list()
        self.columns_variables_float = list()
        self.columns_variables_tiempo = list()
        self.dataset_train = pd.read_csv(path_train, encoding = "ISO-8859-1") #dtype se puede configurar para selecionar por columna el typo 
        self.dataset_test = pd.read_csv(path_test, encoding = "ISO-8859-1")
        

        self.dataset_train_duplicados = self.duplicados_ids_based(self.dataset_train)
        self.dataset_test_duplicados = self.duplicados_ids_based(self.dataset_test)
        self.dataset_train_nan_ids = self.duplicados_ids_based(self.dataset_train)
        self.dataset_test_nan_ids =  self.duplicados_ids_based(self.dataset_train)
        
        #estadisticas, representar cuantos valores hay por columnas
        #valores interdidos por columna, conteo
        self.train_column_NAN = dict()
        self.train_column_INF = dict()
        self.train_column_NINF = dict()
        #informacion
        self.valeurs_interdis = [np.nan, np.inf, -np.inf]
        self.valeurs_interdis_conteo = [self.train_column_NAN, 
                                 self.train_column_INF,
                                 self.train_column_NINF]

    """
    con estza funcion espero reover datos repetidos
    """
    def duplicados_ids_based(self, dataset):
        ids = dataset["Id"]
        return dataset[ids.isin(ids[ids.duplicated()])].sort_values("Id")

    def NAN_ids_based(self, dataset, estadisticas):
        for valeur_interdi, dataset_interdidos_conteo in zip(self.valeurs_interdis, self.valeurs_interdis_conteo):
            dtata_frame =  pd.DataFrame()
            for columns in dataset.columns:
                dataset_nan_column = dataset[dataset.loc[:, columns].isin(valeur_interdi)]
                dataset_interdidos_conteo[columns] = len(dataset_nan_column.index)

    def optener_tipografia_datos(self, dataset):
        for column in dataset.columns:
            if column=="Id" or column=="SalePrice":
                continue
            clase = self.info_dataset.loc[self.info_dataset.variable==(column), "clase"].item()
            if self.info_dataset.loc[self.info_dataset.variable==column, "tipo"].item()=="int":
               self.columns_variables_int.append(column)
            if self.info_dataset.loc[self.info_dataset.variable==column, "tipo"].item()=="categoria":
               self.columns_variables_catagoricas.append(column)
            if self.info_dataset.loc[self.info_dataset.variable==column, "tipo"].item()=="float":
               self.columns_variables_float.append(column)