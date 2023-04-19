# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:20:52 2019

@author: asuazo
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from dataset_load import data_read


class PreProcesing(data_read):

    def __init__(self, **args):
        data_read.__init__(self, **args)
        self.encoded = OrdinalEncoder()

    def encoder_categorias(self, dataset):
        dataset = dataset.copy(deep=True)
        self.encoded.fit(dataset.loc[:,  self.columns_variables_catagoricas ])
        encoded_values = self.encoded.set_params(encoded_missing_value=-1).fit_transform(dataset.loc[:,  self.columns_variables_catagoricas])
        
        dataset.loc[:,  self.columns_variables_catagoricas] = encoded_values
        return dataset

    '''
    removidos todos los datos interdidos
    '''
    def train_test_base(self, dataset):
        dataset = dataset.copy(deep=True)
        new_dataset = dataset.drop_duplicates(subset=['Id'])
        new_dataset = new_dataset[~new_dataset.isin(self.valeurs_interdis).any(1)]
        return new_dataset

    """
    estandarizar datos alrededor de 0, usar despues de remover todos los datos indeseados
    """
    def standarizar_datos(self, dataset):
        sc_X_normalise = StandardScaler()
        for clase, column in self.columns_variables_int.items():
            sc_X_normalise.fit(dataset.loc[:, column])
            dataset.loc[:, column] = sc_X_normalise.fit_transform(dataset.loc[:, column])
        for clase, column in self.columns_variables_float.items():
            sc_X_normalise.fit(dataset.loc[:, column])
            dataset.loc[:, column] = sc_X_normalise.fit_transform(dataset.loc[:, column])
        return dataset