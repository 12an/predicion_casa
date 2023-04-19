# -*- coding: utf-8 -*-

from preprocesing import PreProcesing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

class Model():
    def __init__(self, 
                 no_categorical_x_training,
                 categorical_x_training,
                 x_predictor):
        self.no_categorical_x_training = no_categorical_x_training
        self.categorical_x_training = categorical_x_training
        self.x_predictor = x_predictor
        self.descriptor = dict()
        self.x_predictor_mean = self.x_predictor.mean()
        #join all statistical data relate to the model
        self.descriptor_model = {"R_2":0,
                                 "F_statistic":0,
                                 "RSS":0,
                                 "TSS":0,
                                 "P_value":0,
                                 "error":0}
        self.descriptor_coefficient = list()
        for column in no_categorical_x_training.colomns:
            mean = no_categorical_x_training.loc[:, column].mean()
            self.descriptor_coefficient[column] = {"coefficient":0,
                                       "Standard_error":0,
                                       "T_statistic":0,
                                       "P_value":0,
                                        "Mean":mean}
            
    def get_fit_error(self, to_fit_data, predictor_data_fitted):
        y_predicted = self.train_model.fit(to_fit_data)
        return predictor_data_fitted - y_predicted

    def get_RSS(self,  to_predict_data, predictor_data_fitted):
        y_predicted = self.train_model.fit(to_predict_data)
        error = predictor_data_fitted - y_predicted
        self.descriptor_model["RSS"] = np.sum(np.power(error, 2))
        self.descriptor_model["error"] = error

    def get_TSS(self, y_predictor_mean, y_predicted):
        return np.sum(np.pow(y_predicted - y_predictor_mean))

    def f_statistic(self,  to_fit_data, predictor_data_fitted):
        pass

    def R_2(self,  to_fit_data, predictor_data_fitted):
        a = self.descriptor_model.get("TSS") - self.descriptor_model.get("RSS")
        b = a / self.descriptor_model.get("TSS")
        self.descriptor_model.get("R_2") 
  

    """
to solve the matrix ax=b where x are the predictor, a the data and b the prediction
    """
    def solve_ecuation_matrix():
        pass
        

datos = PreProcesing(**{"path_train":"datos\\" + "train.csv",
                        "path_test":"datos\\" + "test.csv",
                        "path_info":"info\\" + "organizacion_bon_representation.csv"})

#working with a clean data, later to train the final model, fill all the misign or  no conform values
datos.optener_tipografia_datos(datos.dataset_train)
x_train_base = datos.train_test_base(datos.encoder_categorias(datos.dataset_train))
x_test_base = datos.train_test_base(datos.encoder_categorias(datos.dataset_test))

'''
plot correlation matrix excluding all chategorical data
'''

no_chategorical_columns = datos.columns_variables_int + datos.columns_variables_float
dataset_no_categorical = x_train_base.loc[:, no_chategorical_columns]
correlacion_figura = plt.figure(figsize=(20, 20))
plt.matshow(dataset_no_categorical.corr(), fignum=correlacion_figura.number)
plt.xticks(range(dataset_no_categorical.select_dtypes(['number']).shape[1]),
               dataset_no_categorical.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(dataset_no_categorical.select_dtypes(['number']).shape[1]),
               dataset_no_categorical.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=20);


'''
plot correlation scatter against saleprice y excluding all chategorical data
'''
def correlation_scatter():
    global dataset_no_categorical, no_chategorical_columns
    for i in range(0, len(no_chategorical_columns)):
        fig = plt.figure(figsize=(20, 20))
        axe = fig.add_subplot()
        axe.scatter(dataset_no_categorical.iloc[:, i],
                    x_train_base.loc[:, "SalePrice"])
        # Make legend, set axes limits and labels
        axe.set_xlabel(no_chategorical_columns[i])
        axe.set_ylabel("SalePrice")
        plt.show()
    
"""
based in the scatter plot against salesprice
we can notice a lighter polinominal behavioral in:
    -1stFlrSF area of the first floor
    -TotalBsmtSF area of the basement
    -GarageArea area of the garage
"""

#creando lineal regresion basado en variables no categoricals
#add column feature at _pow
to_pow =["1stFlrSF_pow", "TotalBsmtSF_pow", "GarageArea_pow"]
power = dataset_no_categorical.loc[:, ["1stFlrSF", "TotalBsmtSF", "GarageArea"]].pow(2, axis=1)
dataset_no_categorical.loc[:, to_pow] = power.rename(columns={"1stFlrSF": "1stFlrSF_pow",
                                                              "TotalBsmtSF": "TotalBsmtSF_pow",
                                                              "GarageArea":"GarageArea_pow"})
x_training_no_categrical = dataset_no_categorical.loc[:, no_chategorical_columns + to_pow]
y_training = x_train_base.loc[:,"SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(x_training_no_categrical,
                                                    y_training, 
                                                    test_size=0.33, random_state=52)
