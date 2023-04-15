import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



housing = fetch_california_housing()
all_x_train, x_test, all_y_train, y_test = train_test_split(housing.data, housing.target)
 
 
x_train, x_validation, y_train, y_validation = train_test_split(all_x_train, all_y_train)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_validation_scaled = scaler.transform(x_validation)
x_test_scaled = scaler.transform(x_test)



#model.summary()
class Exemple_SubclassingAPI(keras.models.Model):

    def __init__(self, nb_unit_1, nb_unit_2, activation="relu", **kwargs):
        super().__init__(**kwargs) 
        self.hidden1 = keras.layers.Dense(nb_unit_1, activation=activation)
        self.hidden2 = keras.layers.Dense(nb_unit_2, activation=activation)
        self.res = keras.layers.Dense(1)
        

    def call(self, inputs):
        input = inputs
        hidden1 = self.hidden1(input)
        hidden2 = self.hidden2(hidden1)
        res = self.res(hidden2)
        return res

model = Exemple_SubclassingAPI(30,15)



model.compile(loss="mse", optimizer="rmsprop", metrics=["mae"])

ressults = model.fit(x_train_scaled, y_train, epochs=40, validation_data=(x_validation_scaled, y_validation))

mse_test = model.evaluate(x_test_scaled, y_test)
 