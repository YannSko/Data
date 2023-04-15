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


input_1 = keras.layers.Input(shape=x_train_scaled.shape[1:])
input_2 = keras.layers.Input(shape=[5])
input_3 = keras.layers.Input(shape=[3])

hidden1 = keras.layers.Dense(30, activation="relu")(input_1)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat_1 = keras.layers.concatenate([input_2, hidden2])

hidden3 = keras.layers.Dense(30, activation="relu")(concat_1)
concat_2 = keras.layers.concatenate([input_3, hidden3])

output_1 = keras.layers.Dense(1)(concat_2)
output_2 = keras.layers.Dense(1)(hidden3)


model = keras.models.Model(inputs=[input_1, input_2, input_3], outputs=[output_1, output_2])

model.compile(loss="mse", optimizer="rmsprop", metrics=["mae"])

#model.compile(loss=["mse","mae"], loss_weights=[0.9, 0.1], optimizer="rmsprop", metrics=["mae"])


x_train_2, x_train_3 = x_train_scaled[:, :5], x_train_scaled[:, 5:]
x_validation_2, x_validation_3 = x_validation_scaled[:, :5], x_validation_scaled[:, 5:]
x_test_2, x_test_3 = x_test_scaled[:, :5], x_test_scaled[:, 5:]

 

 

ressults = model.fit((x_train_scaled , x_train_2, x_train_3), [y_train, y_train], epochs=5,\
                    validation_data=((x_validation_scaled, x_validation_2, x_validation_3), [y_validation,y_validation]))

res_eval = model.evaluate((x_test_scaled, x_test_2, x_test_3), [y_test, y_test])
 
x_new_1, x_new_2, x_new_3 = x_test_scaled[:3], x_test_scaled[:3, :5], x_test_scaled[:3, 5:]
y_pred = model.predict((x_new_1, x_new_2,x_new_3))
 
print(f"y_pred = {y_pred}")
 


