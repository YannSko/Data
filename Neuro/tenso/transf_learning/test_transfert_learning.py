import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16

from pathlib import Path
import joblib


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


male_path = Path("Images/elephants/train/males")
female_path = Path("Images/elephants/train/females")


train_images = []
train_target = []

height = 128
width = 128

for current_file in male_path.glob("*.JPG"):
    current_img = image.load_img(current_file, target_size=(height,width))
    current_img = image.img_to_array(current_img)  
    train_images.append(current_img)
    train_target.append(0)

for current_file in female_path.glob("*.JPG"):
    current_img = image.load_img(current_file, target_size=(height,width))
    current_img = image.img_to_array(current_img)  
    train_images.append(current_img)
    train_target.append(1)


x_train = np.array(train_images)
y_train = np.array(train_target)


x_train = vgg16.preprocess_input(x_train) 

 
 
pretrained_model= vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(height,width,3)) 

features =  pretrained_model.predict(x_train) 

 
model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=features.shape[1:]))
 
model.add(keras.layers.Dense(256, activation='relu'))
 
model.add(keras.layers.Dense(1, activation='sigmoid'))
 


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
 
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
result = model.fit(features, y_train,\
                   epochs=200,\
                   validation_data=[features, y_train],\
                   callbacks=[early_stopping])


json_model = model.to_json()
model_file = Path("Models/json_model.json")
model_file.write_text(json_model)
model.save_weights("Models/weights.h5")
 











 
