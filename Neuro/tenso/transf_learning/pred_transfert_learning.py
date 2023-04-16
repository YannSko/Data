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

 
male_path = Path("Images/elephants/test/males")
female_path = Path("Images/elephants/test/females")


test_images = []
test_target = []


height = 128
width = 128



for current_file in male_path.glob("*.JPG"):
    current_img = image.load_img(current_file, target_size=(height,width))
    current_img = image.img_to_array(current_img)  
    test_images.append(current_img)
    test_target.append(0)

for current_file in female_path.glob("*.JPG"):
    current_img = image.load_img(current_file, target_size=(height,width))
    current_img = image.img_to_array(current_img)  
    test_images.append(current_img)
    test_target.append(1)


x_test = np.array(test_images)
y_test = np.array(test_target)
x_test = vgg16.preprocess_input(x_test) 

model_file =  Path("Models/json_model.json")
json_model = model_file.read_text()

model = keras.models.model_from_json(json_model) 

model.load_weights("Models/weights.h5")


pretrained_model= vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(height,width,3)) 

features_extracted =  pretrained_model.predict(x_test) 

 
results = model.predict_classes(features_extracted)
 
 
df = pd.DataFrame({"Prédictions":results[:,0],"True":y_test})
print(df) 
 
  
















 
