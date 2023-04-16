import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

model = vgg16.VGG16()

img_src = image.load_img("Images/elephants/train/males/IMG_0585.JPG", target_size=(224,224)) 
img_test = image.img_to_array(img_src)

img_test_lst = np.expand_dims(img_test, axis=0) 

img_test_lst = vgg16.preprocess_input(img_test_lst) 


predictions = model.predict(img_test_lst)


predected_classes = vgg16.decode_predictions(predictions, top=5)

print(predected_classes)








# imagenet classes
# https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a 
