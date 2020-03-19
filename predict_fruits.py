import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from numpy import set_printoptions
import cv2
from os import listdir
from os.path import isfile,join



os.environ["TF_CPP_MIN_LOG_LEVEL"]="3";

filepath="fruits_cnn_0.90.h5"


my_model=load_model(filepath)

print(my_model.summary())



path=r"C:\Users\Ammad\Desktop\Fruit classifier\fruits-360\test\apple.jpg"

image=cv2.imread(path)

cv2.imshow("....",image)

x=image.reshape(1,32,32,3)
x=my_model.predict_classes(image)

print(x)


