from tensorflow.keras import backend as K
import numpy as np
from numpy import loadtxt
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image 
import os
import glob 
import cv2
import random

save_dir="models"

def representative_data_gen(train):
  for input_value in  train.take(30): 
    yield [ input_value]



def saving_tflite(model):
  K.clear_session()
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tfmodel = converter.convert()
  open(os.path.join(save_dir,"tfmodel_normal.tflite"), "wb").write(tfmodel)
  print('normal_saved')

def saving_tflite_weight(model):
  K.clear_session()
  converter = tf.lite.TFLiteConverter.from_keras_model(model)    
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  quantized_model = converter.convert()
  open(os.path.join(save_dir,"tfmodel_quantized_weight.tflite"), "wb").write(quantized_model)
  print('weight_saved')

def saving_tflite_int(model):
  K.clear_session()
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  converter.representative_dataset = representative_data_gen    
  quantized_model = converter.convert()
  open(os.path.join(save_dir,"tfmodel_quantized_int.tflite"), "wb").write(quantized_model)
  print('int_saved ')

def saving_tflite_float(model):
  K.clear_session()
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]
  quantized_model = converter.convert()
  open(os.path.join(save_dir,"tfmodel_quantized_float16.tflite"), "wb").write(quantized_model)
  print('float_saved ')  
    

if __name__ == "__main__":
  X_data =[]
  files = glob.glob ("path/to/the/images.bmp") #could be any image format
  random.shuffle(files)
  #Pick any number of file.
  #But must be equal or more than the value in
  #loop of  representative_data_gen function

  for myFile in files[:100]:    
    image = cv2.imread(myFile)
    image = image.astype(np.float32, copy=False)
  #########remove the comments if requirement for resize the image  ##### 
  # new_width=  256
  # new_height= 256
  # image = cv2.resize(image,(new_width, new_height), )
    X_data.append (image)


  train = tf.convert_to_tensor(np.array(X_data, dtype='float32'))
  train = tf.data.Dataset.from_tensor_slices((train)).batch(1)    
  model = load_model('any_model.h5')

  saving_tflite(model)
  saving_tflite_int(model)
  saving_tflite_float(model)


