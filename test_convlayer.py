from convolutional_layer import ConvolutionalLayer 
import tensorflow as tf
import numpy as np
import cv2
import pyautogui

def convert_img_to_embedding(img, convlayer): #img.shape = (9600, 2806, 3)
    #img = cv2.resize(img, (28, 28))
    sample_input = img.reshape(1, 28, 28, 3)#np.random.rand(11,28, 28, 3)
    output = convlayer(sample_input)
    output = output.numpy()
    output = output.reshape(output.shape[1])#d_model
    return output
#This is waht 

def get_screenshot():
    screencap = pyautogui.screenshot()
    screencap = np.array(screencap)#Load to queue ~
    # screencap = screencap.resize(1, 2)
    screencap = cv2.resize(screencap, (28, 28))
    return screencap #Shape 28, 28, 3



convlayer = ConvolutionalLayer(28, 28)

#Test using tensorflow and numpy
# print(tf.shape(output))
# sample_img = np.random.rand(28, 28, 3)
original_img = get_screenshot()#np.random.rand(900, 2806, 3)
# print(original_img.shape)
output = convert_img_to_embedding(original_img, convlayer)
# screencap = get_screencap_embeddings(convlayer)
print(output.shape) #batch_size, d_model

# print(screencap.shape)