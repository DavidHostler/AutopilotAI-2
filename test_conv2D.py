import os 
import cv2
import numpy as np
from classes.convolutional_layer import ConvolutionalLayer 


img_path = '/home/dolan/Portfolio/dev-portfolio/images/mockup1.png'

img = cv2.imread(img_path)
img = cv2.resize(img, (250, 500))
img = img
#Batch size of 1 
img_batch = np.random.rand(16, 28, 28, 3) #batch_size, width, height, num_channels (colour)
img_batch *= 100


# print(img.shape)

#Send batches of size 64 through the image 
# convolutions = ConvolutionalLayer(img.shape[0], img.shape[1])
convolutions = ConvolutionalLayer(img_batch.shape[0], img_batch.shape[1])
output_batch = convolutions(img_batch)
# output = convolutions(img)
# print(output.shape)

print(output_batch.shape)
#Assume that all outputs of convlayer are integers, which they will be 
#because we're going to feed unnormalized images with activations b/w 0->255.
#Each value shall be a token of the embedding layer!
print('Tokens from the output of the first batch: ', output_batch.numpy().astype(int)[0][:50])
# output_