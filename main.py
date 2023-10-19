#Replace PositionalEmbedding Layer with 

import os 
import cv2
import numpy as np
from classes.positional_embedding import PositionalEmbedding

from classes.convolutional_layer import ConvolutionalLayer
img_path = '/home/dolan/Portfolio/dev-portfolio/images/mockup1.png'

# img = cv2.imread(img_path)
# img = cv2.resize(img, (250, 500))
# img = img
# img_batch = np.array([img.tolist()])
# print(img.shape)


img_batch = np.random.rand(16, 28, 28, 3) #batch_size, width, height, num_channels (colour)
img_batch *= 100 #Convert to integers on the order of 10.0 ^2

#takes in image width, height
convolutions = ConvolutionalLayer(img_batch.shape[1], img_batch.shape[2])
output_batch = convolutions(img_batch)#Send batches of size 16 through the image 

'''
I want embeddings (outputs) of shape 
batch_size, context_window, width, height, as inputs 
to the encoder layer so as to encapsulate time series context by using 
a sequence of contiguous images based on the stream processing queue 
used in the Agent class.

'''
print('shape of outputs from convolutional layer: ', output_batch.shape)
# print(output_batch.shape)
pos_embed = PositionalEmbedding(vocab_size=100, d_model=125)

'''
# # inputs = np.random.rand(10, 16) 
#64,100

# #Send data of size (batch_size * sequence_length) to embed layer from convnet.
# #Use pretrained VGGNet to take advantage of pretrained layers for feature engineering.
'''
encoder_input = pos_embed(output_batch, is_decoder=False)
#Output from embedding layer: (batch_size, sequence_length, embedding_dim)

print('Shape of inputs to encoder: ', encoder_input.shape)