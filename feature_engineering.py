#Replace PositionalEmbedding Layer with 

import os 
import cv2
import numpy as np
from classes.positional_embedding import PositionalEmbedding
from classes.encoder import Encoder
from classes.convolutional_layer import ConvolutionalLayer
img_path = '/home/dolan/Portfolio/dev-portfolio/images/mockup1.png'

#since we are replacing our positional embedding layer with 
#image embeddings directly from the ConvolutionalLayer, 
#the outputs of the layer should have the same dimension as the 
#outputs of the positionalembedding layer.
#d_model will be the size of each image embedding vector
#SIZE = batch_size, seqence_length, d_model

'''FOR TRAINING, NOT INFERENCE. BATCH_SIZE=1 FOR INFERENCE'''
batch_size, img_sequence_length, img_width, img_height, num_channels = 16, 10, 28, 28, 3
d_model = 32
img_batch = np.random.rand(batch_size, img_sequence_length, img_width, img_height, num_channels) #batch_size, width, height, num_channels (colour)
convolutions = ConvolutionalLayer(img_width, img_height)

#This function will be used in training.
#Each batch of image data can be fed into this function and the result will be 
# a direct input to the encoder layer.
def preprocess_img_embeddings(img_batch): #img_sequence_length, img_width, img_height, num_channels
    # batch = img_batch[0] #Let's pick the first batch
    encoder_input = []
    for sequence in img_batch:
        sequence_array = []
        for img in sequence:
            img = img.reshape(1, img_width, img_height, num_channels)
            embeddings = convolutions(img)
            # print('embeddings shape: ', embeddings.shape)
            embeddings = embeddings.numpy().reshape(embeddings.shape[1])
            sequence_array.append(embeddings) #sequence_length, d_model
        print('Sequence Saved!')
        encoder_input.append(sequence_array) #batch_size, sequence_length, d_model

    return np.array(encoder_input)
test = np.random.rand(1, 32)
# test.shape[1]
# test = test.reshape(32)
# print('test: ', test.shape)
to_encoder = preprocess_img_embeddings(img_batch)
#Just what the doctor ordered
print('Preprocessed embeddings: ', to_encoder.shape) #Batch_size, sequence_length, d_model

'''CONCLUSION:
Directly connecting the ConvolutionalLayer to the model during training 
may be inefficient, because of the sheer amount of preprocessing required. 
The convolutional layer takes in data of the shape (batch_size, width, length, num_channels)
and returns a 1D vector output. We need a sequence of these vectors to be fed together 
into the Encoder in order for the Transformer Agent to learn the time-series context of 
the sequential images; this requires image embeddings to be fed in of the shape 
(batch_size, sequence_length, d_model)=> where the hidden dimension of the Encoder is the same as 
the dimension of each image embedding. The size of encoder model then has the same d_model as the 
output from the penultimate layer of whichever pretrained model is used to provide the image 
embeddings.
'''
#batch_size, img_sequence_length, d_model
encoder_input = np.random.rand(batch_size, img_sequence_length, d_model)
print(encoder_input.shape)#sequence_length, batch_size, 

sample_encoder = Encoder(num_layers=4,
                         d_model=32,
                         num_heads=8,
                         dff=20,
                        #  This has to be the same as the size of the output batch from the 
                        # convolutional layer...
                         vocab_size=32,#output_batch.shape[1],
                         dropout_rate=0.1,
                         width=img_width,
                         height=img_height
)

result = sample_encoder(encoder_input)
print(result.shape)  #batch_size, sequence_length, hidden_dim

'''
I want embeddings (outputs) of shape 
batch_size, context_window, width, height, as inputs 
to the encoder layer so as to encapsulate time series context by using 
a sequence of contiguous images based on the stream processing queue 
used in the Agent class.

''' 