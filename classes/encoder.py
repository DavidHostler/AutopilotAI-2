import tensorflow as tf 
from tensorflow import keras 
from keras.layers import Layer, Dense, Dropout
# from positional_embedding import PositionalEmbedding
# from convolutional_layer import ConvolutionalLayer
# from encoder_layer import EncoderLayer
from .positional_embedding import PositionalEmbedding
from .convolutional_layer import ConvolutionalLayer
from .encoder_layer import EncoderLayer
import numpy as np

'''
MODIFICATION TO ENCODER ARCHITECTURE:
in order to achieve computer-vision based performance, 
I'm adding in a Convolutional layer before the positional embedding so as to provide 
context for image data fed in from video.
'''
class Encoder(Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate, width, height):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)
    # Newly added convolutional layer 
    self.convolution = ConvolutionalLayer(width, height) #Use this for now 


    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    # x = self.convolution(x) #In main.py we found that this works
    # x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    # x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.


#Test 

#Sample batch of image data

img_batch = np.random.rand(16, 28, 28, 3) #batch_size, width, height, num_channels (colour)
# img_batch *= 100 #Convert to integers on the order of 10.0 ^2

#takes in image width, height
# convolutions = ConvolutionalLayer(img_batch.shape[1], img_batch.shape[2])
# output_batch = convolutions(img_batch)#Send batches of size 16 through the image 
# output_batch = output_batch.numpy()

# Instantiate the encoder.
sample_encoder = Encoder(num_layers=4,
                         d_model=32,
                         num_heads=8,
                         dff=20,
                        #  This has to be the same as the size of the output batch from the 
                        # convolutional layer...
                         vocab_size=32,#output_batch.shape[1],
                         dropout_rate=0.1,
                         width=img_batch.shape[1],
                         height=img_batch.shape[2])

sample_input = np.random.rand(16, 32)#batch_size * sequence_length 
# result = sample_encoder(sample_input)
# result = sample_encoder(output_batch)
# print(result.shape, type(result.shape)) #batch_size, sequence_length, hidden_dim
# print(output_batch.shape[1], type(output_batch.shape[1]))
