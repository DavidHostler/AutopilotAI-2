import tensorflow as tf
from tensorflow import keras
# from encoder import Encoder
# from decoder import Decoder
# from convolutional_layer import ConvolutionalLayer
# from attention import GlobalSelfAttention
from .encoder import Encoder
from .decoder import Decoder
from .convolutional_layer import ConvolutionalLayer
from .attention import GlobalSelfAttention
import numpy as np 

class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate, width=28, height=28):
    super().__init__()
    self.conv = ConvolutionalLayer(width, height)#Like dis 

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate,
                           width=28, height=28)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)


  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs
    #Only pass inputs through convlayer during inference
    # context = self.conv(context) #Pass through conv layer
    # context = context.numpy()
    context = self.encoder(context)  # (batch_size, context_len, d_model)
    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
    
    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits


'''
#Unit test 

vocab_size=50
d_model=32
batch_size=16
num_heads=4
dropout_rate=0.1
dff=128
num_layers=2
sequence_length=100
width, height = 28, 28

#Encoder inputs 

#Replace img batch with img embeddings when training the transformer.
# img_batch = np.random.rand(16, 28, 28, 3) #batch_size, width, height, num_channels (colour)
# img_batch *= 100 
img_embeddings = np.random.rand(batch_size, sequence_length, d_model)
# img_embeddings = np.random.rand(batch_size, 65, d_model)

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=sequence_length,
    target_vocab_size=vocab_size,
    dropout_rate=dropout_rate)

# encoder_input = img_batch#np.random.rand(batch_size, sequence_length)
encoder_input = img_embeddings #Batched sequences of image embedding vectors

#Test the dropping of the keras mask, 
# decoder_input = np.random.rand(batch_size, 67)
decoder_input = np.random.rand(batch_size, sequence_length)

pred = transformer((encoder_input, decoder_input))
#The next value is going to be the argmax of the pred
print('Shape of transformer outputs: ', pred.shape)
#shape = batch_size, input_sequence_length, vocabulary_size
#where vocabulary_size=number of 'words' or potential outputs in the  
#transformer's vocabulary.
'''