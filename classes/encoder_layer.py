import tensorflow as tf 
from tensorflow import keras 
from keras.layers import Layer, Dense
# from attention import GlobalSelfAttention
# from feed_forward import FeedForward
from .attention import GlobalSelfAttention
from .feed_forward import FeedForward
# import numpy as np

class EncoderLayer(Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

# sample_encoder = Encoder(num_layers=4,
#                          d_model=32,
#                          num_heads=8,
#                          dff=20,
#                          vocab_size=50)
# sample_encoder_layer = EncoderLayer(d_model=32, num_heads=8, dff=20)
# #Move to a unit test later
# sample_input = np.random.rand(16, 100, 32) #Batch_size, sequence_length, hidden_dim 
# print(sample_encoder_layer)

# output = sample_encoder_layer(sample_input)
# print(output.shape)