import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Layer, Dropout
import numpy as np 
# from decoder_layer import DecoderLayer
# from positional_embedding import PositionalEmbedding

from .decoder_layer import DecoderLayer
from .positional_embedding import PositionalEmbedding


class Decoder(Layer):
  def __init__(self,num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                             d_model=d_model)
    self.dropout = Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
    # print('Shape of embedding outputs: ', tf.shape(x))
    x = self.dropout(x)

    # for i in range(self.num_layers):
      # x  = self.dec_layers[i](x, context)

    # self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x


#Unit test 
# d_model=32, num_heads=8, dff=100


# sample_decoder = Decoder(num_layers=4,
#                          d_model=32,
#                          num_heads=4,
#                          dff=10,
#                          vocab_size=50)

#sample_input_from_encoder = np.random.rand(16, 50, 40)#hidden_dim=96
#From embedding layer 
#size =  (batch_size, sequence_length, hidden_dim)
# sample_input_from_posemb = np.random.rand(16, 50, 40)
# sample_decoder(x=sample_input_from_posemb,
#     context=sample_input_from_encoder
# )

# output = sample_decoder(
#     x=sample_input_from_posemb,
#     context=sample_input_from_encoder
# )
'''
vocab_size=50
d_model=32
batch_size=16
num_heads=4
dropout_rate=0.1
dff=64
num_layers=2
sequence_length=50

# From embedding layer #sequence_length, hidden_dims
sample_input_from_posemb = np.random.rand(sequence_length, d_model)#hidden_dim=96
# From encoder layer  size =  (batch_size, sequence_length, hidden_dim)
sample_input_from_encoder = np.random.rand(batch_size, sequence_length, d_model)

sample_decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, dropout_rate)
output = sample_decoder(sample_input_from_posemb, sample_input_from_encoder)

print('DECODER OUTPUT: ', output.shape)
'''