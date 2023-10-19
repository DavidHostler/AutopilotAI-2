import tensorflow as tf 
from tensorflow import keras 
from keras.layers import Layer, Dense, Dropout
# from attention import CausalSelfAttention, CrossAttention
# from feed_forward import FeedForward
from .attention import CausalSelfAttention, CrossAttention
from .feed_forward import FeedForward
import numpy as np
class DecoderLayer(Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x
  

#Unit Test
#hidden_dim, number of attention heads, feed forward dim
sample_decoder_layer = DecoderLayer(d_model=40, num_heads=8, dff=100)
'''Inputs from encoder block and positional embedding layer'''
#From encoder block:
#size = batch_size, sequence_length, hidden_dim
sample_input_from_encoder = np.random.rand(16, 10, 40)#hidden_dim=96
#From embedding layer 
#size =  (batch_size, sequence_length, hidden_dim)
sample_input_from_posemb = np.random.rand(16, 10, 40)


result = sample_decoder_layer(x=sample_input_from_posemb,
context=sample_input_from_encoder
)

print(result.shape) #batch_size, sequence_length, hidden_dim