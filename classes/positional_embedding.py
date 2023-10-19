import tensorflow as tf 
from tensorflow import keras
from keras.layers import Layer, Embedding
import numpy as np
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)
  
class PositionalEmbedding(Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    # (batch_size, input_length)
    self.embedding = Embedding(vocab_size, d_model, mask_zero=True)
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x, is_decoder=True):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    '''
    Get rid of positional encoding for the encoder 
    '''

    # This factor sets the relative scale of the embedding and positonal_encoding.
    if is_decoder:
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
  


#Unit test
batch_size, sequence_length = 16, 50
position_embedding = PositionalEmbedding(vocab_size=50, d_model=32)
inputs = np.random.rand(batch_size, sequence_length)
output = position_embedding(inputs)
print(output.shape) #batch_size, sequence_length, hidden_dims