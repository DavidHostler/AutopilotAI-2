import tensorflow as tf 
from tensorflow import keras
from keras.layers import Layer, Conv2D, MaxPooling2D, Dropout, Flatten, Input, Dense, BatchNormalization 
import numpy as np
 
# img_batch = np.random.rand(16, 224, 224, 3) #batch_size, width, height, num_channels (colour)
# vgg_output = model(img_batch)
# vgg_output.shape (batch_size, sequence_length)

'''
NOTE:
I may likely be replacing this ConvNet with the pretrained VGG model, 
using the pretrained weights for feature engineering.
Simply replace the output layer, freeze all prev layers, and train the output 
to generate our tokens for us.

Outputs of this layer are of the form (batch_size, sequence).
Feed them into the positional_embedding layer.
        
Positional encoding will be not be used as in the original Transformer architecture,
because all pixels in a given image array exist at the same time, unlike tokens in a 
sequence as we have in written language.

The Decoder will still implement positional encoding because it's outputs are inherently 
time series predictions.

BatchNorm is used to regularize data in minibatches during the forward pass. 
MaxPooling2D further reduces the size of forward feed images.
'''

class ConvolutionalLayer(Layer):
    def __init__(self,width,height):
        super().__init__()
        self.width = width
        self.height = height

    def call(self, x):
        # x = tf.convert_to_tensor(x)
        # input_layer = Input(shape=x.shape)
        '''
         #https://keras.io/api/layers/convolution_layers/convolution2d/
         # The inputs are 28x28 RGB images with `channels_last` and the batch
         # size is 4.
         input_shape = (4, 28, 28, 3)
         x = tf.random.normal(input_shape)
         y = tf.keras.layers.Conv2D(
         
        ... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
         print(y.shape)
        (4, 26, 26, 2)
        '''
        x = Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=x.shape[1:])(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = BatchNormalization()(x)

        #x.shape = (None, 8, 62, num_filters) 
        
        x = Flatten()(x)

        # x = Dense(32)

        #Perfect shape to feed into an embedding layer (batch_size * sequence_length)
        return x