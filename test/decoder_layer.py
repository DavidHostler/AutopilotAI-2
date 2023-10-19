from transformer.classes.decoder_layer import DecoderLayer
import numpy as np
sample_decoder_layer = DecoderLayer(d_model=32, num_heads=8, dff=100)
'''#Inputs from encoder block and positional embedding layer


#From encoder block:
#size = batch_size, sequence_length, hidden_dim
sample_input_from_encoder = np.random.rand(16, 10, 96)#hidden_dim=96
#From embedding layer 
#size =  (batch_size, sequence_length, hidden_dim)
sample_input_from_posemb = np.random.rand(16, 10, 96)


result = sample_decoder_layer(x=sample_input_from_posemb,
context=sample_input_from_encoder
)

print(result.shape)
'''