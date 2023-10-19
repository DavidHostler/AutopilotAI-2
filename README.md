This is an updated version of the AutopilotAI portfolio project, based on the paper 
Attention Is All You Need.
In the original project, a recurrent LSTM was pretrained to learn user driving patterns 
in videogames based on their keystroke biometrics. Using the Python multiprocessing module,
I was able to implement concurrency so as to both fine-tune the model and perform 
inference simultaneously, thereby getting around the very short attention window of the 
LSTM.

In this version of the project, I implemented a Transformer architecture based on a similar 
example in the Tensorflow website documentation, implementing the self-attention mechanism 
following a Factory design pattern. 
This design pattern enables implementation of Global self attention in the encoder layers, 
in addition to both Causal and Cross attention in the decoder layers.

I have modified the original Transformer so that rather than taking language tokens (words)
as inputs, the encoder block takes in image data. This required the creation of a ConvolutionalLayer class in order to feed image data into the PositionalEncoding layer.
Rather than feed word tokens into the Decoder, it takes in tokenized keyboard presses.
Each key on the user's keyboard represents a unique token in the tokenizer work index, 
in the same way that words are tokenized. 

A final modifcation I made to the inputs was to replace the conventional data pipeline using 
the tf.data.Datasets module to load in the training and validation batches for training, I 
manually loaded the training data into the model using numpy arrays, enabling me to have 
a better visual of the data in the pipeline right before running my training loop.

In order to avoid making too many modifications to the original Tensorflow Transformer, I followed 
the Open-Closed Principle from SOLID software development principles- The Transformer class is 
open for extension, and closed for modification. Instead of rewriting any input layers in the model to accomodate numpy arrays directly, I simply replaced the conventional model.fit() method 
with my own custom training loop, manually performing Gradient Descent.

### Implications 

This fun project is actually a demonstration of both cutting-edge software development principles 
and applications of the most advanced deep learning techniques applied to a real world computer 
vision application.
