from classes.model import *
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers.legacy import Adam
from classes.optimizer import CustomSchedule

device_lib.list_local_devices()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)
import os 
os.environ["KERAS_BACKEND"] = "tensorflow"

''' 
Write some ETL code to extract the training data in batches, and preprocess the image keypress data into the 
relevant formats
Each batch should contain an image sequence, and two versions of the target sequence. 

The first is "key_press" data, and the second is "labels". Labels is just the keypress data with the 
indices of the elements shifted up by 1.
'''


vocab_size=50
d_model=32
batch_size=16
num_heads=4
dropout_rate=0.1
dff=128
num_layers=2
sequence_length=100
width, height = 28, 28

#Initialize artificial neural network in Tensorflow
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=sequence_length,
    target_vocab_size=vocab_size,
    dropout_rate=dropout_rate)

'''Model training data. 
In order to optimize efficiency, disable eager execution in Tensorflwo by wrapping our 
custom train step in the @tf.function decorator design pattern.

Additionally, run the train loop on the device using CUDA with Tensorflow GPU enabled.
Since we're using a Docker container, this should be a program that any windows PC can run.
'''


    


#Load training data..
batch_size=32#BATCH_SIZE
epochs = 100
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    



'''
Masked Loss and Accuracy
'''

def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)



'''Custom optimizer/ learning rate scheduler
'''
learning_rate = CustomSchedule(d_model)

optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

'''Custom Train step.
Implement graph execution to optimize training performance.
'''
@tf.function
def train_step(image_data, keypress_data, labels, optimizer):
    with tf.GradientTape() as tape:
        logits = transformer((image_data, keypress_data), training=True)
        loss_value = masked_loss(labels, logits)
    grads = tape.gradient(loss_value, transformer.trainable_weights)
    optimizer.apply_gradients(zip(grads, transformer.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

#Do we need to track validation loss for a Generative model?       
@tf.function
def test_step(image_data, keypress_data):
    val_logits = transformer((image_data, keypress_data), training=False)
    val_acc_metric.update_state(y, val_logits)

#Feed in the transformer model, num_epochs, and train_data to this function and run it.
def train(epochs, train_data, optimizer):
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")

        for step, ((image_data, keypress_data), labels) in enumerate(train_data[:]):
            #Call custom train step
            loss_value = train_step(image_data, keypress_data, labels, optimizer)
            
            if step % 100 == 0:
                print(
                    f"Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}"
                )
                print(f"Seen so far: {(step + 1) * batch_size} samples")

'''Uncomment to train the Terminator bot!'''
epochs=10
# train(transformer, 10, train_data, optimizer)