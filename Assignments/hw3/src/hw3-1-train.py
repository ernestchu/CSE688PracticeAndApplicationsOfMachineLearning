import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from IPython.display import Image 
'''
Prepare Dataset
'''
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_images))
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(train_images.shape[0])
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_images))
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

'''
Hyper Parameters
'''
h_dim = 14
latent_dim = 10
num_epochs = 30

'''
Model Design
'''
def encoder(
    h_dim,
    timesteps,
    data_dim,
    out_dim,
    rnn_layer=tf.keras.layers.SimpleRNN
):
    return tf.keras.Sequential([
        tf.keras.layers.Reshape((timesteps, data_dim), input_shape=(timesteps, data_dim, 1)),
        rnn_layer(h_dim, return_sequences=True),
        rnn_layer(h_dim, go_backwards=True),
        tf.keras.layers.Dense(out_dim)
    ])

test = encoder(h_dim, 28, 28, latent_dim)
print("Encoder:")
test.summary()
def decoder(
    h_dim,
    timesteps,
    data_dim,
    out_dim,
    rnn_layer=tf.keras.layers.SimpleRNN
):
    return tf.keras.Sequential([
        tf.keras.layers.RepeatVector(timesteps, input_shape=(data_dim,)),
        rnn_layer(h_dim, return_sequences=True),
        rnn_layer(h_dim, return_sequences=True, go_backwards=True),
        tf.keras.layers.Dense(out_dim),
        tf.keras.layers.Reshape((timesteps, out_dim, 1))
    ])

test = decoder(h_dim, 28, latent_dim, 28)
print("Decoder:")
test.summary()

'''
Training
'''
def train(
    rnn_layer,
    save_weights=None,
    save_fig=None,
    load_weights=None,
    load_fig=None
):
    model = tf.keras.Sequential([
        encoder(h_dim, 28, 28, latent_dim, rnn_layer),
        decoder(h_dim, 28, latent_dim, 28, rnn_layer)
    ])
    
    if save_weights:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='mse'
        )
        history = model.fit(
            ds_train,
            epochs=num_epochs,
            validation_data=ds_test
        )
        model.save_weights(save_weights)
        if save_fig:
            plt.clf()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(save_fig)
    if load_weights:
        model.load_weights(load_weights)
    if load_fig:
        display(Image(filename=load_fig))
    return model

'''
Simple RNN
'''
train(
    tf.keras.layers.SimpleRNN,
    save_weights="../assets/hw3-1/weights/SimpleRNN.h5",
    save_fig="../assets/hw3-1/images/SimpleRNN.png",
)
'''
LSTM
'''
train(
    tf.keras.layers.LSTM,
    save_weights="../assets/hw3-1/weights/LSTM.h5",
    save_fig="../assets/hw3-1/images/LSTM.png",
)
'''
GRU
'''
train(
    tf.keras.layers.GRU,
    save_weights="../assets/hw3-1/weights/GRU.h5",
    save_fig="../assets/hw3-1/images/GRU.png",
)