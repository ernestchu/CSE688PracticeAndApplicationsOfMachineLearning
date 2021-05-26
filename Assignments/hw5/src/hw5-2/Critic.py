import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, Reshape

class Critic:
    def __init__(self, state_dim, lr):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(lr)

    def create_model(self):
        return tf.keras.Sequential([
            InputLayer(self.state_dim),
            Reshape((*self.state_dim, 1)),
            Conv2D(4, 3, activation='relu'),
            Conv2D(16, 3, activation='relu'),
            Conv2D(64, 3, activation='relu'),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])

    def compute_loss(self, v_pred, returns):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(returns, v_pred)

    def train(self, states, returns):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == returns.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(returns))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss