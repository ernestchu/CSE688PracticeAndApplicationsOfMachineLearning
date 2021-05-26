import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, Reshape

class Actor:
    def __init__(self, state_dim, action_dim, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(lr)
        self.entropy_beta = 0.01

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
            Dense(self.action_dim)
        ])

    def compute_loss(self, actions, probs, advantages):
        wce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = wce_loss(actions, probs, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(probs, probs)
        return policy_loss - self.entropy_beta * entropy

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            probs = self.model(states, training=True)
            loss  = self.compute_loss(actions, probs, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss