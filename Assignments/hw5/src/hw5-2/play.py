from snake_env import *
from Actor import Actor
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

env = snake_env()
actor = Actor((env.size, env.size), env.action_space, 0)
actor.model.load_weights('./weights/baseline.h5')
while True:
    state = env.reset()
    done = False
    while not done:
        env.render()
        delay(.1)
        probs = tf.nn.softmax(actor.model(np.expand_dims(state, 0))[0])
        action = np.random.choice(env.action_space, p=probs)
        
        next_state, reward, done, info = env.step(action)
        state = next_state