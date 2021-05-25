from snake_env import *

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, Reshape
from tqdm import tqdm
tf.get_logger().setLevel('FATAL')

import argparse
import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count
tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--log_interval', type=int, default=20)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)

args = parser.parse_args()

CUR_EPISODE = 0


class Actor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.entropy_beta = 0.01

    def create_model(self):
        return tf.keras.Sequential([
            InputLayer(self.state_dim),
            Reshape((*self.state_dim, 1)),
            Conv2D(3, 3, activation='relu'),
            Conv2D(3, 3, activation='relu'),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])

    def compute_loss(self, actions, probs, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        entropy_loss = tf.keras.losses.CategoricalCrossentropy()
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(actions, probs, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(probs, probs)
        return policy_loss - self.entropy_beta * entropy

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            probs = self.model(states, training=True)
            loss = self.compute_loss(actions, probs, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            InputLayer(self.state_dim),
            Reshape((*self.state_dim, 1)),
            Conv2D(3, 3, activation='relu'),
            Conv2D(3, 3, activation='relu'),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self):
        env = snake_env()
        self.state_dim = (env.size, env.size)
        self.action_dim = env.action_space

        self.global_actor = Actor(self.state_dim, self.action_dim)
        self.global_critic = Critic(self.state_dim)
        self.num_workers = cpu_count()

    def train(self, max_episodes=1000):
        workers = []

        for i in tqdm(range(self.num_workers)):
            env = snake_env()
            workers.append(WorkerAgent(
                env, self.global_actor, self.global_critic, max_episodes))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class WorkerAgent(Thread):
    def __init__(self, env, global_actor, global_critic, max_episodes):
        Thread.__init__(self)
        self.lock = Lock()
        self.env = env
        self.state_dim = (self.env.size, self.env.size)
        self.action_dim = self.env.action_space

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self):
        global CUR_EPISODE

        while self.max_episodes >= CUR_EPISODE:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False

            state = self.env.reset()

            while not done:
                # self.env.render()
                probs = self.actor.model.predict(np.reshape(state, (1, *self.state_dim)))
                action = np.random.choice(self.action_dim, p=probs[0])

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, (1, *self.state_dim))
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, (1, *self.state_dim))
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                if len(state_batch) >= args.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)

                    next_v_value = self.critic.model.predict(next_state)
                    td_targets = self.n_step_td_target(
                        rewards, next_v_value, done)
                    advantages = td_targets - self.critic.model.predict(states)
                    
                    with self.lock:
                        actor_loss = self.global_actor.train(
                            states, actions, advantages)
                        critic_loss = self.global_critic.train(
                            states, td_targets)

                        self.actor.model.set_weights(
                            self.global_actor.model.get_weights())
                        self.critic.model.set_weights(
                            self.global_critic.model.get_weights())

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    td_target_batch = []
                    advatnage_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]
            
            if CUR_EPISODE % args.log_interval == 0:
                print(f'Episode: {CUR_EPISODE}, return: {episode_reward}')
            CUR_EPISODE += 1

    def run(self):
        self.train()


def main():
    agent = Agent()
    agent.train()


if __name__ == "__main__":
    main()