from snake_env import *
from Actor import Actor
from Critic import Critic

import tensorflow as tf
from tqdm import tqdm
tf.get_logger().setLevel('FATAL')

import argparse
import numpy as np
import matplotlib.pyplot as plt
# the floating precision is required for the estimate probs have the sum of 1.0
tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--log_interval', type=int, default=500)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)
parser.add_argument('--save_weights', type=str, required=True)
parser.add_argument('--save_figure', type=str, required=True)

args = parser.parse_args()

class Agent:
    def __init__(self):
        self.env = snake_env()
        self.state_dim = (self.env.size, self.env.size)
        self.action_dim = self.env.action_space
        self.actor = Actor(self.state_dim, self.action_dim, args.actor_lr)
        self.critic = Critic(self.state_dim, args.critic_lr)
        self.gamma = args.gamma
        
        # initialize video system only
        self.env.reset()
        self.env.render()

    def MC(self, rewards, dones, next_value):
        '''
        Monte Carlo Estimation
        '''
        rewards = rewards.reshape(-1)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
            
        return returns[:-1].reshape(-1, 1)
    
    def advantage(self, returns, baselines):
        return returns - baselines

    def list_to_batch(self, _list):
        '''
        convert a list of single batches into a batch of len(_list)
        '''
        batch = _list[0]
        for elem in _list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self, max_updates=500, batch_size=64):
        episode_rewards = []
        actor_loss  = 0
        critic_loss = 0
        for up in tqdm(range(max_updates)):
            state_list   = []
            action_list  = []
            reward_list  = []
            done_list    = []
            step_rewards = []
            state        = self.env.reset()

            for ba in range(batch_size):
#                 self.env.render()

                # data collection
                probs  = tf.nn.softmax(self.actor.model(np.expand_dims(state, 0))[0])
                action = np.random.choice(self.action_dim, p=probs)

                next_state, reward, done, _ = self.env.step(action)
                step_rewards.append(reward)
                
                if done:
                    # the end of an episode
                    episode_rewards.append(sum(step_rewards)/len(step_rewards))
                    n_episode = len(episode_rewards)
                    if n_episode % args.log_interval == 0:
                        print(f'\nEpisode: {n_episode}, Avg Reward: {episode_rewards[-1]}')
                        
                    step_rewards = []
                    next_state   = self.env.reset()
                    
                    if max(episode_rewards) == episode_rewards[-1]:
                        self.actor.model.save_weights(args.save_weights)

                # make single batches
                state      = np.expand_dims(state, 0)
                action     = np.expand_dims(action, (0, 1))
                reward     = np.expand_dims(reward, (0, 1))
                done       = np.expand_dims(done, (0, 1))

                
                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)
                done_list.append(done)
                
                state = next_state
                
            # update the batch at once
            # convert list of batches into a batch of len(list)
            states  = self.list_to_batch(state_list)
            actions = self.list_to_batch(action_list)
            rewards = self.list_to_batch(reward_list)
            dones   = self.list_to_batch(done_list)
            
            next_value = self.critic.model(np.expand_dims(state, 0))[0]
            # using state, but actually it's next_state from the end of the loop above
            
            returns = self.MC(rewards, dones, next_value)

            advantages = self.advantage(
                returns,
                self.critic.model.predict(states)
            )

            actor_loss  = self.actor.train(states, actions, advantages)
            critic_loss = self.critic.train(states, returns)
            
        # save figure
        plt.plot(episode_rewards)
        plt.savefig(args.save_figure)

def main():
    agent = Agent()
    agent.train()


if __name__ == "__main__":
    main()