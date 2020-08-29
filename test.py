import tensorflow as tf
import numpy as np
import gym
from buffer import Buffer
from ddpg_agent import Agent


""" Initialization """
agent = Agent()
env = gym.make('BipedalWalker-v3')
total_episodes = 100

# Memory:
buffer = Buffer(buffer_size=10000, batch_size=64, num_states=24, num_action=4)

# Base networks:
actor = agent.get_actor()
critic = agent.get_critic()

# Target networks:
target_actor = agent.get_actor()
target_critic = agent.get_critic()

# Weights are equal initially:
target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

# Optimizers:
critic_optimizer = tf.keras.optimizers.Adam(agent.lr_critic)
actor_optimizer = tf.keras.optimizers.Adam(agent.lr_actor)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

for ep in range(total_episodes):
    previous_state = env.reset()
    episodic_reward = 0
    # print(np.shape(previous_state))
    while True:
        env.render()
        tf_previous_state = tf.expand_dims(tf.convert_to_tensor(previous_state), 0)
        """ Select an action and execute it - receive info: """
        action = agent.act(tf_previous_state, agent.ounoise)
        action = np.reshape(action, newshape=(4, 1))
        # print(np.shape(action))
        state, reward, done, info = env.step(action)
        # Debug:
        # print("-- Output of the env.step: state: {}, reward: {}, done: {}, info: {}".format(
        # np.shape(state), reward, done, info))
        # """ Record new variables """
        action = np.squeeze(action)
        buffer.record((previous_state, action, reward, state))
        episodic_reward += reward
        """ Sample and learn  """
        s_batch, a_batch, r_batch, ns_batch = buffer.batch_sample()
        # Debug
        # print("-- Output of the batch sample: s_batch: {}, a_batch: {}, r_batch: {}, ns_batch: {}".format
        #       (np.shape(s_batch), np.shape(a_batch), np.shape(r_batch), np.shape(ns_batch)))
        agent.learn(s_batch, a_batch, r_batch, ns_batch)

        if done:
            break

        previous_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)
