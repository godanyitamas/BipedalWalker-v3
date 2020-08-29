import tensorflow as tf
import gym
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from ou_noise import OUNoise
from tensorflow.keras import layers
from buffer import Buffer

""" About Bipedal Walker:
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version.
The state's shape is (24,)
"""


# Agent:
class Agent:
    def __init__(self,
                 num_actions=4,  # 2 torque for 2 legs
                 num_states=24,
                 batch_size=64,
                 gamma=0.99,  # Gamma coef. for future rewards
                 tau=1e-3,
                 lr_actor=1e-4,
                 lr_critic=1e-3
                 ):
        self.num_actions = num_actions
        self.num_states = num_states
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        """ Noise generation - returns noise for exploration """

        # self.noise = OUNoise(seed=0, size=(1, 4))
        # self.ounoise = self.noise.sample()
        # print(self.ounoise)

    """ Functions for the neural networks """

    def get_actor(self):
        """
        Predicts an action from the state input
        """
        # Initialize weights between -3e-3 and 3-e3 as I use tanh
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,), name='state_input')
        out = layers.Dense(300, activation="relu", name='act_output_fc1')(inputs)
        # out = layers.BatchNormalization()(out)
        out = layers.Dense(400, activation="relu", name='act_output_fc2')(out)
        # out = layers.BatchNormalization()(out)
        outputs = layers.Dense(4, activation="tanh", kernel_initializer=last_init, name='act_output_fc3')(out)

        model = tf.keras.Model(inputs, outputs)

        return model

    def get_critic(self):
        """
        Predicts the value of the state from the state input and the action
        """
        # State as input with num states neurons (24)
        state_input = layers.Input(shape=(self.num_states), name='state_input')
        state_out = layers.Dense(48, activation="relu", name='state_fc1')(state_input)
        # state_out = layers.BatchNormalization()(state_out)
        state_out = layers.Dense(72, activation="relu", name='state_fc2')(state_out)
        # state_out = layers.BatchNormalization()(state_out)

        # Action as input with num action neurons (4)
        action_input = layers.Input(shape=(self.num_actions,), name='action_input')
        action_out = layers.Dense(72, activation="relu", name='action_fc1')(action_input)
        # action_out = layers.BatchNormalization()(action_out)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])
        # Tensor("concatenate/concat:0", shape=(None, 64), dtype=float32)

        out = layers.Dense(300, activation="relu", name='crt_output_fc1')(concat)
        # out = layers.BatchNormalization()(out)
        out = layers.Dense(400, activation="relu", name='crt_output_fc2')(out)
        # out = layers.BatchNormalization()(out)
        outputs = layers.Dense(1, name='crt_output_fc3')(out)  # Critic has one output as the state-action value

        model = tf.keras.Model([state_input, action_input], outputs, name='Critic')

        return model

    """-----------------------------------"""

    def act(self, state, noise):
        """ Returns a clipped action according to the policy """
        new_action = tf.squeeze(actor(state))
        new_action += noise.sample()
        # DEBUG:
        # print("Act method ran \n"
        #      "New action is: ", new_action)
        return np.clip(new_action, -1, 1)

    def noise_reset(self):
        noise.reset()

    def learn(self, s_batch, a_batch, r_batch, ns_batch):
        """ Uses the sampled batch from buffer to set target values
            Update:
            - Critic
            - Actor
            - Critic_target
            - Actor_target
        """

        """ ----------------- CRITIC ----------------- """
        with tf.GradientTape() as tape:
            target_actions = target_actor(ns_batch)
            """ y = r + gamma * Q(s', a') """
            y = r_batch + self.gamma * target_critic([ns_batch, target_actions])
            critic_value = critic([s_batch, a_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        """ Updating the critic via the optimizer with MSE loss of the y and critic value """
        critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic.trainable_variables)
        )

        """ ----------------- ACTOR ----------------- """
        with tf.GradientTape() as tape:
            actions = actor(s_batch)
            critic_value = critic([s_batch, actions])
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
        """ Actor is corrected by policy grad """
        actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor.trainable_variables)
        )

        """ ----------------- TARGET ----------------- """
        new_weights = []
        target_variables = target_critic.weights
        for i, variable in enumerate(critic.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        target_critic.set_weights(new_weights)

        new_weights = []
        target_variables = target_actor.weights
        for i, variable in enumerate(actor.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        target_actor.set_weights(new_weights)

    def save_log(self):
        # Makes a new folder system with current time as name
        mydir = os.path.join(os.getcwd(), 'Data', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'Models')
        os.makedirs(mydir)
        plotdir = os.path.join(os.getcwd(), 'Data', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        # Creates paths to the weights
        a_path = os.path.join(mydir, 'actor.h5')
        c_path = os.path.join(mydir, 'critic.h5')
        ta_path = os.path.join(mydir, 'target_actor.h5')
        tc_path = os.path.join(mydir, 'target_critic.h5')
        print(a_path)

        # Saves weights for each network
        actor.save_weights(filepath=a_path)
        critic.save_weights(filepath=c_path)
        target_actor.save_weights(filepath=ta_path)
        target_critic.save_weights(filepath=tc_path)

        # Save the plots for learning
        # Plotting graph
        # Episodes versus Avg. Rewards
        plt.figure()
        plt.style.use('seaborn')
        plt.plot(avg_reward_list, label='Avg of previous 100 episodes')
        plt.plot(ep_reward_list, label='Reward per episode', alpha=0.5)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DDPG-BipedalWalker-v3")
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(plotdir, 'avg_plot'))


""" Initialization """
agent = Agent()
env = gym.make('BipedalWalker-v3')
total_episodes = 5000
noise = OUNoise(size=(1, 4), seed=0)

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
        action = agent.act(tf_previous_state, noise)
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
        agent.noise_reset()

    ep_reward_list.append(episodic_reward)
    # Mean of last 100 episodes
    avg_reward = np.mean(ep_reward_list[-100:])
    print("Episode -- {} \t Avg Reward -- {:.3f} \t Reward -- {}".format(ep, avg_reward.round(decimals=3),
                                                                          episodic_reward))
    avg_reward_list.append(avg_reward)

agent.save_log()
