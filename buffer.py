import numpy as np
import tensorflow as tf


class Buffer:
    def __init__(self,
                 buffer_size,
                 batch_size,
                 num_states,
                 num_action
                 ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.counter = 0
        """ Buffer gets a tuple input (s, a, r, s')
            These are stored individually within np.arrays init. here """
        self.s_buffer = np.zeros(shape=(buffer_size, num_states))
        self.a_buffer = np.zeros(shape=(buffer_size, num_action))
        self.r_buffer = np.zeros(shape=(buffer_size, 1))
        self.ns_buffer = np.zeros(shape=(buffer_size, num_states))

    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.counter % self.buffer_size

        self.s_buffer[index] = obs_tuple[0]
        self.a_buffer[index] = obs_tuple[1]
        self.r_buffer[index] = obs_tuple[2]
        self.ns_buffer[index] = obs_tuple[3]

        self.counter += 1

    def batch_sample(self):
        # Get sampling range
        record_range = min(self.counter, self.buffer_size)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.s_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.a_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.r_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.ns_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch
