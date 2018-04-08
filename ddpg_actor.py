import tensorflow as tf
from math import sqrt
import random
import numpy as np

batch_size = 100
LR_A = 0.001
class OU_Process(object):
    def __init__(self,action_dim, theta = 0.15, mu = 0, sigma = 0.2):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.current_x = np.ones(self.action_dim) * self.mu

    def update_process(self):
        dx = self.theta * (self.mu - self.current_x) + self.sigma * np.random.randn(self.action_dim)
        self.current_x = self.current_x + dx

    def return_noise(self):
        self.update_process()
        return self.current_x

class DDPG_Actor(object):
    def __init__(self, state_dim, action_dim,optimizer = None, learning_rate = 0.001, tau = 0.001, sess = None, critic = None):
        self.sess = sess
        self.Ou = OU_Process(action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.l2_reg = 0.01
        self.optimizer = optimizer or tf.train.AdamOptimizer(self.learning_rate)
        self.tau = tau
        self.h1_dim = 400
        self.h2_dim = 300
        self.activation = tf.nn.relu
        self.critic = critic
        self.input_state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name="states")
        self.action_grad = tf.placeholder(tf.float32, shape=[None, self.action_dim], name="action_grad")
        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
        self.action_output = self.__create_actor_network("actor_net")
        self.action_target_output = self.__create_actor_network("actor_target_net")
        self.source_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="actor_net")
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="actor_target_net")

        self.a_learn = self.update_actor_net()

        self.update_target_net_op_list = [self.target_vars[i].assign(self.tau*self.source_vars[i] +(1 - self.tau) * self.target_vars[i])
                                          for i in range(len(self.source_vars))]


    def __create_actor_network(self,scope):
        with tf.variable_scope(scope):
            h1 = tf.layers.dense(self.input_state, units=self.h1_dim, activation=self.activation,
                                kernel_initializer=tf.random_uniform_initializer(minval= - 1/sqrt(self.h1_dim), maxval = 1/sqrt(self.h1_dim)),
                                kernel_regularizer=self.kernel_regularizer,name="hidden_1")
            h2 = tf.layers.dense(h1, units=self.h2_dim, activation=self.activation,
                                kernel_initializer=tf.random_uniform_initializer(minval = -1 /sqrt(self.h2_dim), maxval = 1/sqrt(self.h2_dim)),
                                kernel_regularizer=self.kernel_regularizer,name="hidden_2")
            action_output = tf.layers.dense(h2, units=self.action_dim, activation=tf.nn.tanh,
                                        kernel_initializer=tf.random_uniform_initializer(minval = -3e-3, maxval = 3e-3),
                                        kernel_regularizer=self.kernel_regularizer,use_bias=False,name="action_outputs")
            return action_output

    def select_action(self, s, p=None):
        a = self.sess.run(self.action_output, feed_dict = {self.input_state: s})
        self.noise = self.Ou.return_noise()
        if p is not None:
            return a * p + self.noise * (1 - p)
        else:
            return a + self.noise

    def predict_source_action(self,s):
        a = self.sess.run(self.action_output, feed_dict = {self.input_state:s})
        return a

    def predict_target_action(self,s):
        a = self.sess.run(self.action_target_output, feed_dict = {self.input_state:s})
        return a

    def update_actor_net(self):
        self.policy_gradient = tf.gradients(self.action_output, self.source_vars, -self.action_grad)
        self.grads_and_vars = zip(self.policy_gradient, self.source_vars)
        a_learn = self.optimizer.apply_gradients(self.grads_and_vars)
        return a_learn











