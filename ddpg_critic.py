import tensorflow as tf
from math import sqrt

GAMMA = 0.1
LR_C = 0.0001

class DDPG_Critic(object):
    def __init__(self, state_dim, action_dim, optimizer =None,learning_rate= 0.001,tau =0.001, scope = "", sess = None):
        self.scope = scope
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.l2_reg = 0.01
        self.optimizer = optimizer or tf.train.AdamOptimizer(self.learning_rate)
        self.tau = tau
        self.h1_dim = 400
        self.h2_dim = 100
        self.h3_dim = 300
        self.input_state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name="states")
        self.input_action = tf.placeholder(tf.float32, shape=[None, self.action_dim], name="actions")
        self.activation = tf.nn.relu
        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
        self.y = tf.placeholder(tf.float32, shape = [None, 1], name = "y")

        self.critic_output = self.__create_critic_network("critic_net")
        self.critic_target_output = self.__create_critic_network("critic_target_net")

        self.source_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "critic_net")
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "critic_target_net")

        c_loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.critic_output)
        self.c_learn = tf.train.AdamOptimizer(LR_C).minimize(c_loss, var_list=self.source_vars)


        self.update_target_net_op_list = [self.target_vars[i].assign(self.tau * self.source_vars[i] + (1 - self.tau) * self.target_vars[i])
                                     for i in range(len(self.source_vars))]

    def __create_critic_network(self,scope):
        with tf.variable_scope(scope):
            h1 = tf.layers.dense(self.input_state, units = self.h1_dim, activation = self.activation,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-1/sqrt(self.h1_dim), maxval=1/sqrt(self.h1_dim)),
                                 kernel_regularizer=self.kernel_regularizer,
                                 name = "hidden_1")
            h2 = tf.layers.dense(self.input_action,units = self.h2_dim, activation = self.activation,
                                 kernel_initializer=tf.random_uniform_initializer(minval=-1/sqrt(self.h2_dim), maxval=1/sqrt(self.h2_dim)),
                                 kernel_regularizer=self.kernel_regularizer,
                                 name = "hidden_2")
            h_concat = tf.concat([h1, h2], 1, name="h_concat")
            h3 = tf.layers.dense(h_concat, units=self.h3_dim, activation=self.activation,
                                 kernel_initializer=tf.random_uniform_initializer(minval = -3e-3, maxval = 3e-3),
                                 kernel_regularizer=self.kernel_regularizer,
                                 name="hidden_3")
            q_output = tf.layers.dense(h3, units=1, activation=None,name="q_output")
            return q_output

    def output_q(self, s_batch, a_batch,flag):
        if flag == 0:
            critic_q = self.sess.run(self.critic_output, feed_dict = {self.input_state:s_batch, self.input_action:a_batch})
        else:
            critic_q = self.sess.run(self.critic_target_output, feed_dict = {self.input_state:s_batch, self.input_action: a_batch})
        return critic_q

    def get_action_grad_op(self,s_batch,a_batch):
        self.get_action_grad = tf.reshape(tf.gradients(self.critic_output, self.input_action),(-1,self.action_dim))

#        self.get_action_grad_ = tf.reduce_sum(self.get_action_grad,1)
#        a = self.sess.run(self.get_action_grad, feed_dict = {self.input_state:s_batch, self.input_action:a_batch})
        action_grad = self.sess.run(self.get_action_grad, feed_dict = {self.input_state:s_batch, self.input_action:a_batch})
        return action_grad

