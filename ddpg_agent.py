import tensorflow as tf
import numpy as np
from collections import deque
import ddpg_actor as actor
import ddpg_critic as critic
import random
buffer_size = 10e6
batch_size = 64


class DDPG_Agent(object):
    def __init__(self,state_dim, action_dim):
        self.memory = deque(maxlen=buffer_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = 0.999
        self.modelpath = './model/'
        self.modelpath_ = './model/model.ckpt'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.critic_ = critic.DDPG_Critic(state_dim, action_dim, sess = self.sess)
        self.actor_ = actor.DDPG_Actor(state_dim, action_dim, sess=self.sess,critic = self.critic_)
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.modelpath_)
        if ckpt and ckpt.model_checkpoint_path:
            print "old vars"
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print "new vars"


    def choose_action(self,state):
        return self.actor_.select_action(state)

    def predict_action(self,state):
        return self.actor_.predict_source_action(state)

    def store_memory(self,transition):
        self.memory.append(transition)
        if len(self.memory) > buffer_size:
            self.memory.popleft()
        elif len(self.memory) > batch_size:
            batch = random.sample(self.memory,batch_size)
            s_batch, a_batch, r_batch, next_s_batch, done_batch = self.get_transition_batch(batch)
            target_action_batch = self.actor_.predict_target_action(next_s_batch)
            target_critic = self.critic_.output_q(next_s_batch, target_action_batch, 1)
            y_batch = r_batch + self.discount_factor * target_critic * (1 - done_batch)
            self.sess.run(self.critic_.c_learn,feed_dict = {self.critic_.input_state:s_batch, self.critic_.input_action:a_batch, self.critic_.y:y_batch})
            action_batch_for_grad = self.actor_.predict_source_action(s_batch)
            action_grad_batch = self.critic_.get_action_grad_op(s_batch,action_batch_for_grad)
 #           self.sess.run(self.actor_.update_actor_net(action_grad),feed_dict={self.actor_.input_state:s_batch, self.actor_.action_grad:action_grad})
            self.sess.run(self.actor_.a_learn,feed_dict = {self.actor_.input_state:s_batch, self.actor_.action_grad:action_grad_batch})
            self.sess.run(self.actor_.update_target_net_op_list)
            self.sess.run(self.critic_.update_target_net_op_list)


    def get_transition_batch(self,batch):
        transpose_batch = list(zip(*batch))
        s_batch = np.vstack(transpose_batch[0])
        a_batch = np.vstack(transpose_batch[1])
        r_batch = np.vstack(transpose_batch[2])
        next_s_batch = np.vstack(transpose_batch[3])
        done_batch = np.vstack(transpose_batch[4])
        return s_batch, a_batch, r_batch, next_s_batch, done_batch

    def save_model(self,step):
        self.saver.save(self.sess, self.modelpath_, global_step=step)
