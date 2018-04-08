import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import os
import ddpg_agent as agent
max_learning_ep = 200
max_step_ep = 200
open_render = 0
EPISODES = 10000
TEST_EPS = 1

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    env = gym.make("Pendulum-v0")
#    env = wrappers.Monitor(env, "pendulum-v0/experiment-1", force =True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent_ = agent.DDPG_Agent(state_dim, action_dim)

    #Training
    for episode in range(EPISODES):
        print episode
        state = env.reset()
        for step in range(env.spec.timestep_limit):

            state = np.reshape(state,(1,-1))
 #           print state
            action = np.clip(agent_.choose_action(state), -2.0,2.0)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, (1, -1))
            agent_.store_memory([state,action,reward,next_state,done])
            state = next_state
            if done:
                break
        #Testing
        if episode % 1 == 0 and episode > 0:
            print "test"

            total_reward = 0
            for i in range(TEST_EPS):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    env.render()
                    state = np.reshape(state,(1,3))
                    action = agent_.predict_action(state)
                    state,reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward / TEST_EPS
            print("episode : {}, Evaluation Average Reward : {}".format(episode, avg_reward))

if __name__ == '__main__':
    main()









