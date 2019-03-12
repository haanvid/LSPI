import numpy as np
import gym
import time
import tensorflow as tf

env = gym.make('Hopper-v2')
env.reset()
no_op = np.array([0,0,0])
hi_op = np.array([1,1,1])
lo_op = np.array([-1,-1,-1])

for i in range(10000000000000000):
    env.step(no_op)
    env.render()
    time.sleep(0.05)