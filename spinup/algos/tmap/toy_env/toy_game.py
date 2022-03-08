# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud
A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia
* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding
from matplotlib.pyplot import savefig
import numpy as np
import matplotlib
matplotlib.use('Agg')


def normalization(data):
    return data + abs(np.min(data))


class ToyEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_action = -0.01
        self.max_action = 0.01
        self.low_state = -1
        self.high_state = 1
        self.viewer = None
        self.Points = [np.array([-0.2, -0.2]),
                       np.array([0.2, 0.2]), np.array([-0.5, 0.5])]
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(2,),
            dtype=np.float32
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        
        return [seed]

    def norm(self, x1, y1, x2, y2):
        return (np.abs(x1-x2)**2+np.abs(y1-y2)**2)**(1./2)

    def reward_function(self, agent_pos):
        sum1_x = 0
        sum1_y = 0
        for i in range(len(self.Points)):
            sum1_x += 1/3*self.Points[i][0]
            sum1_y += 1/3*self.Points[i][1]
        pos1 = np.array([sum1_x, sum1_y])
        dis2points = [self.norm(
            each[0], each[1], agent_pos[0], agent_pos[1]) for each in self.Points]
        # print(10*np.linalg.norm(pos1-agent_pos))
        # print(dis2points)
        # print(np.sum(dis2points))
        reward = 2*np.linalg.norm(pos1-agent_pos) - 1*np.sum(dis2points) + 1.8
        return reward

    def clip(self, pos):
        if pos[0] > self.high_state:
            pos[0] = self.high_state
        if pos[0] < self.low_state:
            pos[0] = self.low_state
        if pos[1] > self.high_state:
            pos[1] = self.high_state
        if pos[1] < self.low_state:
            pos[1] = self.low_state
        return pos

    def step(self, action):
        action = np.array(action)
        assert action.shape[0]==2
        
        position = np.array([self.state[0]+action[0], self.state[1]+action[1]])
        position = np.around(position, 3)
        position = self.clip(position)

        # Convert a possible numpy bool to a Python bool.
        # position = np.array([0.2, 0.2])
        done = False
        for i in range(len(self.Points)):
            if (position[0] == self.Points[i][0] and position[1] == self.Points[i][1]):
                done = True

        reward = self.reward_function(position)

        self.state = np.array(position)
        return self.state, reward, done, {}

    def reset(self):
        bkup1 = np.array([self.np_random.uniform(-1,-0.75),self.np_random.uniform(-1,-0.5)])
        #bkup2 = np.array([self.np_random.uniform(0.5,1),self.np_random.uniform(-1,-0.6)])
        bkup3 = np.array([self.np_random.uniform(0.6,1),self.np_random.uniform(0.75,1)])
        bkup4 = np.array([self.np_random.uniform(-1,-0.75),self.np_random.uniform(0.75,1)])
        state_bkups = [bkup1, bkup3, bkup4]
        self.state_choose = self.np_random.choice(3,1)
        self.state = state_bkups[self.state_choose[0]]
        #self.state = bkup2
        self.state = np.around(self.state, 3)
        return np.array(self.state)


if __name__ == '__main__':
    a = ToyEnv()
    a.reset()
    act = np.array([-0.2, -0.2])
    print(a.reward_function(act))
