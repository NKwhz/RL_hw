from bird_class import Game
import pygame
import os
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
from mpl_toolkits.axes_grid1 import AxesGrid
import time
import math
from bird_class import next_state

MAX_STEP = 100
action_size = 4
state_size = 100
Iter = 20000
length = 10000
N = 20
Epsilon = 0.3

current_dir = os.getcwd()
image_paths = [os.path.join(current_dir, "resource/w.png"), 
os.path.join(current_dir, "resource/e.png"), 
os.path.join(current_dir, "resource/s.png"), 
os.path.join(current_dir, "resource/n.png")]

def load_img(p):
    obj = pygame.image.load(p)
    obj = pygame.transform.scale(obj, (40, 40))
    obj = obj.convert_alpha()
    return obj

class Agent:
    def __init__(self):
        self.game = Game(MAX_STEP)
        self.v = np.zeros(state_size)
        self.reward = 10
        self.move_penal = -1
        self.policy = np.full((state_size, action_size), 0.25)
        self.states = []
        self.actions = []
        self.rewards = []
        self.gamma = 0.9
        self.update = set()
        self.seen = set()
        self.epsilon = Epsilon
        self.arrows = [load_img(path) for path in image_paths]

        self.reach_end = 0

        self.g = np.zeros((state_size, action_size))
        self.count = np.zeros((state_size, action_size))

    def step(self, a):
        r = self.reward * self.game.move(a, display=False) + self.move_penal
        return r, self.game.male_pos

    def sample_action(self, state):
        m = random.random()
        s = 0
        for i in range(action_size):
            s += self.policy[state, i]
            if m <= s:
                return int(i)

    def sample_tra(self):
        # print("Sampling trajectory...")
        self.states = []
        self.actions = []
        self.rewards = []
        self.update = set()

        init_pos = 0
        self.game.initialization(init_pos, display=False)
        if not self.game.is_terminal():
            self.states.append(init_pos)
            terminal = False
            s = init_pos
            cnt = 0
            while not terminal:
                a = self.sample_action(s)
                self.actions.append(a)
                r, s = self.step(self.game.actions[a])
                if s == self.game.female_pos:
                    self.reach_end += 1
                self.rewards.append(r)
                cnt += 1
                if self.game.is_terminal() or cnt == self.game.max_step:
                    break
                else:
                    self.states.append(s)
        # print("Sampled {} steps.".format(len(self.states)))

    def visualization(self):
        game = self.game
        game.scene_draw()
        for s in range(state_size):
            ind = np.argmax(self.g[s])
            col = s % 10
            row = s // 10
            coo = (col * 40 , row * 40)
            game.viewer.blit(self.arrows[ind], coo)
        pygame.display.update()

    def train(self, first=True):
        for i in range(Iter):
            if (i + 1) % 1000 == 0:
                print("{} iterations finished.".format(i + 1))
                # self.epsilon = max(self.epsilon - Epsilon / (Iter / 1000), 0)
                # print(self.epsilon)
                print("{} successs trajectories.".format(self.reach_end))
                self.visualization()
                time.sleep(3)
            self.sample_tra()
            seen = set()
            seen_states = set()
            for j in range(len(self.states)):
                if (self.states[j], self.actions[j]) not in seen:
                    self.update.add(j)
                    seen.add((self.states[j], self.actions[j]))
                    seen_states.add(self.states[j])
            g = 0
            
            for j in range(len(self.states) - 1, -1, -1):
                g = self.rewards[j] + self.gamma * g
                if g == math.inf:
                    break
                if j in self.update:
                    g_temp = self.g[self.states[j], self.actions[j]]
                    c = self.count[self.states[j], self.actions[j]]
                    self.g[self.states[j], self.actions[j]] = (g_temp * c + g) / (c + 1)
                    self.count[self.states[j], self.actions[j]] += 1

            # policy update
            for s in seen_states:
                a = np.argmax(self.g[s])
                for j in range(action_size):
                    if j == a:
                        self.policy[s, j] = 1 - self.epsilon * (action_size - 1) / action_size
                    else:
                        self.policy[s, j] = self.epsilon / action_size


    def test(self):
        self.game.initialization()

        while True:
            time.sleep(1)
            s = self.game.male_pos
            a = self.sample_action(s)
            # a = np.argmax()
            a = self.game.actions[a]
            r = self.game.move(a)
            print(self.policy[s])
            if self.game.is_terminal():
                break

if __name__ == '__main__':
    agent = Agent()
    agent.train()
    agent.test()