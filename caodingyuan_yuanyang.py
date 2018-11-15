import os
import pygame
import random
import numpy as np
from pygame.locals import *
import pandas as pd
import time
class Yuanyang:
    def __init__(self):
        self.window = None
        self.states = list(i for i in range(0, 100)) # 状态空间
        self.brick=[3, 13, 23, 33, 63, 73, 83, 93, 6, 16, 26, 36, 46, 66, 76, 86, 96]
        self.endstate=[ 9]
        self.terminal=[3, 13, 23, 33, 63, 73, 83, 93, 6, 16, 26, 36, 46, 66, 76, 86, 96,  9]
        self.state=None
        self.actions = ['e', 's', 'w', 'n']
        self.t = self.trans()
        self.window_size=(400,300)
        self.yuan_xy = (0, 0)
        self.yang_xy = [360, 0]
        self.heart_xy=[360,0]
        self.brick_size = [40, 20]
        self.brick_x = [120,240]*12
        self.brick_y = [0]*2+[20]*2+[40]*2+[60]*2+[80]*2+[100]*2+[180]+[120]+[200]*2+[220]*2+[240]*2+[260]*2+[280]*2
    def print_ifo(self):
        allstates=[i for i in self.states if i not in [3, 13, 23, 33, 63, 73, 83, 93, 6, 16, 26, 36, 46, 66, 76, 86, 96]]
        print(allstates,len(allstates))
        print(self.brick_x)
        print(self.brick_y)
        print(self.t)
        for state in self.states:
            print(self.reward(state))
    def trans(self):
        t=pd.DataFrame(data=None, index=self.states, columns=self.actions)
        for s in self.states :
            if s in self.terminal:continue
            for a in self.actions:
                next_xy = [0, 0]
                if a == "n":
                    next_xy[0] = self.state_to_position(s)[0]
                    next_xy[1] = self.state_to_position(s)[1] - 30
                elif a == "e":
                    next_xy[0] = self.state_to_position(s)[0] + 40
                    next_xy[1] = self.state_to_position(s)[1]
                elif a == "s":
                    next_xy[0] = self.state_to_position(s)[0]
                    next_xy[1] = self.state_to_position(s)[1] + 30
                elif a == "w":
                    next_xy[0] = self.state_to_position(s)[0] - 40
                    next_xy[1] = self.state_to_position(s)[1]
                n_s = self.position_to_state(next_xy)
                if next_xy[0] <= 360 and next_xy[0] >= 0 and next_xy[1] <= 270 and next_xy[1] >= 0:
                    t.loc[s, a] = n_s
        return(t)
    def reward(self, state):
        r = 0.0
        if state in self.endstate:
            r = 1.0
        if state in self.brick:
            r = -1.0
        return r
    def step(self, state, action):
        if state in self.terminal:
            return state, 0, True
        if pd.isna(self.t[action][state]):
            next_state = state
        else:
            next_state = self.t[action][state]
        r=self.reward(next_state)    # 计算回报
        is_terminal = False
        if next_state in self.terminal :
            is_terminal = True
        return next_state, r, is_terminal
    def reset(self):
        state = random.sample(self.states,1)
        return state
    def state_to_position(self, state):
        i = int(state / 10)
        j = state % 10
        position = [0, 0]
        position[0] = 40 * j
        position[1] = 30 * i
        return position
    def position_to_state(self, position):
        i = position[0] / 40
        j = position[1] / 30
        return int(i + 10 * j)
    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size, 0, 32)
            current_dir = os.path.split(os.path.realpath(__file__))[0]
            print(current_dir)
            bird_file = current_dir + "/resource/bird.png"
            obstacle_file = current_dir + "/resource/obstacle.png"
            background_file = current_dir + "/resource/background.png"
            heart_file=current_dir + "/resource/heartttt.jpeg"
            self.yuan = pygame.image.load(bird_file).convert_alpha()
            self.yang = pygame.image.load(bird_file).convert_alpha()
            self.background = pygame.image.load(background_file).convert()
            self.brick = pygame.image.load(obstacle_file).convert()
            # self.heart=pygame.image.load(heart_file).convert()
        self.window.blit(self.background, (0, 0))
        self.window.blit(self.yang, self.yang_xy)
        for i in range(len(self.brick_y)):
            self.window.blit(self.brick, (self.brick_x[i], self.brick_y[i]))
        self.window.blit(self.yuan, self.yuan_xy)
        if self.yuan_xy==self.yang_xy:
            self.window.blit(self.heart, self.heart_xy)
        pygame.display.update()
        time.sleep(0.1)
if __name__=="__main__":
    agent1 = Yuanyang()
    agent1.print_ifo()
    agent1.render()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
