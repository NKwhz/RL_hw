import os
import pygame
import random
import numpy as np
import copy
from pygame.locals import *
import pandas as pd
import time
from matplotlib import pyplot as plt
class Yuanyang:
    def __init__(self):
        self.window = None
        # 横竖10*10 第一个状态也就是第一个方格从零开始编码
        self.states = list(i for i in range(0, 100)) # 状态空间
        self.brick=[3, 13, 23, 33, 63, 73, 83, 93, 6, 16, 26, 36, 46, 66, 76, 86, 96]
        self.q=np.zeros((100,4))
        self.endstate=[ 15]
        # q:每个状态的累计回报期望值
        self.v = np.zeros((1,100))
        self.pai = np.ones((100, 4))/4.0
        self.terminal=[3, 13, 23, 33, 63, 73, 83, 93, 6, 16, 26, 36, 46, 66, 76, 86, 96,  15]
        self.state=None
        self.actions = ['e', 's', 'w', 'n']
        self.t = self.trans()
        self.window_size=(400,300)
        #第一个状态坐标
        self.yuan_xy = (0, 0)
        # 第就个状态坐标
        self.yang_xy = [200, 30]
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
                # 采取n,e,s,w动作后的坐标
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
                # 更新状态动作值函数
                if next_xy[0] <= 360 and next_xy[0] >= 0 and next_xy[1] <= 270 and next_xy[1] >= 0:
                    t.loc[s, a] = n_s
        return(t)
    def reward(self, state):
        # 奖赏函数 只与状态有关，而与动作无关
        r = 0
        if state in self.endstate:
            r = 100
        if state in self.brick:
            r = -100
        return r
    def step(self, state, action):
        if state in self.terminal:
            return state, 0, True
        if pd.isna(self.t[action][state]):
            next_state = state
        else:
            next_state = self.t[action][state]
        # 该回报只与状态有关，与动作无关，所以先计算状态
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
        # 表格位置与状态之间的转换函数，返回新的坐标对应的状态
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
            heart_file=current_dir + "/resource/obstacle.png"
            print (bird_file)
            self.yuan = pygame.image.load(bird_file).convert_alpha()
            self.yang = pygame.image.load(bird_file).convert_alpha()
            self.background = pygame.image.load(background_file).convert()
            self.brick_picture = pygame.image.load(obstacle_file).convert()
            self.heart=pygame.image.load(heart_file).convert()
        self.window.blit(self.background, (0, 0))
        self.window.blit(self.yang, self.yang_xy)
        for i in range(len(self.brick_y)):
            self.window.blit(self.brick_picture, (self.brick_x[i], self.brick_y[i]))
        self.window.blit(self.yuan, self.yuan_xy)
        if self.yuan_xy==self.yang_xy:
            self.window.blit(self.heart, self.heart_xy)
        pygame.display.update()
        time.sleep(0.2)

    def policy_evaluation(self):
        flag=1
        while(flag):
            v_temp=copy.deepcopy(self.v)
            self.v = np.zeros((1,100))
            flag=0
            for s in self.states:
                for action in self.actions:
                    pai=self.pai[s][self.actions.index(action)]
                    next_state, r, is_terminal=self.step(s,action)
                    v_old=0.9*v_temp[0][next_state]
                    self.v[0][s]+=pai*(r+v_old)
            for s in self.states:
                if abs(v_temp[0][s]-self.v[0][s])>0.001:
                    flag=1
                    break
        # print (self.pai[59])
        # print(self.v[0][59])
        for s in self.states:
            for action in self.actions:
                next_state, r, is_terminal = self.step(s, action)
                self.q[s][self.actions.index(action)]=r+0.9*self.v[0][next_state]
        # print(self.v)
        plt.matshow(np.reshape(self.v, (10, 10)), cmap='hot')
        plt.colorbar()
        # savefig("./results/policy_{}.jpg".format(i))
        # plt.show()
        plt.show(block=False)
        time.sleep(1)
        
        plt.close()  
        return self.v, self.q

    def policy_improvement(self):
        print (self.q[59])
        print("-----------")
        for s in self.states:
            a = np.argmax(np.array(self.q[s]))
            for action in self.actions:
                if (self.actions.index(action)==a):
                    self.pai[s][self.actions.index(action)]=1
                else:
                    self.pai[s][self.actions.index(action)]=0
        return self.pai

    def predict(self):
        self.render()
        print(self.pai)
        self.state=0
        # print (np.argmax(np.array(self.pai[0])))
        next_state, r, is_terminal = self.step(self.state, self.actions[np.argmax(np.array(self.pai[0]))])
        while(next_state not in self.terminal):
            self.state=next_state
            self.yuan_xy=self.state_to_position(next_state)
            next_state, r, is_terminal = self.step(next_state, self.actions[np.argmax(np.array(self.pai[next_state]))])
            self.render()
            print (self.state)
            print (self.pai[self.state])
        self.render()
    def get_pai(self):
        return self.pai
    def get_v(self):
        return self.v
    def get_q(self):
        return self.q

if __name__=="__main__":
    agent1 = Yuanyang()
    flag = 1
    while (flag):
        print("++++++++++++")
        flag = 0
        pai_old = copy.deepcopy(agent1.get_pai())
        v_new, q=agent1.policy_evaluation()
        agent1.policy_improvement()
        for s in agent1.states:
            for action in agent1.actions:
                if agent1.pai[s][agent1.actions.index(action)] != pai_old[s][agent1.actions.index(action)]:
                    flag = 1
                    break
    agent1.predict()
    # # agent1.print_ifo()

    # agent1.state=0
    # agent1.actions="e"
    # agent1.state, r, is_terminal=agent1.step(0,agent1.actions[0])
    # agent1.yuan_xy=agent1.state_to_position(agent1.state)
    # # print ("====")
    # # print (next_state)
    # agent1.render()
    # pai = np.ones((1, 4)) / 4.0
    # pai[0][3]=8
    # pai=np.array(pai)
    # print(np.argmax(pai))
    # print(pai[1])
    # print(pai[3][3])
    # actions = ['e', 's', 'w', 'n']
    # for i in actions:
    #     print(actions.index(i))

