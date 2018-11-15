from bird_class import Game
import pygame
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
from mpl_toolkits.axes_grid1 import AxesGrid
import time

MAX_STEP = 10000
action_size = 4
state_size = 100
N = 20

def next_state(s, a):
    if a == 'w':
        if s % 10 > 0:
            s -= 1
    elif a == 'e':
        if s % 10 < 9:
            s += 1
    elif a == 's':
        if s // 10 < 9:
            s += 10
    elif a == 'n':
        if s // 10 > 0:
            s -= 10
    return s

class Agent:
    def __init__(self, n):
        '''
        n is the max number of iterations
        '''
        # self.policy = [0.25, 0.25, 0.25, 0.25]    # probability for every direction, w, e, s, n
        self.game = Game(MAX_STEP)
        self.actions = self.game.actions
        self.n = n
        # self.k = k
        self.gamma = 0.9
        self.state_policy = np.full((state_size, action_size), 0.25)
        self.state_reward = np.zeros(state_size)
        
    def sample(self, state):
        m = random.random()
        s = 0
        for i in range(len(self.state_policy[0])):
            s += self.state_policy[state, i]
            if m <= s:
                return i

    def rewarding(self, s):
        if s in self.game.brick_offset:
            return self.game.brick_reward + self.game.move_reward
        elif s == self.game.female_pos:
            return self.game.success_reward + self.game.move_reward
        else:
            return self.game.move_reward


    def train(self):
        record = []
        for i in range(self.n):
            # self.state_reward = np.zeros(state_size)
            # plt.matshow(self.state_policy, cmap='hot')
            # plt.colorbar()
            # plt.show()   
            # for j in range(self.k):
            temp = np.zeros(state_size)
            for s in range(state_size):
                if s in self.game.brick_offset:
                    temp[s] = self.game.brick_reward
                    continue
                elif s == self.game.female_pos:
                    temp[s] = self.game.success_reward
                    continue
                # for index in range(action_size):
                max_a = -1
                max_v = -np.Inf
                for a in range(len(self.actions)):
                    action = self.actions[a]
                    s_ = next_state(s, action)
                    v_s_ = self.state_reward[s_]
                    if v_s_ > max_v and s_ != s:
                        max_v = v_s_
                        max_a = a
                # index = max_a
                # a = self.actions[index]
                # p = self.state_policy[s, index]
                # self.game.initialization(start=s, display=False)
                # r = self.game.move(direct=a, display=False)
                # # if s == 8:
                # #     print(r)
                # s_ = next_state(s, a)
                r = self.game.move_reward
                temp[s] += r + self.gamma * max_v
            self.state_reward = temp
            record.append(np.reshape(self.state_reward, (10, 10)))
            plt.matshow(np.reshape(self.state_reward, (10, 10)), cmap='hot')
            plt.colorbar()
            # plt.show()
            plt.show(block=False)
            time.sleep(1)
            plt.close()   
            # print(self.state_reward)
        for s in range(self.state_policy.shape[0]):
            max_a = -1
            max_v = -np.Inf
            for a in range(len(self.actions)):
                action = self.actions[a]
                s_ = next_state(s, action)
                v_s_ = self.state_reward[s_]
                if v_s_ > max_v and s_ != s:
                    max_v = v_s_
                    max_a = a
            for a in range(len(self.actions)):
                if a == max_a:
                    self.state_policy[s, a] = 1
                else:
                    self.state_policy[s, a] = 0
        fig = plt.figure()

        grid = AxesGrid(fig, 111,
                        nrows_ncols=(4, 5),
                        axes_pad=0.05,
                        share_all=True,
                        label_mode="L",
                        cbar_location="right",
                        cbar_mode="single",
                        )

        for val, ax in zip(record,grid):
            im = ax.imshow(val, vmin=self.game.brick_reward, vmax=self.game.success_reward)

        grid.cbar_axes[0].colorbar(im)

        # for cax in grid.cbar_axes:
        #     cax.toggle_label(False)
        savefig("./results/value.jpg")
        plt.show(block=False)
        time.sleep(5)
        plt.close()
            

    def test(self):
        self.game.initialization()
        
        while True:
            time.sleep(1)
            s = self.game.male_pos
            a = self.state_policy[s]
            a = np.argmax(a)
            a = self.actions[a]
            r = self.game.move(a)
            print(self.state_policy[s])
            if self.game.male_pos == self.game.female_pos or s in self.game.brick_offset:
                break
        

if __name__ == '__main__':
    agent = Agent(N)
    agent.train()
    agent.test()
    time.sleep(5)
    pygame.display.quit()
    pygame.quit()
