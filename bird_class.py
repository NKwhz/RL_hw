import pygame
import os
import sys
import time
import math

current_dir = os.getcwd()
bird_file = os.path.join(current_dir, "resource/bird.png")
obstacle_file = os.path.join(current_dir, "resource/obstacle.png")
background_file = os.path.join(current_dir, "resource/background.png")

MAX_STEP = 1000
SCREEN = (400, 400)

def load_img(p):
    obj = pygame.image.load(p).convert_alpha()
    return obj

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

class Game:
    def __init__(self, max_step, screen=SCREEN, bird_p=bird_file, o_p=obstacle_file, back_p=background_file):
        self.max_step = max_step
        self.screen_size = screen
        self.viewer = pygame.display.set_mode(self.screen_size, 0, 32)
        self.bird = load_img(bird_p)
        self.obstacle = load_img(o_p)
        self.background = load_img(back_p)
        self.brick_offset = {3, 6, 13, 16, 23, 26, 33, 36, 46, 63, 73, 76, 83, 86, 93, 96}
        self.actions = ['w', 'e', 's', 'n']
        self.bird_co = (0, 5)
        self.bird_offset = 0
        self.male_pos = 0
        self.female_pos = 9
        self.brick_reward = -1000
        self.success_reward = 1000
        self.move_reward = -1
        self.terminal = False

    def initialization(self, start=0, display=True):
        self.male_pos = start
        if display:
            self.scene_draw()
            self.bird_draw(start)
            pygame.display.update()

    def scene_draw(self):
        self.viewer.blit(self.background, (0, 0))
        for o in self.brick_offset:
            col = o % 10
            row = o // 10
            self.viewer.blit(self.obstacle, (col * 40, row * 2 * 20))
            self.viewer.blit(self.obstacle, (col * 40, (row * 2 + 1) * 20))
        self.bird_draw(self.female_pos)

    def bird_draw(self, offset):
        col = offset % 10
        row = offset // 10
        coo = (col * 40 + self.bird_co[0], row * 40 + self.bird_co[1])
        self.viewer.blit(self.bird, coo)

    def is_terminal(self):
        if self.male_pos in self.brick_offset or self.male_pos == self.female_pos:
            return True
        else:
            return False

    def move(self, direct, display=True):

        if direct == 'w':
            if self.male_pos % 10 > 0:
                self.male_pos -= 1
            else:
                return -1
        elif direct == 'e':
            if self.male_pos % 10 < 9:
                self.male_pos += 1
            else:
                return -1
        elif direct == 's':
            if self.male_pos // 10 < 9:
                self.male_pos += 10
            else:
                return -1
        elif direct == 'n':
            if self.male_pos // 10 > 0:
                self.male_pos -= 10
            else:
                return -1
        if display:
            self.scene_draw()
            self.bird_draw(self.male_pos)

            pygame.display.update()    

        # return rewards    
        if self.male_pos in self.brick_offset:
            # self.initialization()
            return -1
        elif self.male_pos == self.female_pos:
            return 1
        
        else:
            return 0
        # return self.move_reward
        
if __name__ is "__main__":
    g = Game(MAX_STEP, (400, 400), bird_file, obstacle_file, background_file)
    g.initialization()
    # d = input()
    time.sleep(3)
    g.move('s')
    time.sleep(3)
    pygame.display.quit()
    pygame.quit()
    pass

    