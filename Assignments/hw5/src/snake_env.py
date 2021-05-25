import numpy as np
from collections import deque
import math
import time

import pygame as pg

FOOD = 100
SNAKE_HEAD = 50
SNAKE_BODY = 20


def delay(second):
    for event in pg.event.get():
        if event.type == pg.QUIT:
            exit()
    time.sleep(second)


class snake_env():
    def __init__(self):
        self.size = 8
        self.snake = deque()
        self.snake_head = []
        self.food = []

        self.screen = None
        self.screen_width = 480
        self.screen_height = 480
        self.block_size = self.screen_height / self.size
        self.bg = None

        self.action_space = 4

        self.eat_reward = 1.
        self.move_reward = 0.
        self.hit_reward = -1.

    def set_reward(self, eat_reward=1., move_reward=0., hit_reward=-1.):
        self.eat_reward = eat_reward
        self.move_reward = move_reward
        self.hit_reward = hit_reward

    def reset(self):
        self.snake = deque()
        self.food = [np.random.randint(
            self.size), np.random.randint(self.size)]
        self.snake_head = [np.random.randint(
            self.size-5) + 5, np.random.randint(self.size)]
        for i in range(4):
            self.snake.append([self.snake_head[0] - (3-i), self.snake_head[1]])

        self.pre_snake_head = [self.snake_head[0]-1, self.snake_head[1]]

        # make state
        state = np.zeros((self.size, self.size))
        state[self.food[0]][self.food[1]] = FOOD
        for snake_body in self.snake:
            state[snake_body[0]][snake_body[1]] = SNAKE_BODY
        state[self.snake_head[0]][self.snake_head[1]] = SNAKE_HEAD

        return state

    def step(self, action):
        next_state = np.zeros((self.size, self.size))
        done = False
        info = []

        if action == 0:
            self.snake_head[0] += 1
            self.snake_head[1] += 0
        elif action == 1:
            self.snake_head[0] += 0
            self.snake_head[1] += 1
        elif action == 2:
            self.snake_head[0] += -1
            self.snake_head[1] += 0
        elif action == 3:
            self.snake_head[0] += 0
            self.snake_head[1] += -1

        # hit wall
        if self.snake_head[0] >= self.size or self.snake_head[1] >= self.size:
            next_state[self.food[0]][self.food[1]] = FOOD
            for snake_body in self.snake:
                next_state[snake_body[0]][snake_body[1]] = SNAKE_BODY

            done = True
            reward = self.hit_reward
            # snake body length
            info = {'length': len(self.snake)}
            return next_state, reward*1., done, info
        elif self.snake_head[0] < 0 or self.snake_head[1] < 0:
            next_state[self.food[0]][self.food[1]] = FOOD
            for snake_body in self.snake:
                next_state[snake_body[0]][snake_body[1]] = SNAKE_BODY

            done = True
            reward = self.hit_reward

            # snake body length
            info = {'length': len(self.snake)}
            return next_state, reward*1., done, info

        # hit it self
        for snake_body in self.snake:
            if snake_body[0] == self.snake_head[0] and snake_body[1] == self.snake_head[1]:
                done = True
                reward = self.hit_reward

                next_state[self.food[0]][self.food[1]] = FOOD
                for snake_body in self.snake:
                    next_state[snake_body[0]][snake_body[1]] = SNAKE_BODY
                next_state[self.snake_head[0]][self.snake_head[1]] = SNAKE_HEAD

                # snake body length
                info = {'length': len(self.snake)}
                return next_state, reward*1., done, info

        # nothing
        self.snake.append([self.snake_head[0], self.snake_head[1]])

        # eat food
        if self.snake_head[0] == self.food[0] and self.snake_head[1] == self.food[1]:
            reward = self.eat_reward
            self.food = [np.random.randint(
                self.size), np.random.randint(self.size)]
        else:
            if self.move_reward == "dist" or self.move_reward == "DIST":
                reward = 1. / \
                    math.sqrt(
                        (self.snake_head[0]-self.food[0])**2. + (self.snake_head[1]-self.food[1])**2.)
            elif self.move_reward == "-dist" or self.move_reward == "-DIST":
                reward = -math.sqrt((self.snake_head[0]-self.food[0])**2. + (
                    self.snake_head[1]-self.food[1])**2.)/(self.size*self.size*1.)
            else:
                reward = self.move_reward
            #reward = 0.
            #reward = -0.1

            self.snake.popleft()

        # state
        next_state[self.food[0]][self.food[1]] = FOOD
        for snake_body in self.snake:
            next_state[snake_body[0]][snake_body[1]] = SNAKE_BODY
        next_state[self.snake_head[0]][self.snake_head[1]] = SNAKE_HEAD

        # snake body length
        info = {'length': len(self.snake)}

        delay(0)

        return next_state, reward*1., done, info

    def draw_board(self, board):
        pg.display.set_caption("Snake RL")
        bg = pg.Surface(self.screen.get_size())
        bg = bg.convert()
        bg.fill((255, 255, 255))

        block_size = self.block_size
        for j in range(self.size):
            for i in range(self.size):
                if board[i][j] == 0:
                    pg.draw.rect(bg, (100, 100, 100),
                                 [i*block_size, j*block_size, block_size, block_size], 1)
                elif board[i][j] == SNAKE_BODY:
                    pg.draw.rect(bg, (0, 0, 255),
                                 [i*block_size, j*block_size, block_size, block_size], 0)
                elif board[i][j] == SNAKE_HEAD:
                    pg.draw.rect(bg, (255, 0, 0),
                                 [i*block_size, j*block_size, block_size, block_size], 0)
                elif board[i][j] == FOOD:
                    pg.draw.rect(bg, (0, 0, 0),
                                 [i*block_size, j*block_size, block_size, block_size], 0)
                else:
                    print(board[i][j])
                    print("ERROR")
        self.screen.blit(bg, (0, 0))

    def render(self, value=None):
        if self.screen == None:
            pg.init()
            pg.font.init()
            self.myfont = pg.font.SysFont('Comic Sans MS', 30)
            # 設定視窗
            self.screen = pg.display.set_mode(
                (self.screen_width, self.screen_height))
            pg.display.set_caption("Snake RL")
            pg.display.update()

        # make state
        state = np.zeros((self.size, self.size))
        state[self.food[0]][self.food[1]] = FOOD
        for snake_body in self.snake:
            state[snake_body[0]][snake_body[1]] = SNAKE_BODY
        state[self.snake_head[0]][self.snake_head[1]] = SNAKE_HEAD

        self.draw_board(state)

        if value != None:
            textsurface = self.myfont.render(
                'Value:' + str(value), False, (0, 0, 0))
            self.screen.blit(textsurface, (0, 0))
        # print(state)

        pg.display.flip()

    def close(self):
        pg.quit()
