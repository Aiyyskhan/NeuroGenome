import math
import pygame
import numpy as np

# general settings
NUM_PLAYERS = 100
NUM_RAYS = 6
TILE = 70
MAX_EPOCH = 50000

MAP_0 = [
    '111111111111111',
    '1.....111....21',
    '1.111..11.11121',
    '1....1..1...111',
    '1111.11.111...1',
    '1....1...1.11.1',
    '1.11.111......1',
    '111111111111111'
]

# game settings
WIDTH = TILE * len(MAP_0[0])
HEIGHT = TILE * len(MAP_0)

# info panel settings
INFO_PANEL_PADDING_LEFT = TILE * 6 + 2
INFO_PANEL_PADDING_TOP = 2
INFO_PANEL_WIDTH = TILE * 3 - 4
INFO_PANEL_HEIGHT = TILE * 2 - 4

# ray casting settings
FOV = math.pi / 2
MAX_DEPTH = WIDTH # // 2
DELTA_ANGLE = FOV / (NUM_RAYS - 1)

# player settings
PLAYER_SPEED = 1
PLAYER_ANGULAR_VELOCITY = 0.2

# colors
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (220,0,0)
GREEN = (0,80,0)
BLUE = (0,0,255)
DARKGRAY = (40,40,40)
PURPLE = (120,0,120)
SKYBLUE = (0,186,255)
YELLOW = (220,220,0)

BACK_COLOR_1 = (63,79,34)
WALL_COLOR_1 = (69,124,144)
WALL_COLOR_2 = (186,136,61)

PLAYER_COLOR_1 = (126,130,45)
RAYS_COLOR_1 = (70,90,80)

TEXT_BACK_COLOR_1 = (10,60,80)
TEXT_COLOR_1 = (253,251,252)