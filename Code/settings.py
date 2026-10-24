from pyray import *
from raylib import *
from random import randint, uniform
from os.path import join

WINDOW_WIDTH, WINDOW_HEIGHT = 1800, 980
ENEMY_SPEED = 150  # Adjust as needed
ENEMY_MAX_HEALTH = 3
ENEMY_SPAWN_INTERVAL = 3  # Time between enemy spawns
BG_COLOR = (15,10,25,255)
PLAYER_SPEED = 500
LASER_SPEED = 600
METEOR_SPEED_RANGE = [300,400]
METEOR_TIMER_DURATION = 0.4
FONT_SIZE = 120