import random
import torch
import pygame

def dont_move():
    return 0, 0

def move_southeast():
    return 1, 1

def move_random():
    return tuple(random.choices([-1, 0, 1], weights = [1, 3, 1], k = 2))

def controlled_action():
    keys = pygame.key.get_pressed()
    dx, dy = 0, 0
    if keys[pygame.K_LEFT]:
        dx += -1
    if keys[pygame.K_RIGHT]:
        dx += 1
    if keys[pygame.K_UP]:
        dy += -1
    if keys[pygame.K_DOWN]:
        dy += 1
    return dx, dy
