"""
University      University of Amstedam
Course:         Complex systems simulation
Authors:        Pjotr Piet, Maan Scipio, Sofia Tete
IDs:            12714933, 15899039, 15830608
Description:    This file imports the FFCA class and runs some standard
scenarios to see the visually see the FFCA interactions.
"""

from FFCA import FFCA
from Grid import Grid, Pos
import time


def test_collision():
    # test agents
    agents = [(Pos(1, 1), 1), (Pos(1, 5), 2)]
    ffca = FFCA(1, 5, 0, agents)
    steps = 10
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        ffca.show()


# test_collision()


def test_small():
    cmax = 40
    rmax = 3
    r = 2
    c = cmax
    left_agents = [(Pos(r, 1 + 2 * i), 1) for i in range(2)]
    right_agents = [(Pos(r, c - 2 * i), 2) for i in range(10)]
    agents = left_agents + right_agents

    ffca = FFCA(rmax, cmax, 0, agents)
    steps = 30
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        ffca.show()


# test_small()


def test_big():
    ffca = FFCA(10, 50, 20, None, spawn_rate=0.03)
    steps = 100
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        ffca.show()


test_big()


def run():
    ffca = FFCA(10, 100, 50)
    ffca.show()
    steps = 1000
    for i in range(steps):
        time.sleep(0.05)
        ffca.step()
        ffca.show()


# run()
