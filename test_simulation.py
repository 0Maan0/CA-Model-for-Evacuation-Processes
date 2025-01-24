'''
run as pytest test_simulation.py in terminal
'''
import pytest 
from pjotr import Grid, Pos, FFCA, to_corridor, EMPTY, OBSTACLE, EXIT, AGENT_1, AGENT_2

def test_to_corridor():
    created_corridor = to_corridor(5, 5)
    expected_corridor = "#######\nE.....E\nE.....E\nE.....E\nE.....E\nE.....E\n#######"
    assert created_corridor == expected_corridor, f"Expected {expected_corridor}, but got {created_corridor}"

def test_grid():
    grid = Grid(5, 5)
    assert grid.grid == [[EMPTY for _ in range(5)] for _ in range(5)], f"Expected {[[EMPTY for _ in range(5)] for _ in range(5)]}, but got {grid.grid}"
test_to_corridor()
