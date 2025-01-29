"""
University      University of Amstedam
Course:         Complex systems simulation
Authors:        Pjotr Piet, Maan Scipio, Sofia Tete
IDs:            12714933, 15899039, 15830608
Description:    This file contains the testing for the base classes used in the
FFCA class. This FFCA class is based on the Grid class and the Pos class, so
this class contains the testing for these base classes. This ensures correct
"""

import unittest
from Grid import Grid, Pos


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.grid = Grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_initialization(self):
        self.assertEqual(self.grid[Pos(0, 0)], 1)
        self.assertEqual(self.grid[Pos(1, 1)], 5)
        self.assertEqual(self.grid[Pos(2, 2)], 9)

    def test_setitem_getitem(self):
        self.grid[Pos(0, 1)] = 10
        self.assertEqual(self.grid[Pos(0, 1)], 10)

    def test_contains(self):
        self.assertIn(Pos(0, 0), self.grid)
        self.assertNotIn(Pos(10, 10), self.grid)

    def test_remove(self):
        self.grid.remove({5})
        self.assertNotIn(Pos(1, 1), self.grid)

    def test_find(self):
        self.assertEqual(self.grid.find(5), Pos(1, 1))
        self.assertIsNone(self.grid.find(100))

    def test_findall(self):
        self.grid[Pos(2, 2)] = 5
        self.assertEqual(set(self.grid.findall(5)), {Pos(1, 1), Pos(2, 2)})

    def test_copy(self):
        grid_copy = self.grid.copy()
        self.assertEqual(grid_copy[Pos(0, 0)], self.grid[Pos(0, 0)])
        self.assertNotEqual(id(grid_copy), id(self.grid))


if __name__ == '__main__':
    unittest.main()
