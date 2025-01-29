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
from Grid import Pos


class TestPos(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(Pos(1, 2) + Pos(3, 4), Pos(4, 6))
        self.assertEqual(Pos(0, 0) + Pos(-1, -1), Pos(-1, -1))
        self.assertEqual(Pos(5, 7) + (3, 2), Pos(8, 9))

    def test_subtraction(self):
        self.assertEqual(Pos(5, 5) - Pos(2, 3), Pos(3, 2))
        self.assertEqual(Pos(0, 0) - Pos(1, 1), Pos(-1, -1))

    def test_multiplication(self):
        self.assertEqual(Pos(2, 3) * Pos(4, 5), Pos(8, 15))
        self.assertEqual(Pos(6, 7) * (1, 2), Pos(6, 14))

    def test_floordiv(self):
        self.assertEqual(Pos(8, 9) // Pos(2, 3), Pos(4, 3))
        self.assertEqual(Pos(10, 10) // (5, 2), Pos(2, 5))

    def test_mod(self):
        self.assertEqual(Pos(10, 15) % Pos(3, 4), Pos(1, 3))
        self.assertEqual(Pos(7, 8) % (2, 5), Pos(1, 3))

    def test_comparison(self):
        self.assertTrue(Pos(1, 2) < Pos(2, 3))
        self.assertFalse(Pos(3, 3) < Pos(3, 2))
        self.assertTrue(Pos(1, 1) == Pos(1, 1))
        self.assertFalse(Pos(2, 2) == Pos(2, 3))

    def test_hash(self):
        pos_set = {Pos(1, 2), Pos(3, 4)}
        self.assertIn(Pos(1, 2), pos_set)
        self.assertNotIn(Pos(5, 6), pos_set)


if __name__ == '__main__':
    unittest.main()