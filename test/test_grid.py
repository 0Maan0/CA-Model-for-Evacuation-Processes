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
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from Grid import Grid, Pos


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.grid = Grid([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])

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


class TestGridColumnUpdates(unittest.TestCase):
    def setUp(self):
        """Initialize a sample grid for testing."""
        self.grid = Grid([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])

    def test_update_existing_column(self):
        """Test updating values in an existing column."""
        new_values = [13, 14, 15]
        for r, val in enumerate(new_values):
            self.grid[Pos(r, 1)] = val  # Updating column index 1

        updated_column = [self.grid[Pos(r, 1)] for r in range(3)]
        self.assertEqual(updated_column, new_values)

    def test_add_new_column(self):
        """Test adding a new column to the grid."""
        new_values = [16, 17, 18]
        new_col_index = 4  # Adding column 4 (previous max was 3)
        for r, val in enumerate(new_values):
            self.grid[Pos(r, new_col_index)] = val  # Adding new column

        self.assertIn(Pos(0, new_col_index), self.grid)
        self.assertIn(Pos(1, new_col_index), self.grid)
        self.assertIn(Pos(2, new_col_index), self.grid)

        updated_column = [self.grid[Pos(r, new_col_index)] for r in range(3)]
        self.assertEqual(updated_column, new_values)

    def test_get_column(self):
        """Test retrieving a column from the grid."""
        col_2 = self.grid.get_column(2)
        self.assertEqual(col_2, [3, 7, 11])  # Expected values from initial grid

    def test_update_column_method(self):
        """Test updating a column using the update_column method."""
        new_values = [20, 21, 22]
        self.grid.update_column(2, new_values)
        updated_column = [self.grid[Pos(r, 2)] for r in range(3)]
        self.assertEqual(updated_column, new_values)

    def test_update_column_invalid_size(self):
        """Test updating a column with incorrect size should raise an error."""
        new_values = [30, 31]  # Too short
        with self.assertRaises(AssertionError):
            self.grid.update_column(1, new_values)


if __name__ == '__main__':
    unittest.main()
