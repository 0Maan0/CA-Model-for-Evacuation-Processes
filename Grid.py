"""
University      University of Amstedam
Course:         Complex systems simulation
Authors:        Pjotr Piet, Maan Scipio, Sofia Tete
IDs:            12714933, 15899039, 15830608
Description:    This file contains the underlying classes to implement our
FFCA. The Pos class is a class that represents a 2d position. This class also
contains the implementation of standard operations on these positions. This
makes the use of 2d positions much more convenient.

Secondly, this file also contains the Grid class. This class represents a 2d
grid using a dictionary: Dict[Pos] -> float. A grid also contains the width
breadth boundaries of the grid. These are used to determine the size of the
grid. Since the grid class is a dictionary, the Grid class also has all the
same standard functions as a dictionary. This class is used to represent a few
different fields in the FFCA, since this class makes use of 4 different 2s
structures. For these 4 structures the Grid class is used.
"""
from typing import List, Set, Iterable, Union
from collections import defaultdict


class Pos:
    """
    Class to represent a 2D position. This class also contains the
    implementation of standard operations on these positions. This makes the
    use of 2D positions much more convenient.
    """
    def __init__(self, r: int, c: int) -> None:
        self.r = r
        self.c = c

    def __add__(self, other: Union['Pos', int, Iterable[int]]) -> 'Pos':
        other = self._convert_to_pos(other)
        if isinstance(other, Pos):
            return Pos(self.r + other.r, self.c + other.c)
        return NotImplemented

    def __sub__(self, other: Union['Pos', int, Iterable[int]]) -> 'Pos':
        other = self._convert_to_pos(other)
        if isinstance(other, Pos):
            return Pos(self.r - other.r, self.c - other.c)
        return NotImplemented

    def __mul__(self, other: Union['Pos', int, Iterable[int]]) -> 'Pos':
        other = self._convert_to_pos(other)
        if isinstance(other, Pos):
            return Pos(self.r * other.r, self.c * other.c)
        return NotImplemented

    def __floordiv__(self, other: Union['Pos', int, Iterable[int]]) -> 'Pos':
        other = self._convert_to_pos(other)
        if isinstance(other, Pos):
            return Pos(self.r // other.r, self.c // other.c)
        return NotImplemented

    def __mod__(self, other: Union['Pos', int, Iterable[int]]) -> 'Pos':
        other = self._convert_to_pos(other)
        if isinstance(other, Pos):
            return Pos(self.r % other.r, self.c % other.c)
        return NotImplemented

    def __lt__(self, other: 'Pos') -> bool:
        return (self.r, self.c) < (other.r, other.c)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Pos):
            return self.r == other.r and self.c == other.c
        return False

    def __hash__(self) -> int:
        return hash((self.r, self.c))

    def __repr__(self) -> str:
        return f"({self.r}, {self.c})"

    def __iter__(self):
        return iter((self.r, self.c))

    def dirs(self) -> List['Pos']:
        return [Pos(-1, 0), Pos(0, 1), Pos(1, 0), Pos(0, -1)]

    def nbs(self) -> List['Pos']:
        return [self + d for d in self.dirs()]

    @staticmethod
    def range(self, other: 'Pos') -> List['Pos']:
        rmin = min(self.r, other.r)
        rmax = max(self.r, other.r)
        cmin = min(self.c, other.c)
        cmax = max(self.c, other.c)
        return [Pos(r, c) for r in range(rmin, rmax + 1) for c in range(cmin, cmax + 1)]

    @staticmethod
    def dist(pos1: 'Pos', pos2: 'Pos') -> int:
        """Calculate the Manhattan distance between two Pos objects."""
        return abs(pos1.r - pos2.r) + abs(pos1.c - pos2.c)

    @staticmethod
    def _convert_to_pos(other: Union['Pos', int, Iterable[int]]) -> Union['Pos', int]:
        """Takes an iterable and converts the iterable to a Pos object."""
        if isinstance(other, Iterable):
            other = list(other)
            if len(other) == 2 and all(isinstance(x, int) for x in other):
                return Pos(*other)
        return other


class Grid:
    """
    Implements a 2d grid structure using a dictionary with the shape:
    Dict[Pos] -> float. The grid also contains the width and breadth of the
    grid. Since the grid is a dictionary, all the same standard functions as a
    dictionary are available.
    """
    def __init__(self, struct) -> None:
        """
        Initializes the grid with a structure using the struct parameter.
        struct: List[List[int]]
        generates: the grid structure
        """

        r = len(struct)
        c = len(struct[0])
        self.Rmin = 0
        self.Rmax = r - 1
        self.Cmin = 0
        self.Cmax = c - 1

        self.grid = defaultdict(lambda: float('inf'))

        # this is for the field convertion
        for r, row in enumerate(struct):
            for c, value in enumerate(row):
                self.grid[Pos(r, c)] = value

    def __getitem__(self, key: Pos) -> float:
        return self.grid[key]

    def __setitem__(self, key: Pos, value: float) -> None:
        self.grid[key] = value

    def __delitem__(self, key: Pos) -> None:
        del self.grid[key]

    def __contains__(self, key: Pos) -> bool:
        return key in self.grid

    def __len__(self) -> int:
        return len(self.grid)

    def __repr__(self) -> str:
        return f"Grid({self.grid})"

    def __iter__(self):
        for r in range(self.Rmin, self.Rmax + 1):
            for c in range(self.Cmin, self.Cmax + 1):
                yield Pos(r, c)

    def keys(self):
        return self.grid.keys()

    def values(self):
        return self.grid.values()

    def _items(self):
        """
        Unsafe items that also returns the wrapped col items
        """
        return self.grid.items()

    def items(self):
        """
        Safe items that only returns the items within the bounds of the grid.
        """
        return [(pos, value) for pos, value in self.grid.items() if
                self.Rmin <= pos.r <= self.Rmax and
                self.Cmin <= pos.c <= self.Cmax]

    def get(self, pos, default=None):
        """
        Safely get the value at position `pos` without modifying the grid.

        :param pos: The position (key) to look up.
        :param default: The default value to return if `pos` is not in the grid.
        :return: The value at `pos` or `default` if `pos` is not found.
        """
        return self.grid[pos] if pos in self.grid else default

    def find(self, target_value: float) -> List[Pos]:
        for pos, value in self.items():
            if value == target_value:
                return pos
        return None

    def findall(self, target_value: float) -> List[Pos]:
        ret = []
        for r in range(self.Rmin, self.Rmax + 1):
            for c in range(self.Cmin, self.Cmax + 1):
                if self.grid[Pos(r, c)] == target_value:
                    ret.append(Pos(r, c))
        return ret

    def to_string(self) -> str:
        result = ""
        for r in range(self.Rmin, self.Rmax + 1):
            row_str = ''.join(str(self.grid.get(Pos(r, c), ' ')) for c in range(self.Cmin, self.Cmax + 1))
            result += row_str + "\n"
        return result

    def show(self) -> None:
        print(self.to_string())

    def remove(self, values: Set[float]) -> None:
        keys_to_remove = [pos for pos, value in self.items() if value in values]
        for key in keys_to_remove:
            del self.grid[key]
        return self

    def copy(self) -> 'Grid':
        """
        Creates and returns a deep copy of the Grid instance.
        The copy includes the same grid data and boundaries.
        """
        copied_grid = Grid([[0]])
        copied_grid.Rmin = self.Rmin
        copied_grid.Rmax = self.Rmax
        copied_grid.Cmin = self.Cmin
        copied_grid.Cmax = self.Cmax
        copied_grid.grid = self.grid.copy()
        return copied_grid

    def calculate_bounds(self: 'Grid') -> List[int]:
        """
        Calcualtes the bounds of the grid.
        grid: the grid to determine the bounds of (Grid)
        """
        positions = list(self.keys())
        r_values = [pos.r for pos in positions]
        c_values = [pos.c for pos in positions]
        return [min(r_values), max(r_values), min(c_values), max(c_values)]

    def get_bounds(self: 'Grid') -> List[int]:
        """
        Returns the bounds of the grid.
        grid: the grid to get the bounds of (Grid)
        """
        return [self.Rmin, self.Rmax, self.Cmin, self.Cmax]

    def map_keys(self: 'Grid', func) -> 'Grid':
        """
        Maps a function over the keys of the grid.
        grid: the grid to map the function over (Grid)
        func: the function to map over the keys (function)
        """
        new_grid = self.copy()
        new_grid.grid = {func(key): value for key, value in self._items()}
        return new_grid

    def get_column(self, c):
        """
        Gets a column from the grid.
        c: the column to get (int)
        returns: the column (List[int])
        """
        assert Pos(0, c) in self.grid, "Column does not exist"
        return [self.grid[Pos(r, c)] for r in range(self.Rmax + 1)]

    def get_row(self, r):
        """
        Gets a row from the grid.
        r: the row to get (int)
        returns: the row (List[int])
        """
        assert Pos(r, 0) in self.grid, "Row does not exist"
        return [self.grid[Pos(r, c)] for c in range(self.Cmax + 1)]

    def update_column(self, c, column):
        """
        Updates a column in the grid.
        c: the column to update (int)
        column: the new values for the column (List[int])
        """
        assert len(column) == self.Rmax + 1, "Wrong size for new column values"
        for r, val in enumerate(column):
            self.grid[Pos(r, c)] = val
