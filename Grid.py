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
from typing import Dict, List, Set, Iterable, Union
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



# TODO: change the initialisation of the Grid, the grid should simply take a
# 2d array of integers and convert that to a grid. The init function should not
# need anything else to change the struct to a grid.
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
        struct: List[List[str]] or List[List[int]] or str
        generates: the grid structure
        """

        # to change the structure of the string to a 2d array for size knowledge
        if isinstance(struct, str):
            struct = [list(row) for row in struct.split('\n')]

        r = len(struct)
        c = len(struct[0])
        self.Rmin = 0
        self.Rmax = r - 2
        self.Cmin = 0
        self.Cmax = c - 2

        self.grid = defaultdict(lambda: float('inf'))

        # we do this for the corridor structure
        if isinstance(struct[0][0], str):
            for r, row in enumerate(struct):
                for c, char in enumerate(row):
                    self.grid[Pos(r, c)] = MAP[char]

        # this is for the field convertion
        elif isinstance(struct[0][0], int):
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
        return iter(self.grid)

    def keys(self):
        return self.grid.keys()

    def values(self):
        return self.grid.values()

    def items(self):
        return self.grid.items()

    def find(self, target_value: float) -> List[Pos]:
        for pos, value in self.items():
            if value == target_value:
                return pos
        return None

    def findall(self, target_value: float) -> List[Pos]:
        return [p for p, v in self.items() if v == target_value]

    def to_string(self) -> str:
        result = ""
        for r in range(self.Rmin, self.Rmax + 1):
            row_str = ''.join(str(self.grid.get(Pos(r, c), ' ')) for c in range(self.Cmin, self.Cmax + 1))
            result += row_str + "\n"
        return result

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
