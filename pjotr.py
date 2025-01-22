"""
CA 2d grid which can be initialized with a lists of obstacles and a list of
exits, given as 2d coords. I also want to be able to update the grid using
similar lists of coords. Then i want the class to initialize the grid using
by calculating the values for every cell. I want the grid to be a l[complex] -> int
Lastly, I also want to make a step function, that moves all the agents in the grid
and updates the grid accordingly.
"""

from typing import Dict, List, Set, Iterable, Union
from collections import defaultdict


class Pos:
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
        if isinstance(other, Iterable):
            other = list(other)
            if len(other) == 2 and all(isinstance(x, int) for x in other):
                return Pos(*other)
        return other

# constants
OBSTACLE = 1000
EXIT = 0
"""
Implements a grid class for the CA, standard value for obstacles is 1000, and
standard value for exits is 0. Standard value for unset cells is inf."""
class Grid:
    def __init__(self, r, c, obstacles: List[Pos], exits: List[Pos]) -> None:
        self.grid = defaultdict(lambda: float('inf'))
        self.Rmin = 0
        self.Rmax = r - 1
        self.Cmin = 0
        self.Cmax = c - 1

        for pos in obstacles:
            self.grid[pos] = OBSTACLE
        for pos in exits:
            self.grid[pos] = 0

        # also add a standard border of obstacles
        for i in range(r):
            self.grid[Pos(i, 0)] = OBSTACLE
            self.grid[Pos(i, c + 2)] = OBSTACLE
        for i in range(c + 2):
            self.grid[Pos(0, i)] = OBSTACLE
            self.grid[Pos(r + 2, i)] = OBSTACLE

        # set all other positions to
        for r in range(self.Rmin, self.Rmax + 1):
            for c in range(self.Cmin, self.Cmax + 1):
                pos = Pos(r, c)
                if self.grid[pos] == OBSTACLE:
                    continue
                if self.grid[pos] == EXIT:
                    continue
                if self.grid[pos] == float('inf'):
                    self.grid[pos] = float('inf')

        # initialise by the following algorithm:
        # 1. set all exits to 0
        # 2. set all obstacles to inf
        # 3. go from exits, and set every cell horizontal or vertical to the exit to 1
        # 4. go from exits, and set every cell diagonal to the exit to 1.5
        # do this for all exits, and check if the value is already set, if so, only update if lower
        # and only update every position only once per exit
        # repeat until all exits are done

        for _exit in exits:
            stack = [(_exit, 0)]
            visited = set()
            while stack:
                pos, val = stack.pop()
                visited.add(pos)
                for nb, new_val in pos.nbs():
                    if grid[nb] == OBSTACLE or nb not in grid or nb in visited:
                        continue

                    d = Pos.dist(_exit, nb)
                    if d == 1:
                        new_val = 1
                    else:
                        new_val = 1.5

                    stack.append((nb, new_val))
                    visited.add(nb)

                    # update the grid with the new value
                    if new_val < grid[nb]:
                        grid[nb] = new_val



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

    def show(self, positions: Set[Pos]=set()) -> None:
        for r in range(self.Rmin, self.Rmax + 1):
            for c in range(self.Cmin, self.Cmax + 1):
                p = Pos(r, c)
                value = self.grid.get(p, float('inf'))
                if value > 100:
                    print('#', end='')
                else:
                    print('.', end='')
            print()
        print()

    def remove(self, values: Set[float]) -> None:
        keys_to_remove = [pos for pos, value in self.items() if value in values]
        for key in keys_to_remove:
            del self.grid[key]
        return self

"""
Structure of the FFCA,
fields:
- initialise 2d structure corridor
- static field agent 1
- static field agent 2
- dynamic field both agents
- step function
"""

class FFCA:
    def __init__(self, r, c, obstacles: List[Pos], exits: List[Pos], agents: List[Pos]):
        self.static_field_1 = None
        self.static_field_2 = None
        self.dynamic_field = None
        self.structure = None

        grid = Grid(r, c, obstacles, exits)
        for r in grid.Rmax:
            in_row = []
            for c in grid.Cmax:
                p = Pos(r, c)
                if grid[p] == AGENT_1:
                    in_row.append(AGENT_1)



    def init_static_field(self, type):
        pass

    def init_dynamic_field(self):
        pass

    def step(self):
        pass