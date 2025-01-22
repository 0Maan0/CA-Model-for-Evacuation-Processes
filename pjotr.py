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
import numpy as np


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
AGENT_1 = 1
AGENT_2 = 2
EMPTY = 3

MAP_TO_STRING = {
    OBSTACLE: '#',
    EXIT: 'E',
    AGENT_1: '1',
    AGENT_2: '2',
    EMPTY: '.',
}

MAP = {
    '#': OBSTACLE,
    'E': EXIT,
    '1': AGENT_1,
    '2': AGENT_2,
    '.': EMPTY,
}


ex = """
########
.      .
.      .
########
"""


"""
Implements a grid class for the CA, standard value for obstacles is 1000, and
standard value for exits is 0. Standard value for unset cells is inf."""
class Grid:
    def __init__(self, struct) -> None:

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
        print(self.Rmax, self.Cmax)
        for r in range(self.Rmin, self.Rmax + 1):
            for c in range(self.Cmin, self.Cmax + 1):
                pos = Pos(r, c)
                print(self.grid[pos], end='')
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
    def __init__(self, r, c):
        # to be determined further
        self.alpha = 0.3
        self.delta = 0.1
        self.ks = 2.5
        self.kd = 2.0

        self.structure = Grid(to_corridor(r, c))
        self.structure = self.init_agents(8)

        # field 1 is defined as 'to the right'
        # field 2 is defined as 'to the left'
        # fields go from high to low
        self.static_field_1 = None
        self.static_field_2 = None
        self.init_static_fields()
        # print(self.static_field_1.grid)
        # print(self.static_field_2.grid)

        # hard part
        self.dynamic_field = None
        self.dynamic_field = self.init_dynamic_field()

        # initialise the grid with agents
        # make step function
    """
    for initialising the grid we want to convert (r, c) into a corridor.
    First, create the docstring for this corridor, then convert that docstring
    to the proper structure.
    """

    # changes inplace
    # no_agents represents the amount of agents of type 1 and the amount of agents of type 2
    def init_agents(self, no_agents):
        assert 2 * no_agents < self.structure.Rmax * self.structure.Cmax, "Too many agents for the grid"
        valid_positions = self.structure.findall(EMPTY)
        pos = np.random.choice(valid_positions, 2*no_agents, replace=False)
        for i in range(no_agents):
            self.structure[pos[i]] = AGENT_1
            self.structure[pos[i+no_agents]] = AGENT_2
        return self.structure

    def init_static_fields(self):
        R, C = self.structure.Rmax, self.structure.Cmax
        self.static_field_1 = Grid([[r for r in range(R, 0, -1)] for _ in range(C)])
        self.static_field_2 = Grid([[r for r in range(1, R + 1)] for _ in range(C)])

    def init_dynamic_field(self):
        R, C = self.structure.Rmax, self.structure.Cmax
        # initialise zeros
        dynamic_field = Grid([[0 for _ in range(C)] for _ in range(R)])
        return dynamic_field

    # moved_cells contains the list of ORIGINAL positions of the moved agents
    # this way we leave behind a trace of the agents
    def update_dynamic_field(self, moved_cells):
        for p in moved_cells:
            self.dynamic_field[p] += 1

        new_dynamic_field = Grid([[0 for _ in range(self.structure.Cmax)] for _ in range(self.structure.Rmax)])

        for pos in self.dynamic_field:
            delta = 0
            for nb in pos.nbs():
                # absorbing boundary conditions (we skip if not in the grid)
                if nb in self.dynamic_field:
                    delta += self.dynamic_field[nb]
            delta -= 4 * self.dynamic_field[pos]
            delta = (1 - self.delta) * (self.dynamic_field[pos] + self.alpha / 4 * delta)
            new_dynamic_field[pos] = delta

        self.dynamic_field = new_dynamic_field

    def step(self):
        pss = {}
        for pos, val in self.structure.items():
            ps = [[0 for _ in range(3)] for _ in range(3)]

            # only considers agents
            if val not in [AGENT_1, AGENT_2]:
                continue

            # loops over all neighbours (moore neighbourhood)
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nb = Pos(pos.r + dr, pos.c + dc)

                    # only considers agents, therefore we don't need the ksi
                    if self.structure[nb] in [AGENT_1, AGENT_2, OBSTACLE, EXIT]:
                        continue

                    # ternary for static field determination
                    # print(self.static_field_1.grid)
                    # print(self.static_field_2.grid)
                    sf = self.static_field_1 if val == AGENT_1 else self.static_field_2
                    field_pos = Pos(nb.r - 1, nb.c - 1)
                    # paper formula

                    p = np.exp(self.ks * sf[field_pos] + self.kd * self.dynamic_field[field_pos])
                    ps[dr + 1][dc + 1] = p

                    # normalisation
                    Z = sum([sum(row) for row in ps])
                    ps = [[p / Z for p in row] for row in ps]
                    # print(ps)
            pss[pos] = ps

        return pss

    # fix
    def show(self):
        r_positions = [p.r for p in self.structure]
        rmin, rmax = min(r_positions), max(r_positions)
        c_positions = [p.c for p in self.structure]
        cmin, cmax = min(c_positions), max(c_positions)
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                pos = Pos(r, c)
                val = self.structure[pos]
                char = MAP_TO_STRING[val]
                print(char, end='')
            print()
        print()

def to_corridor(R, C):
    s = '#' * (C + 2)
    s += '\n'
    for r in range(R):
        s += 'E'
        s += '.' * C
        s += 'E'
        s += '\n'

    s += '#' * (C + 2)

    return s


def test_to_corridor():
    c = to_corridor(5, 5)
    print(c)

test_to_corridor()

ffca = FFCA(5, 5)
# ffca.show()
# ffca.step()

# for pos, value in ffca.structure.grid.items():
#     print(pos, value)
