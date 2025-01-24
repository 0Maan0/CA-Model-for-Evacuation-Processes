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
import time


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
    def __init__(self, r, c, no_agents, agents_list=None):
        # alpha is the strength of the dynamic field
        self.alpha = 0.3
        # delta is the decay rate of the dynamic field
        self.delta = 0.1
        # ks is the strength of the static field
        self.ks = 2.5
        # kd is the decay rate of the dynamic field
        self.kd = 3.0
        # conflict resolution probability
        self.mu = 0.5
        # spawn rate
        self.beta = 0.025
        # horizontal bias
        self.horizontal_bias = 50

        # structure initialisation
        self.structure = Grid(to_corridor(r, c))
        self.structure = self.init_agents(no_agents)
        if agents_list:
            for pos, agent in agents_list:
                self.structure[pos] = agent

        # field 1 is defined as 'to the right'
        # field 2 is defined as 'to the left'
        # fields go from high to low
        self.static_field_1 = None
        self.static_field_2 = None
        self.init_static_fields()

        # hard part
        self.dynamic_field = None
        self.dynamic_field = self.init_dynamic_field()

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
        # print(R, C)
        self.static_field_1 = Grid([[-r for r in range(C+1, -1, -1)] for _ in range(R)])
        self.static_field_2 = Grid([[-r for r in range(0, C + 2)] for _ in range(R)])
        # print(self.static_field_1)
        # print(self.static_field_2)

    def move_agents(self):
        pss = {}
        positions_map = {}
        for pos, val in self.structure.items():
            ps = [[0 for _ in range(3)] for _ in range(3)]

            # only considers agents
            if val not in [AGENT_1, AGENT_2]:
                continue

            # loops over all neighbors (Moore neighborhood)
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nb = pos + Pos(dr, dc)

                    # Skip invalid cells (non-movable positions)
                    if self.structure[nb] in [AGENT_1, AGENT_2, OBSTACLE, float('inf')]:
                        continue

                    # Determine the static field to use based on the agent type
                    sf = self.static_field_1 if val == AGENT_1 else self.static_field_2
                    # mapping from structure position to field position
                    field_pos = nb - Pos(1, 0)
                    assert field_pos in sf
                    assert field_pos in self.dynamic_field

                    # Calculate transition probability using the static and dynamic fields
                    p = np.exp(self.ks * sf[field_pos] + self.kd * self.dynamic_field[field_pos])

                    horizontal_bias = 1

                    # mapping
                    if val == AGENT_1:
                        forward = 1
                    elif val == AGENT_2:
                        forward = -1


                    if dc == forward and dr == 0:
                        horizontal_bias = self.horizontal_bias
                    ps[dr + 1][dc + 1] = horizontal_bias * p

            # Normalization factor
            Z = sum(sum(row) for row in ps)

            # Avoid division by zero if all probabilities are zero
            if Z > 0:
                ps = [[p / Z for p in row] for row in ps]
                assert abs(sum(sum(row) for row in ps) - 1) < 1e-6, "Probabilities do not sum to 1"
            else:
                ps = [[0 for _ in range(3)] for _ in range(3)]
            # print(ps)

            # Store the normalized probabilities for this position
            pss[pos] = ps


            # make distinction between agents that cannot move (surrounde by other agents)
            if sum([sum([p for p in row]) for row in ps]) == 0:
                new_pos = pos
            else:
                new_pos = np.random.choice([pos + Pos(dr, dc) for dr in range(-1, 2) for dc in range(-1, 2)], p=[p for row in ps for p in row])
                positions_map[pos] = new_pos

            # solve conflicts
            new_positions = list(positions_map.values())
            for new_position in new_positions:
                # if there are multiple agents at the same new position
                if new_positions.count(new_position) > 1:

                    # don't resolve conflict
                    if not np.random.random() > self.mu:
                        # not resolve conflict; assign old positions to all agents
                        # with this new position
                        for old_pos, new_pos in positions_map.items():
                            if new_pos == new_position:
                                positions_map[old_pos] = old_pos
                    # resolve conflict
                    else:
                        conflicted_positions = [old_pos for old_pos, new_pos in positions_map.items() if new_pos == new_position]
                        if conflicted_positions:
                            winner = np.random.choice(conflicted_positions, 1)[0]
                        else:
                            continue

                        # pick one agent to move to the conflicted new position
                        for old_pos in conflicted_positions:
                            if old_pos == winner:
                                positions_map[old_pos] = new_position
                            else:
                                positions_map[old_pos] = old_pos

        # actually assign the new positions
        for old_pos, new_pos in positions_map.items():
            if old_pos == new_pos:
                continue
            if self.structure[new_pos] == EXIT:
                self.structure[old_pos] = EMPTY
            else:
                self.structure[new_pos] = self.structure[old_pos]
                self.structure[old_pos] = EMPTY

        return positions_map

    # handle the agent flow; remove agents on the exits and spawn agents with
    # probability beta on the corresponding entrances
    #TODO: can make entrance search more efficient
    def spawn_agents(self):
        # entrances for agent1 and spawning move direction: --->
        entrances1 = self.static_field_2.findall(0)
        for pos in entrances1:
            # map to structure position and move one step into the field
            structure_pos = pos + Pos(1, 1)
            if np.random.random() < self.beta:
                self.structure[structure_pos] = AGENT_1

        # entrances for agent2 and spawning move direction: <---
        entrances2 = self.static_field_1.findall(0)
        for pos in entrances2:
            # map to structure position and move one step into the field
            structure_pos = pos + Pos(1, -1)
            if np.random.random() < self.beta:
                self.structure[structure_pos] = AGENT_2

    def init_dynamic_field(self):
        R, C = self.structure.Rmax, self.structure.Cmax
        # initialise zeros
        dynamic_field = Grid([[0 for _ in range(C + 2)] for _ in range(R)])
        print(dynamic_field)
        return dynamic_field

    # moved_cells contains the list of ORIGINAL positions of the moved agents
    # this way we leave behind a trace of the agents
    def update_dynamic_field(self, moved_cells):
        for p in moved_cells:
            self.dynamic_field[p] += 1

        new_dynamic_field = Grid([[0 for _ in range(self.structure.Cmax + 2)] for _ in range(self.structure.Rmax)])

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

    # spawn new agents at entrances and remove agents on the exits
    def step(self):
        position_map = self.move_agents()
        # extract moved agents
        moved_cells = [pos - Pos(1, 0) for pos, new_pos in position_map.items() if pos != new_pos]
        self.update_dynamic_field(moved_cells)
        self.spawn_agents()

    # quick and dirty show function to test the correctness of the program
    def show(self):
        r_positions = [p.r for p in self.structure.keys()]
        rmin, rmax = min(r_positions), max(r_positions)
        c_positions = [p.c for p in self.structure.keys()]
        cmin, cmax = min(c_positions), max(c_positions)

        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                pos = Pos(r, c)
                val = self.structure[pos]
                char = MAP_TO_STRING.get(val, '?')
                print(char, end='')
            print()
        print()

    # checks that the agents are only removed when they reach an exit
    def validate_removal(self):
        for pos, val in self.structure.items():
            if val in [AGENT_1, AGENT_2]:
                assert self.structure[pos + Pos(1, 0)] == EXIT, "Agent is not on an exit"
        pass


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
    c = to_corridor(10, 100)
    print(c)

# test_to_corridor()

# test agents
# agents = [(Pos(1, 1), 1), (Pos(1, 5), 2)]


#ffca = FFCA(10, 100, 50)
#ffca.show()
#steps = 1000
#for i in range(steps):
  #  time.sleep(0.05)
    # print(f"Step {i}")
  #  ffca.step()
  #  ffca.show()

# for pos, value in ffca.structure.grid.items():
#     print(pos, value)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters for the grid and simulation
grid_size = (20, 20)  # Grid dimensions
num_agents = 30       # Number of agents
num_steps = 50        # Number of steps in the simulation
exit_position = (0, 10)  # Exit location

# Initialize agent positions
def initialize_agents(grid_size, num_agents):
    agents = []
    while len(agents) < num_agents:
        x, y = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])
        if (x, y) not in agents and (x, y) != exit_position:
            agents.append((x, y))
    return agents

# Move agents toward the exit
def move_agents(agents, exit_position, grid_size):
    new_agents = []
    for x, y in agents:
        dx = np.sign(exit_position[0] - x)
        dy = np.sign(exit_position[1] - y)

        # Randomize movement priority
        if np.random.rand() > 0.5:
            new_x, new_y = x + dx, y
        else:
            new_x, new_y = x, y + dy

        # Ensure new positions are within bounds
        new_x = max(0, min(grid_size[0] - 1, new_x))
        new_y = max(0, min(grid_size[1] - 1, new_y))

        # Add the new position if not already occupied
        if (new_x, new_y) not in new_agents and (new_x, new_y) != exit_position:
            new_agents.append((new_x, new_y))
        else:
            new_agents.append((x, y))

    return new_agents

import matplotlib.pyplot as plt
import numpy as np
import imageio

def grid_to_image(grid: Grid) -> np.ndarray:
    # Get the grid size
    rows = grid.Rmax + 2
    cols = grid.Cmax + 2
    
    # Create an empty image (matrix of zeros)
    img = np.zeros((rows, cols, 3), dtype=np.uint8)  # RGB image
    
    # Define color mapping for agents and obstacles
    color_map = {
        OBSTACLE: [0, 0, 0],  # black for obstacles
        EXIT: [0, 255, 0],  # red for exits
        AGENT_1: [255, 0, 0],  # green for agent 1
        AGENT_2: [0, 0, 255],  # blue for agent 2
        EMPTY: [255, 255, 255],  # white for empty cells
    }
    
    # Iterate through the grid and assign colors
    for pos, value in grid.items():
        row, col = pos.r, pos.c
        img[row, col] = color_map.get(value, [255, 255, 255])  # default to white for unknown values
    
    return img

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Sample function to visualize the grid
def visualize_ffca(ffca, steps, output_file="visualization.gif"):
    fig, ax = plt.subplots(figsize=(30, 15))  # Increase the figsize for a larger GIF

    def update(frame):
        ax.clear()
        ax.set_title(f"Step {frame}")
        grid = np.zeros((ffca.structure.Rmax + 2, ffca.structure.Cmax + 2), dtype=int)
        for pos, value in ffca.structure.items():
            grid[pos.r, pos.c] = value

        ax.imshow(grid, cmap="viridis", interpolation="nearest")
        ffca.step()  # Step the FFCA forward

    ani = FuncAnimation(fig, update, frames=steps, interval=50)
    ani.save(output_file, writer="pillow", fps=10)

# Usage
ffca = FFCA(10, 100, 50)  # Adjust parameters as necessary
visualize_ffca(ffca, steps=100, output_file="corridor_simulation.gif")


from IPython.display import Image

# Display the GIF (works in Jupyter notebooks)
Image(filename='simulation.gif')


