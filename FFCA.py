"""
University      University of Amstedam
Course:         Complex systems simulation
Authors:        Pjotr Piet, Maan Scipio, Sofia Tete
IDs:            12714933, 15899039, 15830608
Description:    This file contains the implementation of the FFCA.
The FFCA class represents the FFCA model. It contains two static fields of
type Grid. One for agents of type 1 and one for agents of type 2. It also
contains a dynamic field of type Grid. Then lastly, it also contains the
structure of the grid. The FFCA has a step function that combines all the
update functionality, to update the positions of the agents.
"""

import numpy as np
import time
from collections import defaultdict
from typing import List, Tuple
from Grid import Grid, Pos


# global constants for the FFCA
OBSTACLE = 1000
EXIT = 0
AGENT_1 = 1
AGENT_2 = 2
EMPTY = 3

# mapping for values to string
MAP_TO_STRING = {
    OBSTACLE: '#',
    EXIT: 'E',
    AGENT_1: '1',
    AGENT_2: '2',
    EMPTY: '.',
}

# map for string to values
MAP = {
    '#': OBSTACLE,
    'E': EXIT,
    '1': AGENT_1,
    '2': AGENT_2,
    '.': EMPTY,
}

class FFCA:
    """
    FFCA class, represents the FFCA model. It contains two static fields of
    type Grid. One for agents of type 1 and one for agents of type 2. It also
    contians a dynamic field of type Grid. Then lastly, it also contains the
    structure of the grid. The FFCA has a step function that combines all the
    update functionality, to update the positions of the agents.
    """
    def __init__(self, r, c, agent_count, agents_list=None, spawn_rate=0.025,
                 conflict_resolution_rate=0.5, alpha=0.3, delta=0.2,
                 static_field_strength=2.5, dynamic_field_strength=3.0,
                 horizontal_bias=5000,):
        """
        Initialises the FFCA model with the given parameters.
        r: the amount of rows of the corridor, (int)
        c: the amount of columns of the corridor, (int)
        agent_count: the amount of agents of type 1 and type 2, (int)
        agents_list: a list of tuples with the position and agent type
            (List[Tuple[Pos, int]])
        """
        # alpha is the strength of the dynamic field
        self.alpha = alpha
        # delta is the decay rate of the dynamic field
        self.delta = delta
        # ks is the strength of the static field
        self.ks = static_field_strength
        # kd is the decay rate of the dynamic field
        self.kd = dynamic_field_strength
        # conflict resolution probability
        self.mu = conflict_resolution_rate
        # spawn rate
        self.beta = spawn_rate
        # horizontal bias
        self.horizontal_bias = horizontal_bias

        # structure initialisation
        self.structure = Grid(to_corridor(r, c))
        self.structure = self.init_agents(agent_count)
        if agents_list:
            for pos, agent in agents_list:
                self.structure[pos] = agent

        # field 1 is defined as 'to the right'
        # field 2 is defined as 'to the left'
        # fields go from high to low
        self.static_field_1 = None
        self.static_field_2 = None
        self.init_static_fields()

        # dynamic field initialisation
        self.dynamic_field_1 = None
        self.dynamic_field_2 = None
        self.dynamic_field_1, self.dynamic_field_2 = self.init_dynamic_fields()

    def init_agents(self, agent_count):
        """
        Initialises the agents on the grid. 'agent_count' agents of both kinds
        will be placed at random positions on the grid.
        agent_count: the amount of agents of type 1 and type 2 (int)
        returns: the grid with the agents placed on it (Grid)
        """
        assert 2 * agent_count < self.structure.Rmax * self.structure.Cmax, "Too many agents for the grid"
        valid_positions = self.structure.findall(EMPTY)
        pos = np.random.choice(valid_positions, 2*agent_count, replace=False)
        for i in range(agent_count):
            self.structure[pos[i]] = AGENT_1
            self.structure[pos[i+agent_count]] = AGENT_2
        return self.structure

    def init_static_fields(self):
        """
        Initialises the static fields for the agents. The static fields are
        initialised using the negative euclidean distance to the exits. Since
        agents of type 1 are moving to the right, so the exits are defined on
        the right only. And vice versa for agents of type 2.
        """
        R, C = self.structure.Rmax, self.structure.Cmax
        self.static_field_1 = Grid([[-r for r in range(C+1, -1, -1)] for _ in range(R)])
        self.static_field_2 = Grid([[-r for r in range(0, C + 2)] for _ in range(R)])

    def move_agents(self):
        """
        Moves the agents on the grid. This function is split into logical parts:
        - Generating probabilities
        - Solving conflicts
        - Assigning the actual movements
        Returns:
            A mapping of old positions to new positions (dict)
        """
        # Step 1: Generate movement probabilities for each agent
        positions_map = self._generate_probabilities()

        # Step 2: Solve conflicts between agents moving to the same position
        positions_map = self._solve_conflicts(positions_map)

        # Step 3: Apply the resolved movements to the grid
        self._apply_movements(positions_map)

        return positions_map


    def _generate_probabilities(self):
        """
        Generates the movement probabilities for each agent. Uses moore's
        neighbourhood for the probabilities. The probabilities are calculated
        using the static fields and the dynamic fields as mentioned in the base
        paper.
        """
        pss = {}
        positions_map = {}

        for pos, val in self.structure.items():
            ps = [[0 for _ in range(3)] for _ in range(3)]

            # Only consider agents
            if val not in [AGENT_1, AGENT_2]:
                continue

            # Loop over neighbors (Moore neighborhood)
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nb = pos + Pos(dr, dc)

                    # Skip invalid cells
                    if self.structure[nb] in [AGENT_1, AGENT_2, OBSTACLE, float('inf')]:
                        continue

                    # Determine the static field to use based on the agent type
                    sf = self.static_field_1 if val == AGENT_1 else self.static_field_2
                    df = self.dynamic_field_1 if val == AGENT_1 else self.dynamic_field_2
                    field_pos = nb - Pos(1, 0)
                    assert field_pos in sf
                    assert field_pos in df

                    # Calculate transition probability
                    p = np.exp(self.ks * sf[field_pos] + self.kd * df[field_pos])

                    # Apply horizontal bias for agents
                    horizontal_bias = 1
                    forward = 1 if val == AGENT_1 else -1
                    if dc == forward and dr == 0:
                        horizontal_bias = self.horizontal_bias

                    ps[dr + 1][dc + 1] = horizontal_bias * p

            # Normalize probabilities
            Z = sum(sum(row) for row in ps)
            if Z > 0:
                ps = [[p / Z for p in row] for row in ps]
                assert abs(sum(sum(row) for row in ps) - 1) < 1e-6, "Probabilities do not sum to 1"
            else:
                ps = [[0 for _ in range(3)] for _ in range(3)]

            # Store probabilities and decide the tentative new position
            pss[pos] = ps
            if sum([sum(row) for row in ps]) == 0:  # Agent surrounded
                new_pos = pos
            else:
                new_pos = np.random.choice(
                    [pos + Pos(dr, dc) for dr in range(-1, 2) for dc in range(-1, 2)],
                    p=[p for row in ps for p in row]
                )
                positions_map[pos] = new_pos

        return positions_map


    def _solve_conflicts(self, positions_map):
        """
        Resolves conflicts where multiple agents want to move to the same position.
        positions_map: the mapping of old positions to new positions (dict)
        returns: the resolved mapping of old positions to new positions (dict)
        """
        new_positions = list(positions_map.values())

        for new_position in new_positions:
            if new_positions.count(new_position) > 1:  # Conflict detected
                if not np.random.random() > self.mu:  # Conflict not resolved
                    for old_pos, new_pos in positions_map.items():
                        if new_pos == new_position:
                            positions_map[old_pos] = old_pos
                else:  # Resolve conflict
                    conflicted_positions = [old_pos for old_pos, new_pos in positions_map.items() if new_pos == new_position]
                    if conflicted_positions:
                        winner = np.random.choice(conflicted_positions, 1)[0]
                    else:
                        continue

                    for old_pos in conflicted_positions:
                        if old_pos == winner:
                            positions_map[old_pos] = new_position
                        else:
                            positions_map[old_pos] = old_pos

        return positions_map


    def _apply_movements(self, positions_map):
        """
        Applies the resolved movements to the grid.
        positions_map: the mapping of old positions to new positions (dict)
        """
        for old_pos, new_pos in positions_map.items():
            if old_pos == new_pos:
                continue
            if self.structure[new_pos] == EXIT:
                self.structure[old_pos] = EMPTY
            else:
                self.structure[new_pos] = self.structure[old_pos]
                self.structure[old_pos] = EMPTY

    def spawn_agents(self):
        """
        Spawns agents at the entrances of the grid. The agents are spawned in
        the exit of the other type of agent. Agents also have a probability of
        'beta' to spawn at the entrance.
        """
        entrances1 = self.static_field_2.findall(0)
        for pos in entrances1:
            # map to structure position and move one step into the field
            structure_pos = pos + Pos(1, 1)
            if np.random.random() < self.beta:
                self.structure[structure_pos] = AGENT_1

        entrances2 = self.static_field_1.findall(0)
        for pos in entrances2:
            # map to structure position and move one step into the field
            structure_pos = pos + Pos(1, -1)
            if np.random.random() < self.beta:
                self.structure[structure_pos] = AGENT_2

    def init_dynamic_fields(self):
        """
        Initialises the dynamic field with zeros.
        returns: the dynamic field (Grid)
        """
        R, C = self.structure.Rmax, self.structure.Cmax
        dynamic_field_1 = Grid([[0 for _ in range(C + 2)] for _ in range(R)])
        dynamic_field_2 = Grid([[0 for _ in range(C + 2)] for _ in range(R)])
        return dynamic_field_1, dynamic_field_2

    def update_dynamic_field(self, dynamic_field, moved_cells, agent_type):
        """
        Updates the dynamic field according to the mechanics described in the
        base paper. Models diffusion and decay of the agents using a
        discretised Laplace operator.
        moved_cells: the list of original positions of the moved agents
            (List[Pos])
        """
        for p in moved_cells:
            dynamic_field[p] += 1

        new_dynamic_field = Grid([[0 for _ in range(self.structure.Cmax + 2)] for _ in range(self.structure.Rmax)])

        for pos in dynamic_field:
            delta = 0
            for nb in pos.nbs():
                # absorbing boundary conditions (we skip if not in the grid)
                if nb in dynamic_field and self.structure[nb] == agent_type:
                    delta += dynamic_field[nb]
            delta -= 4 * dynamic_field[pos]
            delta = (1 - self.delta) * (dynamic_field[pos] + self.alpha / 4 * delta)
            new_dynamic_field[pos] = delta
        return new_dynamic_field

    def step(self):
        """
        Combines all the update functions to update the positions of the agents
        and the dynamic field and spawn new agents.
        """
        # extract moved agents
        position_map = self.move_agents()
        moved_cells = [pos - Pos(1, 0) for pos, new_pos in position_map.items() if pos != new_pos]
        moved_cells1 = [pos for pos in moved_cells if self.structure[pos] == AGENT_1]
        moved_cells2 = [pos for pos in moved_cells if self.structure[pos] == AGENT_2]

        # update both dynamic fields
        self.dynamic_field_1 = self.update_dynamic_field(self.dynamic_field_1, moved_cells1, AGENT_1)
        self.dynamic_field_2 = self.update_dynamic_field(self.dynamic_field_2, moved_cells2, AGENT_2)
        self.spawn_agents()

    def show(self):
        """
        Prints the structure of the FFCA in the console.
        """
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

    def validate_removal(self):
        """
        Validates that only the agents are removed from the exits, and ensures
        no other agents are accidentally lost.
        """
        for pos, val in self.structure.items():
            if val in [AGENT_1, AGENT_2]:
                assert self.structure[pos + Pos(1, 0)] == EXIT, "Agent is not on an exit"
        pass

    def agents_in_row(self, structure):
        agent_counts = defaultdict(lambda: {AGENT_1: 0, AGENT_2: 0})
        for pos, val in structure.items():
            if val in [AGENT_1, AGENT_2]:
                agent_counts[pos.r][val] += 1
        N1 = []
        N2 = []
        for row in sorted(agent_counts.keys()):
            N1.append(agent_counts[row][AGENT_1])
            N2.append(agent_counts[row][AGENT_2])
        return N1, N2


def string_to_ints(str):
    """
    Converts a string to a 2d list of integers/floats.
    str: the string to convert (str)
    returns: the 2d list of integers/floats (List[List[int]])
    """
    return [[MAP[c] for c in row] for row in str.split('\n') if row]


def to_corridor(R, C):
    """
    Creates a corridor string with the given dimensions.
    R: the amount of rows of the corridor (int)
    C: the amount of columns of the corridor (int)
    returns: the corridor string (str)
    """
    s = '#' * (C + 2)
    s += '\n'
    for r in range(R):
        s += 'E'
        s += '.' * C
        s += 'E'
        s += '\n'

    s += '#' * (C + 2)

    return string_to_ints(s)
