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
MAP_INT_TO_STRING = {
    OBSTACLE: '#',
    EXIT: 'E',
    AGENT_1: 'R',
    AGENT_2: 'L',
    EMPTY: '.',
}

# map for string to values
MAP_STRING_TO_INT = {
    '#': OBSTACLE,
    'E': EXIT,
    '1': AGENT_1,
    '2': AGENT_2,
    '.': EMPTY,
}

def to_corridor(r, c):
    """
    Creates a corridor with the given dimensions.
    r: the amount of rows of the corridor, (int)
    c: the amount of columns of the corridor, (int)
    returns: the corridor (List[List[int]])
    """
    return [[EMPTY for _ in range(c)] for _ in range(r)]


class FFCA_wrap:
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
                 horizontal_bias=1000, verbose=False):
        """
        Initialises the FFCA model with the given parameters.
        r: the amount of rows of the corridor, (int)
        c: the amount of columns of the corridor, (int)
        agent_count: the amount of agents of type 1 and type 2, (int)
        agents_list: a list of tuples with the position and agent type
            (List[Tuple[Pos, int]])
        """
        # alpha is the diffusion rate of the dynamic field
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
        # or specific 'debug' agents
        if agents_list:
            for pos, agent in agents_list:
                self.structure[pos] = agent

        # wrap the structure (add last col to beginning and first col to end)
        self.update_wrapped_structure()

        # field 1 is defined as 'to the right'
        # field 2 is defined as 'to the left'
        self.static_field_1 = None
        self.static_field_2 = None
        self.static_field_1, self.static_field_2 = self.init_static_fields()

        # dynamic field initialisation
        self.dynamic_field_1 = None
        self.dynamic_field_2 = None
        self.dynamic_field_1, self.dynamic_field_2 = self.init_dynamic_fields()

        # verbose for debugging:
        self.verbose = verbose

        # for validating correctness
        self.initial_agent_count_1 = len(self.structure.findall(AGENT_1))
        self.initial_agent_count_2 = len(self.structure.findall(AGENT_2))

        # positions map to detect movement per iteration
        self.positions_map = None

    def update_column(self, c, column):
        self.structure.update_column(c, column)

    def get_column(self, c):
        return self.structure.get_column(c)

    def get_row(self, r):
        return self.structure.get_row(r)

    def update_wrapped_structure(self):
        """
        Wraps the structure of the grid. Adds/updates the wrapped columns
        to/of the structure
        """
        # add the last column to the beginning
        self.update_column(-1, self.get_column(self.structure.Cmax))
        # add the first column to the end
        self.update_column(self.structure.Cmax + 1, self.get_column(0))

    def init_agents(self, agent_count):
        """
        Initialises the agents on the grid. 'agent_count' agents of both kinds
        will be placed at random positions on the grid.
        agent_count: the amount of agents of type 1 and type 2 (int)
        returns: the grid with the agents placed on it (Grid)
        """
        assert 2 * agent_count < (self.structure.Rmax + 1) * (self.structure.Cmax + 1), \
               "Too many agents for the grid"
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
        # wrapping static field, add one column on both sides
        static_field_1 = Grid([[-r for r in range(C + 2, -1, -1)] for _ in range(R + 1)])
        static_field_2 = Grid([[-r for r in range(0, C + 3)] for _ in range(R + 1)])

        # copy the structure bounds to the static fields
        Rmin, Rmax, Cmin, Cmax = self.structure.get_bounds()
        static_field_1.Rmin = static_field_2.Rmin = Rmin
        static_field_1.Rmax = static_field_2.Rmax = Rmax
        static_field_1.Cmin = static_field_2.Cmin = Cmin
        static_field_1.Cmax = static_field_2.Cmax = Cmax

        # map the keys to the wrapped grid
        static_field_1 = static_field_1.map_keys(lambda pos: pos - Pos(0, 1))
        static_field_2 = static_field_2.map_keys(lambda pos: pos - Pos(0, 1))

        return static_field_1, static_field_2

    def wrap_positions(self, positions_map):
        """
        Wraps the positions of the agents to the 'non'-wrapped part. If an
        agents position would be outside of the structure it will be wrapped
        around to the other side.
        positions_map: the mapping of old positions to new positions
            (dict[Pos, Pos])
        returns: the wrapped positions (dict[Pos, Pos])
        """
        for pos, new_pos in positions_map.items():
            if new_pos.c == self.structure.Cmax + 1:
                positions_map[pos] = Pos(new_pos.r, 0)
            elif new_pos.c == -1:
                positions_map[pos] = Pos(new_pos.r, self.structure.Cmax)
        return positions_map

    def assert_grid_constant_height(self):
        """
        Asserts that the grid has a constant height.
        """
        rmin, rmax, cmin, cmax = self.structure.calculate_bounds()
        assert self.structure.Rmax == rmax and self.structure.Rmin == rmin, \
                "Grid height changed"

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
        positions_map = self._generate_positions_map()

        self.assert_grid_constant_height()

        # step 2: wrap the positions map
        positions_map = self.wrap_positions(positions_map)

        self.assert_grid_constant_height()

        # Step 3: Solve conflicts between agents moving to the same position
        positions_map = self._solve_conflicts(positions_map)

        self.assert_grid_constant_height()

        # Step 4: Apply the resolved movements to the grid
        self._apply_movements(positions_map)

        # Step 5: Update the wrapped structure
        self.update_wrapped_structure()

        self.assert_grid_constant_height()

        # confirm that no agents were lost
        no_agent1 = len(self.structure.findall(AGENT_1))
        no_agent2 = len(self.structure.findall(AGENT_2))
        if not no_agent1 == self.initial_agent_count_1 and \
           no_agent2 == self.initial_agent_count_2:
            print('Lost agents')
            self.show()
            self.show(True)
            print(self.structure)
            print('positions_map:')
            for pos, new_pos in positions_map.items():
                print(f'{pos} -> {new_pos}')

        assert no_agent1 == self.initial_agent_count_1 and \
               no_agent2 == self.initial_agent_count_2, "Lost agents"

        return positions_map

    def find_all_agents(self):
        return self.structure.findall(AGENT_1) + self.structure.findall(AGENT_2)

    def _generate_positions_map(self):
        """
        Generates the position map for each agent. It does so by calculating
        the probabilities for each agent to move to a certain position and
        picking one of these positions based on the probabilities.
        returns: the mapping of old positions to new positions (dict[Pos, Pos])
        """
        pss = {}
        positions_map = {}

        for pos, val in self.structure.items():

            # Only consider agents and only the non wrapped structure
            if val not in [AGENT_1, AGENT_2]:
                continue

            ps = [[0 for _ in range(3)] for _ in range(3)]

            # Loop over neighbors (Moore neighborhood)
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nb = pos + Pos(dr, dc)

                    # Skip non empty cells, since we can't move there
                    if self.structure.get(nb, None) in [OBSTACLE, AGENT_1, AGENT_2, float('inf')]:
                        continue

                    # Determine the fields to use based on the agent type
                    sf = self.static_field_1 if val == AGENT_1 else self.static_field_2
                    df = self.dynamic_field_1 if val == AGENT_1 else self.dynamic_field_2

                    # confirm that the position is also in the fields
                    assert pos in sf
                    assert pos in df

                    # Calculate transition probability based on the formula
                    p = np.exp(self.ks * sf[pos] + self.kd * df[pos])

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
                # confirm that the normalised probabilites sum to 1
                assert abs(sum(sum(row) for row in ps) - 1) < 1e-6, "Probabilities do not sum to 1"

            pss[pos] = ps
            # agent has no valid moves so it stays in the same position
            if Z <= 0:
                new_pos = pos
            # pick one of the positions based on the probabilities
            else:
                new_pos = np.random.choice(
                    [pos + Pos(dr, dc) for dr in range(-1, 2) for dc in range(-1, 2)],
                    p=[p for row in ps for p in row]
                )
            positions_map[pos] = new_pos

        assert all([agent in positions_map for agent in self.find_all_agents()]), \
               "Not all agents have where considered"
        return positions_map

    def _solve_conflicts(self, positions_map):
        """
        Resolves conflicts where multiple agents want to move to the same position.
        positions_map: the mapping of old positions to new positions (dict)
        returns: the resolved mapping of old positions to new positions (dict)
        """
        new_positions = list(positions_map.values())

        for new_position in new_positions:
            if new_positions.count(new_position) > 1:
                # Conflict not resolved
                if not np.random.random() > self.mu:

                    # no one moves
                    for old_pos, new_pos in positions_map.items():
                        if new_pos == new_position:
                            positions_map[old_pos] = old_pos
                # Resolve conflict
                else:
                    conflicted_positions = [old_pos for old_pos, new_pos in positions_map.items() if new_pos == new_position]
                    if conflicted_positions:
                        winner = np.random.choice(conflicted_positions, 1)[0]
                    else:
                        continue

                    # only the winner moves
                    for old_pos in conflicted_positions:
                        if old_pos == winner:
                            positions_map[old_pos] = new_position
                        else:
                            positions_map[old_pos] = old_pos

        new_positions = list(positions_map.values())

        # asserts that we have no agents moving to the same position
        assert len(new_positions) == len(set(new_positions)), \
               "Agents are moving to the same position"

        return positions_map

    def update_postions_map_wrap(self, positions_map):
        """
        This function updates the positions_map so that the agents that would
        move outside of the structure and therefore would wrap around to the
        other side, actually have the wrapped position rather than the 'outside'
        position
        positions_map: the mapping of old positions to new positions (dict)
        returns: the updated mapping of old positions to new positions (dict)
        """
        for pos, new_pos in positions_map.items():
            if new_pos.c == self.structure.Cmax + 1:
                positions_map[pos] = Pos(new_pos.r, 1)
            elif new_pos.c == 0:
                positions_map[pos] = Pos(new_pos.r, self.structure.Cmax)
        return positions_map

    def _apply_movements(self, positions_map):
        """
        Applies the resolved movements to the grid.
        positions_map: the mapping of old positions to new positions (dict)
        """
        for old_pos, new_pos in positions_map.items():
            if old_pos == new_pos:
                continue
            else:
                self.structure[new_pos] = self.structure[old_pos]
                self.structure[old_pos] = EMPTY

    def get_wrapped_structure(self):
        structure_wrapped = self.structure.copy()
        for r in range(self.structure.Rmax):
            # add the first column to the end: (again the horrific mapping)
            structure_wrapped[Pos(r + 1, self.structure.Cmax + 1)] = self.structure[Pos(r + 1, 1)]
            # add the last column to the beginning:
            structure_wrapped[Pos(r + 1, 0)] = self.structure[Pos(r + 1, self.structure.Cmax)]
        return structure_wrapped

    def init_dynamic_fields(self):
        """
        Initialises the dynamic field with zeros.
        returns: the dynamic field (Grid)
        """
        R, C = self.structure.Rmax, self.structure.Cmax
        dynamic_field_1 = Grid([[0 for _ in range(C + 3)] for _ in range(R + 1)])
        dynamic_field_2 = Grid([[0 for _ in range(C + 3)] for _ in range(R + 1)])

        # ensure that the dynamic fields have the same bounds as the structure
        Rmin, Rmax, Cmin, Cmax = self.structure.get_bounds()
        dynamic_field_1.Rmin = dynamic_field_2.Rmin = Rmin
        dynamic_field_1.Rmax = dynamic_field_2.Rmax = Rmax
        dynamic_field_1.Cmin = dynamic_field_2.Cmin = Cmin
        dynamic_field_1.Cmax = dynamic_field_2.Cmax = Cmax

        # map one col to the left for wrapped structure
        dynamic_field_1 = dynamic_field_1.map_keys(lambda pos: pos - Pos(0, 1))
        dynamic_field_2 = dynamic_field_2.map_keys(lambda pos: pos - Pos(0, 1))

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

        new_dynamic_field = self.dynamic_field_1.copy()

        for pos in dynamic_field:
            delta = 0
            for nb in pos.nbs():
                # absorbing boundary conditions (we skip if not in the grid)
                if nb in dynamic_field:
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
        positions_map = self.move_agents()

        moved_cells = {pos: new_pos for pos, new_pos in positions_map.items() if pos != new_pos}
        moved_cells1 = [pos for pos, new_pos in moved_cells.items() if self.structure[new_pos] == AGENT_1]
        moved_cells2 = [pos for pos, new_pos in moved_cells.items() if self.structure[new_pos] == AGENT_2]

        # update both dynamic fields
        self.dynamic_field_1 = self.update_dynamic_field(self.dynamic_field_1, moved_cells1, AGENT_1)
        self.dynamic_field_2 = self.update_dynamic_field(self.dynamic_field_2, moved_cells2, AGENT_2)

    def run(self, steps, delay=0):
        """
        Runs the FFCA model for a given amount of steps.
        steps: the amount of steps to run the model for (int)
        delay: the delay between each step in seconds (float)
        """
        for _ in range(steps):
            self.step()
            self.show()
            time.sleep(delay)

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

    def agents_at_exit(self):
        """
        Determines the amount of agents enering and leaving
        returns: the amount of agents entering and leaving
            (Tuple[List[int], List[int], List[int], List[int]])
        """
        # 1 goes --->
        # 2 goes <---
        agent_1_leaving = 0
        agent_2_leaving = 0
        agent_1_entering = 0
        agent_2_entering = 0

        # invertion is needed since positions map maps from old pos to cur pos
        # and we want to see in cur iteration where the agent came from
        inverted_position_map = {v: k for k, v in self.positions_map.items()}

        first_col = self.get_first_column()
        for pos in first_col:
            if self.structure[pos] == AGENT_1:
                # check in first column if agent1 just entered
                if inverted_position_map[pos].c == self.structure.Cmax:
                    agent_1_entering += 1
                    agent_1_leaving += 1

        last_col = self.get_last_column()
        for pos in last_col:
            if self.structure[pos] == AGENT_2:
                # check in last column if agent1 just entered
                if inverted_position_map[pos].c == 1:
                    agent_2_leaving += 1
                    agent_2_entering += 1

        return agent_1_leaving, agent_2_leaving, agent_1_entering, agent_2_entering

    def global_movement(self):
        """
        Determines the global movement direction of the agents. Forward moving
        agents are counted as positive, backward moving agents are counted as
        negative.
        returns: the global movement direction of the agents (int)
        """
        global_movement_count = 0
        for old_pos, new_pos in self.positions_map_wrapped.items():

            assert self.structure_wrapped[new_pos] in [AGENT_1, AGENT_2], "Agent has not moved yet, run this function after step :)"
            agent_type = self.structure_wrapped[new_pos]
            if agent_type == AGENT_1:
                # agent 1 moves 'forward'
                if new_pos.c > old_pos.c:
                    global_movement_count += 1
                # agent 1 moves 'backwards'
                elif new_pos.c < old_pos.c:
                    global_movement_count -= 1
            elif agent_type == AGENT_2:
                # agent 2 movees 'forwards'
                if new_pos.c < old_pos.c:
                    global_movement_count += 1
                # agent 2 moves 'backwards'
                elif new_pos.c > old_pos.c:
                    global_movement_count -= 1

        return global_movement_count

    def get_amount_agents_not_moved_forward(self):
        """
        Determines the amount of agents that have not moved in the current
        iteration.
        returns: the amount of agents that have not moved (int)
        """
        not_moved_forward = 0
        for old_pos, new_pos in self.positions_map.items():
            agent_type = self.structure_wrapped[new_pos]
            if agent_type == AGENT_1 and new_pos.c >= old_pos.c:
                not_moved_forward += 1
            elif agent_type == AGENT_2 and new_pos.c <= old_pos.c:
                not_moved_forward += 1

        return not_moved_forward

    def show(self, full=False, coords=False):
        self.print_structure(self.structure, full, coords)

    @staticmethod
    def print_structure(grid: Grid, full=False, coords=False) -> None:
        """
        Used to print one of the structures of the FFCA. Either structure, or
        wrapped_structure.

        grid: the grid to print (Grid)
        full: whether to print the full calculated bounds
        coords: whether to show coordinate labels
        """
        rmin, rmax, cmin, cmax = grid.get_bounds()
        if full:
            rmin, rmax, cmin, cmax = grid.calculate_bounds()

        # Print column indices
        if coords:
            print('  ' + ''.join(str(abs(c) % 10) for c in range(cmin - 1, cmax + 2)))

        # Print upper wall
        upper_wall = '#' * (cmax - cmin + 3)
        print((f"{str(abs(rmin - 1))} " if coords else "") + upper_wall)

        for r in range(rmin, rmax + 1):
            if coords:
                print(f"{str(abs(r))} ", end="")  # Row index
            print('E', end='')  # Left boundary
            for c in range(cmin, cmax + 1):
                pos = Pos(r, c)
                val = grid[pos]
                char = MAP_INT_TO_STRING.get(val, '?')
                print(char, end='')
            print('E')  # Right boundary

        # Print bottom wall
        bottom_wall = '#' * (cmax - cmin + 3)
        print((f"{str(rmax + 1)} " if coords else "") + bottom_wall)
        print()


    @staticmethod
    def print_field(grid: Grid, full=False) -> None:
        """
        Print function for both fields (static and dynamic).
        grid: the grid to print (Grid)
        full: whether to print the full grid or just the inner grid (bool)
        """
        rmin, rmax, cmin, cmax = grid.get_bounds()
        # also print the wrapped 'border'
        if full:
            rmin, rmax, cmin, cmax = grid.calculate_bounds()
        print(grid)

        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                pos = Pos(r, c)
                val = grid[pos]
                print(f'{val:.2f}', end=' ')
            print()
        print()
