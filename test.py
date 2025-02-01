import numpy as np
from Grid import Grid, Pos
from FFCA_wrap import FFCA_wrap, OBSTACLE, AGENT_1, AGENT_2, print_grid, print_field
import time

def agent_row_gap(c_start, c_end, row_i, agent_type):
    return [(Pos(row_i, i), agent_type) for i in range(c_start, c_end, 2)]


def test_dynamic_field():
    rows = 3
    cols = 10
    agent_type = AGENT_2
    # spawn one left moving agent in the middle of the grid
    l = agent_row_gap(1, 3, 2, agent_type)

    ffca = FFCA_wrap(rows, cols, 0, agents_list=l, dynamic_field_strength=1)
    steps = 100
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        ffca.show()
        print('dynamic field 1:')
        print_field(ffca.dynamic_field_1)
        print('dynamic field 1:')
        print_field(ffca.dynamic_field_2)


def run_fun():
    # generate one big collumn
    rows = 2
    cols = 4

    # spawn a col of two agents:
    r = [(Pos(1, 2), 1), (Pos(2, 2), 1)]
    walls = [(Pos(1, 1), OBSTACLE), (Pos(2, 1), OBSTACLE)]
    agents = walls + r
    ffca = FFCA_wrap(rows, cols, 0, agents_list=agents)

    ffca = FFCA_wrap(10, 50, 100)

    steps = 1000
    ffca.show()
    tuples = []
    for i in range(steps):
        if i % 100 == 0:
            print(i)
        # time.sleep(0.5)
        ffca.step()
        not_moved = ffca.get_amount_agents_not_moved_forward()
        ffca.show()
        no_agents = ffca.agents_at_exit()


def get_congested_ffca():
    return FFCA_wrap(8, 50, 100)


def get_sparse_ffca():
    return FFCA_wrap(8, 50, 10)

# test if the amount of agents leaving is actually comparable in congestion
# vs no congestion case
def test_agent_counting():
    leaving_count_1 = 0
    leaving_count_2 = 0
    steps = 100
    c = get_congested_ffca()
    s = get_sparse_ffca()
    for i in range(steps):
        print(i)

        c.show()
        s.show()

        c.step()
        s.step()
        leaving_count_1 += sum(c.agents_at_exit()[:2])
        leaving_count_2 += sum(s.agents_at_exit()[:2])

    print('congested count', leaving_count_1)
    print('non congested count', leaving_count_2)


def test_global_movement():
    cummulative_movement_congested = 0
    cummulative_movement_sparse = 0
    leaving_count_2 = 0
    steps = 100
    c = get_congested_ffca()
    s = get_sparse_ffca()
    for i in range(steps):
        print(i)

        c.show()
        s.show()

        c.step()
        s.step()
        global_movement_c = c.global_movement()
        global_movement_s = s.global_movement()

        cummulative_movement_congested += global_movement_c
        cummulative_movement_sparse += global_movement_s

        print('global movement congested', global_movement_c)
        print('global movement sparse', global_movement_s)

    print('cummulative movement congested', cummulative_movement_congested)
    print('cummulative movement sparse', cummulative_movement_sparse)


def test_one():
    rows = 3
    cols = 10
    agent_type = AGENT_1
    r = [(Pos(2, 2), 1)]
    walls = [(Pos(2, 1), OBSTACLE)]
    ffca = FFCA_wrap(rows, cols, 0, agents_list=r + walls)
    steps = 10
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        not_moved = ffca.get_amount_agents_not_moved_forward()
        print(not_moved)
        ffca.show()
        print_field(ffca.dynamic_field_1)


def run():
    agents = 20
    ffca = FFCA_wrap(10, 50, agents, horizontal_bias=10, dynamic_field_strength=4)
    steps = 1000
    ffca.show()
    for i in range(steps):
        ffca.step()
        ffca.show()

        # not_moved_forward = ffca.get_amount_agents_not_moved_forward()
        # print('not moved forward', not_moved_forward, agents * 2)
        # print('global movement', ffca.global_movement())


def run_small():
    # left = [(Pos(1, 2), 1), (Pos(2, 2), 1)]
    no_agents = 1
    ffca = FFCA_wrap(3, 5, no_agents, dynamic_field_strength=4, horizontal_bias=1000)
    steps = 100
    ffca.show()
    for i in range(steps):
        ffca.step()
        ffca.show()

        print(ffca.positions_map)
        agent_positions = ffca.structure.findall(AGENT_1) + ffca.structure.findall(AGENT_2)
        print(agent_positions)
        ffca.all_agents_in_map()


def hb_low():
    agents = 20
    ffca = FFCA_wrap(10, 50, agents, dynamic_field_strength=4, horizontal_bias=1)
    steps = 1000
    ffca.show()
    for i in range(steps):
        ffca.step()
        ffca.show()

def hb_high():
    agents = 20
    ffca = FFCA_wrap(10, 50, agents, dynamic_field_strength=4, horizontal_bias=1000)
    steps = 1000
    ffca.show()
    for i in range(steps):
        ffca.step()
        ffca.show()

def main():
    # run_fun()
    # test_small()
    # test_agent_counting()
    # test_global_movement()
    # test_dynamic_field()
    # test_one()
    run()
    # run_small()
    # hb_low()
    # hb_high()


if __name__ == "__main__":
    main()
