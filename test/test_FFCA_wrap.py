import unittest
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from FFCA_wrap import FFCA_wrap, AGENT_1, AGENT_2
from Grid import Grid, Pos

class TestFFCAWrap(unittest.TestCase):
    def setUp(self):
        """Set up a default FFCA_wrap instance for testing."""
        self.r, self.c = 10, 10  # Grid size
        self.agent_count = 5  # Number of agents per type
        self.ffca = FFCA_wrap(self.r, self.c, self.agent_count)

    def row_of_agents(self, start, end, row_i, agent_type):
        assert row_i < self.r
        return [(Pos(row_i, i), agent_type) for i in range(start, end, 2)]

    def col_of_agents(self, start, end, col_i, agent_type):
        assert col_i < self.c
        return [(Pos(i, col_i), agent_type) for i in range(start, end, 2)]

    def test_initialization(self):
        """Test if the FFCA_wrap initializes correctly."""
        self.assertEqual(self.ffca.structure.Rmax, self.r - 1)
        self.assertEqual(self.ffca.structure.Cmax, self.c - 1)

        # Check if the number of agents is correct
        agent1_count = len(self.ffca.structure.findall(1))
        agent2_count = len(self.ffca.structure.findall(2))
        self.assertEqual(agent1_count, self.agent_count)
        self.assertEqual(agent2_count, self.agent_count)

    def test_agent_movement(self):
        """Test if agents move without errors."""
        initial_positions = set(self.ffca.structure.findall(1) + self.ffca.structure.findall(2))
        self.ffca.step()
        new_positions = set(self.ffca.structure.findall(1) + self.ffca.structure.findall(2))

        self.assertNotEqual(initial_positions, new_positions)

    def test_dynamic_field_update(self):
        """Test if the dynamic field updates correctly."""
        initial_dynamic_field = self.ffca.dynamic_field_1.copy()
        self.ffca.step()
        new_dynamic_field = self.ffca.dynamic_field_1

        # Ensure the dynamic field has changed
        self.assertNotEqual(initial_dynamic_field, new_dynamic_field)

    def test_number_of_agents_constant(self):
        """Test if the number of agents remains constant."""
        initial_agent_count = len(self.ffca.structure.findall(1) + self.ffca.structure.findall(2))

        no_steps = 100
        for _ in range(no_steps):
            self.ffca.step()

        new_agent_count = len(self.ffca.structure.findall(1) + self.ffca.structure.findall(2))
        self.assertEqual(initial_agent_count, new_agent_count)


if __name__ == "__main__":
    unittest.main()
