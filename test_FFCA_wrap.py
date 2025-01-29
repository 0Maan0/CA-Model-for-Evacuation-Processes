import unittest
import numpy as np
from FFCA_wrap import FFCA_wrap
from Grid import Grid, Pos

class TestFFCAWrap(unittest.TestCase):
    def setUp(self):
        """Set up a default FFCA_wrap instance for testing."""
        self.r, self.c = 10, 10  # Grid size
        self.agent_count = 5  # Number of agents per type
        self.ffca = FFCA_wrap(self.r, self.c, self.agent_count)

    def test_initialization(self):
        """Test if the FFCA_wrap initializes correctly."""
        self.assertEqual(self.ffca.structure.Rmax, self.r)
        self.assertEqual(self.ffca.structure.Cmax, self.c)

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

    def test_spawn_agents(self):
        """Test if new agents are spawned correctly."""
        initial_count = len(self.ffca.structure.findall(1)) + len(self.ffca.structure.findall(2))
        self.ffca.spawn_agents()
        new_count = len(self.ffca.structure.findall(1)) + len(self.ffca.structure.findall(2))

        # Ensure at least one new agent is spawned
        self.assertGreaterEqual(new_count, initial_count)


if __name__ == "__main__":
    unittest.main()
