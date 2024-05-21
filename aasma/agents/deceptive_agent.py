import random
import numpy as np
from aasma.agents.agent import Agent

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class DeceptiveAgent(Agent):

    def __init__(self, n_actions: int):
        super(DeceptiveAgent, self).__init__("Deceptive Agent")
        self.n_actions = n_actions

    def action(self, observation, n_agents, agent_idx) -> int:
        """
        return that action that brings the agent closer to its corresponding fake landmark
        """
        landmark_idx = n_agents + agent_idx - 1

        landmark_pos = observation[landmark_idx]
        
        # move towards landmark_pos
        curr_pos = observation[agent_idx]
        return self.direction_to_go(curr_pos, landmark_pos)
    
    def distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def direction_to_go(self, agent_position, landmark_position):
        distances = np.array(landmark_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)
        
        if abs_distances[1] > abs_distances[0]:
            return self._close_horizontally(distances)
        elif abs_distances[1] < abs_distances[0]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    def _close_horizontally(self, distances):
        if distances[1] > 0:
            return RIGHT
        elif distances[1] < 0:
            return LEFT
        else:
            return STAY

    def _close_vertically(self, distances):
        if distances[0] > 0:
            return DOWN
        elif distances[0] < 0:
            return UP
        else:
            return STAY
        