import random
import numpy as np
from aasma.agents.agent import Agent
from aasma.simple_adversary.simple_adversary import SimpleAdversary

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class GreedyAdversary(Agent):

    def __init__(self, n_actions: int):
        super(GreedyAdversary, self).__init__("Greedy Adversary")
        self.n_actions = n_actions

    def action(self, observation, env) -> int:
        """
        return that action that brings the adversary closer to the landmark that has an agent closest to it
        """
        n_good_agents = env.n_good_agents
        adversary_idx = n_good_agents
        landmark_idx = random.randint(0, n_good_agents - 1)

        landmark_pos = env.landmark_pos[landmark_idx]
        dist = env._grid_shape[0] + env._grid_shape[1]
        
        for i in range(n_good_agents):
            for t in range(env.n_landmarks):
                current_agent_pos = observation[i]
                current_landmark_pos = env.landmark_pos[t]
            
                if self.distance(current_agent_pos, current_landmark_pos) < dist:
                    dist = self.distance(current_agent_pos, current_landmark_pos)
                    landmark_pos = current_landmark_pos

        # move towards landmark_pos
        curr_pos = observation[adversary_idx]
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