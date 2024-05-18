import copy
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

from PIL import ImageColor
from PIL import Image, ImageDraw
import gym
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

class SimpleAdversary(gym.Env):
    """
    An altered version of ma_gym.envs.predator_prey.predator_prey.PredatorPrey
    Instead, there are two teams: one with 2 agents (good team) and the other with 1 agent (bad agent), and 2 landmarks (a real and a fake one).
    Instead of trying to catch the other agents, they strive to be as close as possible to the real landmark at the end of a cycle of movement.
    The good agents can tell which is the real landmark, but the bad agent can't. The bad agent can only see the good agents and cannot tell the landmarks apart.
    The good agents are rewarded by closeness to the real landmark and penalized by the bad agent's closeness to the real landmark.
    The bad agent is rewarded by closeness to the real landmark. All rewards are given at the end of the cycle.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(5, 5), n_good_agents=2, max_steps=100):
        self._grid_shape = grid_shape
        self.n_good_agents = n_good_agents
        self.n_bad_agents = 1
        self.n_agents = n_good_agents + self.n_bad_agents
        self.n_landmarks = n_good_agents
        self._max_steps = max_steps
        self._step_count = None

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.landmark_pos = {_: None for _ in range(self.n_landmarks)}
        self._real_landmark_idx = random.randint(0, self.n_landmarks - 1)
        self.viewer = None

        # Observations: [agent positions, landmark positions, step count]
        obs_dim = self.n_agents * 2 + self.n_landmarks * 2 + 1
        self._obs_high = np.array([1.] * obs_dim, dtype=np.float32)
        self._obs_low = np.array([0.] * obs_dim, dtype=np.float32)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.landmark_pos = {}

        self._step_count = 0
        self._real_landmark_idx = random.randint(0, self.n_landmarks - 1)

        self.__init_positions()
        return self._get_obs()

    def step(self, agents_action):
        self._step_count += 1

        for agent_i, action in enumerate(agents_action):
            self.__update_agent_pos(agent_i, action)

        done = self._step_count >= self._max_steps
        rewards = self.__calculate_rewards()
        self._total_episode_reward = [self._total_episode_reward[i] + rewards[i] for i in range(2)]

        return self._get_obs(), rewards, [done] * self.n_agents, {}

    def __init_positions(self):
        for agent_i in range(self.n_agents):
            self.agent_pos[agent_i] = self.__random_pos()

        for landmark_i in range(self.n_landmarks):
            self.landmark_pos[landmark_i] = self.__random_pos()

    def __random_pos(self):
        return [random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1)]

    def __update_agent_pos(self, agent_i, move):
        curr_pos = self.agent_pos[agent_i]
        next_pos = self.__next_pos(curr_pos, move)

        if self.__is_valid(next_pos):
            self.agent_pos[agent_i] = next_pos

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def __is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def __calculate_rewards(self):
        real_landmark_pos = self.landmark_pos[self._real_landmark_idx]
        rewards = []

        # Get positions
        bad_agent_idx = self.n_good_agents
        bad_agent_pos = self.agent_pos[bad_agent_idx]
        good_agents_pos = [self.agent_pos[i] for i in range(self.n_good_agents)]

        # Calculate distances
        bad_distance = self.__distance(bad_agent_pos, real_landmark_pos)
        good_distances = [self.__distance(pos, real_landmark_pos) for pos in good_agents_pos]
        closest_good_distance = min(good_distances)

        # Calculate the maximum possible distance in the grid for normalization
        max_distance = np.sqrt(self._grid_shape[0]**2 + self._grid_shape[1]**2)

        # Reward calculations
        bad_reward = -bad_distance  # The closer the bad agent is, the higher its reward
        good_reward = -closest_good_distance  # Reward the closest good agent's distance
        good_penalty = -np.log((max_distance - bad_distance) + 1)  # Penalize based on bad agent's distance

        # Combine good reward and penalty
        combined_good_reward = good_reward + good_penalty

        # Append rewards for all teams [good agents, bad agent]
        rewards.append(combined_good_reward)
        rewards.append(bad_reward)

        return rewards

    def __distance(self, pos1, pos2):
        return np.sqrt(np.sum(np.square(np.array(pos1) - np.array(pos2))))


    def _get_obs(self):
        obs = []

        for agent_i in range(self.n_agents):
            agent_obs = []
            for i in range(self.n_agents):
                agent_obs.append(self.agent_pos[i])
            for i in range(self.n_landmarks):
                agent_obs.append(self.landmark_pos[i])
            
            obs.append(agent_obs)
        
        return obs

    def render(self, mode='human'):
        img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size = CELL_SIZE, fill='white')
        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR if agent_i < self.n_good_agents else BAD_AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE, fill='white', margin=0.4)

        for landmark_i in range(self.n_landmarks):
            draw_circle(img, self.landmark_pos[landmark_i], cell_size=CELL_SIZE, fill=REAL_LANDMARK_COLOR if landmark_i == self._real_landmark_idx else FAKE_LANDMARK_COLOR)
            write_cell_text(img, text=str(landmark_i + 1), pos=self.landmark_pos[landmark_i], cell_size=CELL_SIZE, fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
BAD_AGENT_COLOR = ImageColor.getcolor('red', mode='RGB')
REAL_LANDMARK_COLOR = ImageColor.getcolor('green', mode='RGB')
FAKE_LANDMARK_COLOR = ImageColor.getcolor('black', mode='RGB')
CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "STAY",
}

PRE_IDS = {
    'agent': 'A',
    'bad_agent': 'BA',
    'wall': 'W',
    'landmark': 'L',
    'fake_landmark': 'FL',
    'empty': '0'
}
