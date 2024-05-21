import numpy as np
import random
import torch
from collections import deque
from aasma.agents.agent import Agent
from aasma.simple_adversary.simple_adversary import SimpleAdversary
from aasma.model import Linear_QNet, QTrainer
from aasma.helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class QLearningAgent(Agent):

    def __init__(self, n_actions: int, state_size, gamma=0.9, epsilon=0):
        super(QLearningAgent, self).__init__("QLearning Agent")
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(state_size, 256, n_actions)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.n_games = 0

    def get_state(self, game, agent_idx):
        agent_pos = game.agent_pos[agent_idx]
        
        for i in range(game.n_landmarks):
            if i == game._real_landmark_idx:
                landmark_pos = game.landmark_pos[i]
            else:
                fake_landmark_pos = game.landmark_pos[i]
        # other_good_agent_pos = [game.agent_pos[i] for i in range(game.n_good_agents) if i != agent_idx]

        state = [
            # current agent position
            agent_pos[0], agent_pos[1],

            # other good agents positions
            # *[pos for pos in other_good_agent_pos],

            # real landmark position
            landmark_pos[0], landmark_pos[1],

            # fake landmark position
            fake_landmark_pos[0], fake_landmark_pos[1],

            # step count
            game._step_count / game._max_steps
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 4)
            print("Random Action: ", action)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()
            print("Predicted Action: ", action)

        return action