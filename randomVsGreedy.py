import torch
import numpy as np
from aasma.simple_adversary.simple_adversary import SimpleAdversary
from aasma.agents.random_agent import RandomAgent
from aasma.helper import plot_score, plot_won
from aasma.agents.greedy_adversary import GreedyAdversary
import time

def action_name(action):
    if action == 0:
        return "DOWN"
    elif action == 1:
        return "LEFT"
    elif action == 2:
        return "UP"
    elif action == 3:
        return "RIGHT"
    elif action == 4:
        return "STAY"
    else:
        return "UNKNOWN"

def train():
    plot_scores = []
    plot_games_won = []
    record = -20
    games_won = 0

    # Parameters
    n_good_agents = 2
    n_bad_agents = 1
    n_agents = n_good_agents + n_bad_agents
    grid_shape = (11, 11)
    max_steps = 20

    # Initialize environment
    env = SimpleAdversary(grid_shape=grid_shape, 
                          n_good_agents=n_good_agents,
                          max_steps=max_steps, random_landmark=False, random_agents=False)

    # Initialize agents
    good_agents = [RandomAgent(n_actions=5) for i in range(n_good_agents)]
    bad_agent = [GreedyAdversary(n_actions=5) for _ in range(n_bad_agents)]
    agents = good_agents + bad_agent

    obs = env.reset()
    done = [False] * n_agents
    round = 1
    games = 1

    #while True:
    while games < 200:
        print("Round: ", round)

        actions = []
        states_old = []

        # Actions for good agents
        for i in range(n_good_agents):
            action = good_agents[i].action()
            actions.append(action)

        # Action for the greedy adversary
        action = bad_agent[0].action(obs[n_good_agents], env)
        actions.append(action)

        # Step the environment
        next_obs, rewards, done, _ = env.step(actions)

        env.render(mode='human')
        obs = next_obs

        # Print actions
        for i, agent in enumerate(agents):
            if i >= n_good_agents:
                print("Adversary: ", obs[0][i], " -> ", action_name(actions[i]))
            else:
                print("Agent ", i+1, " -> ", obs[0][i], " -> ", action_name(actions[i]))

        print("Landmark: ", env.landmark_pos[0])
        print("Good Agents: ", rewards[0])
        print("Bad Agent: ", rewards[1])
        print("\n")

        if round == max_steps:
            if rewards[0] > rewards[1]:
                games_won += 1

            print('Game', games, 'Score', rewards[0] - rewards[1])
            round = 0
            games += 1

            plot_scores.append(rewards[0] - rewards[1])
            plot_games_won.append(games_won/games * 100)
            plot_score(plot_scores, filename="random_scores.jpg", title="Game Scores by Random Agents Team vs Greedy Adversary")
            plot_won(plot_games_won, filename="random_wins.jpg", title="Games Won by Random Agents Team vs Greedy Adversary")

        round += 1

        #print("Press Enter to continue to the next iteration...")
        #input()
    print('Done! Games won:', games_won, 'Total games:', games)

if __name__ == '__main__':
    train()
