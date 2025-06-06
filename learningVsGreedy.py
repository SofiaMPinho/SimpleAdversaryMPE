import torch
import numpy as np
from aasma.simple_adversary.simple_adversary import SimpleAdversary
from aasma.agents.learning_agent import QLearningAgent
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
    good_agents = [QLearningAgent(n_actions=5, agent_id=i) for i in range(n_good_agents)]
    bad_agent = [GreedyAdversary(n_actions=5) for _ in range(n_bad_agents)]
    agents = good_agents + bad_agent

    obs = env.reset()
    done = [False] * n_agents
    round = 0

    while True:
        #while agents[0].n_games < 500:
        round += 1
        print("Round: ", round)

        actions = []
        states_old = []

        # Actions for good agents (RL agents)
        for i in range(n_good_agents):
            state = good_agents[i].get_state(env, i)
            states_old.append(state)
            action = good_agents[i].action(state)
            actions.append(action)

        # Action for the greedy adversary
        action = bad_agent[0].action(obs[n_good_agents], env)
        actions.append(action)

        # Step the environment
        next_obs, rewards, done, _ = env.step(actions)

        # Train RL agents
        for i in range(n_good_agents):
            reward = 10 + rewards[0] - rewards[1] if rewards[0] > rewards[1] else -10 + rewards[0] - rewards[1] # are the good agents winning? by how much?
            state_new = good_agents[i].get_state(env, i)
            good_agents[i].train_short_memory(states_old[i], actions[i], reward, state_new, done[i])
            good_agents[i].remember(states_old[i], actions[i], reward, state_new, done[i])

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
        print("Learning reward: ", reward)
        print("\n")

        if any(done):
            env.reset()
            for agent in good_agents:
                agent.n_games += 1
                agent.train_long_memory()
                if reward > record:
                    agent.save_model()

            if reward > record:
                record = reward

            if rewards[0] > rewards[1]:
                games_won += 1 

            print('Game', good_agents[0].n_games, 'Score', rewards[0] - rewards[1])
            round = 0

            plot_scores.append(rewards[0] - rewards[1])
            plot_games_won.append(games_won/agent.n_games * 100)
            plot_score(plot_scores, title="Game Scores by Learning Agents Team vs Greedy Adversary")
            plot_won(plot_games_won, title="Games Won by Learning Agents Team vs Greedy Adversary")

        # uncomment the following to see the game step by step:
        # print("Press Enter to continue to the next iteration...")
        # input()

if __name__ == '__main__':
    train()
