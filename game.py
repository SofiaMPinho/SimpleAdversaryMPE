import numpy as np
from aasma.random_agent import RandomAgent
from aasma.greedy_adversary import GreedyAdversary
from aasma.simple_adversary.simple_adversary import SimpleAdversary
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


def main():
    # Parameters
    n_good_agents = 2
    n_bad_agents = 1
    n_agents = n_good_agents + n_bad_agents
    grid_shape = (10, 10)
    max_steps = 10

    # Initialize environment
    env = SimpleAdversary(grid_shape=grid_shape, 
                          n_good_agents=n_good_agents,
                          max_steps=max_steps)

    # Initialize agents
    good_agents = [RandomAgent(n_actions=5) for _ in range(n_good_agents)]
    bad_agents = [GreedyAdversary(n_actions=5) for _ in range(n_bad_agents)]
    agents = good_agents + bad_agents

    obs = env.reset()
    done = [False] * n_agents
    round = 0

    while not all(done):
        round += 1
        print("Round: ", round)
        
        actions = []
        for i, agent in enumerate(agents):
            agent.see(obs[i])
            if i < n_good_agents:
                actions.append(agent.action())
            else:
                actions.append(agent.action(obs[i], n_agents, grid_shape))

        print("Actions: ", actions)

        obs, rewards, done, _ = env.step(actions)
        env.render(mode='human')
        
        print("Agent 1: ", obs[0][0], " -> ", action_name(actions[0])) 
        print("Agent 2: ", obs[0][1], " -> ", action_name(actions[1]))
        print("Adversary: ", obs[0][2], " -> ", action_name(actions[2]))
        print("Landmark: ", env.landmark_pos[1])
        print("Good Agents: ", rewards[0])
        print("Bad Agent: ", rewards[1])
        print("\n")

        # wait 2 seconds between steps
        time.sleep(2)


    env.close()

if __name__ == "__main__":
    main()