import numpy as np
from aasma.agents.random_agent import RandomAgent
from aasma.agents.greedy_adversary import GreedyAdversary
from aasma.agents.greedy_agent import GreedyAgent
from aasma.agents.deceptive_agent import DeceptiveAgent
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
    max_steps = 20

    # Initialize environment
    env = SimpleAdversary(grid_shape=grid_shape, 
                          n_good_agents=n_good_agents,
                          max_steps=max_steps)

    # Initialize agents
    # good_agents = [GreedyAgent(n_actions=5) for _ in range(n_good_agents)]
    greedy_agent = [GreedyAgent(n_actions=5,) for _ in range(1)]
    deceptive_agents = [DeceptiveAgent(n_actions=5, ) for _ in range(n_good_agents-1)]

    bad_agents = [GreedyAdversary(n_actions=5) for _ in range(n_bad_agents)]
    agents = greedy_agent + deceptive_agents + bad_agents

    obs = env.reset()
    done = [False] * n_agents
    round = 0

    while not all(done):
        round += 1
        print("Round: ", round)
        
        actions = []
        for i, agent in enumerate(agents):
            agent.see(obs[i])
            if i == 0:
                actions.append(agent.action(obs[i], n_agents, i)) # greedy agent
            elif i < n_good_agents:
                actions.append(agent.action(obs[i], n_agents, i)) # deceptive agents
            else:
                actions.append(agent.action(obs[i], env)) # bad agent

        obs, rewards, done, _ = env.step(actions)
        env.render(mode='human')

        for i, agent in enumerate(agents):
            print("Agent ", i, " -> ", obs[0][i], " -> ", action_name(actions[i]))

            if i >= n_good_agents:
                print("Bad Agent: ", obs[0][i], " -> ", action_name(actions[i]))

        print("Landmark: ", env.landmark_pos[1])
        print("Good Agents: ", rewards[0])
        print("Bad Agent: ", rewards[1])
        print("\n")

        # wait 1 seconds between steps
        time.sleep(1)


    env.close()

if __name__ == "__main__":
    main()