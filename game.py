import numpy as np
from aasma.random_agent import RandomAgent
from aasma.simple_adversary.simple_adversary import SimpleAdversary
import time

def main():
    # Parameters
    n_good_agents = 2
    n_bad_agents = 1
    n_agents = n_good_agents + n_bad_agents
    grid_shape = (5, 5)
    n_landmarks = 2
    max_steps = 100

    # Initialize environment
    env = SimpleAdversary(grid_shape=grid_shape, 
                          n_good_agents=n_good_agents, 
                          n_bad_agents=n_bad_agents, 
                          n_landmarks=n_landmarks, 
                          max_steps=max_steps)

    # Initialize agents
    good_agents = [RandomAgent(n_actions=5) for _ in range(n_good_agents)]
    bad_agents = [RandomAgent(n_actions=5) for _ in range(n_bad_agents)]
    agents = good_agents + bad_agents

    obs = env.reset()
    done = [False] * n_agents

    while not all(done):
        actions = []
        for i, agent in enumerate(agents):
            agent.see(obs[i])
            actions.append(agent.action())

        obs, rewards, done, _ = env.step(actions)
        env.render(mode='human')
        #wait 2 seconds between steps
        time.sleep(2)

    env.close()

if __name__ == "__main__":
    main()
