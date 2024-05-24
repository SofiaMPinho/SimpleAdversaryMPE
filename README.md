# Manipulation of an Agent using Physical Deception

This program simulates a multi-agent environment where t here are two teams: one with 2 agents (good team) and the other with 1 agent (adversary), and 2 landmarks (a real and a fake one).
They strive to be as close as possible to the real landmark at the end of a cycle of movement.
The good agents can tell which is the real landmark, but the adversary cannot.
The good agents are rewarded by closeness to the real landmark and penalized by the bad agent's closeness to the real landmark.
The adversary is rewarded by closeness to the real landmark. All rewards are given at the end of the cycle.

## Installation

1. Enter the Python virtual environment:
    ```shell
    $ source venv/bin/activate
    ```

2. Install the required dependencies:
    ```shell
    $ pip install -r requirements.txt
    ```

## Usage

To observe random agents against the greedy adversary, run the following command:

 ```shell
    $ python3 randomVsGreedy.py
 ```

To observe learning agents against the greedy adversary, run the following:

 ```shell
    $ python3 learningVsGreedy.py
 ```

 note: in line 39 the flag random_agents=False can be changed so the agents spawn in random positions each game.