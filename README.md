[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent, source: Udacity"


This project applies a deep Q learning agent, one of reinforcement learning technique to play a game, 
which consists to navigate (and collect bananas!) in a large, square world.

## Getting Started

1. You should have python 3.7 installed on your machine.

2. Clone the repo into a local directory

    Then cd in it:
    `cd deep-q-networks`


3. To install the dependencies, we advise you to create an environment.
    If you use conda, juste run:
    `conda create --name dqn python=3.7`
    to create `dqn` environment

    Then install requirements files:
    `pip install -r requirements.txt`

4. Run the training file to train the agent, and see progress in training along time
`python dqn.py`


## Reinforcement learning
![Trained Agent][image1]
_(source from udacity: https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)_


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
0. - move forward.
1. - move backward.
2. - turn left.
3. - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.