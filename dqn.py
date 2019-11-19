# import gym
import random
import torch
import numpy as np
from collections import deque
import json


from unityagents import UnityEnvironment
import numpy as np

from dqn_agent import Agent


env = UnityEnvironment(file_name="./Banana.app")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# env = gym.make('LunarLander-v2')
# env.seed(0)

# print('State shape: ', env.observation_space.shape)
# print('Number of actions: ', env.action_space.n)


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]


# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


# from dqn_agent import Agent




filename = 'checkpoint'

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, max_score = 200.0, layers_neurones = 64):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    agent = Agent(state_size=state_size, action_size=action_size, seed=0, layers_neurones = layers_neurones)

    filename = f'./results/n_episodes={n_episodes}, max_t={max_t}, eps_start={eps_start}, eps_end={eps_end}, eps_decay={eps_decay}, max_score = {max_score}, layers_neurones = {layers_neurones}'
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        # state = env.reset()

        env_info = env.reset(train_mode=True)[brain_name]
        score = 0
        for t in range(max_t):
            state = env_info.vector_observations[0]
            action = agent.act(state, eps)

            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            # next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            with open(f'{filename}.json', 'w') as filehandle:
                json.dump(scores, filehandle)            

        if np.mean(scores_window)>=max_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), f'{filename}.pth')
            with open(f'{filename}.json', 'w') as filehandle:
                json.dump(scores, filehandle)            
            break
    torch.save(agent.qnetwork_local.state_dict(), f'{filename}.pth')
    return scores

# scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.9999, layers_neurones=64)
# scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.999, layers_neurones=64)
# scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.995, layers_neurones=64)
# scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.99, layers_neurones=64)
# scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.9, layers_neurones=64)

scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.1, layers_neurones=64)
scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.2, layers_neurones=64)
scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.4, layers_neurones=64)
scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.8, layers_neurones=64)
scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.7, layers_neurones=64)
scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.6, layers_neurones=64)
scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.5, layers_neurones=64)


# scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.995, layers_neurones=32)
# scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.995, layers_neurones=128)
# scores = dqn(max_score=400.0, n_episodes = 2000, eps_decay=0.995, layers_neurones=256)


# plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()