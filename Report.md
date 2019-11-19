## Improvement in agent performance varying hyperparameters:

Insert at the end of dqn file simulations you wish to run, then run `python dqn.py`.
Results will be stored into ./results directory, as json list of episode reward for each episode during training.
Then produce analyses with the Jupyter Notebook 'HyperParametersTuning. Result analysis.ipynb'.

We tuned 2 types of hyperparameters in order to optimize agent training performance:


1. `epsilon parameter decreasing factor`: multiplicative factor (per episode) for decreasing epsilon-greedy action selection.
    A high decrease in epsilon favor the model learning, as we should expect.

    ![epsilon_size_effect](epsilon_size_effect.png)



2. `number of neurones in network layer`

    32 to 64 neurones give better results, well suited to the model states dimensionality.

    ![neurone_size_effect](neurone_size_effect.png)


## Most successfull agent weights:
    We saved pytorch agent weights into file: `successful_agent_weights.pth` at root directory.
    For this best agent, we reach the target of a reward of 13 for less than 250 episodes.
    

## Next steps for improvements :
    Evolutions in DQN might improve current agant performance, like a combination of evolutions of plain vanilha dqn, gathered into a rainbow model.
    Modifying inputs to consider a full pixel image might as well improve performance, as more informations would be available to take decisions. In this case, neural network structure should be modified to contain convolution units to detect pattern in input image.