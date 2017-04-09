# Various RL Algorithms

Modular framework for testing reinforcement various reinforcement algorithnms. Simply pass an OpenAI environment, an agent, and commandline arguments to train.py. 

##### Tabular Q-Learning
- works well with small discrete state and action spaces

##### Linear Q-Learning
- found it too difficult to hand craft feature functions

##### DDPG
- found experience buffer size to be extremely important
- generally converges fairly well
- still need to implement batch norm
