# Various RL Algorithms

Modular framework for testing reinforcement various reinforcement algorithnms. Simply pass an OpenAI environment, an agent, and commandline arguments to train.py. 

##### Tabular Q-Learning
- works well with small discrete state and action spaces
- choice of discretization matters a lot
- not very good for larger state and action spaces (i.e. continuous)

##### Linear Q-Learning
- found it too difficult to hand craft feature functions
- doesn't really work, just moved on to ddpg
- shows power of deep learning - features can be learned

##### DDPG
- found experience buffer size to be extremely important
- generally converges fairly well
- traning is unstable, which makes sense since you're approximating the optimal policy off an approximation of the Q function. 
- still need to implement batch norm (not particularly important though, networks aren't that deep)

NOTE: I didn't have time to recreate things like DQN or A3C because the school year started getting heavier and research started getting more involved. I eventually want to implement these all in PyTorch. 
