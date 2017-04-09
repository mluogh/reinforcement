import sys
sys.path.append('../../ddpg')

from train import start
from exploration import OUPolicy

eps = []
eps.append(OUPolicy(0, 0.15, 0.2))
eps.append(OUPolicy(0, 0.15, 0.2))

env_info = {
    'name': 'Pendulum-v0',
    'action_bounds': [1, 1],
    'exploration_policies': eps 
}

start(env_info)
