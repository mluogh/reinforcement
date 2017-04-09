import sys
sys.path.append('../../ddpg')
sys.path.append('../')
from ddpg import DDPG
from train import train, set_up
from exploration import OUPolicy
import gym
from gym import wrappers

FLAGS = set_up()

action_bounds = [1, 1, 1, 1, 1, 1]

ep = (OUPolicy(0, 0.15, 2))
eps = [ep] * 6

env = gym.make('HalfCheetah-v1')

agent = DDPG(action_bounds, 
	eps,
	env.observation_space.shape[0],
	actor_learning_rate=0.0001,
	critic_learning_rate=0.001,
	retrain=FLAGS.retrain,
	log_dir=FLAGS.log_dir)

train(env, agent, FLAGS)
