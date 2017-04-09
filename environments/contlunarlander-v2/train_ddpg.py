import sys
sys.path.append('../../ddpg')
sys.path.append('../')
from ddpg import DDPG
from train import train, set_up
from exploration import OUPolicy
import gym
from gym import wrappers

FLAGS = set_up()

action_bounds = [1, 1]

eps = []
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))

env = gym.make('LunarLanderContinuous-v2')
env = wrappers.Monitor(env, '/tmp/contlunarlander', force=True)

agent = DDPG(action_bounds, 
	eps,
	env.observation_space.shape[0],
	actor_learning_rate=0.0001,
	critic_learning_rate=0.001,
	retrain=FLAGS.retrain,
	log_dir=FLAGS.log_dir)

train(env, agent, FLAGS)
