import sys
sys.path.append('../../tabular')
from qlearner import QLearner
import gym
import numpy as np
import argparse

env = gym.make('CartPole-v0')

# create bins for this problem
# stole discretizations off the internet, i hate hyperparameters :(
ONE = np.linspace(-4.8, 4.8, 1)
TWO = np.linspace(-10, 10, 1)
THREE = np.linspace(-0.42, 0.42, 3)
FOUR = np.linspace(-0.9, 0.9, 6)

agent = QLearner([ONE, TWO, THREE, FOUR], env.action_space.n, learning_rate=0.01, gamma=0.99, epsilon_decay=0.95)

def train():
    running_av = 0
    for episode in range(FLAGS.num_episodes):
        old_state = None
        state = env.reset()
        done = False
        total_reward = 0
        t = 0

        while t < 300:
            if running_av > FLAGS.show_point:
                env.render()

            old_state = state
            # infer an action
            action = agent.get_action(state)
            # take it
            state, reward, done, _ = env.step(action)
            if done:
                if t < 195:
                    reward = -100
            else:
                total_reward += reward
            # update q vals
            agent.update(old_state, action, reward, state)

            if done:
                break
         
        running_av += 0.05 * (total_reward - running_av)

        if episode % 50 == 0:
            print(running_av)
            print(total_reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_episodes',
        type=int,
        default=1000,
        help='How many episodes to train for'
    )

    parser.add_argument(
        '--show_point',
        type=float,
        default=200,
        help='At what point to render the cart environment'
    )

    FLAGS, unparsed = parser.parse_known_args()
    train()
