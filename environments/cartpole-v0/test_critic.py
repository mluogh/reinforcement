import sys
sys.path.append('../../tabular')
sys.path.append('../../dpg')
from qlearner import QLearner
from critic import Critic
from ddpg import ExperienceBuffer
import gym
import numpy as np
import argparse
import tensorflow as tf

env = gym.make('CartPole-v0')

# create bins for this problem
# stole discretizations off the internet, i hate hyperparameters :(
ONE = np.linspace(-4.8, 4.8, 1)
TWO = np.linspace(-10, 10, 1)
THREE = np.linspace(-0.42, 0.42, 3)
FOUR = np.linspace(-0.9, 0.9, 6)

agent = QLearner([ONE, TWO, THREE, FOUR], env.action_space.n, learning_rate=0.01, gamma=0.99, epsilon_decay=0.95)
sess = tf.Session()

critic = Critic(sess, 1, 4, learning_rate=0.001)
exBuf = ExperienceBuffer(10000)

sess.run(tf.global_variables_initializer())

def train():
    running_av = 0
    for episode in range(FLAGS.num_episodes):
        old_state = None
        state = env.reset()
        done = False
        total_reward = 0
        t = 0
        running_q_av = 0

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
                    # print('fail')
                    # print(t)
                    # if episode > 50:
                        # env.render()
                        # input()
            else:
                total_reward += reward

            agent.update(old_state, action, reward, state)
            exBuf.add([old_state, action, reward, state, done])

            old_states, actions, rewards, new_states, is_terminals = exBuf.get_batch(100)

            new_actions = []
            for i in range(len(old_states)):
                new_actions.append(agent.get_action(state))
            critic.update(old_states, actions, rewards, new_states, new_actions, is_terminals)
            # update q vals

            t += 1
            if done:
                break
         
        running_q_av += 0.05 * (critic.loss_val - running_q_av)
        running_av += 0.05 * (total_reward - running_av)

        if episode % 50 == 0:
            print('Average Loss:', running_q_av)
            print('Average Reward:', running_av)
            print('1 point reward', total_reward)

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
