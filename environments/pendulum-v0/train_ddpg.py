import tensorflow as tf
import sys
import os
import shutil
sys.path.append('../../ddpg')
from ddpg import DDPG
import gym
import numpy as np
import argparse

def train():
    env = gym.make('Pendulum-v0')

    # create bins for this problem
    sess = tf.Session()

    agent = DDPG(sess, [2], 3, actor_learning_rate=0.0001, critic_learning_rate=0.001)

    writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())

    running_av = 0
    running_q_loss = 0

    for episode in range(FLAGS.num_episodes):
        old_state = None
        done = False
        total_reward = 0
        t = 0

        state = env.reset() 

        while t < 300:
            if running_av >= FLAGS.show_point:
                env.render()

            old_state = state

            # infer an action
            action = agent.get_action(np.reshape(state, (1, 3)))
            # print(action)
            # take it

            state, reward, done, _ = env.step(action[0])
           
            # if done:
                # if t < 195:
                    # reward = -100
            # else:
               # total_reward += reward
            total_reward += reward
            # update q vals
            agent.update(old_state, action[0], np.array(reward), state, done)

            t += 1
            if done:
                break
         
        running_av += 0.05 * (total_reward - running_av)
        running_q_loss += 0.05 * (agent.critic.loss_val - running_q_loss)

        writer.add_summary(val_to_summary('reward', total_reward), episode)
        writer.add_summary(val_to_summary('loss', agent.critic.loss_val), episode)
        writer.flush()

        if episode % 50 == 0:
            print('r av', running_av)
            print('loss', running_q_loss)

def val_to_summary(tag, value):
    return tf.Summary(value=[
        tf.Summary.Value(tag=tag, simple_value=value), 
    ])


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

    parser.add_argument(
        '--wipe_logs',
        default=False,
        action='store_true',
        help='Wipe logs or not'
    )

    parser.add_argument(
        '--log_dir',
        default='./logs',
        help='Where to store logs'
    )

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.wipe_logs and os.path.exists(FLAGS.log_dir):
        shutil.rmtree(FLAGS.log_dir)

    train()
