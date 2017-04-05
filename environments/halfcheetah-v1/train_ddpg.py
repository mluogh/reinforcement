import sys
import os
import shutil
sys.path.append('../../ddpg')
from ddpg import DDPG
import gym
import numpy as np
import argparse

def train():
    env = gym.make('HalfCheetah-v1')
    obs_size = env.observation_space.shape[0]

    agent = DDPG([1, 1, 1, 1, 1, 1], 
            obs_size,
            actor_learning_rate=0.0001,
            critic_learning_rate=0.001,
            retrain=FLAGS.retrain,
            log_dir=FLAGS.log_dir)

    for episode in range(1, FLAGS.num_episodes + 1):
        old_state = None
        done = False
        total_reward = 0

        state = env.reset() 

        for t in range(env.spec.max_episode_steps):
            if FLAGS.show:
                env.render()

            old_state = state

            # infer an action
            action = agent.get_action(np.reshape(state, (1, obs_size)), not FLAGS.test)

            # take it
            state, reward, done, _ = env.step(action[0])
           
            total_reward += reward

            if not FLAGS.test:
                # update q vals
                agent.update(old_state, action[0], np.array(reward), state, done)

            if done:
                break

        agent.log_data(total_reward, episode)

        if episode % 100 == 0 and not FLAGS.test:
            print('Saved model at episode', episode)
            agent.save_model(episode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_episodes',
        type=int,
        default=1000,
        help='How many episodes to train for'
    )

    parser.add_argument(
        '--show',
        default=False,
        action='store_true',
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

    parser.add_argument(
        '--retrain',
        default=False,
        action='store_true',
        help='Whether to start training from scratch again or not'
    )

    parser.add_argument(
        '--test',
        default=False,
        action='store_true',
        help='Test more or no (true = no training updates)'
    )

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.wipe_logs and os.path.exists(FLAGS.log_dir):
        shutil.rmtree(FLAGS.log_dir)

    train()
