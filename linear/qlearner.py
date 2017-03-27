import numpy as np

# define a linear qlearner
class QLearner():
    weights = None
    action_space = None
    exploration_decay = 0.99

    def __init__(self, feature_space, action_space, discount=0.99, learning_rate=0.01, epsilon=0.5):
        self.weights = (np.random.rand(action_space, feature_space) * 2 - 1)/ 1000
        self.action_space = action_space
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def update_weights(self, old_state, new_state, reward, done):
        self.epsilon = self.epsilon * self.exploration_decay

        new_max_q = self.infer_q(new_state)[1]

        if not done:
            error = reward + (self.discount * new_max_q) - self.last_q
        else:
            error = reward - self.last_q
        
        self.weights[self.last_action] += self.learning_rate * error * old_state

    def infer_q(self, state):
        # return the action with the largest q value and the actual q value
        q_values = self.weights.dot(state)
        # print(q_values)
        return np.argmax(q_values), np.amax(q_values)

    def get_action(self, state):
        # randomly explore
        if (np.random.rand(1) < self.epsilon):
            self.last_action = np.random.randint(self.action_space)
            self.last_q = self.weights.dot(state)[self.last_action]

        self.last_action, self.last_q = self.infer_q(state)
        return self.last_action
