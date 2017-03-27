import numpy as np

class QLearner():
    # contains tuples of (state, action) with corresponding Q_value
    q_table = {}

    def __init__(self, bins, action_space, learning_rate=0.1, gamma=0.999, epsilon=0.5, epsilon_decay=0.9):
        self.bins = bins
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.99

    def discretize_state(self, state):
        assert len(state) == len(self.bins), 'Bins must be same length as state'
        d_state = []

        for i in range(len(self.bins)):
            d_state.append(int(np.digitize(state[i], self.bins[i])))

        d_state = tuple(d_state)

        return d_state

    def get_q(self, d_state, action):
        if (d_state, action) not in self.q_table:
            self.q_table[(d_state, action)] = 0

        return self.q_table[(d_state, action)]

    def get_action(self, state):
        if np.random.rand(1) < self.epsilon:
            return np.random.randint(self.action_space)

        d_state = self.discretize_state(state)
        return self.get_max_qpair(d_state)[0]

    def get_max_qpair(self, d_state):
        actions = []
        
        for action in list(range(self.action_space)):
            actions.append(self.get_q(d_state, action))

        return np.argmax(actions), np.amax(actions)

    def update(self, old_state, action, new_state, reward):
        self.epsilon = self.epsilon * self.epsilon_decay
        old_state, new_state = self.discretize_state(old_state), self.discretize_state(new_state)

        old_q = self.get_q(old_state, action)
        new_q = self.get_max_qpair(new_state)[1]

        self.q_table[(old_state, action)] = (1 - self.learning_rate) * self.q_table[(old_state, action)] + self.learning_rate * (reward + self.gamma * new_q - old_q)
