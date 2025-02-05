import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, alpha=1e-2, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size  # Size of the state space
        self.action_size = action_size  # Size of the action space
        self.gamma = gamma  # Discount factor
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.memory = deque(maxlen=5000)  # Replay memory
        self.model = self._build_model()  # Neural network model

    def _build_model(self):
        """Build the neural network model"""
        model = Sequential()
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),  # Input layer
            Dense(24, activation='relu'),  # Hidden layer
            Dense(self.action_size, activation='linear')  # Output layer
        ])
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(name="1", units=1024, activation="relu"))
        # model.add(Dense(name="2", units=512, activation="relu"))
        # model.add(Dense(name="3", units=256, activation="relu"))
        # model.add(Dense(name="4", units=128, activation="relu"))
        # model.add(Dense(name="5", units=64, activation="relu"))
        # model.add(Dense(name="6", units=32, activation="relu"))
        # model.add(Dense(name="7", units=16, activation="relu"))
        # model.add(Dense(name="8", units=8, activation="relu"))
        # model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.alpha), loss='mse')  # Compile the model
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store an experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action using epsilon-greedy strategy"""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)  # Random action (exploration)
        else:
            state = np.reshape(state, [1, self.state_size])  # Reshape state for the model
            q_values = self.model.predict(state, verbose=0)  # Predict Q-values
            return np.argmax(q_values[0])  # Choose the action with the highest Q-value

    def replay(self, batch_size):
        """Train the model using a batch of experiences from replay memory"""
        if len(self.memory) < batch_size:
            return  # Not enough experiences to sample a batch

        # Sample a batch of experiences from replay memory
        minibatch = random.sample(self.memory, batch_size)

        # Extract states, actions, rewards, next_states, and done flags from the batch
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for the current states and next states
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        # Update Q-values using the Bellman equation
        for i in range(batch_size):
            if dones[i]:
                # If the episode is done, there is no future reward
                current_q_values[i][actions[i]] = rewards[i]
            else:
                # Use the Bellman equation to update the Q-value
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the model on the updated Q-values
        self.model.fit(states, next_q_values, batch_size=batch_size, verbose=0)

    def decay_epsilon(self):
        """Decay the exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay