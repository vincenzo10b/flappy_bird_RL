import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, alpha=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Fattore di sconto
        self.alpha = alpha  # Tasso di apprendimento
        self.epsilon = epsilon  # Esplorazione iniziale
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=2000)  # Replay memory
        self.model = self._build_model()

    def _build_model(self):
        """Costruisce la rete neurale"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        """Salva l'esperienza nella memoria"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Sceglie un'azione usando epsilon-greedy"""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.numpy())

    def replay(self, batch_size):
        """Addestra il modello usando esperienze casuali dalla memoria"""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state).detach().clone()
            target_f[action] = target
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
            optimizer.zero_grad()
            loss = criterion(self.model(state), target_f)
            loss.backward()
            optimizer.step()

    def decay_epsilon(self):
        """Riduce epsilon per diminuire l'esplorazione"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
