import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        """
        Inizializza l'agente Q-Learning.
        - state_size: Numero totale di stati discreti.
        - action_size: Numero totale di azioni possibili.
        - alpha: Tasso di apprendimento.
        - gamma: Fattore di sconto (quanto l'agente considera le ricompense future).
        - epsilon: Probabilità di esplorazione iniziale.
        - epsilon_decay: Riduzione graduale di epsilon (per meno esplorazione nel tempo).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01  # Limite minimo per epsilon
        self.q_table = np.zeros((state_size, action_size))  # Tabella Q inizializzata a zero

    def select_action(self, state):
        """
        Sceglie un'azione usando una politica epsilon-greedy.
        - Se epsilon è alto: l'agente esplora (sceglie azioni casuali).
        - Altrimenti: sfrutta le migliori azioni conosciute.
        """
        if np.random.rand() < self.epsilon:  # Esplorazione
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])  # Sfruttamento

    def update(self, state, action, reward, next_state, done):
        """
        Aggiorna la tabella Q usando la formula del Q-Learning.
        - state: Stato corrente.
        - action: Azione intrapresa.
        - reward: Ricompensa ricevuta.
        - next_state: Stato successivo.
        - done: True se l'episodio è terminato.
        """
        best_next_action = np.argmax(self.q_table[next_state])  # Miglior azione futura
        td_target = reward + (self.gamma * self.q_table[next_state, best_next_action] * (not done))  # Obiettivo temporale
        td_error = td_target - self.q_table[state, action]  # Errore temporale
        self.q_table[state, action] += self.alpha * td_error  # Aggiorna il valore Q

    def decay_epsilon(self):
        """
        Riduce epsilon per diminuire gradualmente l'esplorazione.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_q_table(self, filename):
        """
        Salva la tabella Q in un file.
        """
        np.save(filename, self.q_table)

    def load_q_table(self, filename):
        """
        Carica la tabella Q da un file.
        """
        self.q_table = np.load(filename)