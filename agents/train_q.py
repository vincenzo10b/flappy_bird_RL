from q_agent import QLearningAgent
from ..envs.flappy_env_simple import FlappyBirdEnv
import matplotlib.pyplot as plt
import numpy as np
import os

# Determina la directory base del progetto
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Percorsi relativi alle altre directory
results_dir = os.path.join(base_dir, "results")  # Directory per salvare i risultati
os.makedirs(results_dir, exist_ok=True)         # Crea la directory se non esiste

q_table_path = os.path.join(results_dir, "q_table.npy")  # File della tabella Q
performance_graph_path = os.path.join(results_dir, "q_learning_performance.png")  # File del grafico


def discretize_state(state):
    """
    Discretizza lo stato continuo (es. posizione, velocità) in un insieme finito di stati.
    """
    bird_y, bird_v, pipe_dist, pipe_y = state
    bird_y_discrete = int(bird_y // 10)
    bird_v_discrete = int(bird_v // 5)
    pipe_dist_discrete = int(pipe_dist // 20)
    pipe_y_discrete = int(pipe_y // 10)
    return bird_y_discrete * 1000 + bird_v_discrete * 100 + pipe_dist_discrete * 10 + pipe_y_discrete

if __name__ == "__main__":
    env = FlappyBirdEnv()  # Crea l'ambiente
    state_size = 10000  # Numero totale di stati discreti
    action_size = 2  # Due azioni: salta o non salta
    agent = QLearningAgent(state_size, action_size)

    # Se esiste una tabella Q salvata, caricala
    if os.path.exists(q_table_path):
        agent.load_q_table(q_table_path)
        print("Tabella Q caricata con successo.")

    episodes = 1000
    rewards = []

    for episode in range(episodes):
        state = discretize_state(env.reset())  # Stato iniziale
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)  # Scegli un'azione
            next_state, reward, done, _ = env.step(action)  # Esegui l'azione nell'ambiente
            next_state = discretize_state(next_state)  # Discretizza lo stato successivo
            agent.update(state, action, reward, next_state, done)  # Aggiorna la tabella Q
            state = next_state
            total_reward += reward

        agent.decay_epsilon()  # Riduci epsilon
        rewards.append(total_reward)  # Salva la ricompensa cumulativa
        print(f"Episode {episode+1}, Total Reward: {total_reward}")

    # Crea la directory results se non esiste
    os.makedirs("results", exist_ok=True)

    # Salva la tabella Q al termine dell'addestramento
    agent.save_q_table(q_table_path)
    print("Tabella Q salvata con successo.")

    # Grafico delle ricompense cumulative
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Performance of Q-Learning Agent")
    plt.savefig(performance_graph_path)
    plt.show()
    print(f"Grafico salvato in {performance_graph_path}.")
