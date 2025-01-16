from ..envs.flappy_env_simple import FlappyBirdEnv
from dqn_agent import DQNAgent
import numpy as np
import os
import matplotlib.pyplot as plt

# Percorsi
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, "results_dqn")
os.makedirs(results_dir, exist_ok=True)

performance_graph_path = os.path.join(results_dir, "dqn_performance.png")

def preprocess_state(state):
    """Prepara lo stato per l'input al modello"""
    return np.array(state, dtype=np.float32)

if __name__ == "__main__":
    env = FlappyBirdEnv()
    state_size = 4  # Dimensione dello stato continuo
    action_size = 2
    agent = DQNAgent(state_size, action_size)

    episodes = 1000
    batch_size = 32
    rewards = []

    for episode in range(episodes):
        state = preprocess_state(env.reset())
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay(batch_size)
        agent.decay_epsilon()
        rewards.append(total_reward)
        print(f"Episode {episode+1}, Total Reward: {total_reward}")

    # Grafico delle ricompense cumulative
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Performance of DQN Agent")
    plt.savefig(performance_graph_path)
    plt.show()
