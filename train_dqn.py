from envs.flappy_env_simple import FlappyBirdEnv
from agents.dqn_agent import DQNAgent
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
    state_size = 3  # Dimensione dello stato continuo
    action_size = 2
    epsilon_decay = 0.995
    agent = DQNAgent(state_size, action_size)

    episodes = 1000
    batch_size = 16
    rewards = []
    max_reward = float("-inf")
    max_score = max_reward = float("-inf")
    for episode in range(episodes):
        state = preprocess_state(env.reset()[0])
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay(batch_size)
        agent.decay_epsilon()
        rewards.append(total_reward)
        score = info["score"]
        max_reward = max(total_reward, max_reward)
        max_score = max(score, max_score)
        print(f"Episode {episode+1}, Total Reward: {total_reward} (Max: {max_reward}), Score: {score} (Max: {max_score}), Epsilon: {agent.epsilon}")

    # Grafico delle ricompense cumulative
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Performance of DQN Agent")
    plt.savefig(performance_graph_path)
    plt.show()
