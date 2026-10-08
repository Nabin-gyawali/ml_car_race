import numpy as np
from env import CarWorld
import sarsa, Qlearning

env = CarWorld(render_mode=None, random_seed=42)
q1 = np.zeros((8, 3))
sarsa.run_sarsa(env, q1, alpha=0.1, epsilon=0.1, gamma=0.9, episodes=1000)
print("SARSA Q-table:")
print(np.round(q1, 2))

env2 = CarWorld(render_mode=None, random_seed=42)
q2 = np.zeros((8, 3))
Qlearning.run_q_learning(env2, q2, alpha=0.1, epsilon=0.1, gamma=0.9, episodes=1000)
print("Q-LEARNING Q-table:")
print(np.round(q2, 2))
