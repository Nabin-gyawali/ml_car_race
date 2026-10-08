from env import CarWorld
import numpy as np



def state_id(state):
    return state[0] * 4 + state[1] * 2 + state[2] * 1

def choose_action(qtable, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(3)  # Random action
    else:
        return np.argmax(qtable[state_id(state)])  # Best action


def run_q_learning(env , Q_table, alpha=0.1, epsilon=0.1, gamma=0.9, episodes=1000):
    qtable = Q_table
    Environment = env
    for episode in range(episodes):
        initial_state , info  = Environment.reset()
        done = False
        curr_state = initial_state
        while not done:
            action = choose_action(qtable, curr_state, epsilon)
            next_state, reward, finished, crashed, info = Environment.step(action)

            if finished or crashed:
                done = True
                target = reward
            else :
                target = reward + gamma * np.max(qtable[state_id(next_state)])
            
            td_error = target - qtable[state_id(curr_state), action]

            # Update Q-table using Q-learning update rule
            qtable[state_id(curr_state), action] += alpha * td_error

            curr_state = next_state


if __name__ == "__main__":
    env = CarWorld(render_mode="human", random_seed=42)
    qtable = np.zeros((8, 3))  # 8 states and 3 actions
    run_q_learning(env, qtable, alpha=0.1, epsilon=0.1, gamma=0.9, episodes=1000)
    print("Trained Q-table:")
    print(qtable)

    # render the environment after training
    initial_obs , info = env.reset()
    done = False
    while not done:
        action = choose_action(qtable, initial_obs, epsilon=0)  # Choose best action (no exploration)
        next_obs, reward, finished, crashed, info = env.step(action)
        env.render()
        if finished or crashed:
            done = True
        initial_obs = next_obs