import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
import imageio.v2 as imageio
import pygame
import env as env_module
env_module.FPS = 15
from env import CarWorld
import sarsa, Qlearning

def train_and_record(name, run_fn, out_base):
    train_env = CarWorld(render_mode=None, random_seed=42)
    qtable = np.zeros((8, 3))
    run_fn(train_env, qtable, alpha=0.1, epsilon=0.1, gamma=0.9, episodes=1000)
    best = []
    for seed in [7, 1, 3, 11, 21, 42, 5, 8]:
        env = CarWorld(render_mode="human", random_seed=seed)
        state, info = env.reset()
        frames = []
        done = False
        finished = crashed = False
        while not done:
            action = int(np.argmax(qtable[(lambda s: s[0]*4 + s[1]*2 + s[2])(state)]))
            state, reward, finished, crashed, info = env.step(action)
            env.render()
            surface = pygame.display.get_surface()
            if surface is not None:
                frames.append(pygame.surfarray.array3d(surface).transpose(1, 0, 2))
            done = finished or crashed
        print(name, "seed", seed, len(frames), finished, crashed)
        if finished and len(frames) > len(best):
            best = frames
    frames = best
    with imageio.get_writer(out_base + ".mp4", fps=15) as w:
        for f in frames:
            w.append_data(f)
    imageio.mimsave(out_base + ".gif", frames[::2], fps=10, loop=0)
    print("saved", out_base)

train_and_record("sarsa", sarsa.run_sarsa, "sarsa_agent")
train_and_record("qlearning", Qlearning.run_q_learning, "qlearning_agent")
