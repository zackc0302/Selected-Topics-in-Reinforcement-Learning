import numpy as np
import torch
import torch.nn as nn
import gym

env = gym.make("ALE/Enduro-v5", render_mode="human")

# if you don't have GUI, you can use the following code to record video
# env = gym.make("ALE/Enduro-v5", render_mode="rgb_array")
# env = gym.wrappers.RecordVideo(env, 'video')

observation, info = env.reset()

total_reward = 0
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    # disable render() when you are recording video
    env.render()
    if terminated or truncated:
        observation = env.reset()

print("Total reward: {}".format(total_reward))
env.close()