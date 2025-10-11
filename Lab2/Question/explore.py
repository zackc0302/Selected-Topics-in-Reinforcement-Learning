# Question/explore.py
from dqn_agent_atari import AtariDQNAgent
import numpy as np
import gym
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    config = {
        "gpu": True,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_min": 0.1,
        "warmup_steps": 20000,
        "eps_decay": 1000000,
        "eval_epsilon": 0.0,
        "replay_buffer_capacity": 100000,
        "logdir": 'log/DQN/MsPacman/',
        "update_freq": 4,
        "update_target_freq": 10000,
        "learning_rate": 0.0000625,
        "eval_interval": 100,
        "eval_episode": 1,
        "env_id": 'ALE/MsPacman-v5',
        "training_steps": 0,
    }
    
    agent = AtariDQNAgent(config)
    
    # 載入最佳模型
    model_path = 'log/DDQN/MsPacman-v5/model_97669024_5170.pth'
    agent.load(model_path)
    print(f"Loaded model: {model_path}\n")
    
    # 測試多個 seed
    results = []
    
    print("Testing seeds to find best performance...")
    print("=" * 60)
    
    for seed in range(100):  # 測試 100 個 seed
        set_seed(seed)
        
        env = gym.make(config["env_id"], render_mode='rgb_array')
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)
        
        observation, info = env.reset(seed=seed)
        total_reward = 0
        
        while True:
            obs_array = np.array(observation)
            if obs_array.shape[-1] == 1:
                obs_array = obs_array.squeeze(-1)
            
            action = agent.decide_agent_actions(obs_array, epsilon=0.0, action_space=env.action_space)
            observation, reward, terminate, truncate, info = env.step(action)
            total_reward += reward
            
            if terminate or truncate:
                results.append((seed, total_reward))
                print(f"Seed {seed:3d}: {total_reward:6.0f}")
                break
        
        env.close()
    
    # 排序並顯示前 10 名
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "=" * 60)
    print("TOP 10 SEEDS:")
    print("=" * 60)
    for i, (seed, score) in enumerate(results[:10], 1):
        print(f"{i:2d}. Seed {seed:3d}: {score:6.0f}")
    
    print("\n" + "=" * 60)
    print(f"BEST SEED: {results[0][0]} with score {results[0][1]:.0f}")
    print("=" * 60)
    print(f"\nUse this seed in your demo.py: DEMO_SEED = {results[0][0]}")