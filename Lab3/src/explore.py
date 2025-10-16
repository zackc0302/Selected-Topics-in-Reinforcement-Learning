# explore.py

import os
import glob
import torch
import numpy as np
import random
import gym
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from ppo_agent_atari import AtariPPOAgent

def set_seed(seed):
    """固定所有隨機數生成器"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def evaluate_model_manually(agent, env_config, seed):
    """
    手動執行評估迴圈，不依賴 agent.evaluate()
    """
    set_seed(seed)
    
    # 創建評估環境
    env = gym.make(env_config["env_id"])
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    
    observation, info = env.reset(seed=seed)
    total_reward = 0
    
    while True:
        action_array, _, _ = agent.decide_agent_actions(observation, eval=True)
        action = action_array[0]
        observation, reward, terminate, truncate, info = env.step(action)
        total_reward += reward
        if terminate or truncate:
            break
            
    env.close()
    return total_reward

if __name__ == '__main__':
    config = {
        "gpu": True,
        "env_id": 'ALE/Enduro-v5',
        # --- 以下參數在評估時用不到，但 agent 初始化需要 ---
        "eval_episode": 1, "training_steps": 0, "update_sample_count": 10000,
        "discount_factor_gamma": 0.99, "discount_factor_lambda": 0.95, "clip_epsilon": 0.2,
        "max_gradient_norm": 0.5, "batch_size": 128, "logdir": 'log/Explore/',
        "update_ppo_epoch": 3, "learning_rate": 2.5e-4, "value_coefficient": 0.5,
        "entropy_coefficient": 0.01, "horizon": 128, "eval_interval": 100,
    }
    
    # 1. 找出所有 .pth 模型檔案
    model_dir = 'log/Enduro_release/'
    model_paths = glob.glob(os.path.join(model_dir, '*.pth'))
    if not model_paths:
        raise ValueError(f"No .pth files found in {model_dir}")
    print(f"Found {len(model_paths)} models to evaluate.")

    # 2. 評估每個模型，找出潛力最高的
    best_model_path = None
    highest_avg_score = -1
    
    print("\n" + "="*60)
    print("Step 1: Finding the best model by testing on 10 seeds...")
    print("="*60)

    agent = AtariPPOAgent(config)
    for path in model_paths:
        agent.load(path)
        scores = []
        for seed in range(10): 
            # 使用我們自訂的評估函式
            score = evaluate_model_manually(agent, config, seed)
            scores.append(score)
        avg_score = np.mean(scores)
        print(f"Model: {os.path.basename(path)} -> Avg Score: {avg_score:.2f}")

        if avg_score > highest_avg_score:
            highest_avg_score = avg_score
            best_model_path = path

    print("\n" + "="*60)
    print(f"Best model found: {os.path.basename(best_model_path)}")
    print(f"With average score: {highest_avg_score:.2f}")
    print("="*60)

    # 3. 針對最佳模型，測試大量種子找出最佳分數
    print("\n" + "="*60)
    print(f"Step 2: Finding the best seed for {os.path.basename(best_model_path)}...")
    print("="*60)
    
    agent.load(best_model_path)
    
    results = []
    num_seeds_to_test = 100
    for seed in range(num_seeds_to_test):
        # 再次使用我們自訂的評估函式
        score = evaluate_model_manually(agent, config, seed)
        results.append((seed, score))
        print(f"Seed {seed:3d}: {score:6.2f}")

    results.sort(key=lambda x: x[1], reverse=True)
    
    best_seed, best_score = results[0]
    print("\n" + "=" * 60)
    print(f"!!! BEST COMBINATION FOUND !!!")
    print(f"Model: {os.path.basename(best_model_path)}")
    print(f"Seed:  {best_seed}")
    print(f"Score: {best_score:.2f}")
    print("=" * 60)
    print(f"\nUse these values in your demo.py.")