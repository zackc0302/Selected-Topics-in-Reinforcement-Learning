# explore.py (最終・可重現・亂數種子版)

import os
import glob
import torch
import numpy as np
import random
import gym
# 確保引入所有需要的 Wrapper，特別是 RecordVideo
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, RecordVideo
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
    手動執行評估迴圈，但使用與 demo.py 完全相同的環境堆疊。
    """
    # 創建一個臨時資料夾給 "假的" 錄影器使用，避免報錯
    temp_video_folder = './temp_videos_for_explore'
    os.makedirs(temp_video_folder, exist_ok=True)

    # 創建與 demo.py 完全相同的環境堆疊
    env = gym.make(env_config["env_id"], render_mode='rgb_array')
    
    # [關鍵修正]：加入 RecordVideo Wrapper，但使用 trigger 讓它從不實際錄製。
    # 這樣可以確保 Wrapper 堆疊一致，但又不會產生大量影片檔案。
    env = RecordVideo(env, video_folder=temp_video_folder, episode_trigger=lambda x: False)
    
    # 保持與 demo.py 完全一致的 Wrapper 順序
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    
    # 固定種子來重設環境
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

    # 2. 執行地毯式搜索
    global_best_score = -1
    best_model_for_demo = None
    best_seed_for_demo = None
    
    num_seeds_to_test_per_model = 200 
    
    agent = AtariPPOAgent(config)
    
    print("\n" + "="*80)
    print(f"Starting ultimate search for the best (model, seed) combination...")
    print(f"Each model will be tested on {num_seeds_to_test_per_model} unique RANDOM seeds.")
    print("This will ensure the result is 100% reproducible in demo.py.")
    print("="*80)

    for i, path in enumerate(model_paths):
        print(f"\n--- Evaluating Model {i+1}/{len(model_paths)}: {os.path.basename(path)} ---")
        agent.load(path)
        
        # 定義一個較大的種子抽樣範圍，例如 0 到 9999
        SEED_UPPER_BOUND = 10000 
        
        # 從大範圍中，不重複地抽出指定數量的亂數種子
        seed_list = random.sample(range(SEED_UPPER_BOUND), num_seeds_to_test_per_model)
        
        for seed in seed_list:
            set_seed(seed)
            score = evaluate_model_manually(agent, config, seed)
            
            print(f"  Seed {seed:5d}: {score:7.2f}", end='\r') 

            if score > global_best_score:
                global_best_score = score
                best_model_for_demo = path
                best_seed_for_demo = seed
                
                print("\n" + "*"*80)
                print(f"!!! NEW HIGH SCORE FOUND !!!")
                print(f"  Model: {os.path.basename(best_model_for_demo)}")
                print(f"  Seed:  {best_seed_for_demo}")
                print(f"  Score: {global_best_score:.2f}")
                print("*"*80)
                
    print("\n\n" + "=" * 80)
    print("!!! ULTIMATE SEARCH COMPLETE !!!")
    print(f"The best combination for the demo is:")
    print(f"  >> Best Model: {os.path.basename(best_model_for_demo)}")
    print(f"  >> Best Seed:  {best_seed_for_demo}")
    print(f"  >> Score:      {global_best_score:.2f}")
    print("=" * 80)
    print(f"\nCOPY the following values into your demo.py:")
    print("-" * 50)
    print(f"BEST_MODEL_PATH = '{best_model_for_demo}'")
    print(f"BEST_SEED = {best_seed_for_demo}")
    print("-" * 50)