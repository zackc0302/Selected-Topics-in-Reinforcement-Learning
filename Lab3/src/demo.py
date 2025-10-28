import torch
import numpy as np
import random
import gym
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, RecordVideo
from ppo_agent_atari import AtariPPOAgent

def set_seed(seed):
    """固定所有隨機數生成器"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    config = {
    "gpu": True,
    "env_id": 'ALE/Enduro-v5',
    # --- agent 初始化用 ---
    "eval_episode": 1, 
    "training_steps": 0, 
    "update_sample_count": 10000,
    "discount_factor_gamma": 0.99, 
    "discount_factor_lambda": 0.95, 
    "clip_epsilon": 0.2,
    "max_gradient_norm": 0.5, 
    "batch_size": 128, 
    "logdir": 'log/Enduro_demo/',
    "update_ppo_epoch": 3, 
    "learning_rate": 2.5e-4, 
    "value_coefficient": 0.5,
    "entropy_coefficient": 0.01, 
    "horizon": 128, 
    "eval_interval": 100,
    }

# ==========================================================
BEST_MODEL_PATH = 'log/Enduro_fix/model_97997648_1655.pth' 
BEST_SEED = 1240                                  
# ==========================================================

agent = AtariPPOAgent(config)
agent.load(BEST_MODEL_PATH)
print(f"Loaded model: {BEST_MODEL_PATH}")
print(f"Using seed: {BEST_SEED}")

set_seed(BEST_SEED)

video_folder = './videos'
video_name_prefix = f'enduro-ppo-seed{BEST_SEED}'

print(f"\nRecording video...")
env = gym.make(config["env_id"], render_mode='rgb_array')
env = RecordVideo(env, video_folder=video_folder, name_prefix=video_name_prefix, episode_trigger=lambda x: True)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

observation, info = env.reset(seed=BEST_SEED)
total_reward = 0

while True:
    action_array, _, _ = agent.decide_agent_actions(observation, eval=True)
    action = action_array[0] 

    observation, reward, terminate, truncate, info = env.step(action)
    total_reward += reward
    
    if terminate or truncate:
        break
        
env.close()

print("\n" + "=" * 60)
print("Demo finished and video recorded!")
print(f"Final score: {total_reward}")
print(f"Video saved in folder: {video_folder}")
print("=" * 60)