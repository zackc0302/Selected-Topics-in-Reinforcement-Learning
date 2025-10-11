from dqn_agent_atari import AtariDQNAgent
import numpy as np
import gym
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, RecordVideo
import random
import torch

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
        "gamma": 0.99,
        "batch_size": 32,
        "eps_min": 0.1,
        "warmup_steps": 20000,
        "eps_decay": 1000000,
        "eval_epsilon": 0.0,  # 改成 0.0，完全不隨機
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
    
    # ========== 在這裡指定你想用的 seed ==========
    DEMO_SEED = 97  # 改成你想要的 seed
    # ===========================================
    
    agent = AtariDQNAgent(config)
    
    # 載入最佳模型
    model_path = 'log/DQN/MsPacman-v5/model_99772916_3414.pth'
    agent.load(model_path)
    print(f"Loaded model: {model_path}")
    print(f"Using seed: {DEMO_SEED}")
    
    # 固定所有隨機性
    set_seed(DEMO_SEED)
    
    # 錄製影片
    print(f"\nRecording video with seed {DEMO_SEED}...")
    env = gym.make(config["env_id"], render_mode='rgb_array')
    env = RecordVideo(env, video_folder='./videos', name_prefix=f'mspacman-seed{DEMO_SEED}', episode_trigger=lambda x: True)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    
    observation, info = env.reset(seed=DEMO_SEED)
    total_reward = 0
    
    while True:
        obs_array = np.array(observation)
        if obs_array.shape[-1] == 1:
            obs_array = obs_array.squeeze(-1)
        
        action = agent.decide_agent_actions(obs_array, epsilon=0.0, action_space=env.action_space)
        observation, reward, terminate, truncate, info = env.step(action)
        total_reward += reward
        
        if terminate or truncate:
            print(f"Video recorded! Final score: {total_reward}")
            break
    
    env.close()
    print(f"Video saved in ./videos/")
    print(f"Final score with seed {DEMO_SEED}: {total_reward}")