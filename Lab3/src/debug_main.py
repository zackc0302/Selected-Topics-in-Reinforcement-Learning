import numpy as np
import torch
from ppo_agent_atari import AtariPPOAgent

if __name__ == "__main__":
    # === 基本設定 ===
    config = {
        "gpu": False,  # debug 建議關掉 GPU
        "training_steps": 10,
        "update_sample_count": 10000,
        "discount_factor_gamma": 0.99,
        "discount_factor_lambda": 0.95,
        "clip_epsilon": 0.2,
        "max_gradient_norm": 0.5,
        "batch_size": 8,
        "logdir": "log/debug/",
        "update_ppo_epoch": 3,
        "learning_rate": 2.5e-6,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "horizon": 8,
        "env_id": "ALE/Enduro-v5",
        "eval_interval": 100,
        "eval_episode": 1,
    }

    # === 初始化 agent ===
    agent = AtariPPOAgent(config)

    # === 建立假觀測資料 ===
    dummy_obs = np.random.rand(4, 84, 84).astype(np.float32)

    print("\n===== INPUT DEBUG =====")
    print("Dummy observation shape:", dummy_obs.shape)
    print("Dummy observation dtype:", dummy_obs.dtype)
    print("========================\n")

    # === 在 decide_agent_actions 內加入中間 print ===
    # 你也可以暫時修改 ppo_agent_atari.py 中的 decide_agent_actions()，例如：
    #
    # print("obs_array:", obs_array.shape, obs_array.dtype)
    # print("obs_tensor:", obs_tensor.shape, obs_tensor.dtype)

    # === 呼叫函式測試 ===
    action, value, logp = agent.decide_agent_actions(dummy_obs, eval=False)

    print("\n===== OUTPUT DEBUG =====")
    print("Action shape:", action.shape, "dtype:", action.dtype)
    print("Action sample:", action)
    print("Value shape:", value.shape, "dtype:", value.dtype)
    print("Value sample:", value)
    print("LogP shape:", logp.shape, "dtype:", logp.dtype)
    print("LogP sample:", logp)
    print("========================\n")
