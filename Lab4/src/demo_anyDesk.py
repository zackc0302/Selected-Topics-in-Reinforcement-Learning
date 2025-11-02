from td3_agent_CarRacing import CarRacingTD3Agent
import numpy as np

def demo_on_seeds(agent, model_path, seeds):
    """
    在指定的 seeds 上測試模型並即時顯示
    
    Args:
        agent: TD3 agent
        model_path: 模型路徑
        seeds: 助教指定的 5 個 seeds
    """
    # 載入模型
    agent.load(model_path)
    print(f"Model loaded from: {model_path}\n")
    
    all_rewards = []
    
    for idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Track {idx+1}/{len(seeds)} | Seed: {seed}")
        print(f"{'='*60}")
        
        total_reward = 0
        
        # 使用 test_env 的 reset（會自動處理 frame stacking）
        state, info = agent.test_env.env.reset(seed=seed)
        
        import cv2
        from collections import deque
        
        # 轉灰階
        state_gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        
        # Frame stacking
        frames = deque(maxlen=4)
        for _ in range(4):
            frames.append(state_gray)
        state = np.stack(frames, axis=0)  # [4, 96, 96]
        
        step = 0
        done = False
        
        while not done and step < 1000:
            # 選擇動作（不加噪音）
            action = agent.decide_agent_actions(state, sigma=0.0)
            
            # 使用原始環境的 step（會回傳 RGB）
            next_state_rgb, reward, terminates, truncates, info = agent.test_env.env.step(action)
            
            # 手動處理 next_state
            next_state_gray = cv2.cvtColor(next_state_rgb, cv2.COLOR_RGB2GRAY)
            frames.append(next_state_gray)
            next_state = np.stack(frames, axis=0)  # [4, 96, 96]
            
            total_reward += reward
            state = next_state
            step += 1
            
            # 即時顯示畫面
            agent.test_env.env.render()
            
            # 判斷結束
            done = terminates or truncates
        
        print(f"Track {idx+1} Results:")
        print(f"   Steps: {step}")
        print(f"   Total Reward: {total_reward:.2f}")
        all_rewards.append(total_reward)
    
    # 最終結果
    avg_reward = np.mean(all_rewards)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    for idx, (seed, reward) in enumerate(zip(seeds, all_rewards)):
        print(f"Track {idx+1} (Seed {seed:4d}): {reward:7.2f}")
    print(f"{'-'*60}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"{'='*60}\n")
    
    return all_rewards, avg_reward


if __name__ == '__main__':
    # 配置(用不到)
    config = {
        "gpu": True,
        "training_steps": 1e6,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 32,
        "warmup_steps": 1000,
        "total_episode": 100000,
        "lra": 4.5e-5,
        "lrc": 4.5e-5,
        "replay_buffer_capacity": 5000,
        "logdir": 'log/CarRacing/demo/',
        "update_freq": 25,
        "eval_interval": 10,
        "eval_episode": 10,
    }
    
    print("Initializing TD3 Agent...")
    agent = CarRacingTD3Agent(config)
    
    demo_seeds = [0, 1, 2, 3, 4]
    
    model_path = "log/CarRacing/td3_firstRun/model_2827069_887.pth"
    
    print(f"Model Path: {model_path}")
    print(f"Testing Seeds: {demo_seeds}\n")
    
    demo_on_seeds(agent, model_path, demo_seeds)
    
    agent.test_env.env.close()
    print("Demo completed.")