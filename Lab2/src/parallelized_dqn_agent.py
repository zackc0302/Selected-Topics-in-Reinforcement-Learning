import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import torch.multiprocessing as mp
import time
import os

### Actor Process 函式 ###
# 這個函式將在獨立的子程序中執行，負責與環境互動並收集經驗
def actor_process(config, replay_buffer_queue, model_state_dict_queue):
    """
    Actor Process:
    1. 創建自己的環境。
    2. 創建本地端的模型。
    3. 定期從主程序接收最新的模型權重。
    4. 與環境互動，並將 (s, a, r, s') 經驗 tuple 傳送回主程序。
    """
    # 1. 初始化環境
    env = gym.make(config["env_id"], render_mode='rgb_array')
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    
    # 2. 初始化本地端的行為網路
    behavior_net = AtariNetDQN(env.action_space.n)
    device = torch.device("cuda" if config["gpu"] and torch.cuda.is_available() else "cpu")
    behavior_net.to(device)

    # 從佇列中取得 Learner 發送的初始模型權重
    try:
        initial_state_dict = model_state_dict_queue.get(timeout=60) 
        behavior_net.load_state_dict(initial_state_dict)
    except mp.queues.Empty:
        print("Actor failed to get initial model weights.")
        return

    observation, info = env.reset()
    while True:
        ### 【補回遺失的程式碼】 ###
        # 1. 檢查是否有來自 Learner 的更新模型權重
        if not model_state_dict_queue.empty():
            try:
                state_dict = model_state_dict_queue.get_nowait()
                behavior_net.load_state_dict(state_dict)
            except mp.queues.Empty:
                pass 

        if random.random() < config["eval_epsilon"]: 
            action = env.action_space.sample()
        else:
            obs_array = np.array(observation)
            if obs_array.shape[-1] == 1:
                obs_array = obs_array.squeeze(-1)
            obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).float().to(device)
            with torch.no_grad():
                q_values = behavior_net(obs_tensor)
                action = q_values.argmax().item()

        next_observation, reward, done, info = env.step(action)
        replay_buffer_queue.put((np.array(observation), [action], [reward], np.array(next_observation), [int(done)]))

        if done:
            observation, info = env.reset()
        else:
            observation = next_observation

class AtariParallelizedDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariParallelizedDQNAgent, self).__init__(config)

		# Learner (主程序) 不再需要自己的訓練環境 `self.env`，因為它不直接與環境互動，只保留 test_env 用於評估
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array") 
		self.test_env = GrayScaleObservation(self.test_env)
		self.test_env = ResizeObservation(self.test_env, shape=84)
		self.test_env = FrameStack(self.test_env, num_stack=4)

		self.behavior_net = AtariNetDQN(self.test_env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.test_env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())

		# 初始化 optimizer (與之前相同)
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)

		# 【新增】平行化所需的屬性
		self.num_actors = config["num_actors"]
		# 創建一個共享佇列，讓所有 Actors 將經驗傳回來
		self.replay_buffer_queue = mp.Queue(maxsize=10000) 
		# 為每個 Actor 創建一個專屬佇列，用於從 Learner 接收最新的模型權重
		self.model_state_dict_queues = [mp.Queue(maxsize=1) for _ in range(self.num_actors)]
		# 將 config 存為屬性，方便傳遞給子程序
		self.config = config
		### 【初始化修改結束】 ###

	### 【覆寫】整個 train() 方法，以實現 Actor-Learner 流程 ###
	def train(self):
		"""
		Learner's training loop:
		1. 啟動 `num_actors` 個 Actor 子程序。
		2. 進入主迴圈，不斷從 `replay_buffer_queue` 接收經驗。
		3. 當 Replay Buffer 滿了之後，開始訓練網路。
		4. 定期將更新後的模型權重，透過 `model_state_dict_queues` 發送給所有 Actors。
		5. 訓練結束後，關閉所有 Actor 子程序。
		"""
		processes = []
		
		# 1. 啟動所有 Actor 子程序
		print(f"Starting {self.num_actors} actor processes...")
		for i in range(self.num_actors):
			# 將初始模型權重放入佇列
			# 注意：需要將模型權重移到 CPU，因為佇列傳輸的是 CPU 物件
			self.model_state_dict_queues[i].put({k: v.cpu() for k, v in self.behavior_net.state_dict().items()})
			
			# 創建並啟動程序
			p = mp.Process(target=actor_process, args=(self.config, self.replay_buffer_queue, self.model_state_dict_queues[i]))
			p.start()
			processes.append(p)

		# 2. Learner 的主迴圈
		while self.total_time_step <= self.training_steps:
			# 從佇列中收集經驗並放入 Replay Buffer
			while not self.replay_buffer_queue.empty():
				experience = self.replay_buffer_queue.get()
				self.replay_buffer.append(*experience)
			
			# 如果 Replay Buffer 中的經驗還不夠 warmup，則稍作等待
			if len(self.replay_buffer) < self.warmup_steps:
				time.sleep(0.1)
				continue
			
			# 進行網路更新
			self.epsilon_decay() # Learner 仍需計算 epsilon 以供記錄
			# 呼叫 base_agent 中的 update 方法，它會處理更新頻率
			self.update() 

			# 定期評估並儲存模型
			if self.total_time_step > 0 and self.total_time_step % self.eval_interval == 0:
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

				# 【重要】將更新後的模型權重發送給所有 Actors
				updated_weights = {k: v.cpu() for k, v in self.behavior_net.state_dict().items()}
				for i in range(self.num_actors):
					if self.model_state_dict_queues[i].empty(): # 只有在 Actor 已經取走舊權重時才發送新的
						self.model_state_dict_queues[i].put(updated_weights)

			self.total_time_step += 1 # Learner 的每一步更新算作一個 time_step
			if self.total_time_step % 1000 == 0:
				print(f"[{self.total_time_step}/{self.training_steps}] Replay Buffer: {len(self.replay_buffer)} Epsilon: {self.epsilon:.4f}")

		# 5. 訓練結束，終止所有 Actor 子程序
		print("Training finished. Terminating actor processes...")
		for p in processes:
			p.terminate()
			p.join()

	# decide_agent_actions 和 update_behavior_network 方法保持不變
	# 因為它們是 Learner 的核心邏輯，與標準 DQN 完全相同
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		if random.random() < epsilon:
			action = action_space.sample()
		else:
			obs_array = np.array(observation)
			if obs_array.shape[-1] == 1:
				obs_array = obs_array.squeeze(-1)
			obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).float().to(self.device)
			with torch.no_grad():
				q_values = self.behavior_net(obs_tensor)
				action = q_values.argmax().item()
		return action
	
	def update_behavior_network(self):
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		
		if state.dim() == 5:
			state = state.squeeze(-1)
		if next_state.dim() == 5:
			next_state = next_state.squeeze(-1)
		
		q_values = self.behavior_net(state)
		q_value = q_values.gather(dim=1, index=action.long())

		with torch.no_grad():
			next_q_values = self.target_net(next_state)
			next_q_value = next_q_values.max(dim=1, keepdim=True)[0]
			q_target = reward + self.gamma * next_q_value * (1 - done)
        
		criterion = nn.SmoothL1Loss()
		loss = criterion(q_value, q_target)
		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)
		self.optim.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), max_norm=1.0)
		self.optim.step()