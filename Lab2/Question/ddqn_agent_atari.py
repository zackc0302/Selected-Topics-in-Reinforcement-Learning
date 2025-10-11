import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
# agent 為 DDQN
class AtariDDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.make(config["env_id"], render_mode='rgb_array')
		self.env = GrayScaleObservation(self.env)
		self.env = ResizeObservation(self.env, shape=84)
		self.env = FrameStack(self.env, num_stack=4)

		### TODO ###
		# initialize testing env
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array") 
		self.test_env = GrayScaleObservation(self.test_env)
		self.test_env = ResizeObservation(self.test_env, shape=84)
		self.test_env = FrameStack(self.test_env, num_stack=4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		
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
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		# 修正維度問題：移除多餘的最後一個維度
		if state.dim() == 5:  # [batch, 4, 84, 84, 1]
			state = state.squeeze(-1)  # [batch, 4, 84, 84]
		if next_state.dim() == 5:
			next_state = next_state.squeeze(-1)

		### TODO ###
		q_values = self.behavior_net(state)
		q_value = q_values.gather(dim=1, index=action.long())

		with torch.no_grad():
			next_actions = self.behavior_net(next_state).max(dim=1, keepdim=True)[1]
			next_q_value = self.target_net(next_state).gather(dim=1, index=next_actions)
			q_target = reward + self.gamma * next_q_value * (1 - done)

		criterion = nn.SmoothL1Loss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DDQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()

		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), max_norm=1.0)
		self.optim.step()