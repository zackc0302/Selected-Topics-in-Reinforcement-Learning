import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.make(config["env_id"], render_mode='rgb_array')
		self.env = GrayScaleObservation(self.env)
		self.env = ResizeObservation(self.env, shape=84)
		self.env = FrameStack(self.env, num_stack=4)
		
		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array") 
		self.test_env = GrayScaleObservation(self.test_env)
		self.test_env = ResizeObservation(self.test_env, shape=84)
		self.test_env = FrameStack(self.test_env, num_stack=4)

		self.net = AtariNet(self.env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		# get action, value, logp from net
		
		obs_array = np.array(observation)

		if obs_array.ndim == 4 and obs_array.shape[-1] == 1:
			obs_array = obs_array.squeeze(-1)  # 移除最後一個維度 -> (4, 84, 84)

		obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(self.device)
		
		if obs_array.shape[-1] == 1:			# FrameStack 會給出 (4, 84, 84, 1)，我們需要 (4, 84, 84)
			obs_array = obs_array.squeeze(-1)  	
		
		if eval:
			with torch.no_grad():
				action, logp, value, _ = self.net(obs_tensor, eval=True)
		else:
			action, logp, value, _ = self.net(obs_tensor)
		
		# Return action, value, and log_prob as numpy arrays
		return action.cpu().numpy(), value.cpu().detach().numpy(), logp.cpu().detach().numpy()


	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				### TODO ###
				# calculate loss and update network
				# Get new log_prob, value, and entropy from the network
				_, logp_pi, v, entropy = self.net(ob_train_batch, a=ac_train_batch)

				# calculate policy loss
				ratio = torch.exp(logp_pi - logp_pi_train_batch)
				surrogate1 = ratio * adv_train_batch
				surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_train_batch
				surrogate_loss = -torch.min(surrogate1, surrogate2).mean()

				# calculate value loss
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(v, return_train_batch)
				
				# calculate total loss
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy.mean()

				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.mean().item()
				total_loss += loss.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print((f"Loss: {total_loss / loss_counter}\n"
			   f"\tSurrogate Loss: {total_surrogate_loss / loss_counter}\n"
			   f"\tValue Loss: {total_v_loss / loss_counter}\n"
			   f"\tEntropy: {total_entropy / loss_counter}"
			   ))
	



