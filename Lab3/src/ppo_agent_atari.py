import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        self.env = gym.make(config["env_id"])
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=84)
        self.env = FrameStack(self.env, num_stack=4)
        
        self.test_env = gym.make(config["env_id"], render_mode="rgb_array") 
        self.test_env = GrayScaleObservation(self.test_env)
        self.test_env = ResizeObservation(self.test_env, shape=84)
        self.test_env = FrameStack(self.test_env, num_stack=4)

        self.net = AtariNet(self.env.action_space.n)
        self.net.to(self.device)
        self.lr = config["learning_rate"]
        self.update_count = config["update_ppo_epoch"]
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)
        
    def decide_agent_actions(self, observation, eval=False):
        obs_array = np.array(observation)
        
        ### --- 修正開始 --- ###
        # 處理來自 Gym Wrapper 的額外維度
        # 預期 shape: (4, 84, 84), C, H, W
        # 實際可能得到: (4, 84, 84, 1)
        if obs_array.ndim == 4 and obs_array.shape[-1] == 1:
            obs_array = obs_array.squeeze(-1)
        ### --- 修正結束 --- ###
        
        # 現在 obs_array 的 shape 是 (4, 84, 84)
        # unsqueeze(0) 後變成 (1, 4, 84, 84)，這正是 CNN 需要的 (N, C, H, W) 格式
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(self.device, dtype=torch.float32)
        
        if eval:
            self.net.eval()
            with torch.no_grad():
                action, action_log_probs, value, _ = self.net(obs_tensor)
        else:
            self.net.train()
            action, action_log_probs, value, _ = self.net(obs_tensor)
        
        logp_pi = action_log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action.cpu().numpy(), value.cpu().detach().numpy(), logp_pi.cpu().detach().numpy()

    def update(self):
        # ... (update 函式維持不變，使用我之前回覆的版本) ...
        loss_counter = 1e-6
        total_surrogate_loss = 0
        total_v_loss = 0
        total_entropy = 0
        total_loss = 0

        batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
        sample_count = len(batches["action"])
        
        # Data conversion to tensors
        obs_batch = torch.from_numpy(batches["observation"]["observation_2d"]).to(self.device, dtype=torch.float32)
        action_batch = torch.from_numpy(batches["action"]).to(self.device, dtype=torch.long)
        return_batch = torch.from_numpy(batches["return"]).to(self.device, dtype=torch.float32)
        adv_batch = torch.from_numpy(batches["adv"]).to(self.device, dtype=torch.float32)
        logp_pi_old_batch = torch.from_numpy(batches["logp_pi"]).to(self.device, dtype=torch.float32)

        for _ in range(self.update_count):
            # Shuffle indices for each epoch
            batch_index = np.random.permutation(sample_count)
            for start in range(0, sample_count, self.batch_size):
                end = start + self.batch_size
                mb_indices = batch_index[start:end]

                # Get mini-batch data
                mb_obs = obs_batch[mb_indices]
                mb_action = action_batch[mb_indices]
                mb_return = return_batch[mb_indices]
                mb_adv = adv_batch[mb_indices]
                mb_logp_pi_old = logp_pi_old_batch[mb_indices]

                # Re-compute outputs from the network for the mini-batch
                _, action_log_probs, v, entropy = self.net(mb_obs, a=mb_action)
                
                # ac_train_batch from buffer is (batch_size, 1), which is correct for gather
                logp_pi = action_log_probs.gather(1, mb_action).squeeze(-1)

                # calculate policy loss (surrogate loss)
                ratio = torch.exp(logp_pi - mb_logp_pi_old)
                surrogate1 = ratio * mb_adv
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_adv
                surrogate_loss = -torch.min(surrogate1, surrogate2).mean()

                # calculate value loss
                v_loss = F.mse_loss(v, mb_return)
                
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