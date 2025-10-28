# Lab3/src/ppo_agent_atari.py
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
        ### TODO ###
        # initialize env
        self.env = gym.make(config["env_id"], render_mode='rgb_array')
        self.env = GrayScaleObservation(self.env, keep_dim=True)
        self.env = ResizeObservation(self.env, shape=84)
        self.env = FrameStack(self.env, num_stack=4)
        
        # ===== 測試環境輸出格式
        # print("\n" + "="*60)
        # print("Testing environment setup...")
        # print("="*60)
        # test_obs, _ = self.env.reset()
        # test_obs_array = np.array(test_obs)
        # print(f"Observation shape from env: {test_obs_array.shape}")
        # print(f"Observation dtype: {test_obs_array.dtype}")
        # print(f"Observation range: [{test_obs_array.min()}, {test_obs_array.max()}]")
        # print("="*60 + "\n")
        
        ### TODO ###
        # initialize test_env
        self.test_env = gym.make(config["env_id"], render_mode="rgb_array") 
        self.test_env = GrayScaleObservation(self.test_env, keep_dim=True)
        self.test_env = ResizeObservation(self.test_env, shape=84)
        self.test_env = FrameStack(self.test_env, num_stack=4)

        self.net = AtariNet(self.env.action_space.n)
        self.net.to(self.device)
        self.lr = config["learning_rate"]
        self.update_count = config["update_ppo_epoch"]
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
    def _process_observation(self, observation):
        """
        統一處理 observation，確保輸出是 (4, 84, 84)
        """
        # Convert LazyFrames to numpy array
        if hasattr(observation, '__array__'):
            obs_array = np.array(observation)
        else:
            obs_array = observation
        
        # 處理不同的可能形狀
        if obs_array.ndim == 4:
            if obs_array.shape == (4, 84, 84, 1):
                # (4, 84, 84, 1) -> (4, 84, 84)
                obs_array = obs_array.squeeze(-1)
            elif obs_array.shape == (1, 4, 84, 84):
                # (1, 4, 84, 84) -> (4, 84, 84)
                obs_array = obs_array.squeeze(0)
            elif obs_array.shape == (84, 84, 4, 1):
                # (84, 84, 4, 1) -> (4, 84, 84)
                obs_array = obs_array.squeeze(-1).transpose(2, 0, 1)
        elif obs_array.ndim == 3:
            if obs_array.shape == (84, 84, 4):
                # (84, 84, 4) -> (4, 84, 84)
                obs_array = obs_array.transpose(2, 0, 1)
            # else: already (4, 84, 84)
        
        # 最終檢查
        if obs_array.shape != (4, 84, 84):
            raise ValueError(f"Cannot convert observation shape {obs_array.shape} to (4, 84, 84)")
        
        return obs_array.astype(np.float32)
        
    def decide_agent_actions(self, observation, eval=False):
        ### TODO ###
        # add batch dimension in observation
        # get action, value, logp from net
        
        # ========[Debug]: 前 5 步印出詳細資訊========
        # if self.total_time_step < 5:
        #     print(f"\n[Step {self.total_time_step}] decide_agent_actions called")
        #     if hasattr(observation, '__array__'):
        #         obs_array = np.array(observation)
        #     else:
        #         obs_array = observation
        #     print(f"  Raw observation shape: {obs_array.shape}")
        #     print(f"  Raw observation range: [{obs_array.min()}, {obs_array.max()}]")
        # ==========================================
        
        # ===使用統一的處理函數(start)===
        try:
            obs_array = self._process_observation(observation)
            # if self.total_time_step < 5:
            #     print(f"  After processing shape: {obs_array.shape}")
            #     print(f"  After processing range: [{obs_array.min()}, {obs_array.max()}]")
        except Exception as e:
            print(f"\nERROR in _process_observation:")
            print(f"  {e}")
            if hasattr(observation, '__array__'):
                obs_array = np.array(observation)
            else:
                obs_array = observation
            print(f"  Original shape was: {obs_array.shape}")
            raise
        
        # Add batch dimension -> (1, 4, 84, 84)
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(self.device)
        # ===使用統一的處理函數(end)===
        
        if eval:
            with torch.no_grad():
                action, logp, value, _ = self.net(obs_tensor, eval=True)
        else:
            action, logp, value, _ = self.net(obs_tensor)
        
        # Return as numpy arrays
        action_np = action.cpu().numpy()
        value_np = value.cpu().detach().numpy()
        logp_np = logp.cpu().detach().numpy()
        
        # if self.total_time_step < 5:
        #     print(f"  Action: {action_np}, Value: {value_np[0]:.3f}, LogP: {logp_np[0]:.3f}")
        
        return action_np, value_np, logp_np
        
    def update(self):
        loss_counter = 0.0001
        total_surrogate_loss = 0
        total_v_loss = 0
        total_entropy = 0
        total_loss = 0

        batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
        sample_count = len(batches["action"])
        
        # ===== 統計資訊（前幾次）======
        # if self.total_time_step < self.update_sample_count * 3:
        #     print(f"\n{'='*60}")
        #     print(f"Update at timestep {self.total_time_step}")
        #     print(f"{'='*60}")
        #     print(f"Sample count: {sample_count}")
            
        #     # Action 分布
        #     actions_flat = batches['action'].flatten().astype(int)
        #     action_counts = np.bincount(actions_flat, minlength=self.env.action_space.n)
        #     action_probs = action_counts / action_counts.sum()
        #     print(f"Action distribution:")
        #     for i, (count, prob) in enumerate(zip(action_counts, action_probs)):
        #         print(f"  Action {i}: {count:5d} ({prob*100:5.2f}%)")
            
        #     print(f"Reward - Non-zero: {np.sum(batches['reward'] != 0)}/{len(batches['reward'])}")
        #     print(f"{'='*60}\n")
        
        batch_index = np.random.permutation(sample_count)
        
        observation_batch = {}
        for key in batches["observation"]:
            observation_batch[key] = batches["observation"][key][batch_index]
        action_batch = batches["action"][batch_index]
        return_batch = batches["return"][batch_index]
        adv_batch = batches["adv"][batch_index]
        v_batch = batches["value"][batch_index]
        logp_pi_batch = batches["logp_pi"][batch_index]

        for epoch in range(self.update_count):
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
                ac_train_batch = ac_train_batch.squeeze(-1)  #  (128, 1) -> (128)
                
                adv_train_batch = torch.from_numpy(adv_train_batch)
                adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
                adv_train_batch = adv_train_batch.squeeze(-1)  #  (128, 1) -> (128)
                
                logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
                logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
                logp_pi_train_batch = logp_pi_train_batch.squeeze(-1)  #  (128, 1) -> (128)
                
                return_train_batch = torch.from_numpy(return_train_batch)
                return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)
                return_train_batch = return_train_batch.squeeze(-1)  #  (128, 1) -> (128)

                #  =====Debug (只在第一次)=====
                # if self.total_time_step < self.update_sample_count * 2 and epoch == 0 and start == 0:
                #     print(f"\n[DEBUG] After squeeze:")
                #     print(f"  ac_train_batch: {ac_train_batch.shape}")
                #     print(f"  adv_train_batch: {adv_train_batch.shape}")
                #     print(f"  logp_pi_train_batch: {logp_pi_train_batch.shape}")
                #     print(f"  return_train_batch: {return_train_batch.shape}")

                ### TODO ###
                # calculate loss and update network
                _, logp_pi, value, entropy = self.net(ob_train_batch, a=ac_train_batch)

                # ===== Debug =====
                # if self.total_time_step < self.update_sample_count * 2 and epoch == 0 and start == 0:
                #     print(f"\n[DEBUG] Network outputs:")
                #     print(f"  logp_pi: {logp_pi.shape}")
                #     print(f"  value: {value.shape}")
                #     print(f"  entropy: {entropy.shape}")

                # calculate policy loss
                ratio = torch.exp(logp_pi - logp_pi_train_batch)
                
                # ===== 改成檢查前 3 個 batch =====
                # if self.total_time_step < self.update_sample_count * 2 and epoch == 0:
                #     if start < 3 * self.batch_size:
                #         batch_num = start // self.batch_size
                #         print(f"\n[DEBUG] Batch {batch_num}:")
                #         print(f"  logp_pi range: [{logp_pi.min().item():.6f}, {logp_pi.max().item():.6f}]")
                #         print(f"  logp_pi_train range: [{logp_pi_train_batch.min().item():.6f}, {logp_pi_train_batch.max().item():.6f}]")
                #         print(f"  logp_pi - logp_pi_train range: [{(logp_pi - logp_pi_train_batch).min().item():.6f}, {(logp_pi - logp_pi_train_batch).max().item():.6f}]")
                #         print(f"  ratio range: [{ratio.min().item():.6f}, {ratio.max().item():.6f}]")
                #         print(f"  ratio mean: {ratio.mean().item():.6f}")
                
                surrogate1 = ratio * adv_train_batch
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_train_batch
                surrogate_loss = -torch.mean(torch.min(surrogate1, surrogate2))

                # calculate value loss
                value_criterion = nn.MSELoss()
                v_loss = value_criterion(value, return_train_batch)

                # calculate total loss
                loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy.mean()

                # ===== Debug =====
                # if self.total_time_step < self.update_sample_count * 2 and epoch == 0 and start == 0:
                #     print(f"\n[DEBUG] Loss components:")
                #     print(f"  surrogate_loss: {surrogate_loss.item():.6f}")
                #     print(f"  v_loss: {v_loss.item():.6f}")
                #     print(f"  entropy bonus: {(self.entropy_coefficient * entropy.mean()).item():.6f}")
                #     print(f"  total loss: {loss.item():.6f}")

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
        print((f"Loss: {total_loss / loss_counter:.3f} | "
            f"Surrogate: {total_surrogate_loss / loss_counter:.3f} | "
            f"Value: {total_v_loss / loss_counter:.3f} | "
            f"Entropy: {total_entropy / loss_counter:.3f}"
            ))
