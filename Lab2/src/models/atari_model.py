# Question/models/atari_model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariNetDQN(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNetDQN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
# Dueling 架構
class DuelingAtariNetDQN(nn.Module):
    def __init__(self, num_actions):
        super(DuelingAtariNetDQN, self).__init__()
        self.num_actions = num_actions
        
        # 共享的 CNN 特徵提取層 (與原本的 DQN 相同)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 價值流 (Value Stream) - 輸出一個純量 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # 輸出維度為 1
        )
        
        # 優勢流 (Advantage Stream) - 輸出每個動作的優勢 A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions) # 輸出維度為動作數量
        )

    def forward(self, obs):
        # 1. 通過共享的 CNN 層
        features = self.cnn(obs / 255.0)
        features = features.view(features.size(0), -1) # Flatten
        
        # 2. 分別通過兩個流
        value = self.value_stream(features)           # Shape: [batch_size, 1]
        advantage = self.advantage_stream(features)    # Shape: [batch_size, num_actions]
        
        # 3. 結合 V(s) 和 A(s, a) 來計算 Q(s, a)
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        # 減去 Advantage 的平均值是為了增加訓練的穩定性，並解決可識別性問題
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values