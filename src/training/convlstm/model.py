"""
ConvLSTM cell and sequence model for spatio-temporal anomaly detection.
Input:  (batch, seq_len, C, H, W)
Output: (batch, 1) — anomaly probability score
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=pad
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMAnomalyDetector(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, img_size: int = 64):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.img_size = img_size

        # Spatial encoder — reduce resolution before LSTM
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),           # /4
            nn.ReLU(),
        )
        enc_size = img_size // 4

        self.convlstm = ConvLSTMCell(32, hidden_channels, kernel_size=3)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        device = x.device

        h = torch.zeros(B, self.hidden_channels, H // 4, W // 4, device=device)
        c = torch.zeros(B, self.hidden_channels, H // 4, W // 4, device=device)

        for t in range(T):
            frame = x[:, t]                 # (B, C, H, W)
            enc   = self.encoder(frame)     # (B, 32, H/4, W/4)
            h, c  = self.convlstm(enc, h, c)

        return self.classifier(h)           # (B, 1)
