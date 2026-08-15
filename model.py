import torch
import torch.nn as nn

class GeoCNN(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()

        self.features = nn.Sequential(
            # input: 3 x 224 x 224
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)  # 256 x 1 x 1
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)  # outputs [lat_norm, lng_norm]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x