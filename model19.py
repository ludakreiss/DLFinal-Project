# model_classify.py
import torch
import torch.nn as nn

class GeoClassifier(nn.Module):
    def __init__(self, num_countries):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: input has 3 channels (RGB). Pick an output channel count (e.g. 32).
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),   # must match out_channels above
            nn.ReLU(inplace=True),

            # Block 2: input channels = whatever Block 1 output. Increase channels (e.g. double it).
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Add 2 more blocks yourself, following the same pattern,
            # each time: in_channels = previous out_channels, and increase out_channels again.
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Finally: squash whatever spatial size remains down to 1x1
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Linear layer: in_features must match your LAST conv block's out_channels.
            # out_features = num_countries (this is your final prediction — one score per country)
            nn.Linear(in_features=256, out_features=num_countries)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

