import torch
import torch.nn as nn
import torch.nn.functional as F


class AndreaNet(nn.Module):
    def __init__(self, channels: int = 2, classes: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.fc1 = nn.Linear(32 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, classes)
        self.softmax = nn.Softmax(dim=1)
        self.classes = classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.classes > 1:
            x = self.softmax(x)
        return x
