from torch.nn.modules.pooling import MaxPool2d
import config

import torch
from torch import nn
from torch.nn import functional as F

import os

PEOPLE_NUM = len(os.listdir(config.DATA_TRAIN))


class FaceRecNet(nn.Module):
    def __init__(self) -> None:
        super(FaceRecNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
            )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=2
            )
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=2,
            stride=1,
            padding=2
            )
        self.pool = MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 5000)
        self.fc3 = nn.Linear(5000, 800)
        self.fc4 = nn.Linear(800, PEOPLE_NUM)

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        # print("x.size()", x.size())
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # print("x", x)
        # print("softmax", F.softmax(x, dim=1))
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    frn = FaceRecNet()
    x = torch.randn(1, 3, 50, 50)
    frn.forward(x)
