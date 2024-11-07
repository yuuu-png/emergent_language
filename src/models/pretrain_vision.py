import os

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import random
import numpy as np


class Vision(nn.Module):
    def __init__(self):
        """Default Backbone for MNIST
        This is a simple CNN with 2 Convolutional layers and 1 Fully Connected layer.
        """
        super(Vision, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return x


class PretrainNet(nn.Module):
    def __init__(self, vision_module):
        super(PretrainNet, self).__init__()
        self.vision_module = vision_module
        self.fc = nn.Linear(500, 10)

    def forward(self, x):
        x = self.vision_module(x)
        x = self.fc(F.leaky_relu(x))
        return x


def main(batch_size=32, model_path="models/vision_pretrained.pth"):
    if os.path.exists(model_path):
        vision = Vision()
        vision.load_state_dict(torch.load(model_path))
        return vision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    transform = transforms.ToTensor()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("/data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("/data", train=False, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        **kwargs,
    )

    vision = Vision()
    class_prediction = PretrainNet(
        vision
    )  #  note that we pass vision - which we want to pretrain
    optimizer = torch.optim.Adam(vision.parameters())
    class_prediction = class_prediction.to(device)

    for epoch in range(10):
        mean_loss, n_batches = 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = class_prediction(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            mean_loss += loss.mean().item()
            n_batches += 1

        print(f"Train Epoch: {epoch}, mean loss: {mean_loss / n_batches}")

    torch.save(vision.state_dict(), model_path)

    return vision


if __name__ == "__main__":
    main()
