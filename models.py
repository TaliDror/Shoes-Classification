import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        n = 16
        self.kernel_size = 3
        padding = int((self.kernel_size - 1) / 2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n, kernel_size=self.kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=2 * n, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(2 * n)
        self.conv3 = nn.Conv2d(in_channels=2 * n, out_channels=4 * n, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(4 * n)
        self.conv4 = nn.Conv2d(in_channels=4 * n, out_channels=8 * n, kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm2d(8 * n)


        self.feature_size = 8 * n * 28 * 14

        self.fc1 = nn.Linear(in_features=self.feature_size, out_features=100)
        self.bn_fc1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, inp):
        device = inp.device
        inp = self.conv1(inp)
        inp = self.bn1(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)

        inp = self.conv2(inp)
        inp = self.bn2(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)

        inp = self.conv3(inp)
        inp = self.bn3(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)

        inp = self.conv4(inp)
        inp = self.bn4(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)

        inp = inp.reshape(inp.size(0), -1)

        inp = self.fc1(inp)
        inp = self.bn_fc1(inp)
        inp = F.relu(inp)
        inp = self.fc2(inp)
        inp = F.log_softmax(inp, dim=1)

        return inp


class CNNChannel(nn.Module):
    def __init__(self):
        super(CNNChannel, self).__init__()
        n = 64
        self.kernel_size = 3
        padding = int((self.kernel_size - 1) / 2)

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=n, kernel_size=self.kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=2 * n, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(2 * n)
        self.conv3 = nn.Conv2d(in_channels=2 * n, out_channels=4 * n, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(4 * n)
        self.conv4 = nn.Conv2d(in_channels=4 * n, out_channels=8 * n, kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm2d(8 * n)

        self.feature_size = 8 * n * 28 * 7  # (channels * height * width)

        self.fc1 = nn.Linear(in_features=self.feature_size, out_features=100)
        self.bn_fc1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, inp):
        device = inp.device

        left_shoes = inp[:, :, :224, :]
        right_shoes = inp[:, :, 224:, :]

        assert left_shoes.shape == (inp.size(0), 3, 224, 224)
        assert right_shoes.shape == (inp.size(0), 3, 224, 224)

        inp = torch.cat((left_shoes, right_shoes), dim=1)

        if inp.dtype != torch.float32:
            inp = inp.float()

        inp = self.conv1(inp)
        inp = self.bn1(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)

        inp = self.conv2(inp)
        inp = self.bn2(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)

        inp = self.conv3(inp)
        inp = self.bn3(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)

        inp = self.conv4(inp)
        inp = self.bn4(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)

        inp = inp.reshape(inp.size(0), -1)

        inp = self.fc1(inp)
        inp = self.bn_fc1(inp)
        inp = F.relu(inp)
        inp = self.fc2(inp)
        inp = F.log_softmax(inp, dim=1)

        return inp