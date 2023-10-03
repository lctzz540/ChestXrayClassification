import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm(x)
        x = self.maxpool(x)
        return x


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.batchnorm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x


class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(16, 32)
        self.conv4 = ConvBlock(32, 64)
        self.conv5 = ConvBlock(64, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.conv6 = ConvBlock(128, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = FCBlock(12544, 512, 0.7)
        self.fc2 = FCBlock(512, 128, 0.5)
        self.fc3 = FCBlock(128, 64, 0.3)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.dropout1(x)
        x = self.conv6(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x
