import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_channels, p_size=0, kernel_size=5):
        super(Model, self).__init__()
        self.p_size = p_size
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size)
        self.batchnorm1 = nn.BatchNorm2d(32, 1e-3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size)
        self.batchnorm2 = nn.BatchNorm2d(32, 1e-3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size)
        self.batchnorm3 = nn.BatchNorm2d(64, 1e-3)

        self.conv4 = nn.Conv2d(64, 64, kernel_size)
        self.batchnorm4 = nn.BatchNorm2d(64, 1e-3)

        self.conv5 = nn.Conv2d(64, 64, kernel_size)
        self.batchnorm5 = nn.BatchNorm2d(64, 1e-3)

        self.conv6 = nn.Conv2d(64, 64, kernel_size)
        self.batchnorm6 = nn.BatchNorm2d(64, 1e-3)

        self.conv7 = nn.Conv2d(64, 64, kernel_size)
        self.batchnorm7 = nn.BatchNorm2d(64, 1e-3)

        self.conv8 = nn.Conv2d(64, 64, kernel_size)
        self.batchnorm8 = nn.BatchNorm2d(64, 1e-3)

        self.conv9 = nn.Conv2d(64, 64, kernel_size)
        self.batchnorm9 = nn.BatchNorm2d(64, 1e-3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_pass(self, x):
        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))

        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))

        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))

        x = self.conv4(x)
        x = F.relu(self.batchnorm4(x))

        x = self.conv5(x)
        x = F.relu(self.batchnorm5(x))

        x = self.conv6(x)
        x = F.relu(self.batchnorm6(x))

        x = self.conv7(x)
        x = F.relu(self.batchnorm7(x))

        x = self.conv8(x)
        x = F.relu(self.batchnorm8(x))

        x = self.conv9(x)
        x = self.batchnorm9(x)
        return x


    def forward(self, left_patch, right_patch):
        left_patch = self.forward_pass(left_patch)  # Left patch of size 37x37
        left_patch = left_patch.view(left_patch.size(0), 1, 64)

        right_patch = self.forward_pass(right_patch)  # Right patch of size 37x237
        right_patch = right_patch.squeeze().view(right_patch.size(0), 64, self.p_size)

        pred = left_patch.bmm(right_patch).view(right_patch.size(0), self.p_size)
        pred = self.logsoftmax(pred)

        return left_patch, right_patch, pred


    def predict(self, x):
        with torch.no_grad():
            x = nn.ReflectionPad2d(18)(x)
            x = self.conv1(x)
            x = F.relu(self.batchnorm1(x))

            x = self.conv2(x)
            x = F.relu(self.batchnorm2(x))

            x = self.conv3(x)
            x = F.relu(self.batchnorm3(x))

            x = self.conv4(x)
            x = F.relu(self.batchnorm4(x))

            x = self.conv5(x)
            x = F.relu(self.batchnorm5(x))

            x = self.conv6(x)
            x = F.relu(self.batchnorm6(x))

            x = self.conv7(x)
            x = F.relu(self.batchnorm7(x))

            x = self.conv8(x)
            x = F.relu(self.batchnorm8(x))

            x = self.conv9(x)
            x = self.batchnorm9(x)

        return x
