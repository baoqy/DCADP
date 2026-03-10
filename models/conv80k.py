import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv80k(nn.Module):
    """
    可剪枝参数约 8 万的卷积网络，CIFAR-10默认输入
    """
    def __init__(self, dataset='cifar10', enable_dropout=False, do_log_soft=True):
        super(Conv80k, self).__init__()
        self.enable_dropout = enable_dropout
        self.do_log_soft = do_log_soft

        if dataset == 'cifar10':
            in_channels = 3
            num_classes = 10
        else:
            raise NotImplementedError("Currently only CIFAR-10 is supported")

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 80, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU(inplace=True)

        # ✔ 经过 GAP 后，通道数=80，输入维度应为 80
        self.fc1 = nn.Linear(80, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        x = F.adaptive_avg_pool2d(x, 1)  # -> (batch, 80, 1, 1)
        x = torch.flatten(x, 1)          # -> (batch, 80)

        x = self.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        if self.do_log_soft:
            return F.log_softmax(x, dim=1)
        return x
