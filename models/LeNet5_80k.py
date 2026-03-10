import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5_Precise80K(nn.Module):
    """
    精确控制为80,000可剪枝参数的LeNet-5
    """
    def __init__(self, num_classes=10, dataset='cifar10'):
        super(LeNet5_Precise80K, self).__init__()
        
        in_channels = 3 if dataset == 'cifar10' else 1
        
        # 第一层卷积：精心设计通道数
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 第二层卷积：主要参数来源
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        # 第三层卷积：补充参数
        self.conv3 = nn.Conv2d(64, 48, kernel_size=3, padding=1)
        
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(48)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(48 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x