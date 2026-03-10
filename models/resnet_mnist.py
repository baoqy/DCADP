import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

# 假设你已经有 BasicBlock 和 ResNetCifar 定义
from models.resnet_cifar10 import BasicBlock, ResNetCifar

NUM_CLASSES = 10
CHECKPOINT_DIR = "./checkpoints"

# ===============================
#   创建保存目录
# ===============================
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# ===============================
#   定义 MNIST 专用 ResNet20
# ===============================
class ResNetMNIST(ResNetCifar):
    def __init__(self, block, layers, num_classes=NUM_CLASSES):
        self.nlayers = 0
        self.layer_gates = []
        for layer in range(3):
            self.layer_gates.append([])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 16
        nn.Module.__init__(self)

        # conv1 输入通道改为 1
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(self.layer_gates[0], block, 16, layers[0])
        self.layer2 = self._make_layer(self.layer_gates[1], block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 64, layers[2], stride=2)

        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, (2. / n)**0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def resnet20_mnist(**kwargs):
    return ResNetMNIST(BasicBlock, [3,3,3], **kwargs)