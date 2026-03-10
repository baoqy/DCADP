import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv40k(nn.Module):
    def __init__(self, args=None, dataset='cifar10'):
        super(Conv40k, self).__init__()

        # ---------------------------
        # 默认设定（与 MlpNet 风格一致）
        # ---------------------------
        if args is None:
            enable_dropout = False
            disable_bias = True
            do_log_soft = True
        else:
            enable_dropout = args.enable_dropout
            disable_bias = args.disable_bias
            do_log_soft = not args.disable_log_soft

        self.do_log_soft = do_log_soft
        self.enable_dropout = enable_dropout

        # ---------------------------
        # 网络结构（约 40k 参数）
        # ---------------------------
        # CIFAR-10 输入通道 = 3
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1,
            bias=not disable_bias,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1,
            bias=not disable_bias,
        )

        # CIFAR10 经过 2 次 stride=1 的卷积 => 尺寸仍是 32×32
        # 再经过 2×2 maxpool 两次 => 8×8
        self.fc1 = nn.Linear(16 * 8 * 8, 32, bias=not disable_bias)
        self.fc2 = nn.Linear(32, 10, bias=not disable_bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)   # 32 → 16

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)   # 16 → 8

        x = x.view(x.size(0), -1)  # flatten

        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        if self.do_log_soft:
            return F.log_softmax(x, dim=1)
        else:
            return x
