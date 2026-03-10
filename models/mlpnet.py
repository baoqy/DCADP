import torch.nn as nn
import torch.nn.functional as F

class MlpNet(nn.Module):
    def __init__(self, args, dataset='cifar10'):
        super(MlpNet, self).__init__()
        if dataset == 'cifar10':
            input_size = 3072 # 32 x 32 x 3
        elif dataset == 'mnist':
            input_size = 784

        if args is None:
            enable_dropout = False
            nh1 = 40
            nh2 = 20
            nc = 10
            disable_bias = True
            do_log_soft = True
        else:
            enable_dropout = args.enable_dropout
            nh1 = args.num_hidden_nodes1
            nh2 = args.num_hidden_nodes2
            nc = args.num_classes
            disable_bias = args.disable_bias
            do_log_soft = not args.disable_log_soft
        
        self.do_log_soft = do_log_soft
        print("Do log softmax: ", do_log_soft)
        self.fc1 = nn.Linear(input_size, nh1, bias=not disable_bias)
        self.fc2 = nn.Linear(nh1, nh2, bias=not disable_bias)
        self.fc3 = nn.Linear(nh2, nc, bias=not disable_bias)
        self.enable_dropout = enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        
        if self.do_log_soft:
            return F.log_softmax(x)
        else:
            return x

class MlpNet_Double(nn.Module):
    def __init__(self, dataset='mnist', enable_dropout=False, do_log_soft=True):
        super(MlpNet_Double, self).__init__()
        if dataset == 'mnist':
            input_size = 784
        else:
            raise NotImplementedError
        
        nh1 = 80
        nh2 = 40
        nc = 10

        self.fc1 = nn.Linear(input_size, nh1)
        self.fc2 = nn.Linear(nh1, nh2)
        self.fc3 = nn.Linear(nh2, nc)
        self.enable_dropout = enable_dropout
        self.do_log_soft = do_log_soft

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        if self.do_log_soft:
            return F.log_softmax(x, dim=1)
        return x


class MLPBIG(nn.Module):
    def __init__(self):
        super(MLPBIG, self).__init__()

        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)