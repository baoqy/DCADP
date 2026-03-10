class MlpNet_100K(nn.Module):
    def __init__(self, args=None, dataset='cifar10'):
        super(MlpNet_100K, self).__init__()

        if dataset == 'cifar10':
            input_size = 32*32*3
        elif dataset == 'mnist':
            input_size = 28*28
        else:
            raise ValueError("Unknown dataset")

        # 设置隐藏层参数，保证总参数大约10万
        if args is None:
            nh1 = 128
            nh2 = 128
            nc = 10
            enable_dropout = False
            disable_bias = True
            do_log_soft = True
        else:
            nh1 = args.num_hidden_nodes1
            nh2 = args.num_hidden_nodes2
            nc = args.num_classes
            enable_dropout = args.enable_dropout
            disable_bias = args.disable_bias
            do_log_soft = not args.disable_log_soft

        self.do_log_soft = do_log_soft
        print("Do log softmax:", do_log_soft)
        print("MLP-100K structure:", input_size, "→", nh1, "→", nh2, "→", nc)

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

        # 推理阶段输出概率
        if not self.training and self.do_log_soft:
            return F.log_softmax(x, dim=1)
        else:
            return x  # 训练阶段返回 logits