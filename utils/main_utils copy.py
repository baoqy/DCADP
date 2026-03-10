import torch
import sys
import numpy as np
import os
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.models import resnet50 as torch_resnet50
from models.resnet_cifar10 import resnet20
from models.resnet_cifar10 import resnet20_small
from models.resnet_cifar10 import resnet20_tiny
from models.resnet_cifar10 import resnet8
from models.resnet_mnist import resnet20_mnist
from models.wideresnet_cifar import Wide_ResNet
from models.mlpnet import MlpNet
from models.mlpnet import MlpNet_Double
from models.conv40k import Conv40k
from models.conv80k import Conv80k
from models.LeNet5_80k import LeNet5_Precise80K
from models.mobilenet import mobilenet
from collections import OrderedDict
import json
import torch.distributed as dist
import torch.nn.functional as F

from CHITA_opt.L0_card_const import Heuristic_CD_PP,Active_IHTCDLS_PP,Heuristic_LS,Heuristic_LSBlock,evaluate_obj

def sync_weights(model, rank, world_size):
    for param in model.parameters():
        if rank == 0:
            # Rank 0 is sending it's own weight
            # to all it's siblings (1 to world_size)
            for sibling in range(1, world_size):
                dist.send(param.data, dst=sibling)
        else:
            # Siblings must recieve the parameters
            dist.recv(param.data, src=0)

def sync_mask(pruner, rank, world_size):
    if rank == 0:
        # Rank 0 is sending it's own weight
        # to all it's siblings (1 to world_size)
        for sibling in range(1, world_size):
            dist.send(pruner.mask.data, dst=sibling)
    else:
        # Siblings must recieve the parameters
        dist.recv(pruner.mask.data, src=0)

def flatten_tensor_list(tensors):
    flattened = []
    for tensor in tensors:
        flattened.append(tensor.view(-1))
    return torch.cat(flattened, 0)


def print_parameters(model):
    for name, param in model.named_parameters(): 
        print(name, param.shape)

def load_model(path, model):
    tmp = torch.load(path, map_location='cpu')
    if 'state_dict' in tmp:
        tmp = tmp['state_dict']
    if 'model' in tmp:
        tmp = tmp['model']
    for k in list(tmp.keys()):
        if 'module.' in k:
            tmp[k.replace('module.', '')] = tmp[k]
            del tmp[k]
    model.load_state_dict(tmp)


def imagenet_get_datasets(data_dir):

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
    ]
    
    train_transform += [
        transforms.ToTensor(),
        normalize,
    ]
    train_transform = transforms.Compose(train_transform)

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset


@torch.no_grad()
def get_pvec(model, params):
    state_dict = model.state_dict()
    return torch.cat([
        state_dict[p].reshape(-1) for p in params
    ])

@torch.no_grad()
def get_sparsity(model, params):
    pvec = get_pvec(model,params)
    return (pvec == 0).float().mean()

@torch.no_grad()
def get_blocklist(model,params,block_size):
    i_w = 0
    block_list = [0]
    state_dict = model.state_dict()
    for p in params:
        param_size = np.prod(state_dict[p].shape)
        if param_size <block_size:
            block_list.append(i_w+param_size)
        else:
            num_block = int(param_size/block_size)
            block_subdiag = list(range(i_w,i_w+param_size+1,int(param_size/num_block))) 
            block_subdiag[-1] = i_w+param_size
            block_list += block_subdiag   
        i_w += param_size
    return block_list

@torch.no_grad()
def set_pvec(w, model, params,device, nhwc=False):
    state_dict = model.state_dict()
    i = 0
    for p in params:
        count = state_dict[p].numel()
        if type(w) ==  torch.Tensor :
            state_dict[p] = w[i:(i + count)].reshape(state_dict[p].shape)
        else:
            state_dict[p] = torch.Tensor(w[i:(i + count)]).to(device).reshape(state_dict[p].shape)
        i += count
    model.load_state_dict(state_dict)

@torch.no_grad()
def get_gvec(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad.reshape(-1) for p in params
    ])
@torch.no_grad()
def get_gvec1(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad_sample.reshape(named_parameters[p].grad_sample.shape[0],-1) for p in params
    ],dim=1)

@torch.no_grad()
def get_gps_vec(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad_sample.reshape(named_parameters[p].grad_sample.shape[0],-1) for p in params
    ],dim=1)
@torch.no_grad()
def apply_mask(mask, model, params,device):
    state_dict = model.state_dict()
    i = 0
    for p in params:
        param = state_dict[p]
        count = param.numel()
        state_dict[p] *= mask[i:(i + count)].to(device).reshape(param.shape).float()
        i += count
    model.load_state_dict(state_dict)
    
@torch.no_grad()
def zero_grads(model):
    for p in model.parameters():
        p.grad = None

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_acc(model,dataloader,device='cpu',verbose=False):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    i = 0
    with torch.no_grad():
        for data in dataloader:
            i+=1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images=images
            labels=labels
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if verbose and i%10 == 0:
                print(total,correct)

            del images,labels,outputs

    return 100 * correct / total

def compute_kl_loss(model, dataloader, device='cpu', verbose=False):
    avg_loss = 0
    i = 0
    with torch.no_grad():
        for data in dataloader:
            i += 1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            # 模型预测分布
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=1)
            
            # 将真实标签转为独热编码形式
            true_probs = F.one_hot(labels, num_classes=outputs.size(1)).float().to(device)
            true_probs /= true_probs.sum(dim=1, keepdim=True)  # 确保归一化

            # 计算 KL 散度损失
            loss = F.kl_div(log_probs, true_probs, reduction='batchmean')
            avg_loss += loss.item()

            if verbose and i % 100 == 0:
                print(f"Batch {i}: KL Loss = {loss.item()}")

            del images, labels, outputs

    return avg_loss / i

def compute_loss(model,criterion,dataloader,device='cpu',verbose=False):
    avg_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    i = 0
    with torch.no_grad():
        for data in dataloader:
            i+=1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images=images
            labels=labels
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = criterion(outputs, labels).item()
            avg_loss += loss
            if verbose and i%100 ==0:
                print('computing loss', i)

            del images,labels,outputs

    return avg_loss / i


def generate_schedule(num_stages, base_level,sparsity_level,schedule):
    repeat=1
    if num_stages == 1:
        return [sparsity_level]
    if schedule == 'exp':
        sparsity_multiplier = (sparsity_level - base_level)*np.power(2, num_stages-1)/(np.power(2, num_stages-1) - 1)
        l =[base_level + sparsity_multiplier*((np.power(2, stage) - 1)/np.power(2, stage)) for stage in range(num_stages)]
        return [x for x in l for _ in range(repeat)]
    elif schedule == 'poly':
        l= [sparsity_level + (base_level-sparsity_level)*np.power(1 - (stage/(num_stages-1)), 3) for stage in range(num_stages)]
        return [x for x in l for _ in range(repeat)]
    elif schedule == 'const':
        return [sparsity_level for stage in range(num_stages)]
    elif schedule == 'linear':
        return [base_level + stage*(sparsity_level - base_level)/(num_stages-1) for stage in range(num_stages)]
    elif schedule == 'MFAC':
        sparsity_multiplier = ((1. - sparsity_level) / (1. - base_level)) ** (1./num_stages)
        return [1. - ((1. - base_level) * (sparsity_multiplier**(stage+1))) for stage in range(num_stages)]

def mnist_get_datasets(data_dir):
    # same used in hessian repo!
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                                   download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                                  transform=test_transform)

    return train_dataset, test_dataset

def model_factory(arch,dset_path,pretrained=True):
    if arch == 'mlpnet':
        model = MlpNet(args=None,dataset='mnist')
        train_dataset,test_dataset = mnist_get_datasets(dset_path)
        criterion = torch.nn.functional.nll_loss

        state_trained = torch.load('checkpoints/mnist_25_epoch_93.97.ckpt',map_location=torch.device('cpu'))['model_state_dict']
        new_state_trained = OrderedDict()
        for k in state_trained:
            if 'mask' in k:
                continue
            new_state_trained[k.split('.')[1]+'.'+k.split('.')[3]] = state_trained[k]
        if pretrained:
            model.load_state_dict(new_state_trained)

        modules_to_prune = []
        for name, param in model.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)
        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'mlpnet_double':
        model = MlpNet_Double(dataset='mnist')
        train_dataset,test_dataset = mnist_get_datasets(dset_path)
        criterion = torch.nn.functional.nll_loss
        if pretrained:
            state_trained = torch.load('checkpoints/mlpnet_double_mnist.pth', map_location=torch.device('cpu'))['model_state_dict']
            model.load_state_dict(state_trained)
            #state_trained = torch.load('checkpoints/mnist_25_epoch_93.97.ckpt',map_location=torch.device('cpu'))['model_state_dict']
            new_state_trained = OrderedDict()

        for k, v in state_trained.items():
            if 'mask' in k:
                continue
            new_state_trained[k] = v  # 不做复杂拆分，直接保留 key

        if pretrained:
            model.load_state_dict(new_state_trained)

        modules_to_prune = []
        for name, param in model.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)
        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'conv40k':
        # 尝试加载预训练权重
        pretrained_path = 'checkpoints/conv40k_cifar10.pth'  # 你训练好的 ResNet8 权重
        model = Conv40k()
    
        if pretrained:
            try:
                # 详细调试信息
                print(f"=== 调试模型加载 ===")
                print(f"模型文件路径: {pretrained_path}")
                print(f"文件存在: {os.path.exists(pretrained_path)}")
            
                if os.path.exists(pretrained_path):
                    # 检查文件大小
                    file_size = os.path.getsize(pretrained_path) / 1024 / 1024
                    print(f"文件大小: {file_size:.2f} MB")
                
                    # 加载并检查内容
                    state_trained = torch.load(pretrained_path, map_location='cpu')
                    print(f"加载的对象类型: {type(state_trained)}")
                
                    if isinstance(state_trained, dict):
                        print(f"字典键: {list(state_trained.keys())}")
                    
                        # 检查是否是完整checkpoint
                        if 'model_state_dict' in state_trained:
                            print("这是一个完整checkpoint，包含model_state_dict")
                            state_dict = state_trained['model_state_dict']
                        elif 'state_dict' in state_trained:
                            print("这是一个完整checkpoint，包含state_dict") 
                            state_dict = state_trained['state_dict']
                        else:
                            print("这可能是直接的state_dict")
                            state_dict = state_trained
                    else:
                        print("这不是字典，可能是直接的state_dict")
                        state_dict = state_trained
                
                    # 打印模型参数信息
                    print(f"当前模型参数:")
                    model_keys = list(model.state_dict().keys())
                    for key in model_keys[:3]:  # 显示前3个
                        print(f"  {key}: {model.state_dict()[key].shape}")
                
                    print(f"保存的模型参数:")
                    state_keys = list(state_dict.keys())
                    for key in state_keys[:3]:  # 显示前3个
                        print(f"  {key}: {state_dict[key].shape}")
                
                    # 尝试加载
                    model.load_state_dict(state_dict, strict=True)
                    print("✓ 模型加载成功")
                
                else:
                    print("文件不存在，无法加载预训练权重")
                
            except Exception as e:
                print(f"✗ 加载失败: {e}")
                print("将从头开始训练")

        # 数据增强
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_random_transforms = True
        if train_random_transforms:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True, transform=test_transform)

        criterion = torch.nn.functional.cross_entropy

        # 可剪枝模块
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'conv80k':
        # 尝试加载预训练权重
        pretrained_path = 'checkpoints/conv80k_cifar10.pth'  # 你训练好的 ResNet8 权重
        model = Conv80k()
        if pretrained:
            try:
                state_trained = torch.load(pretrained_path, map_location='cpu')
                model.load_state_dict(state_trained)
            except:
                print("Warning: No pretrained Conv80k found, training from scratch.")

        # 数据增强
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_random_transforms = True
        if train_random_transforms:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True, transform=test_transform)

        criterion = torch.nn.functional.cross_entropy

        # 可剪枝模块
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'LeNet5_80k':
        # 尝试加载预训练权重
        pretrained_path = 'checkpoints/LeNet5_80k_cifar10.pth'  # 你训练好的 ResNet8 权重
        model = LeNet5_Precise80K()
        if pretrained:
            try:
                state_trained = torch.load(pretrained_path, map_location='cpu')
                model.load_state_dict(state_trained)
            except:
                print("Warning: No pretrained LeNet5_80k found, training from scratch.")

        # 数据增强
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_random_transforms = True
        if train_random_transforms:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True, transform=test_transform)

        criterion = torch.nn.functional.cross_entropy

        # 可剪枝模块
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'resnet20':
        state_trained = torch.load('checkpoints/resnet20_cifar10.pth.tar',map_location=torch.device('cpu'))['state_dict']
        new_state_trained = OrderedDict()
        for k in state_trained:
            new_state_trained[k[7:]] = state_trained[k]

        model = resnet20()
        if pretrained:
            model.load_state_dict(new_state_trained)

        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_random_transforms=True

        if train_random_transforms:
            train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True,transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True,transform=test_transform)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, param in model.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'res20_mnist':

        pretrained = True     # 是否加载训练好的模型
        # ===============================
        #  加载训练好的 MNIST ResNet20
        # ===============================
        state_trained = torch.load('checkpoints/resnet20_mnist.pth', map_location=torch.device('cpu'))
        # 如果保存的是 state_dict，直接使用，如果是字典中包含 'state_dict'，需要提取
        if 'state_dict' in state_trained:
            state_trained = state_trained['state_dict']

        new_state_trained = OrderedDict()
        for k in state_trained:
            # 去掉可能的 'module.' 前缀（如果用 DataParallel 保存过）
            new_key = k[7:] if k.startswith('module.') else k
            new_state_trained[new_key] = state_trained[k]

        # 构建模型
        model = resnet20_mnist()
        if pretrained:
            model.load_state_dict(new_state_trained)

        # ===============================
        #  数据集及转换
        # ===============================
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
        ])

        train_random_transforms = True
        if train_random_transforms:
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            train_transform = test_transform

        train_dataset,test_dataset = mnist_get_datasets(dset_path)
        #train_dataset = datasets.MNIST(root=dset_path, train=True, download=True, transform=train_transform)
        #test_dataset = datasets.MNIST(root=dset_path, train=False, download=True, transform=test_transform)

        # ===============================
        #  损失函数
        # ===============================
        criterion = torch.nn.functional.cross_entropy

        # ===============================
        #  需要稀疏化/剪枝的参数
        # ===============================
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        # ===============================
        #  输出
        # ===============================
        print("Model, datasets, criterion, modules_to_prune 已准备好")
        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'mobilenetv1':
        model = mobilenet()
        train_dataset,test_dataset = imagenet_get_datasets(dset_path)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                modules_to_prune.append(name+'.weight')


        if pretrained:
            path = 'checkpoints/MobileNetV1-Dense-STR.pth'
            state_trained = torch.load(path,map_location=torch.device('cpu'))['state_dict']
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key',key)
            model.load_state_dict(new_state_trained,strict=False)

        return model,train_dataset,test_dataset,criterion,modules_to_prune

    elif arch == 'resnet50':
        model = torch_resnet50(weights=None)
        train_dataset,test_dataset = imagenet_get_datasets(dset_path)
        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                modules_to_prune.append(name+'.weight')
        print('Pruning modeules',modules_to_prune)
        if pretrained:
            
            path = 'checkpoints/ResNet50-Dense.pth'
            #path = 'checkpoints/resnet50-19c8e357.pth'
            
            state_trained = torch.load(path,map_location=torch.device('cpu'))['state_dict']
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key',key)
            model.load_state_dict(new_state_trained,strict=False)
        return model,train_dataset,test_dataset,criterion,modules_to_prune
            #model.load_state_dict(torch.load(path))
    elif arch == 'resnet8':
        # 尝试加载预训练权重
        pretrained_path = 'checkpoints/resnet8_cifar10.pth'  # 你训练好的 ResNet8 权重
        model = resnet8()
        if pretrained:
            try:
                state_trained = torch.load(pretrained_path, map_location='cpu')
                model.load_state_dict(state_trained)
            except:
                print("Warning: No pretrained ResNet8 found, training from scratch.")

        # 数据增强
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_random_transforms = True
        if train_random_transforms:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True, transform=test_transform)

        criterion = torch.nn.functional.cross_entropy

        # 可剪枝模块
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'resnet20_small':
        # 尝试加载预训练权重
        pretrained_path = 'checkpoints/resnet20_small_cifar10.pth'  # 你训练好的 ResNet8 权重
        model = resnet20_small()
        if pretrained:
            try:
                state_trained = torch.load(pretrained_path, map_location='cpu')
                model.load_state_dict(state_trained)
            except:
                print("Warning: No pretrained ResNet20_small found, training from scratch.")

        # 数据增强
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_random_transforms = True
        if train_random_transforms:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True, transform=test_transform)

        criterion = torch.nn.functional.cross_entropy

        # 可剪枝模块
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'resnet20_tiny':
        # 尝试加载预训练权重
        pretrained_path = 'checkpoints/resnet20_tiny_cifar10.pth'  # 你训练好的 ResNet8 权重
        model = resnet20_tiny()
        if pretrained:
            try:
                state_trained = torch.load(pretrained_path, map_location='cpu')
                model.load_state_dict(state_trained)
            except:
                print("Warning: No pretrained ResNet20_tiny found, training from scratch.")

        # 数据增强
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_random_transforms = True
        if train_random_transforms:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True, transform=test_transform)

        criterion = torch.nn.functional.cross_entropy

        # 可剪枝模块
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        return model,train_dataset,test_dataset,criterion,modules_to_prune


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)