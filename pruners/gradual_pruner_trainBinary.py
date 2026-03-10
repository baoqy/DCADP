from utils.main_utils import *
import time
import torch.distributed as dist
import gc
import copy
import torch.nn as nn
import numpy as np
import json
import torch


class DirectEqualMagnitudeNetwork(nn.Module):
    """直接等幅值网络 - 参数本身就是等幅值"""
    def __init__(self, base_model, fixed_sign_pattern=None, magnitude_value=None, device=None):
        super().__init__()
        self.base_model = base_model
        self.device = device if device is not None else next(base_model.parameters()).device
        
        # 存储符号（固定不变）
        self.signs = {}
        
        # 初始化符号
        if fixed_sign_pattern is None:
            for name, param in base_model.named_parameters():
                flat_name = name.replace('.', '_')
                sign_tensor = torch.sign(param.data)
                zero_mask = (sign_tensor == 0)
                if zero_mask.any():
                    random_sign = torch.where(
                        torch.randn_like(sign_tensor[zero_mask]) > 0,
                        torch.tensor(1.0, device=self.device),
                        torch.tensor(-1.0, device=self.device)
                    )
                    sign_tensor[zero_mask] = random_sign
                
                self.register_buffer(f'sign_{flat_name}', sign_tensor)
                self.signs[name] = getattr(self, f'sign_{flat_name}')
        else:
            for name, sign in fixed_sign_pattern.items():
                flat_name = name.replace('.', '_')
                sign_tensor = sign.to(self.device)
                self.register_buffer(f'sign_{flat_name}', sign_tensor)
                self.signs[name] = getattr(self, f'sign_{flat_name}')
        
        # 幅值参数（可学习）
        if magnitude_value is None:
            # 计算初始幅值
            total_abs = 0.0
            total_params = 0
            for param in base_model.parameters():
                total_abs += param.data.abs().sum().item()
                total_params += param.numel()
            magnitude_value = total_abs / total_params if total_params > 0 else 0.01
        
        self.magnitude = nn.Parameter(torch.tensor(magnitude_value, device=self.device))
        
        # 立即设置参数为等幅值
        self._apply_equal_magnitude()
        
        print(f"DirectEqualMagnitudeNetwork initialized with magnitude: {magnitude_value:.6f}")
    
    def _apply_equal_magnitude(self):
        """设置所有参数为 sign * magnitude"""
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in self.signs:
                    param.data = self.signs[name] * self.magnitude.abs()
    
    def forward(self, x):
        # 参数已经是等幅值，直接使用
        return self.base_model(x)
    
    def apply_sparsity_mask(self, mask_dict):
        """应用稀疏性掩码，保持等幅值属性"""
        with torch.no_grad():
            current_magnitude = self.magnitude.abs()
            for name, param in self.base_model.named_parameters():
                if name in mask_dict and name in self.signs:
                    param_mask = mask_dict[name].to(self.device)
                    sign = self.signs[name]
                    # 设置：非零位置 = sign * magnitude，零位置 = 0
                    param.data = sign * current_magnitude * param_mask
    
    def update_signs_from_params(self):
        """从当前参数更新符号"""
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in self.signs:
                    new_sign = torch.sign(param.data)
                    zero_mask = (new_sign == 0)
                    if zero_mask.any():
                        random_sign = torch.where(
                            torch.randn_like(new_sign[zero_mask]) > 0,
                            torch.tensor(1.0, device=self.device),
                            torch.tensor(-1.0, device=self.device)
                        )
                        new_sign[zero_mask] = random_sign
                    self.signs[name].data.copy_(new_sign)
    
    def get_magnitude(self):
        return self.magnitude.item()
    
    def set_magnitude(self, value):
        with torch.no_grad():
            self.magnitude.data = torch.tensor(value, device=self.device)
        self._apply_equal_magnitude()
    
    def restore_original_weights(self, original_params):
        """恢复原始权重（用于剪枝前）"""
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])
    
    def save_current_weights(self):
        """保存当前权重"""
        current_weights = {}
        for name, param in self.base_model.named_parameters():
            current_weights[name] = param.data.clone()
        return current_weights
    
    def verify_equal_magnitude(self, tolerance=1e-6):
        """验证等幅值属性"""
        all_nonzero_values = []
        total_params = 0
        nonzero_params = 0
        
        current_magnitude = self.get_magnitude()
        
        for name, param in self.base_model.named_parameters():
            if name in self.signs:
                param_abs = param.data.abs()
                total_params += param.numel()
                
                # 检测非零位置
                nonzero_mask = param_abs > tolerance
                nonzero_count = nonzero_mask.sum().item()
                nonzero_params += nonzero_count
                
                if nonzero_count > 0:
                    nonzero_values = param_abs[nonzero_mask]
                    all_nonzero_values.append(nonzero_values.flatten())
        
        if all_nonzero_values:
            all_values = torch.cat(all_nonzero_values)
            max_diff = (all_values - current_magnitude).abs().max().item()
            std_abs = all_values.std().item()
            
            is_equal = max_diff < tolerance
            
            return is_equal, current_magnitude, std_abs, total_params, nonzero_params
        return True, current_magnitude, 0.0, total_params, nonzero_params


class GradualPruner:
    def __init__(self, multi_stage_pruner, train_dataloader, test_dataloader, criterion,
                 params, reset_optimizer, momentum, weight_decay, results, filename, seed, 
                 mask=None, model=None, device=None, first_epoch=0, distributed=False, 
                 rank=-1, world_size=1, use_equal_magnitude=True):
        
        set_seed(seed)
        assert not distributed or rank >= 0
        
        self.pruner = multi_stage_pruner
        self.device = device
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.params = params
        self.reset_optimizer = reset_optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.results = results
        self.filename = filename
        self.use_equal_magnitude = use_equal_magnitude
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        
        # 等幅值网络包装
        if self.use_equal_magnitude:
            self.equal_mag_model = DirectEqualMagnitudeNetwork(model, device=self.device)
            self.model = self.equal_mag_model.base_model  # 原始模型
            # 保存初始参数（用于剪枝前恢复）
            self.original_params_before_pruning = None
        else:
            self.model = model
            self.equal_mag_model = None
        
        # 获取model_without_ddp的引用
        if self.distributed:
            self.model_without_ddp = self.pruner.pruner.model
        else:
            self.model_without_ddp = self.model
        
        # 初始化掩码
        if mask is None:
            self.mask = torch.ones_like(get_pvec(self.model_without_ddp, self.params)).cpu() != 0
        else:
            self.mask = mask
        
        self.nonzero_count = self.mask.sum().item()
        self.mask = self.mask.to(self.device)
        
        # 将掩码转换为字典格式
        self.mask_dict = self._convert_mask_to_dict()
        
        # 优化器设置
        if self.use_equal_magnitude and self.equal_mag_model is not None:
            # 只优化幅值参数
            self.optim = torch.optim.SGD(
                [self.equal_mag_model.magnitude], 
                lr=0, 
                momentum=self.momentum, 
                weight_decay=self.weight_decay
            )
            print(f"Optimizing only magnitude parameter, initial value: {self.equal_mag_model.get_magnitude():.6f}")
        else:
            self.optim = torch.optim.SGD(
                self.model.parameters(), 
                lr=0, 
                momentum=self.momentum, 
                weight_decay=self.weight_decay
            )
        
        self.runloss = 0
        self.step = 0
        self.first_epoch = first_epoch
        
        sparsity = (~self.mask.cpu().numpy()).mean()
        print(f'Initial Model sparsity: {sparsity:.4f} at rank {self.rank}')
        
        if self.use_equal_magnitude and self.equal_mag_model is not None:
            magnitude_val = self.equal_mag_model.get_magnitude()
            print(f'Using direct equal magnitude training with initial magnitude: {magnitude_val:.6f}')
            
            # 验证初始等幅值属性
            self._verify_and_print_magnitude("Initialization")
    
    def _convert_mask_to_dict(self):
        """将扁平化的掩码转换为参数字典格式"""
        mask_dict = {}
        
        # 获取模型参数
        param_vec = get_pvec(self.model_without_ddp, self.params)
        
        # 获取每个参数的形状和大小
        param_shapes = {}
        param_sizes = {}
        for name, param in self.model_without_ddp.named_parameters():
            param_shapes[name] = param.shape
            param_sizes[name] = param.numel()
        
        # 将扁平掩码分割为各个参数
        start_idx = 0
        for name, param in self.model_without_ddp.named_parameters():
            if name in param_sizes:
                size = param_sizes[name]
                end_idx = start_idx + size
                param_mask = self.mask[start_idx:end_idx].reshape(param_shapes[name])
                mask_dict[name] = param_mask
                start_idx = end_idx
        
        return mask_dict
    
    def _verify_and_print_magnitude(self, stage=""):
        """验证并打印幅值统计信息"""
        if not self.use_equal_magnitude or self.equal_mag_model is None:
            return
        
        is_equal, magnitude, std_abs, total_params, nonzero_params = \
            self.equal_mag_model.verify_equal_magnitude()
        
        sparsity = (total_params - nonzero_params) / total_params if total_params > 0 else 0
        
        print(f"\n=== Equal Magnitude Verification ({stage}) ===")
        print(f"  Current magnitude: {magnitude:.8f}")
        print(f"  Total parameters: {total_params}")
        print(f"  Nonzero parameters: {nonzero_params}")
        print(f"  Sparsity: {sparsity:.4f}")
        print(f"  Absolute std: {std_abs:.8f}")
        print(f"  Equal magnitude: {'✓' if is_equal else '✗'}")
        
        if not is_equal:
            print(f"  WARNING: Parameters do NOT have equal magnitude!")
    
    def train(self):
        """训练一个epoch - 参数始终是等幅值的"""
        torch.cuda.empty_cache()
        gc.collect()
        
        self.model.train()
        
        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # 清零梯度
            self.optim.zero_grad()
            
            # 前向传播（参数已经是等幅值）
            if self.use_equal_magnitude and self.equal_mag_model is not None:
                output = self.equal_mag_model(x)
            else:
                output = self.model(x)
            
            # 计算损失
            loss = self.criterion(output, y)
            
            # 反向传播
            loss.backward()
            
            # 优化（如果是等幅值训练，只更新幅值）
            self.optim.step()
            
            # 重新应用等幅值（确保参数正确）
            if self.use_equal_magnitude and self.equal_mag_model is not None:
                self.equal_mag_model._apply_equal_magnitude()
            
            # 应用稀疏性掩码
            if self.use_equal_magnitude and self.equal_mag_model is not None:
                self.equal_mag_model.apply_sparsity_mask(self.mask_dict)
            else:
                apply_mask(self.mask, self.model_without_ddp, self.params, self.device)
            
            # 更新运行损失
            self.runloss = 0.99 * self.runloss + 0.01 * loss.item()
            self.step += 1
            
            # 定期打印信息
            if self.step % 100 == 0:
                magnitude_val = None
                if self.use_equal_magnitude and self.equal_mag_model is not None:
                    magnitude_val = self.equal_mag_model.get_magnitude()
                
                if magnitude_val is not None:
                    print(f'step {self.step:06d}: loss={loss.item():.3f}, magnitude={magnitude_val:.6f} at rank {self.rank}')
                else:
                    print(f'step {self.step:06d}: loss={loss.item():.3f} at rank {self.rank}')
            
            # 每500步验证一次等幅值属性
            if self.step % 500 == 0 and self.use_equal_magnitude:
                self._verify_and_print_magnitude(f"Step {self.step}")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    def _extract_max_abs_from_w_pruned(self, w_pruned):
        """从w_pruned中提取最大绝对值"""
        max_abs = 0.01  # 默认最小值
        
        if w_pruned is None:
            return max_abs
        
        if isinstance(w_pruned, dict):
            # w_pruned是字典格式
            for name, tensor in w_pruned.items():
                if tensor is not None and torch.is_tensor(tensor):
                    if tensor.numel() > 0:
                        current_max = tensor.abs().max().item()
                        max_abs = max(max_abs, current_max)
        elif torch.is_tensor(w_pruned):
            # w_pruned是张量
            if w_pruned.numel() > 0:
                max_abs = w_pruned.abs().max().item()
        else:
            # 尝试其他格式
            try:
                if hasattr(w_pruned, '__len__') and len(w_pruned) > 0:
                    # 假设是列表或元组
                    for item in w_pruned:
                        if torch.is_tensor(item):
                            if item.numel() > 0:
                                current_max = item.abs().max().item()
                                max_abs = max(max_abs, current_max)
            except:
                pass
        
        return max_abs
    
    def prune(self, nepochs, lr_schedule, prunepochs, sparsities, base_level_=0.1):
        """主训练和剪枝循环"""
        # 初始化结果记录
        for key in ['epoch', 'pruning_res', 'running_loss', 'acc', 'lr', 
                    'momentum', 'weight_decay', 'magnitude_value']:
            self.results[-1][key] = []
        
        for epoch in range(self.first_epoch, nepochs):
            if self.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # 剪枝步骤
            if epoch in prunepochs and (not self.distributed or self.rank == 0):
                if self.reset_optimizer:
                    # 重置优化器
                    if self.use_equal_magnitude and self.equal_mag_model is not None:
                        self.optim = torch.optim.SGD(
                            [self.equal_mag_model.magnitude], 
                            lr=0, 
                            momentum=self.momentum, 
                            weight_decay=self.weight_decay
                        )
                    else:
                        self.optim = torch.optim.SGD(
                            self.model.parameters(), 
                            lr=0, 
                            momentum=self.momentum, 
                            weight_decay=self.weight_decay
                        )
                
                epoch_index = prunepochs.index(epoch)
                base_level = sparsities[epoch_index-1] if epoch_index > 0 else base_level_
                
                self.model.eval()
                if self.use_equal_magnitude:
                    self.equal_mag_model.eval()
                
                # 关键：在剪枝前保存当前参数
                if self.use_equal_magnitude and self.equal_mag_model is not None:
                    # 保存当前等幅值参数
                    self.original_params_before_pruning = self.equal_mag_model.save_current_weights()
                    print(f"Saved current parameters before pruning at epoch {epoch}")
                
                # 执行剪枝（使用当前参数计算w_pruned）
                w_pruned, mask, k = self.pruner.prune(self.mask, sparsities[epoch_index], base_level)
                self.mask = mask.to(self.device)
                
                # 更新掩码字典
                self.mask_dict = self._convert_mask_to_dict()
                
                # 更新等幅值网络
                if self.use_equal_magnitude and self.equal_mag_model is not None:
                    # 1. 从w_pruned中提取最大绝对值
                    max_abs = self._extract_max_abs_from_w_pruned(w_pruned)
                    print(f"Extracted max absolute value from w_pruned: {max_abs:.6f}")
                    
                    # 2. 恢复剪枝前的参数（如果需要）
                    if self.original_params_before_pruning is not None:
                        # 恢复参数
                        self.equal_mag_model.restore_original_weights(self.original_params_before_pruning)
                        print(f"Restored original parameters before re-initialization")
                    
                    # 3. 使用w_pruned中的最大绝对值重新初始化幅值
                    self.equal_mag_model.set_magnitude(max_abs)
                    print(f"Re-initialized magnitude to: {self.equal_mag_model.get_magnitude():.6f}")
                    
                    # 4. 更新符号模式（使用w_pruned或当前参数）
                    # 如果有w_pruned，使用w_pruned的符号
                    if w_pruned is not None and isinstance(w_pruned, dict):
                        # 使用w_pruned更新符号
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if name in self.equal_mag_model.signs and name in w_pruned:
                                    pruned_tensor = w_pruned[name]
                                    if pruned_tensor is not None and torch.is_tensor(pruned_tensor):
                                        new_sign = torch.sign(pruned_tensor)
                                        zero_mask = (new_sign == 0)
                                        if zero_mask.any():
                                            random_sign = torch.where(
                                                torch.randn_like(new_sign[zero_mask]) > 0,
                                                torch.tensor(1.0, device=self.device),
                                                torch.tensor(-1.0, device=self.device)
                                            )
                                            new_sign[zero_mask] = random_sign
                                        self.equal_mag_model.signs[name].data.copy_(new_sign)
                        print(f"Updated signs from w_pruned")
                    else:
                        # 使用当前参数更新符号
                        self.equal_mag_model.update_signs_from_params()
                        print(f"Updated signs from current parameters")
                    
                    # 5. 重新应用等幅值
                    self.equal_mag_model._apply_equal_magnitude()
                    
                    # 6. 应用稀疏性掩码
                    self.equal_mag_model.apply_sparsity_mask(self.mask_dict)
                    
                    print(f"After pruning at epoch {epoch}: magnitude = {self.equal_mag_model.get_magnitude():.6f}")
                    
                    # 清理
                    self.original_params_before_pruning = None
                
                del mask
                pruning_res = copy.deepcopy(self.pruner.results)
                self.pruner.reset_pruner()
                
                self.model.train()
                if self.use_equal_magnitude:
                    self.equal_mag_model.train()
                
                if epoch == prunepochs[-1]:
                    self.pruner = None
            else:
                pruning_res = []
            
            # 分布式同步
            if self.distributed:
                dist.barrier()
                sync_weights(self.model, self.rank, self.world_size)
                sync_mask(self, self.rank, self.world_size)
                
                if self.use_equal_magnitude and self.equal_mag_model is not None:
                    # 同步幅值
                    dist.broadcast(self.equal_mag_model.magnitude.data, src=0)
                    
                    # 同步后重新应用等幅值
                    self.equal_mag_model._apply_equal_magnitude()
                    self.equal_mag_model.apply_sparsity_mask(self.mask_dict)
                
                sparsity = (~(get_pvec(self.model.module, self.params).cpu() != 0).numpy()).mean()
                mask_sparsity = (~self.mask.cpu().numpy()).mean()
                print(f'Done syncing at rank {self.rank}, '
                      f'sparsity={sparsity:.4f}, mask_sparsity={mask_sparsity:.4f}')
            
            # 设置学习率
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr_schedule[epoch]
            
            print(f'starting epoch {epoch} -- lr {lr_schedule[epoch]} at rank {self.rank}')
            
            # 训练一个epoch
            start_epoch = time.time()
            self.train()
            end_epoch = time.time()
            
            # 评估
            if self.distributed:
                dist.barrier()
            
            self.model.eval()
            if self.use_equal_magnitude:
                self.equal_mag_model.eval()
            
            print(f'epoch {epoch} - time: {end_epoch-start_epoch:.2f}s at rank {self.rank}')
            
            if not self.distributed or self.rank == 0:
                # 使用当前参数计算准确率（已经是等幅值）
                acc = compute_acc(self.model, self.test_dataloader, self.device)
                print('epoch ',epoch, ' - acc :',acc, ' - time :',end_epoch-start_epoch,' at rank',self.rank)
                magnitude_val = None
                if self.use_equal_magnitude and self.equal_mag_model is not None:
                    magnitude_val = self.equal_mag_model.get_magnitude()
                """
                print(f'epoch {epoch} - acc: {acc:.4f}, '
                      f'magnitude={magnitude_val:.6f if magnitude_val else "N/A"}, '
                      f'time: {end_epoch-start_epoch:.2f}s')
                """
                # 验证等幅值属性
                if self.use_equal_magnitude and self.equal_mag_model is not None:
                    self._verify_and_print_magnitude(f"Epoch {epoch}")
                
                # 保存检查点
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'mask': self.mask.cpu(),
                    'sparsity': (~self.mask.cpu().numpy()).mean(),
                    'magnitude': magnitude_val,
                    'acc': acc
                }
                
                if self.use_equal_magnitude and self.equal_mag_model is not None:
                    # 保存符号模式
                    sign_pattern = {}
                    for name, sign_tensor in self.equal_mag_model.signs.items():
                        sign_pattern[name] = sign_tensor.cpu()
                    checkpoint['sign_pattern'] = sign_pattern
                
                PATH = f'{self.filename}_epoch{epoch}.pth'
                torch.save(checkpoint, PATH)
                
                # 记录结果
                self.results[-1]['epoch'].append(epoch)
                self.results[-1]['pruning_res'].append(pruning_res)
                self.results[-1]['running_loss'].append(self.runloss)
                self.results[-1]['acc'].append(acc)
                self.results[-1]['lr'].append(lr_schedule[epoch])
                self.results[-1]['momentum'].append(self.momentum)
                self.results[-1]['weight_decay'].append(self.weight_decay)
                self.results[-1]['magnitude_value'].append(magnitude_val)
                
                # 保存结果到文件
                with open(self.filename, "w") as file:
                    json.dump(self.results, file, cls=NpEncoder)


def sync_mask(trainer, rank, world_size):
    """同步所有进程的mask"""
    if world_size <= 1:
        return
    
    mask_tensor = trainer.mask.cpu()
    dist.broadcast(mask_tensor, src=0)
    trainer.mask = mask_tensor.to(trainer.device)
    
    # 更新掩码字典
    if hasattr(trainer, '_convert_mask_to_dict'):
        trainer.mask_dict = trainer._convert_mask_to_dict()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# 辅助函数（确保这些函数在utils.main_utils中）
def apply_mask(mask, model, params, device):
    """应用稀疏性掩码到模型参数"""
    @torch.no_grad()
    def _apply_mask():
        state_dict = model.state_dict()
        i = 0
        for p in params:
            param = state_dict[p]
            count = param.numel()
            state_dict[p] *= mask[i:(i + count)].to(device).reshape(param.shape).float()
            i += count
        model.load_state_dict(state_dict)
    _apply_mask()


def get_pvec(model, params):
    """获取模型参数向量"""
    param_list = []
    for p in params:
        param = model.state_dict()[p]
        param_list.append(param.view(-1))
    return torch.cat(param_list)


def zero_grads(model):
    """清零模型梯度"""
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()


def compute_acc(model, test_loader, device):
    """计算模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return correct / total


def sync_weights(model, rank, world_size):
    """同步分布式训练的模型权重"""
    if world_size <= 1:
        return
    
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True