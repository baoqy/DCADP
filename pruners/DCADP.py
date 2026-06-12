from utils.main_utils import *
import time
import re
from utils.flops_utils import get_flops
from contextlib import nullcontext
from torch.utils.data import DataLoader

from DCADP_opt.L0_card_const import Active_refBBDCA_PP
from DCADP_opt.L0_card_const import SMSB_OneStep
from DCADP_opt.L0_card_const import OneS_SB
from DCADP_opt.L0_card_const import TriSB

import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
class DCADP:

    def __init__(self,model,params,prun_dataloader,ngrads,fisher_mini_bsz,criterion,lambda2,num_iterations,device,algo='SMSB',dis_num=0):
        self.model = model
        self.params = params 
        self.prun_dataloader = prun_dataloader
        self.criterion = criterion
        self.ngrads = ngrads
    
        self.lambda2 = lambda2*ngrads/2 #self.lambda2 is the lambda in the regression formulation
        self.num_iterations = num_iterations 
        self.device = device 

        self.fisher_mini_bsz = fisher_mini_bsz
        self.algo = algo
        self.grads = None
        self.results = dict()
        self.dis_num = dis_num

    def update_model(self,new_w):
        set_pvec(new_w, self.model,self.params,self.device)

    def compute_flops(self,input_res):
        self.model.eval()
        self.results['flops'] = get_flops(input_res,self.device,self.model)

    def reset_pruner(self):
        self.results = dict()
        self.grads=None
        
 
    def prune(self,mask,sparsity,grads=None):
        original_weight = get_pvec(self.model, self.params)
        if mask is None:
            mask = torch.ones_like(original_weight).cpu() != 0
        w1 = original_weight.to('cpu').numpy().astype(np.float64)
        d = len(w1)
        k = int((1-sparsity)*original_weight.numel())
        Total_number = original_weight.numel()
        zero_grads(self.model)
        self.model.eval()

        if grads is None and self.grads is None:
            ##Deactive syncing if runnining distribitued DP so that the other processes dont wait
            with self.model.no_sync() if isinstance(self.model,torch.nn.parallel.DistributedDataParallel) else nullcontext() as gs:
                #grads = torch.zeros((self.ngrads, d), device='cpu')
                grads = torch.zeros((self.ngrads, d), device='cpu', dtype=torch.float64)
                start_grad_comp = time.time()
                for i, batch in enumerate(self.prun_dataloader):
                    if i%100 ==0:
                        print('Computing gradients',i)
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    loss = self.criterion(self.model(x), y)
                    loss.backward()
                    grads[i] = get_gvec(self.model, self.params).to('cpu').double()
                    zero_grads(self.model)
                    #print('i:',i,'ngrads:',self.ngrads)
                    if (i + 1) % self.ngrads == 0:
                        break
                
            grads = grads.numpy()
            grads = grads.astype(np.float64)
            end_grad_comp = time.time()
            print('Grad computation took ',end_grad_comp - start_grad_comp)
        
        self.grads = grads
        w1 = w1.astype(self.grads.dtype)
        y=grads@w1
        
        """
        #小规模数据生成

        n = 100
        np.random.seed(42)
        w1 = np.random.uniform(-1, 1, n)
        grads = np.random.rand(n,n)

        self.grads = grads
        y = grads@w1
        k = 10
        """

        print('Starting Optimization')
        start_algo = time.time()
        print('sp:',sparsity,'k:',k)
        if self.algo == 'SMSB':

            print('sp:',sparsity,'k:',k,'dis num:',self.dis_num)
            print('OneStep S-Sparse-Binary Start')

            if self.dis_num == 1:
                w_pruned, s, obj, sol_time, epsilon = OneS_SB(y, grads, w1, self.lambda2, k, init_tol=1e-3, rho_start1 = 1e-2, rho_start2 = 1, rho_ratio=2, rhoStartRatio=1e-2, ConvergenceRatio=0.99)
            else:
                w_pruned, s, obj,  BBL1K2sol_time, epsilon = SMSB_OneStep(self.dis_num, y, grads, w1, self.lambda2, k, init_tol=1e-3,  rho_ratio=2, rhoStartRatio=1e-2, ConvergenceRatio=0.99, max_sub=1)
            
        elif self.algo == 'TriSB':

            print('OneStep Tri-Sparse-Binary Start')
            w_pruned,  objs, sol_time, epsilon = TriSB(y, grads, w1, self.lambda2, k, init_tol=1e-3, rho_ratio=2, rhoStartRatio=1e-2, ConvergenceRatio=0.99,max_sub=1)

        elif self.algo == 'Active_refBBDCA':

            print('Active_refBBDCA(Only Sparse) Start')
            w_pruned, obj, r, DCAtotal_iter, DCAtot_time, epsilon = Active_refBBDCA_PP(y, grads, self.lambda2, w1, k, rho_delta=1e-2, init_tol = 1e-3, rho_start=1e-2, rho_ratio=2, init_step=1e-1, ArmijoM=10, act_max_itr=10, sea_max_itr=10, kmip=1.5)

            
        end_algo = time.time()

        self.results['sparsity']=(sparsity)
        new_nz = (w_pruned[w1 == 0] != 0).sum()
        self.results['new_non_zeros']=(new_nz)
        self.results['obj']=(obj)
        self.results['prun_runtime']=(end_algo - start_algo)

        new_mask = torch.from_numpy(w_pruned != 0)
        
        return w_pruned, new_mask