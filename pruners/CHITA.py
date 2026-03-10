from utils.main_utils import *
import time
import re
from utils.flops_utils import get_flops
from contextlib import nullcontext
from torch.utils.data import DataLoader

from CHITA_opt.L0_card_const import Active_refBBDCA_PP
from CHITA_opt.L0_card_const import SMSB_OneStep
from CHITA_opt.L0_card_const import OneS_SB
from CHITA_opt.L0_card_const import TriSB

import scipy.sparse.linalg
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt
import math

class CHITA:

    def __init__(self,model,params,prun_dataloader,
    ngrads,fisher_mini_bsz,criterion,lambda2,num_iterations,
    device,algo='SMSB'):
        '''
         This object changes the model. 
        After prune is called, the attribute results is filled with the following keys :
        'norm_w_wbar','sparsity','new_non_zeros','trace_C','trace_H',
        'gradient_norm','obj','prun_runtime','norm_w'
        '''
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
        print('gards shape:',grads.shape)
        print('bit:',w1.dtype)
        """
        """
        #小规模数据生成
        
        #一个小的数据集,更低稀疏度的表现
        #每一个rho的解的情况(支集)
        #支集, 取到1的速度, 两者的影响
        #更快的rho的增长, 但更精确的求解
        n = 100
        #w1 = np.random.rand(n)
        np.random.seed(42)
        
        w1 = np.random.uniform(-1, 1, n)#均匀分布
        #w1 = np.array([0.01,1])
        #grads = np.array([[1,10],[0.1,0]])
        #w1 = np.random.normal(loc=0, scale=1, size=n)#高斯分布
        #w1 = np.random.exponential(scale=1, size=n)#指数分布
        #w1 = np.random.poisson(lam=3, size=n).astype(np.float64)#泊松分布
        #w1 = np.random.standard_cauchy(size=n)#柯西分布
        #print('w1:',w1)
        #grads = np.random.rand(self.ngrads, n)
        grads = np.random.rand(n,n)
        
        #print('grads:',grads)
        self.grads = grads
        y = grads@w1
        #k = int((1-sparsity)*n)
        k = 2
        """

        print('Starting Optimization')
        start_algo = time.time()

        if self.algo == 'SMSB':

            print('sp:',sparsity,'k:',k)
            print('OneStep S-Sparse-Binary Start:')
            st = time.time()

            M = 256
            if M == 1:
                w_pruned, s, objs, sol_time, epsilon = OneS_SB(y, grads, w1, self.lambda2, k, init_tol=1e-3, rho_start1 = 1e-2, rho_start2 = 1, rho_ratio=2, rhoStartRatio=1e-2, ConvergenceRatio=0.99)
            else:
                w_pruned, s, obj,  BBL1K2sol_time, epsilon = SMSB_OneStep(M, y, grads, w1, self.lambda2, k, init_tol=1e-3,  rho_ratio=2, rhoStartRatio=1e-2, ConvergenceRatio=0.99, max_sub=1)

            ratio = math.floor(math.log10(num)) + 1
            w_pruned = self.normalize_to_8bit(w_pruned)/ratio
            One_Step_time = time.time() - st
            NonZero_Indices = w_pruned.nonzero()[0]
            r = y - grads@w_pruned
            f_original= 0.5*np.dot(r,r) + self.lambda2*np.dot(w_pruned-w1,w_pruned-w1)
            obj = [f_original]

            Ktest = len(NonZero_Indices)
            mtest = 0
            top_K_idx = np.argsort(-np.abs(w_pruned))[:Ktest]
            for i in range(Ktest-1):
                idx = top_K_idx[i]
                idx_next = top_K_idx[i+1]
                if abs(abs(w_pruned[idx_next]) - abs(w_pruned[idx])) > 0:
                    mtest += 1
            print('f:',f_original,'m:',M,'mtest:',mtest,'Nonzero:',len(NonZero_Indices),'s-bit:',s.dtype)
        elif self.algo == 'TriSB':

            print('sp:',sparsity,'k:',k)
            print('OneStep Tri-Sparse-Binary Start:')
            st = time.time()
            w_pruned, objs, sol_time, epsilon, iter_number = BBL1K2_OneStep(y, grads, w1, self.lambda2, k, init_tol=1e-3, rho_delta = 1e-2, scalar=1, rho_ratio=2, rho_start=1e-2, init_step=1)
    
            #w_pruned,  objs, sol_time, epsilon = TriSB(y, grads, w1, self.lambda2, k, init_tol=1e-3, rho_ratio=2, rhoStartRatio=1e-2, ConvergenceRatio=0.99,max_sub=1)
            """
            w_pruned = np.zeros_like(w1)
            top_K_idx = np.argsort(-np.abs(w1))[:k]
            w_pruned[top_K_idx] = np.sign(w1[top_K_idx])
            """
            One_Step_time = time.time() - st
            NonZero_Indices = w_pruned.nonzero()[0]
            r = y - grads@w_pruned
            f_original= 0.5*np.dot(r,r) + self.lambda2*np.dot(w_pruned-w1,w_pruned-w1)
            obj = [f_original]

            Ktest = len(NonZero_Indices)
            top_K_idx = np.argsort(-np.abs(w_pruned))[:Ktest]
            OneRes = np.sum(np.abs(w_pruned[top_K_idx])) - k
            print('f:',f_original,'Nonzero:',Ktest,'OneRes:',OneRes)
        if self.algo == 'Active_refBBDCA':
            print('sp:',sparsity,'K:',k)
            print('Active_refBBDCA(Only Sparse) Start:')
            time_start = time.time()
            DCAw_pruned, obj, r, DCAtotal_iter, DCAtot_time, epsilon = Active_refBBDCA_PP(y, grads, self.lambda2, w1, k, rho_delta=1e-2, init_tol = 1e-3, rho_start=1e-2, rho_ratio=2, init_step=1e-1, ArmijoM=10, act_max_itr=10, sea_max_itr=10, kmip=1.5)
            w_pruned = DCAw_pruned
            DCAtotal_time = time.time() - time_start
            DCANonZero_Indices = DCAw_pruned.nonzero()[0]
            r = y - grads@DCAw_pruned
            DCAf_original= 0.5*np.dot(r,r) + self.lambda2*np.dot(DCAw_pruned-w1,DCAw_pruned-w1)
            print('Nonzero-Idx:',DCANonZero_Indices)
            print('Active-ref-BB-DCA f:',DCAf_original,'time:',DCAtotal_time,'Non-Zero Num:',len(DCANonZero_Indices))


        end_algo = time.time()

        #set_pvec(w_pruned, self.model,self.params,self.device)

        #self.results['trace_C'] = (trace_C)
        #self.results['trace_H']=(trace_H)
        #self.results['gradient_norm']=(gradient_norm)
        self.results['norm_w_wbar']=(np.linalg.norm(w_pruned-w1,ord=2))
        self.results['sparsity']=(sparsity)
        new_nz = (w_pruned[w1 == 0] != 0).sum()
        self.results['new_non_zeros']=(new_nz)
        self.results['obj']=(obj)
        self.results['prun_runtime']=(end_algo - start_algo)
        self.results['norm_w']=(np.linalg.norm(w_pruned,ord=2))
        #self.results['test_acc']=(compute_acc(self.model,self.test_dataloader,self.device))

        new_mask = torch.from_numpy(w_pruned != 0)
        
        return w_pruned, new_mask