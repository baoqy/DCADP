from errno import ENETUNREACH
import numpy as np
import numpy.linalg as la
import numba as nb
from time import time
from sklearn.utils import extmath
from collections import namedtuple
import warnings
from numba.core.errors import NumbaDeprecationWarning, \
    NumbaPendingDeprecationWarning, NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
from numba import prange
import math
from scipy.sparse.linalg import cg, LinearOperator
import cvxpy as cp
from models.resnet_cifar10 import resnet20
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import argparse
import os
from numba.typed import List
from torch.utils.data import DataLoader
from utils.lr_schedules import cosine_lr_restarts,mfac_lr_schedule
from models.mlpnet import MlpNet
from scipy.linalg import svd, qr
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='mlpnet')
parser.add_argument('--dset', type=str, default='mnist')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--exp_id',type=str,default='')

parser.add_argument('--train_batch_size', type=int, default=500)
parser.add_argument('--test_batch_size', type=int, default=500)

parser.add_argument('--fisher_subsample_size', type=int, nargs='+')
parser.add_argument('--fisher_mini_bsz', type=int, nargs='+')

parser.add_argument('--num_iterations', type=int, nargs='+')
parser.add_argument('--num_stages', type=int, nargs='+')
parser.add_argument('--seed', type=int, nargs='+')
parser.add_argument('--first_order_term', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--sparsity', type=float, nargs='+')
parser.add_argument('--base_level', type=float, default=0.1) ##In correspondance with sparsity
parser.add_argument('--dis_num',type=int, default=0)
parser.add_argument('--outer_base_level',type=float,default=0.5)
parser.add_argument('--l2', type=float, nargs='+')
parser.add_argument('--sparsity_schedule', type=str, nargs='+')
parser.add_argument('--algo', type=str, nargs='+')
parser.add_argument('--block_size', type=int, nargs='+') ##Set to -1 if algo does not use this

parser.add_argument('--weight_decay',type=float,default=0.00003751757813)
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--max_lr',type=float)
parser.add_argument('--min_lr',type=float)
parser.add_argument('--ft_min_lr',type=float,default=-1)
parser.add_argument('--ft_max_lr',type=float,default=-1)
parser.add_argument('--prune_every',type=int)
parser.add_argument('--nprune_epochs',type=int)
parser.add_argument('--nepochs',type=int)
parser.add_argument('--gamma_ft',type=float,default=-1)
parser.add_argument('--warm_up', type=int, default=0)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--first_epoch', type=int, default=0)
parser.add_argument('--schedule', type=str, default='cosine_lr_restarts')
parser.add_argument('--pretrained', type=lambda x: (str(x).lower() == 'true'), default=True)

parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')

args = parser.parse_args()
arch = args.arch
dset = args.dset
num_workers = args.num_workers
exp_name = args.exp_name
pretrained = args.pretrained
momentum = args.momentum
weight_decay = args.weight_decay


@nb.njit(cache=True)
def evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2):
    beta_sub1 = beta - beta_tilde1
    beta_sub2 = beta - beta_tilde2
    #return 0.5*(r@r) + lambda2*(beta_sub2@beta_sub2) + lambda1*(np.sum(np.abs(beta_sub1))) + alpha@beta
    return 0.5*(r@r) + lambda2*(beta_sub2@beta_sub2)


def skl_svd(X):
    return extmath.randomized_svd(X,n_components=1)[1][0]


def initial_active_set(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M=np.inf,buget=None,kimp=2.,act_itr=1):
    
    p = beta.shape[0]
    buget = p if buget is None else buget
    ksupp = int(np.max([np.min([kimp*k, buget, p]),k]))
    beta_tmp, r_tmp = np.copy(beta), np.copy(r)
    for i in range(act_itr):
        beta_tmp,r_tmp = hard_thresholding(y,X,beta_tmp,r_tmp,ksupp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M)
    active_set = set(np.where(beta_tmp)[0])    
    active_set = np.array(sorted(active_set),dtype=int)
    
    return active_set


#original objuec function
@nb.njit(cache=True)
def evaluate_original(r, w_t, w_bar, lambda2):

    return 0.5*r@r + lambda2*((w_t - w_bar)@(w_t - w_bar))


#penalty object function
@nb.njit(cache=True)
def evaluate_penaltyNewDCA(r, w_t, w_bar, lambda2, rho1, rho2, K, s_sum, top_K_idx):
    w_sub = w_t - w_bar
    vec = w_t[top_K_idx]
    L1norm = np.sum(np.abs(w_t))
    Knorm = np.sum(np.abs(vec))
    #print('f_original:',0.5*np.dot(r,r) + lambda2*np.dot(w_sub,w_sub),'rho1*:',rho1 * (L1norm - Knorm),'rho2/K*:',rho2/K * (K*scalar - Knorm))
    return 0.5*np.dot(r,r) + lambda2*np.dot(w_sub,w_sub) + rho1 * (L1norm - Knorm) + rho2/K * (s_sum - Knorm)

#update v, s in ADMM
@nb.njit(cache=True)
def prox_l1_linf(z, rho1, rho2, rho):
    """
    Solve:
        min_v rho1*||v||_1 + rho2*||v||_inf + (rho/2)*||v - z||_2^2

    Parameters
    ----------
    z : ndarray, shape (n,)
    rho1 : float >= 0
    rho2 : float >= 0
    rho  : float > 0     # quadratic penalty

    Returns
    -------
    v : ndarray, shape (n,)
        Exact minimizer
    """

    # proximal parameters
    lam1 = rho1 / rho
    lam2 = rho2 / rho

    # soft-threshold
    u = np.maximum(np.abs(z) - lam1, 0.0)

    """
    # trivial cases
    if u.max() == 0 or lam2 == 0:
        return np.sign(z) * u, -1
    """
    # find t
    u_sorted = np.sort(u)[::-1]
    cumsum = np.cumsum(u_sorted)

    t = 0.0
    for k in range(len(u_sorted)):
        t_candidate = (cumsum[k] - lam2) / (k + 1)
        if k == len(u_sorted) - 1 or t_candidate >= u_sorted[k + 1]:
            t = max(t_candidate, 0.0)
            break

    # clip
    v = np.sign(z) * np.minimum(u, t)
    return v, t


@nb.njit(cache=True)
def SM_proj_l1_linf(z, K, M, lengths,sub_rho1, sub_rho2, rho, offsets):


    z_abs = np.abs(z)

    top_K_idx = np.argpartition(-z_abs, K-1)[:K]

    top_K_idx = top_K_idx[np.argsort(-z_abs[top_K_idx])]

    z_new=np.zeros_like(z)
    s_arr=np.zeros(M)

    for i in range(M):

        start=offsets[i]
        end=offsets[i+1]

        idx_block=top_K_idx[start:end]

        z_piece=z_abs[idx_block]

        zProj,sProj=prox_l1_linf(
            z_piece,
            sub_rho1,
            sub_rho2/K*lengths[i],
            rho
        )

        s_arr[i]=sProj

        for j in range(end-start):
            idx=idx_block[j]
            z_new[idx]=np.sign(z[idx])*zProj[j]

    return z_new,s_arr,top_K_idx

#update w in ADMM
def solve_w_subproblem(b_part, X, w_bar, s_vec, u, v, lambda2, rho, w_init=None, tol=1e-8, maxiter=50):
    """
    Solve:
        (X^T X + lambda2 I)(w - w_bar) - s_vec + rho (w + u - v) = 0

    i.e.
        (X^T X + (lambda2 + rho) I) w
        = (X^T X + lambda2 I) w_bar + s_vec - rho (u - v)

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Data matrix (m << n), X^T X is NOT formed.
    w_bar : ndarray, shape (n,)
    s_vec : ndarray, shape (n,)
    u, v : ndarray, shape (n,)
        ADMM dual / auxiliary variables
    lambda2 : float
    rho : float
    w_init : ndarray, optional
        Warm start for CG (strongly recommended)
    tol : float
        CG tolerance
    maxiter : int
        Max CG iterations

    Returns
    -------
    w : ndarray, shape (n,)
        Solution of the w-subproblem
    """

    n = w_bar.shape[0]

    # Linear operator: A w = X^T (X w) + (2*lambda2 + rho) w
    def matvec(w):
        return X.T @ (X @ w) + (2*lambda2 + rho) * w

    A = LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=np.float64
    )

    # Right-hand side
    b = b_part  - rho * (u-v)
    """
    b = (
        X.T @ (X @ w_bar)
        + 2*lambda2 * w_bar
        + s_vec
        - rho * (u - v)
    )
    """
    # Conjugate Gradient solve
    w, info = cg(
        A,
        b,
        x0=w_init,
        tol=tol,
        maxiter=maxiter
    )
    """
    if info != 0:
        raise RuntimeError(f"CG did not converge, info = {info}")
    """
    #res = np.linalg.norm(A @ w - b)
    #print('CG res:',res)
    return w

#update v, subproblem of s=1, w = -1, 0, 1, ADMM
def shrink_clip(w, u, rho1, rho):
    """
    Solve:
        min_{||v||_inf <= 1} 
        rho1 * ||v||_1 + (rho/2) * ||v - (w + u)||^2

    Parameters
    ----------
    w_hat : np.ndarray
    u     : np.ndarray
    rho1  : float
    rho   : float

    Returns
    -------
    v : np.ndarray
        Optimal solution
    """
    a = w + u
    lam = rho1 / rho

    # soft-threshold
    v = np.sign(a) * np.maximum(np.abs(a) - lam, 0.0)

    # clip to [-1, 1]
    v = np.clip(v, -1.0, 1.0)

    return v

#subproblem of s = 1, w = -1, 0, 1, ADMM
def TriProblem_ADMMSubProblem(y, grads,  w_start, w_bar,  s_vec, lambda2,  ADMMrho, sub_rho1, sub_rho2, K,  tol = 1e-4, epsilon=1e-4, adaptive_rho=True):
    w = w_start.copy()
    v = w_start.copy()
    u = np.zeros_like(w)
    rho = ADMMrho#这个rho的选取确保不会过稀疏
    mu = 10
    max_iter = 500
    itr = 0
    f_old = 0
    while itr < max_iter:
        itr += 1
        w_old = w.copy()
        v_old = v.copy()

        w = solve_w_subproblem(X=grads, w_bar=w_bar, s_vec=s_vec, u=u, v=v_old, lambda2=lambda2, rho=rho, w_init=w_old, tol=1e-4,maxiter=50)
        #v, s = SM_proj_l1_linf(w + u, v_old, K, M, base_len, sub_rho1, sub_rho2, rho, lengths)
        v = shrink_clip(w, u, sub_rho1, rho)
        u = u + w - v
        r_norm = np.linalg.norm(w - v)
        d_norm = rho * np.linalg.norm(w - w_old)
        if adaptive_rho:
            if r_norm > mu * d_norm:
                rho *= 2
                u /= 2
            elif d_norm > mu * r_norm:
                rho /= 2
                u *= 2
        r_new = y - grads@v

        f_new = 0.5 * r_new @ r_new + lambda2 * np.dot(w-w_bar, w-w_bar) + sub_rho1 * np.sum(np.abs(v)) - np.dot(v, s_vec)
        #if r_norm < tol and d_norm < tol and abs(f_new-f_old) <= epsilon:
        if abs(f_new-f_old) <= epsilon:
            #print('ADMM Acc break')
            break
        f_old = f_new

    return v

#given S = 1, w = -1, 0, 1
def TriSB(y, grads, w_bar, lambda2, K, init_tol=1e-3, rho_ratio=2, rhoStartRatio=1e-2, ConvergenceRatio=0.99,max_sub=50):
    
    st = time()
    itr_rho = 1
    ArmijoM = 4
    objs = []
    epsilon = init_tol
    
    L = 1.05*(skl_svd(grads)**2 + lambda2*2)
    
    """
    len_tmp = int((len(w_bar) - K)/2) + K
    top_tmpK_idx = np.argsort(-np.abs(w_bar))[:len_tmp]
    top_K_idx = top_tmpK_idx[:K]
    w_rho_tmp = np.clip(w_bar, -1, 1)
    w_rho = np.zeros_like(w_bar)
    w_rho[top_tmpK_idx] = w_rho_tmp[top_tmpK_idx]
    r_rho = y - grads@w_rho
    abs_w = np.abs(w_rho)
    L1norm = np.sum(abs_w)
    Knorm = np.sum(abs_w[top_K_idx])

    f_original = 0.5*np.dot(r_rho,r_rho) + lambda2*np.dot(w_rho-w_bar,w_rho-w_bar)
    penaltySparse = L1norm - Knorm
    penaltyBinary = K - Knorm
    sub_rho1 = f_original / penaltySparse * rhoStartRatio
    sub_rho2 = f_original / penaltyBinary  * K * rhoStartRatio * 1000
    """
    top_K_idx = np.argsort(-np.abs(w_bar))[:K]
    """
    w_rho = np.zeros_like(w_bar)
    w_rho[top_K_idx] = np.sign(w_bar[top_K_idx])
    """
    w_rho = w_bar.copy()
    
    r_rho = y - grads@w_rho
    
    sub_rho1 = 100/K
    sub_rho2 = sub_rho1*K
    #sub_rho2 = 0
    #Notrho2Flag = True
    print('sub_rho1 start:',sub_rho1,'sub_rho2/K start:',sub_rho2/K)

    signSupp = np.sign(w_rho[top_K_idx])
    signSupp_old = signSupp.copy()
    f_old = 0
    while True:
        
        w_now = w_rho.copy()
        r_now = r_rho.copy()
        f_new = 0
        epsilon /= itr_rho
        epsilon = max(epsilon, 1e-5)
        itr_sub = 0
        st = time()
        alpha = sub_rho1 + sub_rho2/K
        itr_in = 1
        s_vec = np.zeros_like(w_bar)
        while True:
            itr_sub += 1

            grad_K2norm = np.zeros_like(w_now)
            grad_K2norm[top_K_idx] = np.sign(w_now[top_K_idx])
            s_vec =  alpha * grad_K2norm 

            #ADMM求解子问题
            #ADMMrho = sub_rho1 / max(np.min(np.abs(w_now[top_K_idx])) * 10, 1e-5)#这个ADMM里的rho的选取确保不会过稀疏
            #w_new, s_new = SM_ADMMSubProblem(y, grads, w_now, w_bar, s_now, s_vec, lambda2, ADMMrho, sub_rho1, sub_rho2, K, M, base_len, lengths, tol = 1e-4, epsilon=epsilon, adaptive_rho=True)
            ADMMrho = 1e-1
            w_new = TriProblem_ADMMSubProblem(y, grads,  w_now, w_bar, s_vec, lambda2,  ADMMrho, sub_rho1, sub_rho2, K, tol = 1e-4, epsilon=1e-4, adaptive_rho=True)
            
            top_K_idx_new = np.argpartition(np.abs(w_new), -K)[-K:]
            ADMMtime = time() - st
            r_new = y - grads@w_new
            f_new = evaluate_penaltyNewDCA(r_new, w_new, w_bar, lambda2, sub_rho1, sub_rho2, K, K, top_K_idx_new)
            print('itr_sub:',itr_sub,'ADMM time:',ADMMtime,'F:',f_new,'max:',np.max(np.abs(w_new)))
            top_K_idx = top_K_idx_new
            if abs(f_new - f_old) <= max(epsilon, 1e-4) or itr_sub >= max_sub:
    
                
                f_rho = f_new
                w_rho = w_new.copy()
                r_rho = r_new.copy()
          
                f_old = f_new
                break

            f_old = f_new
            w_now = w_new
 
            r_now = r_new
        
        abs_w = np.abs(w_rho)
        top_K_idx = np.argsort(-abs_w)[:K]
        test_w = np.abs(abs_w - 1)
        s_count = np.sum(test_w <= 1e-6)
        below_sCount = np.sum(test_w > 1e-6)
        
        number_nonzero = np.sum(abs_w > 0)
        L1norm = np.sum(np.abs(w_rho))
        Knorm = np.sum(abs_w[top_K_idx])

        f_original = 0.5 * (y-grads@w_rho) @ (y-grads@w_rho) + lambda2 * (w_rho-w_bar) @ (w_rho-w_bar)
        print('sub_rho1:',sub_rho1,'sub_rho2/K:',sub_rho2/K,'f_original:',f_original,'f_penalty:',f_rho,'Non-Zero:',number_nonzero)
        SparseRes = L1norm - Knorm
        BinaryRes = K - Knorm

        SparseRatio = 1 - SparseRes/Knorm
        BinaryRatio = 1 - BinaryRes/Knorm
        print('SparseRes:',SparseRes,'BinaryRes:',BinaryRes,'Sparse Ratio:',SparseRatio,'Binary Ratio:',BinaryRatio)
        print('1 Count:', s_count, 'below 1 count:', below_sCount)
    
        if SparseRes > 1:
            sub_rho1 *= rho_ratio
        #if BinaryRes > 1 and SparseRes <= 1:
        if BinaryRes > 1 :
            sub_rho2 *= rho_ratio
        if SparseRes <= 1 and BinaryRes <= 1:
            print('Cons Res Satisfy')
            w_new = np.zeros_like(w_bar)
            for i in range(K):
                idx = top_K_idx[i]
                w_new[idx] = np.sign(w_rho[idx])
            w_rho = w_new
            break
    
        itr_rho += 1
        epsilon = init_tol/itr_rho

    sol_time = time()-st

    return w_rho,  objs, sol_time, epsilon

#subproblem of SMSB, ADMM
def SM_ADMMSubProblem(arch, offsets, b_part1, max_sub, top_K_idx, y, grads,  w_start, v_start, w_bar, s_start, lambda2,  ADMMrho, sub_rho1, sub_rho2, K, M, base_len, lengths, tol = 1e-4, epsilon=1e-4, adaptive_rho=True):

    w = w_start.copy()
    v = v_start.copy()

    u = np.zeros_like(w)
    s = s_start.copy()
    rho = ADMMrho
    mu = 10
    if arch == 'mlpnet':
        max_iter = 200
    else:
        max_iter = 50 #MobileNetV1,Res20

    if arch == 'resnet50' or 'mobilenetv1':
        tol = 1e-3
        
    itr = 0
    itr_sub = 0
    BinaryRes_old = 0
    Knorm_old = 0
    while True:
        w = v.copy()
        alpha = sub_rho1 + sub_rho2/K
        grad_K2norm = np.zeros_like(w)
        grad_K2norm[top_K_idx] = np.sign(w[top_K_idx])
        s_vec =  alpha * grad_K2norm 
        b_part = b_part1 + s_vec
        itr_sub += 1
        itr = 0

        while itr < max_iter:
            itr += 1
            w_old = w.copy()
            v_old = v.copy()

            w = solve_w_subproblem(b_part, X=grads, w_bar=w_bar, s_vec=s_vec, u=u, v=v_old, lambda2=lambda2, rho=rho, w_init=w_old, tol=1e-4,maxiter=50)
            v, s, top_K_idx = SM_proj_l1_linf(w+u, K, M, lengths, sub_rho1, sub_rho2, rho, offsets)
            u = u + w - v
            r_norm = np.linalg.norm(w-v)
            d_norm = rho*np.linalg.norm(v-v_old)

            r_rel = r_norm / max(np.linalg.norm(w),np.linalg.norm(v),1.0)
            d_rel = d_norm / max(np.linalg.norm(w),np.linalg.norm(v),1.0)

            if adaptive_rho:
                if r_norm > mu * d_norm:
                    rho *= 2
                    u /= 2
                elif d_norm > mu * r_norm:
                    rho /= 2
                    u *= 2
            abs_v = np.abs(v)
            L1norm = np.sum(np.abs(v))
            Knorm = np.sum(abs_v[top_K_idx])
            SparseRes = L1norm - Knorm
            BinaryRes = np.sum(lengths * s) - Knorm
            if Knorm > 0:
                SparseRatio = 1 - SparseRes/Knorm
                BinaryRatio = 1 - BinaryRes/Knorm
            else:
                SparseRatio = 0
                BinaryRatio = 0
                
            print('itr:',itr,'d_rel:',d_rel,'r_rel',r_rel,'rho:',rho,'Knorm:',Knorm)
            print('SparseRes:',SparseRes,'SparseRatio:',SparseRatio,'BinaryRes:',BinaryRes,'BinaryRatio:',BinaryRatio)

            #print('SparseRes_w:',SparseRes_w,'SparseRatio_w:',SparseRatio_w,'BinaryRes_w:',BinaryRes_w,'BinaryRatio_w:',BinaryRatio_w)
            #if  SparseRatio > 0.999 and BinaryRatio > 0.99 and r_rel < 1e-1 and d_rel < 1e-1:
            if  SparseRatio > 0.999 and BinaryRatio > 0.99 and abs(BinaryRes-BinaryRes_old) <= max(BinaryRes, BinaryRes_old)*1e-2 and abs(Knorm-Knorm_old) <= max(Knorm, Knorm_old)*1e-2 and itr > 1:
                break
            
            if r_rel < tol and d_rel < tol:
                break
            BinaryRes_old = BinaryRes
            Knorm_old = Knorm
        if itr_sub >= max_sub:
            break
    #print('itr:',itr,'d_rel:',d_rel,'r_rel',r_rel,'rho:',rho,'Knorm:',Knorm)
    print('SparseRes:',SparseRes,'SparseRatio:',SparseRatio,'BinaryRes:',BinaryRes,'BinaryRatio:',BinaryRatio)
    return w, v, s, SparseRatio, BinaryRatio, SparseRes, BinaryRes, top_K_idx

#only one s value, ADMM subproblem
def ADMMSubProblem(b_part, y, grads,  w_start, w_bar, s_start, s_vec, lambda2,  ADMMrho, sub_rho1, sub_rho2, tol = 1e-4, epsilon=1e-4, adaptive_rho=True):
    w = w_start.copy()
    v = np.zeros_like(w)
    u = np.zeros_like(w)
    s = s_start
    rho = ADMMrho#这个rho的选取确保不会过稀疏
    mu = 10
    max_iter = 500
    itr = 0

    while itr < max_iter:
        itr += 1
        w_old = w.copy()
        v_old = v.copy()

        w = solve_w_subproblem(b_part, X=grads, w_bar=w_bar, s_vec=s_vec, u=u, v=v, lambda2=lambda2, rho=rho, w_init=w_old, tol=1e-4,maxiter=50)

        v, s = prox_l1_linf(w + u, sub_rho1, sub_rho2, rho)
        u = u + w - v
        r_norm = np.linalg.norm(w - v)
        d_norm = rho * np.linalg.norm(w - w_old)
        if adaptive_rho:
            if r_norm > mu * d_norm:
                rho *= 2
                u /= 2
            elif d_norm > mu * r_norm:
                rho /= 2
                u *= 2
        #r_new = y - grads@w
        #f_new = 0.5 * r_new @ r_new + lambda2 * np.dot(w-w_bar, w-w_bar) + sub_rho1 * np.sum(np.abs(v)) + sub_rho2 * s  - np.dot(w, s_vec)
        if r_norm < tol and d_norm < tol:
            break
 
    return w, s

@nb.njit(cache=True)
def NewCompute_nbar(what_abs, shat, sort_idx):
    n_bar = -1
    sum_ = 0
    s = 0
    n = len(what_abs)
    for i in range(1, n+1):
        val = what_abs[sort_idx[i-1]]
        sum_ += val
        lambda_bar = (sum_ - i*shat) / (i+1)
        s =  shat  + lambda_bar

        # 检查跳出条件
        if i < n:
            next_val = what_abs[sort_idx[i]]
            if next_val < s:
                n_bar = i
                break
        else:
            n_bar = n
            break

    return n_bar, s

#||w||_{inf} <= s
@nb.njit(cache=True)
def sLinf_Proj(what, shat, K, sort_idx=None):
    what_abs = np.abs(what)
    Linfnorm = np.max(what_abs)
    if sort_idx is None:
        sort_idx = np.argsort(-what_abs)
    
    if Linfnorm <= shat:
        return what, shat, sort_idx[:K]
    else:
        n_bar, s = NewCompute_nbar(what_abs, shat, sort_idx)
        w = what.copy()
        nbar_idx = sort_idx[:n_bar]
        w[nbar_idx] = np.sign(what[nbar_idx])*s
        return w, s, sort_idx[:K]

@nb.njit(cache=True)
def SM_LinfProj(what, shat, K, M, base_len):
    what_abs = np.abs(what)
    top_all_idx = np.argsort(-what_abs)
    top_K_idx = top_all_idx[:K]
    sort_what_abs = what_abs[top_all_idx]#这里sort_what_abs里是绝对值且从大到小
    
    # 用 numba.typed.List 替代普通 list
    Pieces = List()
    
    #前面的M-1段都在前K个里，每段长度为base_len
    start = 0
    if M > 1:
        for i in range(M-1):
            end = start + base_len
            Pieces.append(sort_what_abs[start:end])#每个Piece里都是绝对值，从大到小
            start = end
    
        #进行投影时为了保序，将其余 K - (M-1)*base_len + n-K个全部放入最后一段
        #Pieces.append(sort_what_abs[(M-1)*base_len:])
        Pieces.append(sort_what_abs[(M-1)*base_len:K])
    else:
        Pieces.append(sort_what_abs)

    # 预分配 w 数组
    w_new = np.zeros_like(what)
    s_arr = np.zeros_like(shat)
    
    ptr = 0
    for i in range(M):
        what_piece = Pieces[i]#每个Piece里都是绝对值，从大到小
        shat_element = shat[i]
        wProj, sProj, _ = sLinf_Proj(what_piece, shat_element, len(what_piece))#每个sProj里都是绝对值，从大到小

        s_arr[i] = sProj
        for j in range(len(wProj)):
            idx = top_all_idx[ptr]#对应在what中的索引为idx
            w_new[idx] = np.sign(what[idx]) * wProj[j]
            ptr += 1
    
    return w_new, s_arr, top_K_idx

#M value s, DCADP problem
def SMSB(arch, M, y, grads, w_bar, lambda2, K, init_tol=1e-3, rho_start1 = 1e-2, rho_start2 = 1, rho_ratio=2, SparseRhoRatio=1e-4, BinaryRhoRatio=1e-2,ConvergenceRatio=0.99,max_sub=5):
    
    st = time()
    itr_rho = 1
    ConverFlag = False
    objs = []
    epsilon = init_tol
    
    base_len = K // M
    remainder = K % M

    lengths = np.full(M, base_len, dtype=np.int64)

    if remainder > 0:
        lengths[-remainder:] += 1
    print('base len:',base_len,'final block:',lengths[-1])
    if arch == 'mobilenetv1' or arch == 'resnet50':
        Kexp = K#MobileNetV1,Res50
    else:
        #Kexp = min(int(K*1.5), int(len(w_bar)*0.9))#Res20,MLPNet
        Kexp = K
    top_Kexp_idx = np.argpartition(np.abs(w_bar), -Kexp)[-Kexp:]
    top_K_idx = np.argpartition(np.abs(w_bar), -Kexp)[-K:]
    top_K_idx = top_K_idx[np.argsort(-np.abs(w_bar[top_K_idx]))]
    
    idx_arr = []
    start = 0
    
    for i in range(M):
        length = lengths[i]
        end = start + length
        idx_block = top_K_idx[start:end]
        idx_arr.append(idx_block)
        start += length
    s_rho = np.zeros(M)
    block = 0
    for i in range(M):
        idx = top_K_idx[block]
        s_rho[i] = abs(w_bar[idx])
        block += lengths[i]

    w_rho = np.zeros_like(w_bar)
    w_rho[top_Kexp_idx] = w_bar[top_Kexp_idx]

    abs_w = np.abs(w_rho)
    L1norm = np.sum(abs_w)
    Knorm = np.sum(abs_w[top_K_idx])

    SparseRes = L1norm - Knorm
    BinaryRes = np.sum(lengths * s_rho) - Knorm
    r_rho = y - grads@w_rho
    f_original = 0.5*np.dot(r_rho,r_rho) + lambda2*np.dot(w_rho-w_bar,w_rho-w_bar)
    print('arch:',arch)
    if arch == 'mobilenetv1' or arch == 'resnet50':
        sub_rho1 = 1e-5
        sub_rho2 = 1
    else:
        #MLPNet:SparseRhoRatio=1e-2,BinaryRhoRatio=1e-1
        if SparseRes > 0:
            sub_rho1 = f_original/SparseRes * SparseRhoRatio 
        else:
            sub_rho1 = 1e-5
        if BinaryRes > 0:
            sub_rho2 = f_original/BinaryRes * BinaryRhoRatio * K
        else:
            sub_rho2 = 1
        
    print('sub_rho1 start:',sub_rho1,'sub_rho2 start/K:',sub_rho2/K,'num of w:',len(w_bar),'f_ori:',f_original,'Convergence Ratio:',ConvergenceRatio)
    print('SparseRes:',SparseRes,'BinaryRes:',BinaryRes)
    F_old = 0
    f_ori_old = 0
    
    w_rho = np.zeros_like(w_bar)
    w_rho[top_Kexp_idx] = w_bar[top_Kexp_idx]
    w_app = w_rho.copy()

    b_part1 = grads.T @ (grads @ w_bar) + 2*lambda2 * w_bar
    offsets=np.zeros(M+1,dtype=np.int64)
    for i in range(M):
        offsets[i+1]=offsets[i]+lengths[i]

    while True:
        
        w_now = w_rho.copy()
        s_now = s_rho.copy()
        
        ADMMrho = sub_rho1 / max(np.min(np.abs(w_now[top_K_idx])) * 10, 1e-5)
        w_app, w_rho, s_rho, SparseRatio, BinaryRatio, SparseRes, BinaryRes,top_K_idx = SM_ADMMSubProblem(arch, offsets, b_part1, max_sub, top_K_idx, y, grads, w_app, w_now, w_bar, s_now, lambda2, ADMMrho, sub_rho1, sub_rho2, K, M, base_len, lengths, tol = 1e-3, epsilon=1e-3, adaptive_rho=True)

        idx_arr = []
        start = 0
        for i in range(M):
            length = lengths[i]
            end = start + length
            idx_block = top_K_idx[start:end]
            idx_arr.append(idx_block)
            start += length

        sub_rho1 *= rho_ratio
        sub_rho2 *= rho_ratio
        
        if arch == 'mobilenetv1' or arch == 'resnet50':
            if SparseRatio > ConvergenceRatio and BinaryRatio > ConvergenceRatio:
                ConverFlag = True
        else:
            if SparseRes < 1e-1 and BinaryRes < 1e-1:
                ConverFlag = True
                
        if ConverFlag:
            print('Cons Res Satisfy')
            w_sign = np.zeros_like(w_bar)
            w_sign[top_K_idx] = np.sign(w_rho[top_K_idx])

            s_final = solve_sVec_CG(b_part1, grads,w_sign,w_bar,idx_arr,lambda2,a0=s_rho,rtol=1e-5,atol=1e-8,maxiter=100)
            w_new = w_rho.copy()
            ptr = 0
            for j in range(len(lengths)):
                s = s_final[j]
                L = lengths[j]
                for i in range(L):
                    idx = top_K_idx[ptr]
                    w_new[idx] = np.sign(w_rho[idx]) * s
                    ptr += 1
            """
            r_rho = y - grads@w_rho
            f_ori_rho = 0.5*(r_rho@r_rho) + lambda2*np.dot(w_rho-w_bar, w_rho-w_bar)
            
            r_new = y - grads@w_new
            f_ori_new = 0.5*(r_new@r_new) + lambda2*np.dot(w_new-w_bar, w_new-w_bar)
            print('f_rho final:',f_ori_rho,'unique value:',len(np.unique(np.abs(w_rho))))
            print('f_new final:',f_ori_new,'unique value:',len(np.unique(np.abs(w_new))))
            print('||w_rho - w_new||_1:',np.sum(np.abs(w_rho-w_new)),'||w_rho - w_new||_inf:',np.max(np.abs(w_rho-w_new)),'||w_rho - w_new||^2:',np.dot(w_rho-w_new,w_rho-w_new))

            cnt = np.sum(np.sign(w_rho) != np.sign(w_new))
            print('diff sign count:',cnt)
            print('||s_sol - s_smsb||_inf:',np.max(np.abs(s_final-s_rho)),'||s_sol-s_smsb||_1:',np.sum(np.abs(s_final-s_rho)))
            print('s_final < 0 count:',np.sum(s_final<0),'s_rho < 0 count:',np.sum(s_rho < 0))
            """
            break
            
    sol_time = time()-st
    print('sol time:',sol_time)
    return w_new, s_final, objs, sol_time, epsilon

def solve_sVec_CG(qw_bar, X,w,w_bar,idx_arr,lambda2,a0=None,rtol=1e-8,atol=1e-12,maxiter=100):


    M = len(idx_arr)
    n = len(w_bar)

    def expand(a):

        z = np.zeros(n)

        for i, idx in enumerate(idx_arr):
            z[idx] = w[idx] * a[i]

        return z

    def reduce(u):

        out = np.zeros(M)

        for i, idx in enumerate(idx_arr):
            out[i] = np.dot(w[idx], u[idx])

        return out


    # --------------------------
    # H*v=W^T Q W v
    # --------------------------
    def Hv(v):

        z = expand(v)

        qz = (
            X.T @ (X @ z)
            +2*lambda2*z
        )

        return reduce(qz)


    H = LinearOperator(
        (M,M),
        matvec=Hv,
        dtype=np.float64
    )


    # --------------------------
    # b=W^T Q w_bar
    # --------------------------
    #qw_bar = (X.T @ (X @ w_bar) +2*lambda2*w_bar)

    b = reduce(qw_bar)


    # --------------------------
    # CG solve
    # --------------------------
    a, info = cg(H,b,x0=a0,rtol=rtol,atol=atol,maxiter=maxiter)


    return a


#Only one s value DCADP problem
def OneS_SB(y, grads, w_bar, lambda2, K, init_tol=1e-3, rho_start1 = 1e-2, rho_start2 = 1, rho_ratio=2, rhoStartRatio=1e-2, ConvergenceRatio=0.99, max_sub=1):
    
    st = time()
    itr_rho = 1
    objs = []
    epsilon = init_tol

    L = 1.05*(skl_svd(grads)**2 + lambda2*2)
    top_all_idx = np.argsort(np.abs(w_bar))[::-1]
    top_K_idx = top_all_idx[:K]
    u = np.zeros_like(w_bar)
    u[top_K_idx] = np.sign(w_bar[top_K_idx])
    A_u = grads.T @ (grads @ u) + lambda2 * u    # 等于 (X^T X + lambda I) * u
    A_w = grads.T @ (grads @ w_bar) + lambda2 * w_bar
    alpha = u.dot(A_u)    # u^T A u
    beta  = u.dot(A_w)    # u^T A wbar
    s_rho = beta / alpha
    w_rho = np.zeros_like(w_bar)
    w_rho[top_K_idx] = w_bar[top_K_idx]
    r_rho = y - grads@w_rho


    abs_w = np.abs(w_bar)
    L1norm = np.sum(abs_w)
    Knorm = np.sum(abs_w[top_K_idx])

    f_original = 0.5*np.dot(r_rho,r_rho) + lambda2*np.dot(w_rho-w_bar,w_rho-w_bar)

    penaltySparse = L1norm - Knorm
    penaltyBinary = K*s_rho - Knorm
    sub_rho1 = f_original / penaltySparse * rhoStartRatio
    sub_rho2 = f_original / penaltyBinary  * K * rhoStartRatio
    print('s_start:',s_rho,'sub_rho1 start:',sub_rho1,'sub_rho2/K start:',sub_rho2/K)
    
    #top_K_idx_old = top_K_idx.copy()
    signSupp = np.sign(w_rho[top_K_idx])
    signSupp_old = signSupp.copy()
    f_old = 0
    mu = 1e-12
    b_part1 = grads.T @ (grads @ w_bar) + 2*lambda2 * w_bar
    while True:
        
        w_now = w_rho.copy()
        s_now = s_rho
        r_now = r_rho.copy()

        f_new = 0
        epsilon /= itr_rho
        epsilon = max(epsilon, 1e-5)
        itr_sub = 0
        alpha = (sub_rho1 + sub_rho2/K)

        s_vec = np.zeros_like(w_bar)
        while True:
            itr_sub += 1

            grad_K2norm = np.zeros_like(w_now)
            grad_K2norm[top_K_idx] = np.sign(w_now[top_K_idx])
            s_vec =  alpha * grad_K2norm 
            ADMMrho = sub_rho1 / np.min(np.abs(w_now[top_K_idx])) * 10
            b_part = b_part1 + s_vec

            w_new, s_new = ADMMSubProblem(b_part, y, grads, w_now, w_bar, s_now, s_vec, lambda2, ADMMrho, sub_rho1, sub_rho2, tol = 1e-6, epsilon=epsilon, adaptive_rho=True)
            
            if itr_sub >= max_sub:
                w_rho = w_new.copy()
                s_rho = s_new
                break
            
            w_now = w_new
            s_now = s_new
   
        
        abs_w = np.abs(w_rho) 

        top_K_idx = np.argpartition(abs_w, -K)[-K:]
        L1norm = np.sum(np.abs(w_rho))
        Knorm = np.sum(abs_w[top_K_idx])

        SparseRatio = 1 - (L1norm - Knorm)/Knorm
        BinaryRatio = 1 - (K*s_rho - Knorm)/Knorm

        if SparseRatio > ConvergenceRatio and BinaryRatio > ConvergenceRatio:
            print('Constrain Nearly Satisfy') 
            u = np.zeros_like(w_rho)

            u[top_K_idx] = np.sign(w_rho[top_K_idx])
            A_u = grads.T @ (grads @ u) + lambda2 * u    # (X^T X + lambda I) * u
            A_w = grads.T @ (grads @ w_bar) + lambda2 * w_bar
            alpha = u.dot(A_u)    # u^T A u
            beta  = u.dot(A_w)    # u^T A wbar
            s_rho = beta / alpha
            w_rho = u * s_rho
            break
        elif SparseRatio > ConvergenceRatio:
            sub_rho2 *= rho_ratio
        elif BinaryRatio > ConvergenceRatio:
            sub_rho1 *= rho_ratio
        else:
            sub_rho1 *= rho_ratio
            sub_rho2 *= rho_ratio    

        itr_rho += 1
        epsilon = init_tol/itr_rho
    sol_time = time()-st

    return w_rho, s_rho, objs, sol_time, epsilon


#update of w in DCA
@nb.njit(cache=True)
def DCA_update(w, r, w_bar, rho, grads, step, lambda2, K):
    # Calculate the w_t in DCA
    w_t = np.zeros_like(w_bar)
    lambda1 = 1/(2*step)

    grad1 = grads.T@r + 2*(-lambda2 + lambda1) * (w-w_bar)
    #grad1 = grads.T@r + (-2*lambda2 + lambda1) * (w-w_bar)
 
    top_K_idx = np.argpartition(np.abs(w), -K)[-K:]
    grad2 = np.zeros_like(w)
    for idx in top_K_idx:
        grad2[idx] = rho*np.sign(w[idx])
    
    s = grad1 + grad2
    
    # Using the Soft Threshold to solve the subproblem
    for i in range(w_bar.shape[0]):
        # Simplified conditional logic using a single computation
        term = w_bar[i] + s[i] * step
        if term >= rho * step:
            w_t[i] = w_bar[i] + (s[i] - rho) * step
        elif term <= -rho * step:
            w_t[i] = w_bar[i] + (s[i] + rho) * step
        else:
            w_t[i] = 0

    return w_t


#init Active Set
def DCAinitial_active_set(y,grads,w_bar,r,k,step,lambda2,sub_rho,M=np.inf,kimp=1.5,act_itr=1, s= np.inf):
    
    p = w_bar.shape[0]
    #将支集容量拓宽至稀疏度的kimp倍
    ksupp = int(np.max([np.min([kimp*k, p]),k]))
    if ksupp == p:
        ksupp = int(p-1)
    w_tmp, r_tmp = np.copy(w_bar), np.copy(r)
    for i in range(act_itr):
        w_tmp = DCA_update(w_tmp, r_tmp, w_bar, sub_rho, grads, step, lambda2, ksupp)
    active_set = set(np.where(w_tmp)[0])    
    active_set = np.array(sorted(active_set),dtype=int)
    
    return active_set

#Active Set + BB-Step DCA
def Active_BBDCA_PP(y, grads, lambda2, w_bar,  K, rho_delta, init_tol = 1e-6, rho_start=1e-2, rho_ratio=2, init_step=1, ArmijoM=10, act_max_itr=5, sea_max_itr=10):
    nnz_idx = np.where(np.linalg.norm(grads, axis=0)**2)[0]
    w_new = np.zeros_like(w_bar)
    if len(nnz_idx) > K:
        w, f, r, act_cur_itr, tot_time, total_iter, epsilon = Active_BBDCA(y, grads[:,nnz_idx], lambda2, w_bar[nnz_idx],  K, rho_delta, init_tol = init_tol, rho_start=rho_start, rho_ratio=rho_ratio, init_step=init_step, ArmijoM=ArmijoM, act_max_itr=act_max_itr, sea_max_itr=sea_max_itr)
        w_new[nnz_idx] = np.copy(w)
    else:
        w_new[nnz_idx] = np.copy(w_bar[nnz_idx])
        f = 0
        r = np.zeros_like(y)
        act_cur_itr=0
        tot_time=0
        total_iter = 0
    #print('f:',f,'iter_number:',total_iter)
    return w_new, f, r, total_iter, tot_time, epsilon

#f_ref-BBstep-Armijo
def ref_BBDCA(y, grads, lambda2, w_bar,  K, rho_delta, init_tol = 1e-6, rho_start=1e-8, rho_ratio=2, init_step=1, ArmijoM=10):
    st = time()
    lambda1_tmp = 1.05*(skl_svd(grads)**2+lambda2*2)

    init_step = 1/(lambda1_tmp*100)
    iter_rho = 1
    iter_number = 0

    sub_rho = 1e-5
    objs = []
    L = ArmijoM
    epsilon = init_tol
    f_now = np.inf
    Kexp = min(int(K*2), int(len(w_bar)*0.9))
    top_Kexp_idx = np.argpartition(np.abs(w_bar), -Kexp)[-Kexp:]
    top_K_idx = np.argpartition(np.abs(w_bar), -K)[-K:]
    w_now = np.zeros_like(w_bar)
    w_now[top_Kexp_idx] = w_bar[top_Kexp_idx]
    SparseRes = np.sum(np.abs(w_now)) - np.sum(np.abs(w_now[top_K_idx]))
    r_now = y - grads@w_now
    f_ori = 0.5*r_now@r_now + lambda2*np.dot(w_now-w_bar,w_now-w_bar)
    sub_rho = f_ori/SparseRes * 1e-2
    if sub_rho == 0:
        sub_rho = 1e-2
    print('sub_rho:',sub_rho,'f_ori:',f_ori,'SparseRes:',SparseRes,'||w_now-w_bar||_:',np.sum(np.abs(w_now-w_bar)))
    while True:
        #w_rho = w_now.copy()
        l = 0
        itr_in = 0
        
        top_K_idx = np.argpartition(np.abs(w_now), -K)[-K:]
        penalty_norm = np.sign(w_now)
        penalty_norm[top_K_idx] = 0
        g_now = -grads.T @ r_now + 2 * lambda2 * (w_now - w_bar)
        g_penaltynow = g_now + sub_rho * penalty_norm
        

        rhoFlag = False
        while True:

            if itr_in != 0:
                M = min(itr_in, ArmijoM)
                f_ref = np.max(objs[-M:])
                
            if itr_in == 0:
                f_ref = np.inf
                step = init_step
                #step = 0.1
            else:
                
                top_K_idx = np.argpartition(np.abs(w_now), -K)[-K:]
                penalty_norm = np.sign(w_now)
                penalty_norm[top_K_idx] = 0
                g_now = -grads.T @ r_now + 2 * lambda2 * (w_now - w_bar)
                g_penaltynow = g_now + sub_rho * penalty_norm
                sBB = w_now - w_old
                yBB = g_penaltynow - g_penaltyold
                sy = np.dot(sBB, yBB)
                
                if sy <= 0 or np.dot(sBB, sBB) == 0:
                    step = 0.1
                else:
                    if sy <= 0:
                        print('sy:',sy)
                        step = 0.1
                    elif iter_number % 2 == 0:
<<<<<<< HEAD
                    #elif itr_in % 2 == 0:
=======
>>>>>>> 601517b13a297935f9cc63bd6cdcf23cb62481d1
                        step = sy /np.dot(yBB, yBB)
                    else:
                        step = np.dot(sBB,sBB)/sy
                #step = max(1e-3, step)

            while True:
                w_trial = DCA_update(w_now, r_now, w_bar, sub_rho, grads, step, lambda2, K)
                
                if iter_number == 0:
                    w_temp = np.zeros_like(w_trial)
                    top_K_idx = np.argpartition(np.abs(w_trial), -K)[-K:]
                    w_temp[top_K_idx] = w_trial[top_K_idx]
                    w_trial = w_temp
                
                d = w_trial - w_now
                r_trial = y-grads@w_trial
                top_K_idx = np.argpartition(np.abs(w_trial), -K)[-K:]
                Knorm = np.sum(np.abs(w_trial[top_K_idx]))
                SparseRes = np.sum(np.abs(w_trial)) - Knorm
                f_ori = evaluate_original(r_trial, w_trial, w_bar, lambda2)
                f_trial =  f_ori + sub_rho * SparseRes
                
                iter_number += 1

                if f_trial <= f_ref + 1e-4 * np.dot(g_penaltynow, d):
                    itr_in += 1
                    objs.append(f_trial)

                    break
                else:
                    step *= 0.5

            if np.dot(w_trial-w_now, w_trial-w_now) <= epsilon and abs(f_trial - f_now) <= min(1e-2, 1e-3*abs(f_now)):
                rhoFlag = True

            f_now = f_trial
            w_old = w_now


            w_now = w_trial
            r_now = r_trial
            g_penaltyold = g_penaltynow
            number_nonzero = np.count_nonzero(w_trial)

            if rhoFlag:
                break
            
        if number_nonzero <= K:
            break
        else:
            sub_rho *= rho_ratio
            iter_rho += 1
            epsilon = init_tol/iter_rho
            w_now = w_trial.copy()
            r_now = y - grads@w_now
    
    sol_time = time()-st

    return w_trial, r_trial, iter_number, sol_time, sub_rho, epsilon

def solve_subset_zero_cg(
        X,
        idx,
        w_bar,
        lambda2,
        w0=None,
        rtol=1e-8,
        atol=1e-12,
        maxiter=100):

    """
    Solve:

    min 0.5*(w-w_bar)^T(X.T@X+2*lambda2*I)(w-w_bar)

    subject to:
        w[i]=0, i not in idx

    """

    idx = np.asarray(idx, dtype=np.int64)

    n = len(w_bar)
    K = len(idx)

    # -----------------------------
    # RHS: (Q @ w_bar)_S
    # -----------------------------
    
    Xw = X @ w_bar

    Qw_bar = X.T @ Xw + 2 * lambda2 * w_bar

    b = Qw_bar[idx]


    # -----------------------------
    # Linear operator:
    # (X_S.T X_S + 2lambda I)v
    # -----------------------------

    def matvec(v):

        tmp = X[:, idx] @ v

        out = X[:, idx].T @ tmp

        out += 2 * lambda2 * v

        return out


    A = LinearOperator(
        shape=(K, K),
        matvec=matvec,
        dtype=w_bar.dtype
    )


    # -----------------------------
    # initial point
    # -----------------------------

    if w0 is None:
        x0 = None
    else:
        x0 = w0[idx]


    # -----------------------------
    # CG
    # -----------------------------

    wS, info = cg(
        A,
        b,
        x0=x0,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter
    )


    # -----------------------------
    # recover full vector
    # -----------------------------

    w = w0.copy()

    w[idx] = wS


    return w

#Active Set+BB+ref DCA
def Active_refBBDCA(y, grads, lambda2, w_bar,  K, rho_delta, init_tol = 1e-6, rho_start=1e-2, rho_ratio=2, init_step=1, ArmijoM=10, act_max_itr=5, sea_max_itr=10, kmip=1.5):
    st = time()
    p = w_bar.shape[0]
    lambda1 = 0.5*1.05*(skl_svd(grads)**2+lambda2*2)
    if init_step is None:
        step = None
    else:
        step = init_step
    r = y - grads@w_bar
    act_cur_itr = 0
    total_iter = 0
    sub_rho = rho_start
    w = w_bar
    active_set = DCAinitial_active_set(y, grads, w_bar,r,K,1/(2*lambda1),lambda2,sub_rho=rho_start,M=np.inf,kimp=kmip,act_itr=1)

    while act_cur_itr < act_max_itr:
        grads_act = grads[:,active_set]
        w_act = w[active_set]
        step_act = step
        L_act = 0.5*1.05*(skl_svd(grads_act)**2+lambda2*2)
        w_act, r_act, iter_act, time_act, sub_rho, epsilon = ref_BBDCA(y, grads_act, lambda2, w_act, K, rho_delta, init_tol = init_tol, rho_start=rho_start, rho_ratio=rho_ratio, init_step=step_act, ArmijoM=ArmijoM)
        total_iter += iter_act
        
        step_search = 0.5*1/(2*L_act)
        w = np.zeros(p)
        w[active_set] = w_act
        r = y - grads@w
        f = evaluate_original(r, w, w_bar, lambda2)
        active_set_set = set(active_set)
        search_flag = False
        search_cur_itr = 0
        outliers = set()
        w_update,r_update = w,r
    
        while search_cur_itr < sea_max_itr:
            w_tmp = DCA_update(w, r, w_bar, sub_rho, grads, step_search, lambda2, K)
            r_tmp = y - grads@w_tmp
            total_iter += 1
            f_new = evaluate_original(r_tmp, w, w_bar, lambda2)    
            outliers = set(np.where(w_tmp)[0]) - active_set_set
            search_cur_itr += 1
            #print(f_new,f,len(outliers),search_cur_itr)
            if len(outliers) >= 1 and f_new < f and np.count_nonzero(w_tmp) <= K:
                search_flag = True
                w_update = w_tmp
                r_update = r_tmp
            elif f_new >= f:
                w = w_update 
                r = r_update
                break
            step_search /= 2
            
        if not search_flag:
            break
        active_set = np.array(sorted(active_set_set | outliers))
        act_cur_itr += 1
        
    f = evaluate_original(r, w, w_bar, lambda2)
    tot_time = time()-st
    
    return w, f, r, act_cur_itr, tot_time, total_iter,epsilon

#Active Set+BB+ref DCA
def Active_refBBDCA_PP(y, grads, lambda2, w_bar,  K, rho_delta, init_tol = 1e-6, rho_start=1e-2, rho_ratio=2, init_step=1, ArmijoM=10, act_max_itr=5, sea_max_itr=10, kmip=1.5):
    nnz_idx = np.where(np.linalg.norm(grads, axis=0)**2)[0]
    w_new = np.zeros_like(w_bar)
    if len(nnz_idx) > K:
        
        w, f, r, act_cur_itr, tot_time, total_iter, epsilon = Active_refBBDCA(y, grads[:,nnz_idx], lambda2, w_bar[nnz_idx],  K, rho_delta, init_tol = init_tol, rho_start=rho_start, rho_ratio=rho_ratio, init_step=init_step, ArmijoM=ArmijoM, act_max_itr=act_max_itr, sea_max_itr=sea_max_itr, kmip=kmip)
        w_new[nnz_idx] = np.copy(w)
    else:
        w_new[nnz_idx] = np.copy(w_bar[nnz_idx])
        f = 0
        r = np.zeros_like(y)
        act_cur_itr=0
        tot_time=0
        total_iter = 0
    return w_new, f, r, total_iter, tot_time, epsilon
