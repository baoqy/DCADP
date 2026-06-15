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


#原问题的目标函数值
@nb.njit(cache=True)
def evaluate_original(r, w_t, w_bar, lambda2):
    #print('r*r:',r@r,'lambda2*||w-w_bar||^2:',lambda2*((w_t - w_bar)@(w_t - w_bar)))
    
    return 0.5*r@r + lambda2*((w_t - w_bar)@(w_t - w_bar))
    #return 0.5*np.sum(r**2) + lambda2*np.sum(w_sub2**2) 

#新DCA方法的子问题目标函数
@nb.njit(cache=True)
def evaluate_penaltyNewDCA(r, w_t, w_bar, lambda2, rho1, rho2, K, s_sum, top_K_idx):
    w_sub = w_t - w_bar
    vec = w_t[top_K_idx]
    L1norm = np.sum(np.abs(w_t))
    Knorm = np.sum(np.abs(vec))
    #print('f_original:',0.5*np.dot(r,r) + lambda2*np.dot(w_sub,w_sub),'rho1*:',rho1 * (L1norm - Knorm),'rho2/K*:',rho2/K * (K*scalar - Knorm))
    return 0.5*np.dot(r,r) + lambda2*np.dot(w_sub,w_sub) + rho1 * (L1norm - Knorm) + rho2/K * (s_sum - Knorm)

#ADMM求解子问题过程的v,s的更新
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

#ADMM求解子问题中分段约束的v,s更新
@nb.njit(cache=True)
def SM_proj_l1_linf(z, v, K, M, base_len, sub_rho1, sub_rho2, rho, lengths):
    #分段以v的排序进行，取出对应的z
    #v_abs = np.abs(v)
    z_abs = np.abs(z)
    top_all_idx = np.argsort(-z_abs)
    top_K_idx = top_all_idx[:K]
    sort_z_abs = z_abs[top_all_idx]#这里sort_what_abs里是绝对值且从大到小
    
    # 用 numba.typed.List 替代普通 list
    Pieces = List()
    
    #前面的M-1段都在前K个里，每段长度为base_len
    start = 0
    if M > 1:
        for i in range(M-1):
            end = start + base_len
            Pieces.append(sort_z_abs[start:end])#每个Piece里都是绝对值，从大到小
            start = end
    
        #进行投影时为了保序，将其余 K - (M-1)*base_len + n-K个全部放入最后一段
        Pieces.append(sort_z_abs[(M-1)*base_len:])
    else:
        Pieces.append(sort_z_abs)

    # 预分配 w 数组
    z_new = np.zeros_like(z)
    s_arr = np.zeros(M)
    
    ptr = 0
    for i in range(M):
        z_piece = Pieces[i]#每个Piece里都是绝对值，从大到小
        #shat_element = shat[i]
        #zProj, sProj, _ = sLinf_Proj(z_piece, shat_element, len(z_piece))#每个sProj里都是绝对值，从大到小
        if i < M-1:
            block_size = base_len
        else:
            block_size = lengths[-1]
        zProj, sProj = prox_l1_linf(z_piece, sub_rho1, sub_rho2/K * block_size, rho)
        s_arr[i] = sProj
        for j in range(len(zProj)):
            idx = top_all_idx[ptr]#对应在what中的索引为idx
            z_new[idx] = np.sign(z[idx]) * zProj[j]
            ptr += 1
    
    return z_new, s_arr

#ADMMD里w更新
def solve_w_subproblem(X, w_bar, s_vec, u, v, lambda2, rho, w_init=None, tol=1e-8, maxiter=50):
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
    b = (
        X.T @ (X @ w_bar)
        + 2*lambda2 * w_bar
        + s_vec
        - rho * (u - v)
    )

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

#三值问题，ADMM求解子问题时v的更新
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

#三值问题，取定s=1的取值时，ADMM求解子问题
def TriProblem_ADMMSubProblem(y, grads,  w_start, w_bar,  s_vec, lambda2,  ADMMrho, sub_rho1, sub_rho2, K,  tol = 1e-4, epsilon=1e-4, adaptive_rho=True):
    w = w_start.copy()
    #v = np.zeros_like(w)
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

#三值问题给定s=1, PPA-FISTA求解Sparse-Binary-新DCA问题的子问题
def PPAsol_subproblem(y, grads, w_start, w_bar,  r_start, s_vec, L, lambda2, K, sub_rho1, sub_rho2, epsilon):
    w_now = w_start.copy()
    w_old = w_start.copy()

    w_rho = w_start.copy()

    r_rho = r_start.copy()
    f_rho = 0
    r_now = r_start.copy()

    objs = []
    step_BB = 0 
    step_old = 0
    f_old = 0.5*r_now@r_now + lambda2*np.dot(w_now-w_bar,w_now-w_bar)  - np.dot(w_now, s_vec)
    f_trial = 0
    itr_in = 1
    c = 1e-4

    while True:

        step = 1/L
        itr_search = 0
        w_FISTA = w_now + (itr_in - 2)/(itr_in + 1)*(w_now - w_old)
        r_FISTA = y - grads@w_FISTA
        f_FISTA = 0.5 * r_FISTA @ r_FISTA + lambda2 * np.dot(w_FISTA-w_bar, w_FISTA-w_bar)  - np.dot(w_FISTA, s_vec)
        gradf_wFISTA = -grads.T@r_FISTA + 2 * lambda2 * (w_FISTA - w_bar)  - s_vec
        while True:
            itr_search += 1
            
            w_hat = w_FISTA - gradf_wFISTA * step
            what_prox = np.sign(w_hat) * np.maximum(np.abs(w_hat) - step * sub_rho1, 0)
            #w_trial, s_trial, top_K_idx = sLinf_Proj(what_prox, s_hat, K)
            #d = w_trial - w_now
            w_trial = np.clip(what_prox, -1, 1)
            r_trial = y - grads @ w_trial
            f_trial = 0.5 * r_trial @ r_trial + lambda2 * np.dot(w_trial-w_bar, w_trial-w_bar)  - np.dot(w_trial, s_vec)
            #f_trial = 0.5 * r_trial @ r_trial + lambda2 * np.dot(w_trial-w_bar, w_trial-w_bar) + sub_rho1 * np.sum(np.abs(w_trial)) + sub_rho2 * s_trial  - np.dot(w_trial, s_vec)

            if abs(step) <= 1e-8:
            #if step <= 1e-4 * min(1/L, s_now/sub_rho):
                step = 1/L
                w_hat = w_FISTA - gradf_wFISTA * step
                what_prox = np.sign(w_hat) * np.maximum(np.abs(w_hat) - step * sub_rho1, 0)
                w_trial = np.clip(what_prox,-1,1)
                r_trial = y - grads @ w_trial
                f_trial = 0.5 * r_trial @ r_trial + lambda2 * np.dot(w_trial-w_bar, w_trial-w_bar) + sub_rho1 * np.sum(np.abs(w_trial))  - np.dot(w_trial, s_vec)
                print('Sub Armijo Failed,f_trial:',f_trial)
                break

            if f_trial <= f_FISTA + np.dot(gradf_wFISTA, w_trial - w_FISTA) + (np.dot(w_trial-w_FISTA,w_trial-w_FISTA))/(2*step):
                itr_in += 1
                objs.append(f_trial)
                break
            else:
                step *= 0.5

        if abs(f_trial-f_old) <= max(epsilon, 1e-5):
        #if abs(f_trial-f_old) <= 1e-4:
            w_rho = w_trial
            r_rho = r_trial
            f_rho = f_trial
            break
        
        f_old = f_trial
        w_old = w_now.copy()
        r_old = r_now.copy()
        w_now = w_trial.copy()
        r_now = r_trial.copy()

    return w_rho, r_rho,  f_rho
#三值问题，使用ADMM求解最优二值化目标值的一步稀疏+量化方法
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
            """
            grad_Squarenorm = np.zeros_like(w_now)
            grad_Squarenorm[top_K_idx] = 2*w_now[top_K_idx]
            s_vec = sub_rho1 * grad_K2norm + sub_rho2/K * grad_Squarenorm
            """
            #ADMM求解子问题
            #ADMMrho = sub_rho1 / max(np.min(np.abs(w_now[top_K_idx])) * 10, 1e-5)#这个ADMM里的rho的选取确保不会过稀疏
            #w_new, s_new = SM_ADMMSubProblem(y, grads, w_now, w_bar, s_now, s_vec, lambda2, ADMMrho, sub_rho1, sub_rho2, K, M, base_len, lengths, tol = 1e-4, epsilon=epsilon, adaptive_rho=True)
            ADMMrho = 1e-1
            w_new = TriProblem_ADMMSubProblem(y, grads,  w_now, w_bar, s_vec, lambda2,  ADMMrho, sub_rho1, sub_rho2, K, tol = 1e-4, epsilon=1e-4, adaptive_rho=True)
            
            #w_new, r_new, _ = PPAsol_subproblem(y, grads, w_now, w_bar,  r_now, s_vec, L, lambda2, K, sub_rho1, sub_rho2, epsilon)
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
        #print('Knorm:',Knorm)
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
    
        """
        if SparseRes <= 1 and Notrho2Flag:
            sub_rho2 = 100
            Notrho2Flag = False
        """
        """
        if SparseRatio < 0.97:
            sub_rho1 *= rho_ratio
        if BinaryRatio < 0.97:
            sub_rho2 *= rho_ratio
        if SparseRatio >= 0.97 and BinaryRatio >= 0.97:
            print('Cons Res Satisfy')
            w_new = np.zeros_like(w_bar)
            for i in range(K):
                idx = top_K_idx[i]
                w_new[idx] = np.sign(w_rho[idx])
            w_rho = w_new
            break
        """
        itr_rho += 1
        epsilon = init_tol/itr_rho

    sol_time = time()-st

    return w_rho,  objs, sol_time, epsilon

#M个s的取值时，ADMM求解子问题
def SM_ADMMSubProblem(y, grads,  w_start, w_bar, s_start, s_vec, lambda2,  ADMMrho, sub_rho1, sub_rho2, K, M, base_len, lengths, tol = 1e-4, epsilon=1e-4, adaptive_rho=True):
    w = w_start.copy()
    #v = np.zeros_like(w)
    v = w_start.copy()
    u = np.zeros_like(w)
    s = s_start
    rho = ADMMrho#这个rho的选取确保不会过稀疏
    mu = 10
    max_iter = 500
    itr = 0
    f_old = 0
    while itr < max_iter:
        itr += 1
        w_old = w.copy()
        v_old = v.copy()
        #vec_tmp = grads.T @ (grads @ w_bar) + 2*lambda2 * w_bar + s_vec - rho * (u-v)
        #w = solve_XTX_lambda_inv(grads, V, S, vec_tmp, 2*lambda2, cache=None)
        w = solve_w_subproblem(X=grads, w_bar=w_bar, s_vec=s_vec, u=u, v=v_old, lambda2=lambda2, rho=rho, w_init=w_old, tol=1e-4,maxiter=50)
        v, s = SM_proj_l1_linf(w + u, v_old, K, M, base_len, sub_rho1, sub_rho2, rho, lengths)
        #v, s = prox_l1_linf(w + u, sub_rho1, sub_rho2, rho)
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
        s_sum = np.sum(lengths * s)
        f_new = 0.5 * r_new @ r_new + lambda2 * np.dot(w-w_bar, w-w_bar) + sub_rho1 * np.sum(np.abs(v)) + sub_rho2/K * s_sum  - np.dot(v, s_vec)
        #if r_norm < tol and d_norm < tol and abs(f_new-f_old) <= epsilon:
        if abs(f_new-f_old) <= epsilon:
            #print('ADMM Acc break')
            break
        f_old = f_new
    #return w, s
    return v, s
#只有一个s的取值时，ADMM求解子问题
def ADMMSubProblem(y, grads,  w_start, w_bar, s_start, s_vec, lambda2,  ADMMrho, sub_rho1, sub_rho2, tol = 1e-4, epsilon=1e-4, adaptive_rho=True):
    w = w_start.copy()
    v = np.zeros_like(w)
    u = np.zeros_like(w)
    s = s_start
    rho = ADMMrho#这个rho的选取确保不会过稀疏
    mu = 10
    max_iter = 500
    itr = 0
    f_old = 0
    while itr < max_iter:
        itr += 1
        w_old = w.copy()
        v_old = v.copy()
        #vec_tmp = grads.T @ (grads @ w_bar) + 2*lambda2 * w_bar + s_vec - rho * (u-v)
        #w = solve_XTX_lambda_inv(grads, V, S, vec_tmp, 2*lambda2, cache=None)
        w = solve_w_subproblem(X=grads, w_bar=w_bar, s_vec=s_vec, u=u, v=v, lambda2=lambda2, rho=rho, w_init=w_old, tol=1e-4,maxiter=50)
        #v, s = SM_proj_l1_linf(w + u, K, M, base_len, sub_rho1, sub_rho2, rho)
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
        r_new = y - grads@w
        f_new = 0.5 * r_new @ r_new + lambda2 * np.dot(w-w_bar, w-w_bar) + sub_rho1 * np.sum(np.abs(v)) + sub_rho2 * s  - np.dot(w, s_vec)
        if r_norm < tol and d_norm < tol and abs(f_new-f_old) <= epsilon:
            #print('ADMM Acc break')
            break
        f_old = f_new
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

#仅做||w||_{inf} <= s的投影
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

#对于分段无穷范数的约束投影，shat=[shat1,shat2,...,shatm]
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
        Pieces.append(sort_what_abs[(M-1)*base_len:])
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

def SMSB_OneStep(M, y, grads, w_bar, lambda2, K, init_tol=1e-3, rho_start1 = 1e-2, rho_start2 = 1, rho_ratio=2, rhoStartRatio=1e-2, ConvergenceRatio=0.99,max_sub=5):
    
    st = time()
    itr_rho = 1
    
    objs = []
    epsilon = init_tol
    
    ####将起始点从w_bar更换为根据w_bar的前K支集求解的SB解
    base_len = K // M#将长度为K的数组均分成M段，每段应该有base_len个数字
    remainder = K % M#剩下的remainder个全部放进最后一个数组
    lengths = np.full(M, base_len, dtype=np.int64)  # 前 M-1 段都是 base_len
    lengths[-1] += remainder#最后一段长度是base_len + remainder
    top_K_idx = np.argsort(-np.abs(w_bar))[:K]
    a = np.max(np.abs(w_bar))
    b = abs(w_bar[top_K_idx[-1]])
    s_rho = np.linspace(a, b, M)

    w_rho = np.copy(w_bar)
    w_rho[top_K_idx] = b
    r_rho = y - grads@w_rho

    abs_w = np.abs(w_rho)
    L1norm = np.sum(abs_w)
    Knorm = np.sum(abs_w[top_K_idx])

    f_original = 0.5*np.dot(r_rho,r_rho) + lambda2*np.dot(w_rho-w_bar,w_rho-w_bar)

    penaltySparse = L1norm - Knorm
    penaltyBinary = np.sum(lengths * s_rho) - Knorm
    """
    #对于MLP,SP=0.3, M=4
    sub_rho1 = f_original / penaltySparse * rhoStartRatio
    sub_rho2 = f_original / penaltyBinary  * K * rhoStartRatio * 1e-2
    """
    """
    #MLP
    sub_rho1 = f_original / penaltySparse * rhoStartRatio * 1e-2 #MLP,当过稀疏时，*1e-2减小sub_rho1，同时减小sub_rho2
    sub_rho2 = f_original / penaltyBinary  * K * rhoStartRatio * 1e-2
    """
    """
    #LeNet SP=0.3,0.4,0.5, M=64, 127
    sub_rho1 = f_original / penaltySparse * rhoStartRatio * 1e-1
    sub_rho2 = f_original / penaltyBinary  * K * rhoStartRatio * 10
    """
    """
    #LeNet SP=0.6,0.7,M=64, 127
    sub_rho1 = f_original / penaltySparse * rhoStartRatio * 1e-1
    sub_rho2 = f_original / penaltyBinary  * K * rhoStartRatio
    """
    
    #Res SP=0.3,0.4,0.5, M=64, 127
    sub_rho1 = f_original / penaltySparse * rhoStartRatio * 1e-1
    sub_rho2 = f_original / penaltyBinary  * K * rhoStartRatio * 10
    
    """
    #Res SP=0.6, 0.7, M=64, 127
    sub_rho1 = f_original / penaltySparse * rhoStartRatio
    sub_rho2 = f_original / penaltyBinary  * K * rhoStartRatio
    """
    #print('sub_rho1 start:',sub_rho1,'sub_rho2 start:',sub_rho2)

    while True:
        
        w_now = w_rho.copy()
        s_now = s_rho
        epsilon /= itr_rho
        epsilon = max(epsilon, 1e-5)
        itr_sub = 0
        st = time()
        alpha = (sub_rho1 + sub_rho2/K)

        s_vec = np.zeros_like(w_bar)
        while True:
            itr_sub += 1
  
            grad_K2norm = np.zeros_like(w_now)
            grad_K2norm[top_K_idx] = np.sign(w_now[top_K_idx])
            s_vec =  alpha * grad_K2norm 
            
            #ADMM求解子问题
            ADMMrho = sub_rho1 / max(np.min(np.abs(w_now[top_K_idx])) * 10, 1e-5)#这个ADMM里的rho的选取确保不会过稀疏
            w_new, s_new = SM_ADMMSubProblem(y, grads, w_now, w_bar, s_now, s_vec, lambda2, ADMMrho, sub_rho1, sub_rho2, K, M, base_len, lengths, tol = 1e-4, epsilon=epsilon, adaptive_rho=True)

            if  itr_sub >= max_sub:
                w_rho = w_new.copy()
                s_rho = s_new.copy()
                break

            w_now = w_new
            s_now = s_new

        
        abs_w = np.abs(w_rho)
        top_K_idx = np.argsort(-abs_w)[:K]

        number_nonzero = np.sum(abs_w > 0)
        L1norm = np.sum(np.abs(w_rho))
        Knorm = np.sum(abs_w[top_K_idx])

        SparseRes = L1norm - Knorm
        BinaryRes = np.sum(lengths * s_rho) - Knorm

        if SparseRes > 1e-1:
            sub_rho1 *= rho_ratio
        if BinaryRes > 1e-1:
            sub_rho2 *= rho_ratio
        if SparseRes <= 1e-1 and BinaryRes <= 1e-1:
            print('Cons Res Satisfy')
            w_new = np.zeros_like(w_bar)
            ptr = 0
            for j in range(len(lengths)):
                s = s_rho[j]
                if j <= M-2:
                    L = lengths[j]
                else:
                    L = base_len + remainder
                for i in range(L):
                    idx = top_K_idx[ptr]
                    w_new[idx] = np.sign(w_rho[idx]) * s
                    ptr += 1
            w_rho = w_new
            break
        
        itr_rho += 1
        epsilon = init_tol/itr_rho

    sol_time = time()-st

    return w_rho, s_rho, objs, sol_time, epsilon

#只有一个s值时的SB问题求解
def OneS_SB(y, grads, w_bar, lambda2, K, init_tol=1e-3, rho_start1 = 1e-2, rho_start2 = 1, rho_ratio=2, rhoStartRatio=1e-2, ConvergenceRatio=0.99, max_itr=1):
    
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
    
    while True:
        
        w_now = w_rho.copy()
        s_now = s_rho


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
            ADMMrho = sub_rho1 / np.min(np.abs(w_now[top_K_idx])) * 10#这个ADMM里的rho的选取确保不会过稀疏
            #ADMM求解子问题
            w_new, s_new = ADMMSubProblem(y, grads, w_now, w_bar, s_now, s_vec, lambda2, ADMMrho, sub_rho1, sub_rho2, tol = 1e-6, epsilon=epsilon, adaptive_rho=True)
     
            if itr_sub >= max_itr:
                w_rho = w_new.copy()
                s_rho = s_new

                break

            w_now = w_new
            s_now = s_new

        top_K_idx = np.argpartition(abs_w, -K)[-K:]
        L1norm = np.sum(np.abs(w_rho))
        Knorm = np.sum(abs_w[top_K_idx])

        SparseRatio = 1 - (L1norm - Knorm)/Knorm
        BinaryRatio = 1 - (K*s_rho - Knorm)/Knorm
        
        if SparseRatio >= 1 and BinaryRatio > ConvergenceRatio:
            print('Sparse Satisfy') 
            u = np.zeros_like(w_rho)
            u[top_K_idx] = np.sign(w_rho[top_K_idx])
            A_u = grads.T @ (grads @ u) + lambda2 * u    # 等于 (X^T X + lambda I) * u
            A_w = grads.T @ (grads @ w_bar) + lambda2 * w_bar
            alpha = u.dot(A_u)    # u^T A u
            beta  = u.dot(A_w)    # u^T A wbar
            s_rho = beta / alpha
            w_rho = u * s_rho
            abs_w = np.abs(w_rho) 
            s_Count = np.sum(np.abs((abs_w - s_rho)) <= 1e-6)

            break
        if SparseRatio > ConvergenceRatio and BinaryRatio > ConvergenceRatio:
            print('Constrain Nearly Satisfy') 
            u = np.zeros_like(w_rho)

            u[top_K_idx] = np.sign(w_rho[top_K_idx])
            A_u = grads.T @ (grads @ u) + lambda2 * u    # 等于 (X^T X + lambda I) * u
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

#旧DCA的罚形式目标函数值
@nb.njit(cache=True)
def penalty_OldDCA(r, w_t, w_bar, sub_rho, K, lambda2):
 
    top_K_idx = np.argpartition(np.abs(w_t), -K)[-K:]
    w_tabs = np.abs(w_t)
    w_tKnorm = np.sum(w_tabs[top_K_idx])
  
    w_sub = w_t - w_bar
    return 0.5*np.dot(r,r) + lambda2*np.dot(w_sub,w_sub) + sub_rho*(np.linalg.norm(w_t,1) - w_tKnorm)

#计算旧DCA的罚函数的梯度
@nb.njit(cache=True)
def gradient_OldDCA(grads, r, w_t, w_bar, lambda2, sub_rho, K):
    grad1 = -grads.T @ r + 2 * lambda2 * (w_t - w_bar)

    grad2 = sub_rho * np.sign(w_t)
    top_K_idx = np.argpartition(np.abs(w_t), -K)[-K:]
    grad2[top_K_idx] = 0

    return grad1 + grad2

#计算原始DCA方法的更新解，合并了s的计算，步长step = 1/(2*lambda1)
@nb.njit(cache=True)
def DCA_update(w, r, w_bar, rho, grads, step, lambda2, K):
    # Calculate the w_t in DCA
    w_t = np.zeros_like(w_bar)
    lambda1 = 1/(2*step)
    #lambda1 = -1/step
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


#初始化一个支集作为Active Set策略的开始
def DCAinitial_active_set(y,grads,w_bar,r,k,step,lambda2,sub_rho,M=np.inf,kimp=1.5,act_itr=1, s= np.inf):
    
    p = w_bar.shape[0]
    #将支集容量拓宽至稀疏度的kimp倍
    ksupp = int(np.max([np.min([kimp*k, p]),k]))
    w_tmp, r_tmp = np.copy(w_bar), np.copy(r)
    for i in range(act_itr):
        w_tmp = DCA_update(w_tmp, r_tmp, w_bar, sub_rho, grads, step, lambda2, ksupp)
    active_set = set(np.where(w_tmp)[0])    
    active_set = np.array(sorted(active_set),dtype=int)
    
    return active_set

#Active Set策略+BB步长的DCA在非零初始支集开始的函数总入口
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

#使用f_ref选取策略的BB法+Armijo
def ref_BBDCA(y, grads, lambda2, w_bar,  K, rho_delta, init_tol = 1e-6, rho_start=1e-8, rho_ratio=2, init_step=1, ArmijoM=10):
    st = time()
    lambda1_tmp = 1.05*(skl_svd(grads)**2+lambda2*2)

    init_step = 1/lambda1_tmp
    iter_rho = 1
    iter_number = 0

    w_now = w_bar.copy()
    r_now = y - grads@w_now
    #w_rho = w_bar.copy()
    sub_rho = rho_start
    objs = []
    L = ArmijoM
    epsilon = init_tol
    f_now = np.inf
    #step_old = 0
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
                    elif itr_in % 2 == 0:
                        step = sy /np.dot(yBB, yBB)
                    else:
                        step = np.dot(sBB,sBB)/sy
                #step = max(1e-3, step)

            #itr_search = 0
            while True:
                #itr_search += 1
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
                f_trial = evaluate_original(r_trial, w_trial, w_bar, lambda2) + sub_rho * (np.sum(np.abs(w_trial)) - Knorm)
                
                iter_number += 1

                if f_trial <= f_ref + 1e-4 * np.dot(g_penaltynow, d):
                    itr_in += 1
                    objs.append(f_trial)
                    #step_old = step
                    break
                else:
                    step *= 0.5

            if np.dot(w_trial-w_now, w_trial-w_now) <= epsilon and abs(f_trial - f_now) <= min(1e-2, 1e-2*abs(f_now)):
                rhoFlag = True

            if itr_in == 1:
                f_best = f_trial
                f_c = f_best
            
            if f_trial < f_best:
                f_best = f_trial
                f_c = f_trial
                l = 0
            else:
                f_c = max(f_c, f_trial)
                l += 1
                if l == L:
                    f_ref = f_c
                    f_c = f_trial
                    l = 0
                    
            f_now = f_trial
            w_old = w_now
            #r_old = r_now

            w_now = w_trial
            r_now = r_trial
            g_penaltyold = g_penaltynow
            
            if rhoFlag:
                break
            
        number_nonzero = np.count_nonzero(w_trial)
        if number_nonzero <= K:
            break
        else:
            sub_rho *= rho_ratio
            iter_rho += 1
            epsilon = init_tol/iter_rho
    
    sol_time = time()-st

    return w_trial, r_trial, iter_number, sol_time, sub_rho, epsilon

#使用Active Set策略+BB步长+ref策略的DCA
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
    #print('len(active_set):',len(active_set))
    while act_cur_itr < act_max_itr:
        grads_act = grads[:,active_set]
        w_act = w[active_set]
        step_act = step
        L_act = 0.5*1.05*(skl_svd(grads_act)**2+lambda2*2)
        #step_act = 1/(2*L_act)
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

#Active Set策略+BB步长+ref策略的DCA在非零初始支集的函数总入口
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
    #print('f:',f,'iter_number:',total_iter)
    return w_new, f, r, total_iter, tot_time, epsilon
