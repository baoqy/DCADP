#!/bin/bash
#SBATCH -J SMSB_MLP
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --gres=gpu:1

hostname

TASK_ID=0
EXP_ID="0_0"

CHECKPOINT_PATH=""
FIRST_EPOCH=0

echo $TASK_ID
echo $EXP_ID

algos=("Active_refBBDCA" "SMSB")
algo=${algos[1]}


nums_stages=(1 16 16)

sparsity_schedule="poly"

training_schedules=("cosine_fast_works_098" "cosine_fast1" "cosine_one")
training_schedule=${training_schedules[TASK_ID%3]}
num_stages=${nums_stages[TASK_ID%3]}

if [ $training_schedule == "cosine_fast" ] 
then 
    max_lr=0.05
    min_lr=0.000005
    prune_every=15
    nprune_epochs=7
    nepochs=100
    warm_up=0
    ft_max_lr=0.0005
    ft_min_lr=0.00005
    gamma_ft=0.5
fi
if [ $training_schedule == "cosine_fast_works_098" ] 
then 
    max_lr=0.1
    min_lr=0.00001
    prune_every=12
    nprune_epochs=7
    nepochs=100
    warm_up=0
    ft_max_lr=0.1
    ft_min_lr=0.00001
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_fast1" ] 
then 
    max_lr=0.05
    min_lr=0.000005
    prune_every=12
    nprune_epochs=7
    nepochs=100
    warm_up=0
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_fast_works" ] 
then 
    max_lr=0.05
    min_lr=0.000005
    prune_every=15
    nprune_epochs=7
    nepochs=100
    warm_up=0
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_fast_gamma" ] 
then 
    max_lr=0.05
    min_lr=0.000005
    prune_every=15
    nprune_epochs=7
    nepochs=150
    warm_up=0
    gamma_ft=0.8
fi
if [ $training_schedule == "cosine_one" ] 
then 
    max_lr=0.256
    min_lr=0.000005
    prune_every=1
    nprune_epochs=1
    nepochs=100
    warm_up=5
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_slow" ]
then
    max_lr=0.005
    min_lr=0.000005
    prune_every=4
    nprune_epochs=16
    nepochs=100
    warm_up=0
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_fast_slr" ]
then
    max_lr=0.005
    min_lr=0.000005
    prune_every=12
    nprune_epochs=7
    nepochs=100
    warm_up=0
    gamma_ft=0.9
fi

echo $max_lr

seed=2

fisher_subsample_sizes=(500)

fisher_subsample_size=${fisher_subsample_sizes[0]}

l2s=(0.0001 0.001)
l2=${l2s[0]}

fisher_mini_bszs=(1)
fisher_mini_bsz=16


### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$((12073 + TASK_ID))
export WORLD_SIZE=1
echo $MASTER_PORT

#export MASTER_ADDR=$master_addr


python3 -u trainLeNet5.py