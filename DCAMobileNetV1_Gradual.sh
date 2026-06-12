#!/bin/bash
#SBATCH -J MobileNetV1
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

TASK_ID=0
EXP_ID="0_0"

echo $TASK_ID
echo $EXP_ID

algos=("Active_refBBDCA" "SMSB")
algo=${algos[0]}


nums_stages=(1 16 16)

sparsity_schedule="poly"

training_schedules=("cosine_fast_works_098" "cosine_fast1" "cosine_one")
training_schedule=${training_schedules[TASK_ID%3]}
num_stages=${nums_stages[TASK_ID%3]}


if [ $training_schedule == "cosine_fast_works_098" ] 
then 
    max_lr=0.1
    min_lr=0.00001
    prune_every=12
    nprune_epochs=7
    nepochs=100
    warm_up=0
    ft_max_lr=0.05
    ft_max_lr=0.1
    ft_min_lr=0.00001
    gamma_ft=-1
fi

echo $max_lr


fisher_subsample_sizes=(500)
fisher_subsample_size=${fisher_subsample_sizes[0]}

l2s=(0.0001 0.001)
l2=${l2s[0]}

fisher_mini_bszs=(1)
fisher_mini_bsz=16

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$((12073 + TASK_ID))
echo $MASTER_PORT

#export MASTER_ADDR=$master_addr

export OMP_NUM_THREADS=24


CHECKPOINT_PATH="/share/home/fanxilai_lsec/SMSB/checkpoints/MobileNetV1-Dense-STR.pth"
FIRST_EPOCH=0
seed=2


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=1 run_experiment_gradual.py \
--arch mobilenetv1 --dset imagenet --num_workers 24 \
--exp_name test --exp_id ${seed} --test_batch_size 256 --train_batch_size 256 \
--fisher_subsample_size ${fisher_subsample_size} --fisher_mini_bsz ${fisher_mini_bsz} \
--num_iterations 1 --num_stages ${num_stages} --seed ${seed} \
--sparsity 0.95 --base_level 0.3 --dis_num 0 \
--outer_base_level 0.3 --l2 ${l2} --sparsity_schedule ${sparsity_schedule} \
--algo ${algo} \
--max_lr ${max_lr} --min_lr ${min_lr} --prune_every ${prune_every} \
--nprune_epochs ${nprune_epochs} --nepochs ${nepochs} \
--gamma_ft ${gamma_ft} --warm_up ${warm_up} \
--ft_max_lr ${ft_max_lr} --ft_min_lr ${ft_min_lr} \
--first_epoch ${FIRST_EPOCH} --checkpoint_path ${CHECKPOINT_PATH}null