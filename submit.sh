#!/bin/bash -l

# SLURM SUBMIT SCRIPT
##SBATCH --nodelist=moselle,mouette,zeuxine,sitelle
##SBATCH --nodelist=venturi,skoda,volvo,simca
##SBATCH --nodelist=fiat,ford,jaguar,lada
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=0-02:00:00
#SBATCH --output=job/log.out
#SBATCH --error=job/R-log.err
#SBATCH --job-name=lr_finder


# activate conda env
source /users/eleves-a/2018/duc-thinh.ngo/venv/bin/activate
cd /users/eleves-a/2018/duc-thinh.ngo/recvis21_a3

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
srun python main_lightning.py \
            --stage 1 \
            --mode supervise \
            --step-size 50 \
            --batch-size 16 \
            --lambda-u 1. \
            --lr 0.1 \
            --weight-decay 0.1 \
            --gpus 1 \
            --num-nodes 4 \
            --cpus 8 \
            --model swin_large_patch4_window12_384_in22k \
            --experiment $1 \
            --epochs 15