#!/bin/bash
#SBATCH --job-name=molgen
#SBATCH --output=molgen_%a.out  # %a 会变成数组ID (0,1,2,3,4)
#SBATCH --error=molgen_%a.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4     # 每个子任务只运行 1 个进程
#SBATCH --cpus-per-task=2       # 【关键】每个任务占用 2 个 CPU 核
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

# export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# torchrun 启动
uv run python -m torch.distributed.run --nproc_per_node=4 /home/wxl/molgen/main.py