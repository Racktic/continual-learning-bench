#!/bin/bash
# 节点 B: rl 分区 8 卡, dp=8, 跑 2 遍纯能力 baseline(seed 2,3)。与节点A(A100 salloc, seed 0,1)合成 pass@4。
#SBATCH --job-name=swecl-baseline-B
#SBATCH --partition=rl
#SBATCH --qos=rl_qos
#SBATCH --account=aviralku
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/driver_nodeB-%j.out
set -uo pipefail
bash /home/qixinx/continual-learning-bench/scripts/cluster/run_swecl_passes.sh 8 nodeB 2 2
