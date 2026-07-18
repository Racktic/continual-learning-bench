#!/bin/bash
# Qwen3.5-4B × swe_bench_cl 纯能力 baseline (pass@4)。
# 4 卡 dp=4, 4 遍 --baseline-only (seed 0..3, 全量 240 题) → 与 9B baseline 同一口径。
# 跑完用 passk_report.py 聚合 → passk_detail → enrich (见文件尾注释)。
# 用法: sbatch scripts/cluster/swecl_baseline_4b.sbatch
#SBATCH --job-name=swecl-baseline-4b
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/swecl-baseline-4b-%j.out
set -uo pipefail

export SWECL_MODEL_PATH=/data/user_data/qixinx/Qwen3.5-4B
export SWECL_MODEL_NAME=qwen3.5-4b

# dp=4, tag=4b, npass=4, seed_base=0  → seeds 0,1,2,3 = pass@4
SWECL_TAG="${SWECL_TAG:-4b}"; SWECL_NPASS="${SWECL_NPASS:-4}"; SWECL_SEED0="${SWECL_SEED0:-0}"
bash /home/qixinx/continual-learning-bench/scripts/cluster/run_swecl_passes.sh 4 "$SWECL_TAG" "$SWECL_NPASS" "$SWECL_SEED0"

# ── 跑完聚合 (手动, 或看 README) ──
#   .venv/bin/python scripts/swecl/passk_report.py --glob \
#     'results/swe_bench_cl/live/<4b run 时间戳>*/baseline.json'
#   → 得到 4B 的 solved4/avg4, 存成 passk_detail_4b.jsonl, 再据此建 4B gain baseline。
