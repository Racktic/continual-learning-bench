#!/bin/bash
# 用本地 Qwen3.5-4B 跑一个任务。在计算节点上执行。
# 用法: bash scripts/cluster/run_qwen.sh <task> [schedule]
#   例: bash scripts/cluster/run_qwen.sh sales_prediction smoke_test
#       bash scripts/cluster/run_qwen.sh codebase_adaptation default
set -euo pipefail
export PYTHONUNBUFFERED=1   # 底线: 过程实时可见(进度行不许被缓冲憋住)
TASK=${1:?用法: run_qwen.sh <task> [schedule]}
SCHEDULE=${2:-smoke_test}
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
source scripts/cluster/env.sh
[ -n "${OPENAI_API_BASE:-}" ] || { echo "serve 未就绪, 先: sbatch scripts/cluster/serve_qwen35_4b.sbatch"; exit 1; }

exec uv run clbench run \
  --task "$TASK" --system icl \
  --task-params "{\"schedule\":\"$SCHEDULE\"}" \
  --system-params '{"model":"openai/qwen3.5-4b","provider_mode":"litellm_chat"}' \
  --live-dashboard --no-live-server --verbose-runs
# --live-dashboard 必须开: 逐题写 live 快照到 results/<task>/live/, 不开就全程无进度可看
