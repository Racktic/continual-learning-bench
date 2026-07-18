#!/bin/bash
# icl × Qwen3.5-9B × codebase_adaptation 的 stateless baseline 重复采样(算 pass@k)。
# 每遍 = clbench --baseline-only 独立跑一次 19 题(各自新 group id, 互不覆盖)。
# 用法: sbatch [--job-name=...] scripts/cluster/baseline_qwen35_9b_repeat.sbatch <重复次数>
#SBATCH --job-name=baseline-q35-9b-repeat
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=36:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/baseline-q35-9b-repeat-%j.out

set -uo pipefail
REPS=${1:?用法: sbatch ... baseline_qwen35_9b_repeat.sbatch <重复次数>}
export PYTHONUNBUFFERED=1   # 底线: 过程实时可见
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-9B
MODEL_NAME=qwen3.5-9b
LOGDIR=/data/user_data/qixinx/clbench/logs
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"
echo "===== baseline-repeat x$REPS thinking=${CLB_THINKING:-0} host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

# CLB_THINKING=1: 默认模板+reasoning parser(开思考); 否则 nothink 模板(默认)
if [ "${CLB_THINKING:-0}" = "1" ]; then
  TEMPLATE_ARGS=(--reasoning-parser qwen3)
else
  TEMPLATE_ARGS=(--chat-template "$REPO/scripts/cluster/qwen35_chat_template_nothink.jinja")
fi
nohup apptainer exec --nv --bind /data,/home/qixinx --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python -m sglang.launch_server --model-path "$MODEL" --served-model-name "$MODEL_NAME" \
    "${TEMPLATE_ARGS[@]}" \
    --tp 1 --dp-size 4 --host 0.0.0.0 --port "$PORT" --mem-fraction-static 0.85 --trust-remote-code \
    > "$LOGDIR/serve_${MODEL_NAME}_${SLURM_JOB_ID}.log" 2>&1 &
SRV_PID=$!
ok=0
for _ in $(seq 1 90); do
  curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SRV_PID 2>/dev/null || break
  sleep 10
done
[ "$ok" = 1 ] || { echo "!!! serve not ready"; tail -80 "$LOGDIR/serve_${MODEL_NAME}_${SLURM_JOB_ID}.log"; exit 1; }
echo "serve ready: $(hostname):$PORT"

export CLBENCH_CONTAINER_BACKEND=singularity
export CLBENCH_SIF_DIR=/data/user_data/qixinx/clbench/sifs
export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export TMPDIR=/data/user_data/qixinx/tmp
export APPTAINER_CACHEDIR=/data/user_data/qixinx/tmp/apptainer_cache
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=dummy
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR"
cd "$REPO"

RC_ALL=0
for i in $(seq 1 "$REPS"); do
  echo "===== baseline rep $i/$REPS ====="
  "$REPO/.venv/bin/clbench" run --task codebase_adaptation --system icl \
    --task-params '{"schedule":"default"}' \
    --system-params '{"model":"openai/qwen3.5-9b","provider_mode":"litellm_chat","max_tokens":240000}' \
    --baseline-only --max-workers 12 \
    --live-dashboard --no-live-server --verbose-runs || { echo "!!! rep $i failed"; RC_ALL=1; }
done
echo "===== baseline-repeat exit: $RC_ALL ====="
kill $SRV_PID 2>/dev/null
exit $RC_ALL
