#!/bin/bash
# 一次作业补齐 ace 和 mem0 的真 baseline(qwen3.5-9B × codebase_adaptation, 各19题独立推理)。
# summary 两组的 baseline 直接复用 icl 的(无状态管线相同), 不必重跑。
# 之后用 scripts/cluster/inject_baseline.py 注入各组 viewer 包点亮 gain。
# 用法: sbatch scripts/cluster/baselines_qwen35_9b.sbatch
#SBATCH --job-name=baselines-q35-9b
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/baselines-q35-9b-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1   # 底线: 过程实时可见
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-9B
MODEL_NAME=qwen3.5-9b
LOGDIR=/data/user_data/qixinx/clbench/logs
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"
echo "===== baselines host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

# --tool-call-parser qwen: mem0 需要; 对其他系统无影响
nohup apptainer exec --nv --bind /data,/home/qixinx --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python -m sglang.launch_server --model-path "$MODEL" --served-model-name "$MODEL_NAME" \
    --chat-template "$REPO/scripts/cluster/qwen35_chat_template_nothink.jinja" \
    --tool-call-parser qwen \
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
# mem0 embedder 用真 key(嵌入端点在 mem0 参数里显式钉死 api.openai.com, 不受下面变量影响)
set -a; source /home/qixinx/miles/.env >/dev/null 2>&1; set +a
[ -n "${OPENAI_API_KEY:-}" ] || { echo "!!! OPENAI_API_KEY 未读到"; kill $SRV_PID 2>/dev/null; exit 3; }
# litellm 路径(icl_summary 做题 / mem0 做题+抽取)
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
# ace 直连 openai SDK 路径
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR"
cd "$REPO"
CLB="$REPO/.venv/bin/clbench"
RC_ALL=0

echo "===== 1/2: ace baseline ====="
"$CLB" run --task codebase_adaptation --system ace \
  --task-params '{"schedule":"default"}' \
  --system-params '{"model":"qwen3.5-9b","api_provider":"openai"}' \
  --baseline-only --max-workers 12 \
  --live-dashboard --no-live-server --verbose-runs || RC_ALL=1

echo "===== 2/2: mem0 baseline ====="
"$CLB" run --task codebase_adaptation --system mem0 \
  --task-params '{"schedule":"default"}' \
  --system-params '{"model":"openai/qwen3.5-9b","embedding_base_url":"https://api.openai.com/v1","assume_function_calling":true}' \
  --baseline-only --max-workers 12 \
  --live-dashboard --no-live-server --verbose-runs || RC_ALL=1

echo "===== baselines exit: $RC_ALL ====="
kill $SRV_PID 2>/dev/null
exit $RC_ALL
