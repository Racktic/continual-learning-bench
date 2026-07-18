#!/bin/bash
# icl × Qwen3.5-4B × codebase_adaptation 的 stateless baseline(19题独立推理)。
# 4B 的 summary 两组(replace/augment)复用此 baseline(无状态管线与 icl 相同),
# 之后用 scripts/cluster/inject_baseline.py 注入两组 viewer 包点亮 gain。
# 用法: sbatch scripts/cluster/baseline_qwen35_4b_icl.sbatch
#SBATCH --job-name=baseline-q35-4b-icl
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/baseline-q35-4b-icl-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1   # 底线: 过程实时可见
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-4B
MODEL_NAME=qwen3.5-4b
LOGDIR=/data/user_data/qixinx/clbench/logs
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"
echo "===== baseline-4b-icl host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

nohup apptainer exec --nv --bind /data,/home/qixinx --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python -m sglang.launch_server --model-path "$MODEL" --served-model-name "$MODEL_NAME" \
    --chat-template "$REPO/scripts/cluster/qwen35_chat_template_nothink.jinja" \
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

echo "===== icl baseline (qwen3.5-4b) ====="
"$REPO/.venv/bin/clbench" run --task codebase_adaptation --system icl \
  --task-params '{"schedule":"default"}' \
  --system-params '{"model":"openai/qwen3.5-4b","provider_mode":"litellm_chat","max_tokens":240000}' \
  --baseline-only --max-workers 12 \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
echo "===== baseline exit: $RC ====="
kill $SRV_PID 2>/dev/null
exit $RC
