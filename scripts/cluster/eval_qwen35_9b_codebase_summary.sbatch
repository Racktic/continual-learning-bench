#!/bin/bash
# Qwen3.5-9B × codebase_adaptation × icl_summary 正式评测(window=1, 无 baseline)。
# 用法: sbatch [--job-name=...] [--dependency=afterany:<id>] \
#         scripts/cluster/eval_qwen35_9b_codebase_summary.sbatch <replace|augment>
# 5 runs(官方 default schedule 自带), --skip-baseline(gain 离线对齐 icl 组的 baseline 计算)。
#SBATCH --job-name=eval-qwen35-9b-summary
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/eval-qwen35-9b-summary-%j.out

set -uo pipefail
SUMMARY_MODE=${1:?用法: sbatch ... eval_qwen35_9b_codebase_summary.sbatch <replace|augment> [window(默认1)]}
case "$SUMMARY_MODE" in replace|augment) ;; *) echo "非法 summary_mode: $SUMMARY_MODE"; exit 2;; esac
SUMMARY_WINDOW=${2:-1}

export PYTHONUNBUFFERED=1   # 底线: 进度实时可见
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
# 可通过 sbatch --export=ALL,CLB_MODEL=...,CLB_MODEL_NAME=... 换模型（默认 9B）
MODEL=${CLB_MODEL:-/data/user_data/qixinx/Qwen3.5-9B}
MODEL_NAME=${CLB_MODEL_NAME:-qwen3.5-9b}
LOGDIR=/data/user_data/qixinx/clbench/logs
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"
echo "===== summary-$SUMMARY_MODE host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

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

echo "===== 正式: default × icl_summary($SUMMARY_MODE, window=$SUMMARY_WINDOW) × 5 runs, 无 baseline ====="
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system icl_summary \
  --task-params '{"schedule":"default"}' \
  --system-params "{\"model\":\"openai/${MODEL_NAME}\",\"provider_mode\":\"litellm_chat\",\"max_tokens\":240000,\"summary_mode\":\"$SUMMARY_MODE\",\"summary_window\":$SUMMARY_WINDOW}" \
  --skip-baseline \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
echo "===== clbench exit code: $RC ====="
kill $SRV_PID 2>/dev/null
exit $RC
