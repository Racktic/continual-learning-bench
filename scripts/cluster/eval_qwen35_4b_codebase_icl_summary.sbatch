#!/bin/bash
# Qwen3.5-4B × codebase_adaptation × icl_summary(记忆模式)正式评测, 官方 default schedule。
# 记忆模式由 CLB_SUMMARY_MODE 决定: augment(历史+记忆) 或 replace(仅记忆, 跨题清空历史)。
# 一个作业内: 4 卡 serve(dp=4) → 就绪后跑 clbench(19 题 × 5 runs × 12 workers)。
# 用法: sbatch --export=ALL,CLB_SUMMARY_MODE=augment scripts/cluster/eval_qwen35_4b_codebase_icl_summary.sbatch
#SBATCH --job-name=eval-4b-iclsum
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/eval-4b-iclsum-%j.out

set -uo pipefail
MODE="${CLB_SUMMARY_MODE:-replace}"
case "$MODE" in augment|replace) ;; *) echo "!! CLB_SUMMARY_MODE 必须是 augment 或 replace, 收到 '$MODE'"; exit 2;; esac
# 底线: 实时可见。PYTHONUNBUFFERED=1 刷进度; clbench --live-dashboard 每题落快照。
export PYTHONUNBUFFERED=1
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-4B
MODEL_NAME=qwen3.5-4b
LOGDIR=/data/user_data/qixinx/clbench/logs
EP_FILE=/data/user_data/qixinx/clbench/serve_endpoint_4b_iclsum_${MODE}.txt
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"; rm -f "$EP_FILE"
echo "===== serve $MODEL_NAME host=$(hostname) port=$PORT job=$SLURM_JOB_ID mode=$MODE ====="

# ── serve: 4 卡 data-parallel ──
if [ "${CLB_THINKING:-0}" = "1" ]; then
  TEMPLATE_ARGS=(--reasoning-parser qwen3)
else
  TEMPLATE_ARGS=(--chat-template "$REPO/scripts/cluster/qwen35_chat_template_nothink.jinja")
fi
nohup apptainer exec --nv --bind /data,/home/qixinx --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python -m sglang.launch_server --model-path "$MODEL" --served-model-name "$MODEL_NAME" \
    "${TEMPLATE_ARGS[@]}" \
    --tp 1 --dp-size 4 --host 0.0.0.0 --port "$PORT" --mem-fraction-static 0.85 --trust-remote-code \
    > "$LOGDIR/serve_${MODEL_NAME}_iclsum_${MODE}_${SLURM_JOB_ID}.log" 2>&1 &
SRV_PID=$!
ok=0
for _ in $(seq 1 90); do
  curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SRV_PID 2>/dev/null || break
  sleep 10
done
[ "$ok" = 1 ] || { echo "!!! serve not ready"; tail -80 "$LOGDIR/serve_${MODEL_NAME}_iclsum_${MODE}_${SLURM_JOB_ID}.log"; exit 1; }
echo "$(hostname):$PORT" > "$EP_FILE"
echo "serve ready: $(cat $EP_FILE)"

# ── 评测环境 ──
export CLBENCH_CONTAINER_BACKEND=singularity
export CLBENCH_SIF_DIR=/data/user_data/qixinx/clbench/sifs
export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export TMPDIR=/data/user_data/qixinx/tmp
export APPTAINER_CACHEDIR=/data/user_data/qixinx/tmp/apptainer_cache
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=dummy
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR"

# ── 正式评测: icl_summary, summary_mode=$MODE, 官方 default(runs=5, max_workers=12, permute) ──
cd "$REPO"
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system icl_summary \
  --task-params '{"schedule":"default"}' \
  --system-params "{\"model\":\"openai/qwen3.5-4b\",\"summary_model\":\"openai/qwen3.5-4b\",\"provider_mode\":\"litellm_chat\",\"max_tokens\":240000,\"summary_mode\":\"$MODE\"}" \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
echo "===== clbench exit code: $RC (mode=$MODE) ====="
kill $SRV_PID 2>/dev/null
exit $RC
