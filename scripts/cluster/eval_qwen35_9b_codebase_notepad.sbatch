#!/bin/bash
# Qwen3.5-9B × codebase_adaptation 官方 default × icl_notepad 系统。
# 流程: 4 卡 serve(dp=4, 关thinking) → 先 smoke_test(2题)验证 notepad 链路 → 通过后跑正式 default。
# 用法: sbatch scripts/cluster/eval_qwen35_9b_codebase_notepad.sbatch
#SBATCH --job-name=eval-qwen35-9b-codebase-notepad
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/eval-qwen35-9b-codebase-notepad-%j.out

set -uo pipefail
# 底线: 实验过程必须实时可见 — PYTHONUNBUFFERED 让进度行实时进本日志;
# clbench 带 --live-dashboard 逐题写轨迹快照到 results/<task>/live/。
export PYTHONUNBUFFERED=1
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-9B
MODEL_NAME=qwen3.5-9b
LOGDIR=/data/user_data/qixinx/clbench/logs
EP_FILE=/data/user_data/qixinx/clbench/serve_endpoint_9b.txt
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"; rm -f "$EP_FILE"
echo "===== serve $MODEL_NAME host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

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
echo "$(hostname):$PORT" > "$EP_FILE"
echo "serve ready: $(cat $EP_FILE)"

export CLBENCH_CONTAINER_BACKEND=singularity
export CLBENCH_SIF_DIR=/data/user_data/qixinx/clbench/sifs
export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export TMPDIR=/data/user_data/qixinx/tmp
export APPTAINER_CACHEDIR=/data/user_data/qixinx/tmp/apptainer_cache
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=dummy
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR"
cd "$REPO"
SYS_PARAMS='{"model":"openai/qwen3.5-9b","provider_mode":"litellm_chat","max_tokens":240000}'

echo "===== phase 1: smoke_test (2题, 验证 notepad 链路) ====="
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system icl_notepad \
  --task-params '{"schedule":"smoke_test"}' \
  --system-params "$SYS_PARAMS" \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
[ $RC -eq 0 ] || { echo "!!! smoke 失败(rc=$RC), 不跑正式, 保卡待查"; kill $SRV_PID 2>/dev/null; exit $RC; }

echo "===== phase 2: 正式 default (19题 x 5runs + baseline) ====="
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system icl_notepad \
  --task-params '{"schedule":"default"}' \
  --system-params "$SYS_PARAMS" \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
echo "===== clbench exit code: $RC ====="
kill $SRV_PID 2>/dev/null
exit $RC
