#!/bin/bash
# serve Qwen3.5-4B (单卡) 给 continual-learning-bench 用。
# 用法: sbatch scripts/cluster/serve_qwen35_4b.sbatch
# 就绪后把 "host:port" 写到 /data/user_data/qixinx/clbench/serve_endpoint.txt。
#SBATCH --job-name=serve-qwen35-4b-clbench
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/serve-qwen35-4b-%j.out

set -uo pipefail
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-4B
MODEL_NAME=qwen3.5-4b
EP_FILE=/data/user_data/qixinx/clbench/serve_endpoint.txt
LOGDIR=/data/user_data/qixinx/clbench/logs
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"; rm -f "$EP_FILE"
echo "===== serve $MODEL_NAME host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

nohup apptainer exec --nv --bind /data,/home/qixinx --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python -m sglang.launch_server --model-path "$MODEL" --served-model-name "$MODEL_NAME" \
    --tp 1 --host 0.0.0.0 --port "$PORT" --mem-fraction-static 0.85 --trust-remote-code \
    > "$LOGDIR/serve_${MODEL_NAME}_${SLURM_JOB_ID}.log" 2>&1 &
SRV_PID=$!
ok=0
for _ in $(seq 1 60); do
  curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SRV_PID 2>/dev/null || break
  sleep 10
done
[ "$ok" = 1 ] || { echo "!!! serve not ready"; tail -80 "$LOGDIR/serve_${MODEL_NAME}_${SLURM_JOB_ID}.log"; exit 1; }
echo "$(hostname):$PORT" > "$EP_FILE"
echo "serve ready: $(cat $EP_FILE)"
wait $SRV_PID
