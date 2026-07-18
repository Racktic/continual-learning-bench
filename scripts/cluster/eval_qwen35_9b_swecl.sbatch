#!/bin/bash
# Qwen3.5-9B × swe_bench_cl 全量评测。
# 一个作业内: 4 卡 serve(dp=4) → 就绪 → 先 smoke(4题) → 通过则跑全量(240题 × 4 runs)。
# 用法: sbatch scripts/cluster/eval_qwen35_9b_swecl.sbatch
#   CLB_SMOKE_ONLY=1 只跑 smoke; CLB_SKIP_BASELINE=1 跳过 baseline。
#SBATCH --job-name=eval-qwen35-9b-swecl
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/eval-qwen35-9b-swecl-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1                       # 底线: 进度实时刷进日志
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-9B
MODEL_NAME=qwen3.5-9b
LOGDIR=/data/user_data/qixinx/clbench/logs
EP_FILE=/data/user_data/qixinx/clbench/serve_endpoint_swecl_9b.txt
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"; rm -f "$EP_FILE"
echo "===== serve $MODEL_NAME host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

# ── serve: 4 卡 dp, nothink 模板(与 codebase 评测一致) ──
if [ "${CLB_THINKING:-0}" = "1" ]; then
  TEMPLATE_ARGS=(--reasoning-parser qwen3)
else
  TEMPLATE_ARGS=(--chat-template "$REPO/scripts/cluster/qwen35_chat_template_nothink.jinja")
fi
nohup apptainer exec --nv --bind /data,/home/qixinx --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python -m sglang.launch_server --model-path "$MODEL" --served-model-name "$MODEL_NAME" \
    "${TEMPLATE_ARGS[@]}" \
    --tp 1 --dp-size 4 --host 0.0.0.0 --port "$PORT" --mem-fraction-static 0.85 --trust-remote-code \
    > "$LOGDIR/serve_swecl_${MODEL_NAME}_${SLURM_JOB_ID}.log" 2>&1 &
SRV_PID=$!
ok=0
for _ in $(seq 1 90); do
  curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SRV_PID 2>/dev/null || break
  sleep 10
done
[ "$ok" = 1 ] || { echo "!!! serve not ready"; tail -80 "$LOGDIR/serve_swecl_${MODEL_NAME}_${SLURM_JOB_ID}.log"; exit 1; }
echo "$(hostname):$PORT" > "$EP_FILE"
echo "serve ready: $(cat $EP_FILE)"

# ── 评测环境(swe_bench_cl 只跑 singularity;SIF 绝对路径已在数据集 image_name 里) ──
export CLBENCH_CONTAINER_BACKEND=singularity
export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export TMPDIR=/data/user_data/qixinx/tmp
export APPTAINER_CACHEDIR=/data/user_data/qixinx/tmp/apptainer_cache
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=dummy
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR"
cd "$REPO"

SYSPARAMS='{"model":"openai/qwen3.5-9b","provider_mode":"litellm_chat","max_tokens":240000}'
run_clbench () {  # $1=schedule
  "$REPO/.venv/bin/clbench" run \
    --task swe_bench_cl --system icl \
    --task-params "{\"schedule\":\"$1\"}" \
    --system-params "$SYSPARAMS" \
    $([ "${CLB_SKIP_BASELINE:-0}" = "1" ] && echo --skip-baseline) \
    --live-dashboard --no-live-server --verbose-runs
}

# ── 先 smoke(4题) ──
echo "===== SMOKE (4 题) ====="
run_clbench smoke; SMK=$?
echo "===== smoke exit: $SMK ====="
if [ "$SMK" != 0 ] || [ "${CLB_SMOKE_ONLY:-0}" = "1" ]; then
  kill $SRV_PID 2>/dev/null; exit $SMK
fi

# ── 全量: all schedule(240题 × 4 runs replicate) ──
echo "===== FULL (240题 × 4 runs) ====="
run_clbench all; RC=$?
echo "===== clbench exit code: $RC ====="
kill $SRV_PID 2>/dev/null
exit $RC
