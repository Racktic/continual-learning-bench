#!/bin/bash
# Qwen3.5-4B × codebase_adaptation 官方 default schedule 正式评测。
# 一个作业内: 4 卡 serve(dp=4) → 就绪后跑 clbench(19 题 × 5 runs × 12 workers)。
# 用法: sbatch scripts/cluster/eval_qwen35_9b_codebase.sbatch
#SBATCH --job-name=eval-qwen35-4b-codebase
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/eval-qwen35-4b-codebase-%j.out

set -uo pipefail
# 底线要求: 实验过程必须实时可见。两道保险:
# 1) PYTHONUNBUFFERED=1 — 进度行实时刷进本 .out 日志(否则 Python 缓冲会憋住几小时);
# 2) clbench 带 --live-dashboard — 每完成一题就更新 results/<task>/live/ 下的轨迹快照。
export PYTHONUNBUFFERED=1
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-4B
MODEL_NAME=qwen3.5-4b
LOGDIR=/data/user_data/qixinx/clbench/logs
EP_FILE=/data/user_data/qixinx/clbench/serve_endpoint_4b.txt
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"; rm -f "$EP_FILE"
echo "===== serve $MODEL_NAME host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

# ── serve: 4 卡 data-parallel, 每卡一个 9B 副本, 吃 12 路并发 ──
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
echo "$(hostname):$PORT" > "$EP_FILE"
echo "serve ready: $(cat $EP_FILE)"

# ── 评测环境(apptainer 后端 + 本机回环端点) ──
export CLBENCH_CONTAINER_BACKEND=singularity
export CLBENCH_SIF_DIR=/data/user_data/qixinx/clbench/sifs
export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export TMPDIR=/data/user_data/qixinx/tmp
export APPTAINER_CACHEDIR=/data/user_data/qixinx/tmp/apptainer_cache
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=dummy
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR"

# ── 正式评测: 官方 default schedule(runs=5, max_workers=12, permute 由 schedule 自带) ──
# max_tokens=240000: icl 的历史截断预算, 给 262144 的模型上下文留输出余量。
cd "$REPO"
EXTRA_ARGS=()
[ "${CLB_SKIP_BASELINE:-0}" = "1" ] && EXTRA_ARGS+=(--skip-baseline)
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system icl \
  --task-params '{"schedule":"default"}' \
  --system-params '{"model":"openai/qwen3.5-4b","provider_mode":"litellm_chat","max_tokens":240000}' \
  "${EXTRA_ARGS[@]}" \
  --live-dashboard --no-live-server --verbose-runs
# --live-dashboard 必须开: 每完成一题就把部分轨迹写进 results/<task>/live/, 否则全程无进度可看
# (轨迹要等全部跑完才落盘 traces/); --no-live-server 只关本地查看服务器, 不影响快照落盘。
RC=$?
echo "===== clbench exit code: $RC ====="
kill $SRV_PID 2>/dev/null
exit $RC
