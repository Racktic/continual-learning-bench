#!/bin/bash
# Qwen3.5-9B × codebase_adaptation × ACE 系统。全部 LLM 调用(Generator/Reflector/Curator)
# 走本地 sglang(OPENAI_BASE_URL 补丁, vendors/ace/utils.py)。
# 流程: serve → smoke_test(2题, 验证 playbook 真的在更新) → 正式 default(5 runs, 无 baseline)。
# 用法: sbatch scripts/cluster/eval_qwen35_9b_codebase_ace.sbatch
#SBATCH --job-name=eval-q35-9b-ace
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/eval-q35-9b-ace-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1   # 底线: 过程实时可见
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-9B
MODEL_NAME=qwen3.5-9b
LOGDIR=/data/user_data/qixinx/clbench/logs
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"
echo "===== ace host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

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
# ACE 直连 openai SDK: OPENAI_BASE_URL 指向本地(补丁生效点), key 任意非空即可
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=dummy
# 保险: 若 ace 适配层任何地方经 litellm, 也指到本地
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR"
cd "$REPO"
# 注意: ACE 绕过 litellm, model 用 served 名(不带 openai/ 前缀)
SYS_PARAMS='{"model":"qwen3.5-9b","api_provider":"openai"}'

if [ "${SKIP_SMOKE:-0}" = "1" ]; then
  echo "===== SKIP_SMOKE=1: 跳过冒烟(已在 job 9012608 验证), 直入正式 ====="
else
echo "===== phase 1: smoke_test (2题, 验证 playbook 更新) ====="
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system ace \
  --task-params '{"schedule":"smoke_test"}' \
  --system-params "$SYS_PARAMS" \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
[ $RC -eq 0 ] || { echo "!!! smoke 失败(rc=$RC)"; kill $SRV_PID 2>/dev/null; exit $RC; }

# 闸门: 冒烟 traces 的 artifacts 里 playbook 必须非空
TRACE_DIR=$(ls -td "$REPO"/results/codebase_adaptation/traces/* | head -1)
PB=$(ls "$TRACE_DIR"/artifacts/*/final_playbook.md 2>/dev/null | head -1)
if [ -n "$PB" ] && [ "$(wc -c < "$PB")" -gt 200 ]; then
  echo "SMOKE_PASS: playbook 产物 $PB ($(wc -c < "$PB") 字节)"
else
  echo "SMOKE_FAIL: final_playbook.md 缺失或过小"; kill $SRV_PID 2>/dev/null; exit 42
fi
fi

echo "===== phase 2: 正式 default (5 runs, 无 baseline) ====="
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system ace \
  --task-params '{"schedule":"default"}' \
  --system-params "$SYS_PARAMS" \
  --skip-baseline \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
echo "===== clbench exit code: $RC ====="
kill $SRV_PID 2>/dev/null
exit $RC
