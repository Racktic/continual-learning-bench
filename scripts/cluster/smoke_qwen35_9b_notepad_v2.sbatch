#!/bin/bash
# notepad v2 修复的独立冒烟: 1 卡 serve 9B + smoke_test(2题) + 自动验证笔记写入。
# 用法: sbatch scripts/cluster/smoke_qwen35_9b_notepad_v2.sbatch
#SBATCH --job-name=smoke-notepad-v2
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/smoke-notepad-v2-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-9B
MODEL_NAME=qwen3.5-9b
LOGDIR=/data/user_data/qixinx/clbench/logs
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"
echo "===== smoke-notepad-v2 host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

nohup apptainer exec --nv --bind /data,/home/qixinx --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python -m sglang.launch_server --model-path "$MODEL" --served-model-name "$MODEL_NAME" \
    --chat-template "$REPO/scripts/cluster/qwen35_chat_template_nothink.jinja" \
    --tp 1 --host 127.0.0.1 --port "$PORT" --mem-fraction-static 0.85 --trust-remote-code \
    > "$LOGDIR/serve_${MODEL_NAME}_smoke_${SLURM_JOB_ID}.log" 2>&1 &
SRV_PID=$!
ok=0
for _ in $(seq 1 90); do
  curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SRV_PID 2>/dev/null || break
  sleep 10
done
[ "$ok" = 1 ] || { echo "!!! serve not ready"; tail -60 "$LOGDIR/serve_${MODEL_NAME}_smoke_${SLURM_JOB_ID}.log"; exit 1; }
echo "serve ready: 127.0.0.1:$PORT"

export CLBENCH_CONTAINER_BACKEND=singularity
export CLBENCH_SIF_DIR=/data/user_data/qixinx/clbench/sifs
export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export TMPDIR=/data/user_data/qixinx/tmp
export APPTAINER_CACHEDIR=/data/user_data/qixinx/tmp/apptainer_cache
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=dummy
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR"
cd "$REPO"

export NP="$(cat "$REPO/scripts/cluster/notepad_system_prompt.txt")"
SYS_PARAMS=$(python3 -c "import json,os;print(json.dumps({'model':'openai/qwen3.5-9b','provider_mode':'litellm_chat','max_tokens':240000,'system_prompt':os.environ['NP']}))")

echo "===== smoke_test (2题) ====="
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system icl_notepad \
  --task-params '{"schedule":"smoke_test"}' \
  --system-params "$SYS_PARAMS" \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
[ $RC -eq 0 ] || { echo "!!! smoke 失败(rc=$RC)"; kill $SRV_PID 2>/dev/null; exit $RC; }

LIVE_DIR=$(ls -td "$REPO"/results/codebase_adaptation/live/* | head -1)
WROTE=$(python3 - <<PYEOF
import json, glob
w = 0
for f in glob.glob("$LIVE_DIR/*.json"):
    try: data = json.load(open(f))
    except: continue
    for t in data.get("interactions", []):
        a = (t.get("response") or {}).get("action") or {}
        if isinstance(a, dict) and a.get("notepad_update"): w += 1
print(w)
PYEOF
)
echo "===== notepad 写入次数: $WROTE (live: $LIVE_DIR) ====="
kill $SRV_PID 2>/dev/null
[ "$WROTE" -gt 0 ] && { echo "SMOKE_PASS: 笔记本已被使用"; exit 0; } || { echo "SMOKE_FAIL: 笔记本仍未被写"; exit 42; }
