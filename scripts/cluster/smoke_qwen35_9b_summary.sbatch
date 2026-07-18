#!/bin/bash
# icl_summary(summary-replace) 冒烟: 1 卡 serve 9B + smoke_test(2题) + 验证总结真的生成。
# 用法: sbatch scripts/cluster/smoke_qwen35_9b_summary.sbatch
#SBATCH --job-name=smoke-summary
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/smoke-summary-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-9B
MODEL_NAME=qwen3.5-9b
LOGDIR=/data/user_data/qixinx/clbench/logs
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"
echo "===== smoke-summary host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

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

echo "===== smoke_test (2题, summary-replace, window=3) ====="
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system icl_summary \
  --task-params '{"schedule":"smoke_test"}' \
  --system-params '{"model":"openai/qwen3.5-9b","provider_mode":"litellm_chat","max_tokens":240000,"summary_mode":"replace","summary_window":3}' \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
[ $RC -eq 0 ] || { echo "!!! smoke 失败(rc=$RC)"; kill $SRV_PID 2>/dev/null; exit $RC; }

# 闸门: 轨迹里必须出现非空 summary(第2题的首轮 metadata 应带上第1题产出的总结)
LIVE_DIR=$(ls -td "$REPO"/results/codebase_adaptation/live/* | head -1)
python3 - <<PYEOF
import json, glob, sys
n = 0
latest = ""
for f in glob.glob("$LIVE_DIR/*.json"):
    try: data = json.load(open(f))
    except: continue
    for t in data.get("interactions", []):
        md = (t.get("response") or {}).get("metadata") or {}
        s = md.get("summary_text")
        if s:
            n += 1
            latest = s
print(f"带非空 summary 的轮数: {n}")
if latest:
    print("=== 最新一版 summary 全文 ===")
    print(latest)
sys.exit(0 if n > 0 else 42)
PYEOF
RC=$?
kill $SRV_PID 2>/dev/null
[ $RC -eq 0 ] && echo "SMOKE_PASS: 总结机制工作" || echo "SMOKE_FAIL: 未生成总结"
exit $RC
