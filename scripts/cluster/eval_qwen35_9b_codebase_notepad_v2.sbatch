#!/bin/bash
# Qwen3.5-9B × codebase_adaptation × icl_notepad —— v2: 修复"笔记本对模型不可见"。
# 背景: 我们的 litellm_chat+sglang 路径里 schema 字段描述不进模型上下文, 模型不知道
# notepad_update 字段的存在(v1 实测 196 轮 0 次写入)。官方 GPT/Claude 路径下 schema 说明
# 会被 API/prompt 注入, 模型可见。本版用 icl_notepad 现成的 system_prompt 参数把说明补上,
# 零代码改动, 对齐官方路径的信息条件。
# 流程: serve → smoke(2题,人工检查笔记是否被写) → 正式 default。
# 用法: sbatch scripts/cluster/eval_qwen35_9b_codebase_notepad_v2.sbatch
#SBATCH --job-name=eval-qwen35-9b-codebase-notepad-v2
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/eval-qwen35-9b-codebase-notepad-v2-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1   # 底线: 进度实时可见
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

# 笔记本使用说明(v1 里模型不可见的信息, 在此显式注入)——与冒烟脚本共享同一份文案
export NP="$(cat "$REPO/scripts/cluster/notepad_system_prompt.txt")"
SYS_PARAMS=$(python3 -c "import json,os;print(json.dumps({'model':'openai/qwen3.5-9b','provider_mode':'litellm_chat','max_tokens':240000,'system_prompt':os.environ['NP']}))")

echo "===== phase 1: smoke_test (2题, 验证笔记是否真的被写) ====="
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system icl_notepad \
  --task-params '{"schedule":"smoke_test"}' \
  --system-params "$SYS_PARAMS" \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
[ $RC -eq 0 ] || { echo "!!! smoke 失败(rc=$RC)"; kill $SRV_PID 2>/dev/null; exit $RC; }

# 自动验证: smoke 轨迹里 notepad_update 必须有非空写入, 否则不进正式阶段
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
echo "smoke 阶段 notepad 写入次数: $WROTE"
[ "$WROTE" -gt 0 ] || { echo "!!! 修复无效: smoke 里笔记本仍未被写, 不跑正式"; kill $SRV_PID 2>/dev/null; exit 42; }

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
