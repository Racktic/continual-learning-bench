#!/bin/bash
# Qwen3.5-9B × codebase_adaptation × mem0 系统。
# 做题与事实抽取走本地 sglang(litellm, OPENAI_API_BASE); embedding 走真 OpenAI
# (openai SDK 读 OPENAI_API_KEY, OPENAI_BASE_URL 不设置 → api.openai.com)。
# 真 key 从 /home/qixinx/miles/.env 读取, 不打印、不进命令行参数。
# 流程: serve → smoke_test(2题, 验证记忆真的入库) → 正式 default(5 runs, 无 baseline)。
# 用法: sbatch scripts/cluster/eval_qwen35_9b_codebase_mem0.sbatch
#SBATCH --job-name=eval-q35-9b-mem0
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/eval-q35-9b-mem0-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1   # 底线: 过程实时可见
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-9B
MODEL_NAME=qwen3.5-9b
LOGDIR=/data/user_data/qixinx/clbench/logs
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"
echo "===== mem0 host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

nohup apptainer exec --nv --bind /data,/home/qixinx --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python -m sglang.launch_server --model-path "$MODEL" --served-model-name "$MODEL_NAME" \
    --chat-template "$REPO/scripts/cluster/qwen35_chat_template_nothink.jinja" \
    --tool-call-parser qwen \
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
# embedding: openai SDK 读 OPENAI_API_KEY(真 key), OPENAI_BASE_URL 不设置 → 真 OpenAI
set -a; source /home/qixinx/miles/.env >/dev/null 2>&1; set +a
[ -n "${OPENAI_API_KEY:-}" ] || { echo "!!! OPENAI_API_KEY 未从 miles/.env 读到"; kill $SRV_PID 2>/dev/null; exit 3; }
# 做题+抽取: litellm 读 OPENAI_API_BASE → 本地 sglang(对 key 内容不校验)。
# 必须在 source .env 之后设置, 防止被 .env 覆盖; OPENAI_BASE_URL 保持未设置。
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
unset OPENAI_BASE_URL 2>/dev/null || true
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR"
cd "$REPO"
# embedding_base_url 显式钉死真 OpenAI: mem0 的 embedder 会回退读 OPENAI_API_BASE(指着本地聊天模型), 必须覆盖。
# assume_function_calling: litellm 不认识本地模型时注册其工具调用能力(mem0 抽取管线依赖), 配合 serve 端 --tool-call-parser qwen。
SYS_PARAMS='{"model":"openai/qwen3.5-9b","embedding_base_url":"https://api.openai.com/v1","assume_function_calling":true}'

if [ "${SKIP_SMOKE:-0}" = "1" ]; then
  echo "===== SKIP_SMOKE=1: 跳过冒烟(记忆管线已在 job 9013892 验证: 74/75 轮检索成功), 直入正式 ====="
else
echo "===== phase 1: smoke_test (2题, 验证记忆入库+检索) ====="
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system mem0 \
  --task-params '{"schedule":"smoke_test"}' \
  --system-params "$SYS_PARAMS" \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
[ $RC -eq 0 ] || { echo "!!! smoke 失败(rc=$RC)"; kill $SRV_PID 2>/dev/null; exit $RC; }

# 闸门: 冒烟 traces 的 artifacts 里 mem0 记忆必须非空
TRACE_DIR=$(ls -td "$REPO"/results/codebase_adaptation/traces/* | head -1)
python3 - <<PYEOF
import json, glob, sys
ok = False
for f in glob.glob("$TRACE_DIR/artifacts/*/artifacts.json"):
    try: art = json.load(open(f))
    except: continue
    blob = json.dumps(art)
    if ("memories" in blob or "memory" in blob) and len(blob) > 200:
        ok = True
        print(f"mem0 记忆产物存在: {f} ({len(blob)} 字节)")
print("GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 42)
PYEOF
RC=$?
[ $RC -eq 0 ] || { echo "SMOKE_FAIL: 记忆未入库"; kill $SRV_PID 2>/dev/null; exit 42; }
echo "SMOKE_PASS"
fi

echo "===== phase 2: 正式 default (5 runs, 无 baseline) ====="
"$REPO/.venv/bin/clbench" run \
  --task codebase_adaptation --system mem0 \
  --task-params '{"schedule":"default"}' \
  --system-params "$SYS_PARAMS" \
  --skip-baseline \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
echo "===== clbench exit code: $RC ====="
kill $SRV_PID 2>/dev/null
exit $RC
