#!/bin/bash
# Qwen3.5-9B 开 thinking mode 的 smoke test(codebase_adaptation smoke_test 调度, 2题)。
# 关键差异 vs 正式 nothink 配置:
#   1) 不挂 nothink jinja(用模型默认模板, enable_thinking=true)
#   2) --reasoning-parser qwen3: 把 <think> 段剥进 reasoning_content, 约束解码只作用于正文
# 预检: 先 curl 一次 json_schema+thinking 请求, 验证 reasoning_content 非空 且 content 是合法 JSON。
# 用法: sbatch scripts/cluster/smoke_qwen35_9b_thinking.sbatch
#SBATCH --job-name=smoke-q35-9b-thinking
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/data/user_data/qixinx/clbench/logs/smoke-q35-9b-thinking-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1   # 底线: 过程实时可见
REPO=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
MODEL=/data/user_data/qixinx/Qwen3.5-9B
MODEL_NAME=qwen3.5-9b
LOGDIR=/data/user_data/qixinx/clbench/logs
PORT=$((22000 + SLURM_JOB_ID % 8000))
mkdir -p "$LOGDIR"
echo "===== smoke-thinking host=$(hostname) port=$PORT job=$SLURM_JOB_ID ====="

# 默认模板 + reasoning parser(无 --chat-template)
nohup apptainer exec --nv --bind /data,/home/qixinx --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python -m sglang.launch_server --model-path "$MODEL" --served-model-name "$MODEL_NAME" \
    --reasoning-parser qwen3 \
    --tp 2 --host 0.0.0.0 --port "$PORT" --mem-fraction-static 0.85 --trust-remote-code \
    > "$LOGDIR/serve_${MODEL_NAME}_thinking_${SLURM_JOB_ID}.log" 2>&1 &
SRV_PID=$!
ok=0
for _ in $(seq 1 90); do
  curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SRV_PID 2>/dev/null || break
  sleep 10
done
[ "$ok" = 1 ] || { echo "!!! serve not ready"; tail -80 "$LOGDIR/serve_${MODEL_NAME}_thinking_${SLURM_JOB_ID}.log"; exit 1; }
echo "serve ready: $(hostname):$PORT"

echo "===== 预检 1/2: 纯 thinking(无 schema) ====="
python3 - "$PORT" <<'EOF'
import json, sys, urllib.request
port = sys.argv[1]
def chat(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))
r = chat({"model":"qwen3.5-9b","messages":[{"role":"user","content":"What is 17*23? Answer with just the number."}],"max_tokens":2048})
m = r["choices"][0]["message"]
rc = m.get("reasoning_content") or ""
print(f"reasoning_content 长度={len(rc)} | content={m.get('content','')[:80]!r} | finish={r['choices'][0]['finish_reason']}")
assert len(rc) > 0, "FAIL: 没有思考内容, thinking 未开启"
print("预检1 PASS: thinking 已开启且被 parser 剥离")
EOF
[ $? -eq 0 ] || { echo "!!! 预检1失败"; kill $SRV_PID; exit 2; }

echo "===== 预检 2/2: thinking + json_schema 约束共存 ====="
python3 - "$PORT" <<'EOF'
import json, sys, time, urllib.request
port = sys.argv[1]
schema = {"type":"object","properties":{"thought":{"type":"string"},"command":{"type":"string"}},
          "required":["thought","command"],"additionalProperties":False}
payload = {"model":"qwen3.5-9b",
  "messages":[{"role":"user","content":"You are in /testbed. List python files. Respond with a JSON object with fields 'thought' and 'command'."}],
  "max_tokens":8192,
  "response_format":{"type":"json_schema","json_schema":{"name":"action","schema":schema,"strict":True}}}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
t0=time.time(); r = json.load(urllib.request.urlopen(req, timeout=900)); dt=time.time()-t0
m = r["choices"][0]["message"]; rc = m.get("reasoning_content") or ""; c = m.get("content") or ""
fin = r["choices"][0]["finish_reason"]
print(f"耗时{dt:.0f}s | reasoning长度={len(rc)} | finish={fin} | content前120: {c[:120]!r}")
parsed = json.loads(c)  # content 必须是合法 JSON
assert set(parsed) == {"thought","command"}, f"FAIL: 字段不对 {list(parsed)}"
assert fin == "stop", f"FAIL: finish_reason={fin}(疑似无限生成被截断)"
print(f"预检2 PASS: thinking(思考{len(rc)}字符)与 json_schema 共存, JSON 合法")
EOF
[ $? -eq 0 ] || { echo "!!! 预检2失败: thinking 与约束解码冲突, 需换方案(绕过 response_format)"; kill $SRV_PID; exit 3; }

export CLBENCH_CONTAINER_BACKEND=singularity
export CLBENCH_SIF_DIR=/data/user_data/qixinx/clbench/sifs
export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export TMPDIR=/data/user_data/qixinx/tmp
export APPTAINER_CACHEDIR=/data/user_data/qixinx/tmp/apptainer_cache
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=dummy
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR"
cd "$REPO"

echo "===== smoke: schedule=smoke_test × icl × thinking ====="
"$REPO/.venv/bin/clbench" run --task codebase_adaptation --system icl \
  --task-params '{"schedule":"smoke_test"}' \
  --system-params '{"model":"openai/qwen3.5-9b","provider_mode":"litellm_chat","max_tokens":240000}' \
  --skip-baseline \
  --live-dashboard --no-live-server --verbose-runs
RC=$?
echo "===== smoke exit: $RC ====="
kill $SRV_PID 2>/dev/null
exit $RC
