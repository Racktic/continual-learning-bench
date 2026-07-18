# continual-learning-bench 集群运行环境(apptainer 后端 + 本地 sglang 模型)。
# 用法: source scripts/cluster/env.sh
# 前提: 在计算节点上(登录节点没有 apptainer); serve 作业已就绪(serve_endpoint.txt 存在)。

export CLBENCH_CONTAINER_BACKEND=singularity
export CLBENCH_SIF_DIR=/data/user_data/qixinx/clbench/sifs
export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export TMPDIR=/data/user_data/qixinx/tmp
export APPTAINER_CACHEDIR=/data/user_data/qixinx/tmp/apptainer_cache
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR"

_EP_FILE=/data/user_data/qixinx/clbench/serve_endpoint.txt
if [ -f "$_EP_FILE" ]; then
  export OPENAI_API_BASE="http://$(cat "$_EP_FILE")/v1"
  export OPENAI_API_KEY=dummy
  echo "clbench env ready; model endpoint: $OPENAI_API_BASE"
else
  echo "WARNING: $_EP_FILE 不存在 — serve 作业未就绪(先 sbatch scripts/cluster/serve_qwen35_4b.sbatch), OPENAI_API_BASE 未设置" >&2
fi

# 跑 icl + 本地模型的关键参数:
#   --system-params '{"model":"openai/qwen3.5-4b","provider_mode":"litellm_chat"}'
# provider_mode=litellm_chat 必须带: 否则 openai/ 前缀会被当成真 OpenAI 走 Responses API。
