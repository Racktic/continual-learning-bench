#!/bin/bash
# env re-test of matplotlib/django/sympy baseline with Opus 4.6, to verify the
# three env fixes (matplotlib pytest-rerunfailures, django/sympy pytest hint).
# Runs as an --overlap step on the live sequence job so no new allocation needed.
set -euo pipefail
cd /home/qixinx/continual-learning-bench
set -a; source /home/qixinx/miles/.env; set +a
export CLBENCH_CONTAINER_BACKEND=singularity MSWEA_SINGULARITY_EXECUTABLE=apptainer PYTHONUNBUFFERED=1
L=/dev/shm/qixinx/env_retest
export TMPDIR=$L/tmp APPTAINER_TMPDIR=$L/aptmp APPTAINER_CACHEDIR=$L/apcache SINGULARITY_TMPDIR=$L/aptmp
mkdir -p "$TMPDIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
df -h /dev/shm | tail -1
.venv/bin/clbench run --task swe_bench_cl --system icl \
  --task-params '{"schedule":"env_retest"}' \
  --system-params '{"model":"anthropic/claude-opus-4-6","provider_mode":"litellm_chat","max_tokens":32000}' \
  --baseline-only --max-workers 4 --live-dashboard --no-live-server --verbose-runs
