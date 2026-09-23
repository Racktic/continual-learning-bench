#!/usr/bin/env bash
# 重建 SWE-smith 全量池子并找出 top53 之外的 held-out repo(babel,2026-09-20)。
# 与 notes/SWESMITH_DATA_CONSTRUCTION.md §3-4 的原流程完全一致,只是换到 babel 路径:
#   HF snapshot(SWE-bench/SWE-smith parquet) -> build_pool.py -> pool.jsonl + repo_stats.tsv
# 产物小文件拷回 /data/user_data/qixinx/swe_smith/pool_v2/ 以便后续步骤复用。
set -uo pipefail
unset SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS GIT_SSL_CAINFO PIP_CERT
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
CLB=/home/qixinx/continual-learning-bench
SCR=/scratch/qixinx
PQ=$SCR/swesmith_parquet
POOL=$SCR/swesmith_pool_v2
KEEP=/data/user_data/qixinx/swe_smith/pool_v2
mkdir -p "$PQ" "$POOL" "$KEEP" "$SCR/hf_home"
export HF_HOME=$SCR/hf_home
echo "=== $(date) host=$(hostname) ==="

apptainer exec --bind /data,/home/qixinx,/scratch "$SIF" env HF_HOME=$SCR/hf_home python3 - "$PQ" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
dst = sys.argv[1]
p = snapshot_download(repo_id="SWE-bench/SWE-smith", repo_type="dataset",
                      allow_patterns=["*.parquet"], local_dir=dst, max_workers=8)
tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
print(f"[hf] downloaded to {p}  {tot/1e9:.2f} GB")
for r, _, fs in os.walk(p):
    for f in sorted(fs)[:10]:
        print("   ", os.path.relpath(os.path.join(r, f), p), f"{os.path.getsize(os.path.join(r,f))/1e6:.1f} MB")
PY
rc=$?; echo "[hf] rc=$rc"; [ $rc -ne 0 ] && exit $rc

# HF snapshot 把 parquet 放在 <dir>/data/ 下,而 build_pool.py 只 glob <parquet-dir>/*.parquet
PQD=$PQ; [ -d "$PQ/data" ] && PQD=$PQ/data
echo "=== build_pool.py(五道过滤,与原流程同) parquet-dir=$PQD ==="
apptainer exec --bind /data,/home/qixinx,/scratch "$SIF" bash -c "
  cd $CLB && python3 scripts/swesmith/build_pool.py --parquet-dir $PQD --out-dir $POOL" || exit 3
cp -f "$POOL/pool.jsonl" "$POOL/repo_stats.tsv" "$KEEP/" 2>/dev/null
ls -la "$KEEP"
echo "=== $(date) DONE ==="
