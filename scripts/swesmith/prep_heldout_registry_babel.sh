#!/usr/bin/env bash
# held-out repo 的判分数据与 episode 生成(babel,2026-09-20)。SIF 拉完后跑这一个脚本即可:
#   1. extract_test_patches.py  每 repo 一次 apptainer exec,算 branch->main 的测试 diff
#   2. convert_pool.py          组装 generic PR runtime 格式的 registry(image_name 指向本地 SIF)
#   3. gen_episodes_v4.py --report  各候选段配方能产多少条 episode(不落盘,给人挑)
# 用法: bash scripts/swesmith/prep_heldout_registry_babel.sh
set -uo pipefail
unset SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS GIT_SSL_CAINFO PIP_CERT
CLB=/home/qixinx/continual-learning-bench
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
SIFDIR=/data/group_data/rl/yuxiaoq/qixinx/swesmith_sifs
POOL=/scratch/qixinx/swesmith_pool_v2/pool.jsonl
PQ=/scratch/qixinx/swesmith_parquet/data
OUT=/data/user_data/qixinx/swe_smith/heldout_v4
PREP=/home/qixinx/heldout_prep
export APPTAINER_TMPDIR=/scratch/qixinx/apptainer_tmp APPTAINER_CACHEDIR=/scratch/qixinx/apptainer_cache TMPDIR=/scratch/qixinx/tmp
mkdir -p "$OUT/test_patches" "$TMPDIR"
# repo 名单 = 全部 held-out repo 里 SIF 真的存在的那些(不依赖某一批的 pull_list)
python3 -c "
import json
rows = json.load(open('$PREP/heldout_repos.json'))
open('$PREP/repos.txt','w').write('\n'.join(r['repo'] for r in rows) + '\n')
"
echo "=== $(date) host=$(hostname) held-out 候选 repo: $(wc -l < "$PREP/repos.txt") ==="
: > "$PREP/repos_ready.txt"
while read -r r; do
  [ -f "$SIFDIR/${r##*__}.sif" ] && echo "$r" >> "$PREP/repos_ready.txt"
done < "$PREP/repos.txt"
READY=$(tr '\n' ' ' < "$PREP/repos_ready.txt")
echo "SIF 就绪: $(wc -l < "$PREP/repos_ready.txt") 个 -> $READY"
[ -s "$PREP/repos_ready.txt" ] || { echo "❌ 没有可用 SIF,退出"; exit 2; }

echo; echo "=== 1/3 extract_test_patches.py ==="
python3 "$CLB/scripts/swesmith/extract_test_patches.py" --pool "$POOL" --sif-dir "$SIFDIR" \
  --out-dir "$OUT/test_patches" --repos $READY || exit 3
wc -l "$OUT"/test_patches/*.jsonl 2>/dev/null | tail -3

echo; echo "=== 2/3 convert_pool.py ==="
python3 "$CLB/scripts/swesmith/convert_pool.py" --pool "$POOL" --test-patch-dir "$OUT/test_patches" \
  --sif-dir "$SIFDIR" --out "$OUT/heldout_registry.jsonl" --repos $READY || exit 4
echo "registry 行数: $(wc -l < "$OUT/heldout_registry.jsonl")"

echo; echo "=== 3/3 gen_episodes_v4.py --report(各段配方产量) ==="
apptainer exec --bind /data,/home/qixinx,/scratch "$SIF" bash -c "
  cd $CLB/scripts/swesmith && python3 gen_episodes_v4.py --instances $OUT/heldout_registry.jsonl \
    --parquet-dir $PQ --repos-file $PREP/repos_ready.txt --report"
echo "=== $(date) DONE ==="
