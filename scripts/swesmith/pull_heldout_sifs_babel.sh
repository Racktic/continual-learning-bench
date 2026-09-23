#!/usr/bin/env bash
# 为 held-out repo(top53 之外)拉 per-repo SIF(babel,2026-09-20)。
# 镜像名直接取 pool.jsonl 的 image_name 字段(jyangballin/swesmith.x86_64.<owner>_1776_<repo>.<commit8>),
# 落点与现有 53 个同目录、同命名规则(<repo>.<commit8>.sif),这样 extract_test_patches / convert_pool /
# 运行时的 CLBENCH_SIF_DIR 都不用改。已存在的跳过,可反复重跑。
# 用法: N_REPOS=12 bash scripts/swesmith/pull_heldout_sifs_babel.sh
set -uo pipefail
unset SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS GIT_SSL_CAINFO PIP_CERT
N=${N_REPOS:-12}
SIFDIR=/data/group_data/rl/yuxiaoq/qixinx/swesmith_sifs
POOL=/scratch/qixinx/swesmith_pool_v2/pool.jsonl
RANK=/home/qixinx/heldout_prep/heldout_repos.json
SCR=/scratch/qixinx
export APPTAINER_TMPDIR=$SCR/apptainer_tmp APPTAINER_CACHEDIR=$SCR/apptainer_cache TMPDIR=$SCR/tmp
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$TMPDIR" "$SIFDIR"
echo "=== $(date) host=$(hostname) 拉前 $N 个 held-out repo 的 SIF ==="

python3 - "$RANK" "$POOL" "$N" > /home/qixinx/heldout_prep/pull_list.tsv <<'PY'
import json, sys
rank, pool, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = json.load(open(rank))[:n]
want = {r["repo"] for r in rows}
img = {}
for line in open(pool):
    d = json.loads(line)
    s = d["repo"].split("/")[-1]
    if s in want and s not in img:
        img[s] = d["image_name"]
for r in rows:
    print(f'{r["repo"]}\t{img.get(r["repo"], "MISSING")}\t{r["n"]}\t{r["funcs"]}')
PY
cat /home/qixinx/heldout_prep/pull_list.tsv
echo

ok=0; fail=0; skip=0
while IFS=$'\t' read -r repo image n funcs; do
  sif="$SIFDIR/${repo##*__}.sif"
  if [ -f "$sif" ]; then echo "  [skip] $repo -> $(basename "$sif") 已存在"; skip=$((skip+1)); continue; fi
  [ "$image" = "MISSING" ] && { echo "  [fail] $repo 没找到 image_name"; fail=$((fail+1)); continue; }
  echo "  [pull] $repo  <- docker://$image"
  if timeout 2400 apptainer pull --force "$sif" "docker://$image" >>/home/qixinx/heldout_prep/pull.log 2>&1; then
    echo "         ✓ $(du -h "$sif" | cut -f1)  ($n 题 / $funcs 函数)"; ok=$((ok+1))
  else
    echo "         ❌ 失败,尾部日志:"; tail -3 /home/qixinx/heldout_prep/pull.log; rm -f "$sif"; fail=$((fail+1))
  fi
done < /home/qixinx/heldout_prep/pull_list.tsv

echo; echo "=== $(date) 完成: ok=$ok skip=$skip fail=$fail | SIF 总数 $(ls "$SIFDIR"/*.sif 2>/dev/null | wc -l) ==="
