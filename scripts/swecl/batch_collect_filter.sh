#!/bin/bash
# 对 dataset jsonl 每题跑 collect_filter_one, 并行, 汇总 collectable map
# 用法: batch_collect_filter.sh <in.jsonl> <out_collectable.jsonl> [par=4]
set -uo pipefail
JSONL=$1; OUT=$2; PAR=${3:-4}
SC=/scratch/qixinx/swecl; mkdir -p "$SC/cflines"
HERE=$(dirname "$0")
rm -f "$SC/cflines"/*.json
python3 -c "
import json
for i,l in enumerate(open('$JSONL')):
    open('$SC/cflines/%04d.json'%i,'w').write(l)
print('拆分', i+1, '题')
"
: > "$OUT"
ls $SC/cflines/*.json | xargs -P "$PAR" -I{} bash "$HERE/collect_filter_one.sh" {} >> "$OUT" 2>/dev/null
python3 -c "
import json
rows=[json.loads(l) for l in open('$OUT') if l.strip()]
ok=[r for r in rows if 'error' not in r]
err=[r for r in rows if 'error' in r]
tot_drop=sum(len(r.get('dropped_f2p',[]))+len(r.get('dropped_p2p',[])) for r in ok)
print(f'===== collectable 汇总: {len(rows)} 题, 正常 {len(ok)}, 错误 {len(err)}, 共丢弃畸形 node {tot_drop} =====')
for r in ok:
    nd=len(r.get('dropped_f2p',[]))+len(r.get('dropped_p2p',[]))
    if nd: print(f\"  {r['instance_id']}: 丢 {nd} 条\")
for r in err: print(f\"  ERR {r['instance_id']}: {r['error']}\")
"
