#!/bin/bash
# 批量: 对 jsonl 每题跑 pull_and_validate_one, 并行 N 路, 汇总 clean/dirty
# 用法: batch_pull_validate.sh <subset.jsonl> <out_report.jsonl> [parallel=4]
set -uo pipefail
JSONL=$1; REPORT=$2; PAR=${3:-4}
SC=/scratch/qixinx/swecl; mkdir -p "$SC/lines"
HERE=$(dirname "$0")
rm -f "$SC/lines"/*.json   # 清旧拆分文件, 避免混入上轮实例
# 拆成单行文件
python3 -c "
import json,sys
for i,l in enumerate(open('$JSONL')):
    t=json.loads(l); open('$SC/lines/%04d.json'%i,'w').write(l)
print('拆分', i+1, '题')
"
ls $SC/lines/*.json | xargs -P "$PAR" -I{} bash "$HERE/pull_and_validate_one.sh" {} >> "$REPORT" 2>/dev/null
echo "===== 汇总 ====="
python3 -c "
import json
rows=[json.loads(l) for l in open('$REPORT') if l.strip()]
clean=[r for r in rows if r.get('clean')]
dirty=[r for r in rows if not r.get('clean')]
print(f'总 {len(rows)} | clean {len(clean)} | dirty {len(dirty)}')
if clean:
    import statistics
    print(f'  clean SIF 平均 {statistics.mean(r[\"sif_mb\"] for r in clean):.0f}MB, 单题平均 {statistics.mean(r[\"elapsed\"] for r in clean):.0f}s')
for r in dirty:
    print(f'  DIRTY {r[\"instance_id\"]}: {r.get(\"reason\")} {r.get(\"diag\",\"\")}')
"
