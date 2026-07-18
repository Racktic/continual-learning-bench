#!/bin/bash
# 单题: 在 SIF 里打 test_patch+gold, collect-only 出可选中的 F2P/P2P node,
# 输出一行 JSON: {instance_id, f2p:[可选中], p2p:[可选中], dropped_f2p:[...], dropped_p2p:[...]}
# 用法: collect_filter_one.sh <instance_json_file>
set -uo pipefail
LF=$1
SIFDIR=/data/group_data/rl/yuxiaoq/qixinx/swebench_sifs
SC=/scratch/qixinx/swecl
export APPTAINER_CACHEDIR=$SC/cache APPTAINER_TMPDIR=$SC/tmp TMPDIR=$SC/tmp
mkdir -p "$SC/cfpatches"

python3 - "$LF" "$SIFDIR" "$SC" <<'PY'
import json, sys, os, subprocess, shlex
lf, sifdir, sc = sys.argv[1], sys.argv[2], sys.argv[3]
t = json.load(open(lf)); iid = t["instance_id"]
sif = f"{sifdir}/{iid}.sif"
pdir = f"{sc}/cfpatches/{iid}"; os.makedirs(pdir, exist_ok=True)
open(f"{pdir}/test.patch","w").write(t["test_patch"])
open(f"{pdir}/gold.patch","w").write(t["patch"])
f2p, p2p = t["FAIL_TO_PASS"], t["PASS_TO_PASS"]
open(f"{pdir}/all.txt","w").write("\n".join(f2p+p2p))
tc = t.get("test_command", "/opt/miniconda3/envs/testbed/bin/python -m pytest")

def out(**kw): kw["instance_id"]=iid; print(json.dumps(kw)); sys.exit(0)

if not os.path.exists(sif):
    out(error="sif_missing")

# 容器内: 打 test_patch+gold, 按【文件】collect-only(不按 node 选择, 避免 not-found abort),
# 拿到每个测试文件的全部真实 node id。畸形 node(如 test_x[]) 不在真实集合里 → 后续被交集剔除。
runner = (
    "import subprocess\n"
    "ts=[x for x in open('/P/all.txt').read().split(chr(10)) if x]\n"
    # 只从含 '::' 的真 node 提取文件；像 '[100%]' 这种无 '::' 的假条目(SWE-bench 把 pytest
    # 进度输出误收进 P2P)会让 pytest 当文件名收集而整体 abort，必须排除。
    "files=sorted({x.split('::',1)[0] for x in ts if '::' in x})\n"
    f"base={shlex.split(tc)!r}\n"
    "c=subprocess.run(base+['--collect-only','-q','--color=no',"
    "'--continue-on-collection-errors']+files,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)\n"
    "print('=COLLECTED_START=')\n"
    "print(c.stdout.decode())\n"
)
cmd = ("cd /testbed && git apply /P/test.patch && git apply /P/gold.patch && "
       f"python -c {shlex.quote(runner)}")
r = subprocess.run(["apptainer","exec","--contain","--writable-tmpfs","--bind",f"{pdir}:/P",
                    sif,"bash","-c",cmd], capture_output=True, text=True, timeout=600)
txt = r.stdout + r.stderr
body = txt.split("=COLLECTED_START=",1)[-1]
collected = set()
for ln in body.splitlines():
    s = ln.strip()
    if "::" in s and " " not in s:
        collected.add(s)

def keep(node):
    if node in collected: return True
    return any(c.endswith(node) or node.endswith(c) for c in collected)

if not collected:
    out(error="collect_empty", raw_tail=txt[-300:])

kf2p=[x for x in f2p if keep(x)]; kp2p=[x for x in p2p if keep(x)]
out(f2p=kf2p, p2p=kp2p,
    dropped_f2p=[x for x in f2p if x not in kf2p],
    dropped_p2p=[x for x in p2p if x not in kp2p])
PY
