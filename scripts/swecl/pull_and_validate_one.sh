#!/bin/bash
# 单题: pull官方镜像→SIF(本地盘)→空跑+gold跑双向验证(退出码)→clean拷/data,dirty删
# 用法: pull_and_validate_one.sh <instance_json_file>
# 输出 stdout 一行 JSON: {instance_id, pull_rc, sif_mb, clean, reason, diag, elapsed}
set -uo pipefail
LINE_FILE=$1
SC=/scratch/qixinx/swecl
DEST=/data/group_data/rl/yuxiaoq/qixinx/swebench_sifs
export APPTAINER_CACHEDIR=$SC/cache APPTAINER_TMPDIR=$SC/tmp TMPDIR=$SC/tmp
mkdir -p "$SC/sifs" "$SC/cache" "$SC/tmp" "$SC/patches" "$DEST"

python3 - "$LINE_FILE" "$SC" <<'PY'
import json, sys, os, subprocess, time, shlex
lf, sc = sys.argv[1], sys.argv[2]
t=json.load(open(lf)); iid=t["instance_id"]; img=t["image_name"]
sif=f"{sc}/sifs/{iid}.sif"; dest=f"/data/group_data/rl/yuxiaoq/qixinx/swebench_sifs/{iid}.sif"
pdir=f"{sc}/patches/{iid}"; os.makedirs(pdir,exist_ok=True)
open(f"{pdir}/test.patch","w").write(t["test_patch"]); open(f"{pdir}/gold.patch","w").write(t["patch"])
f2p,p2p=t["FAIL_TO_PASS"],t["PASS_TO_PASS"]
open(f"{pdir}/f2p.txt","w").write("\n".join(f2p)); open(f"{pdir}/p2p.txt","w").write("\n".join(p2p))
def out(**k): k["instance_id"]=iid; print(json.dumps(k)); sys.exit(0)
t0=time.time()
if not os.path.exists(sif):
    r=subprocess.run(["apptainer","pull","--force",sif,f"docker://{img}"],
                     capture_output=True,text=True,timeout=1200)
    if r.returncode!=0: out(pull_rc=r.returncode,clean=False,reason="pull_failed",
                            err=r.stderr[-200:],elapsed=round(time.time()-t0))
sif_mb=round(os.path.getsize(sif)/1e6)
def run(apply_gold, targets):
    steps="cd /testbed && git apply /P/test.patch"
    if apply_gold: steps+=" && git apply /P/gold.patch"
    # 容器内 python: ①测试名含空格/方括号/%(参数化), 传 list 绕开 shell 拆词;
    # ②先 collect-only 过滤掉 pytest 选不中的畸形 node(如空参数 test_x[]), 否则单条
    #   not-found 会让整条 pytest abort(exit 4) 带崩其余上百条; 过滤数量打印 DROP=N。
    runner=(
        "import subprocess,sys\n"
        f"ts=[x for x in open('/P/{targets}').read().split(chr(10)) if x]\n"
        "c=subprocess.run(['python','-m','pytest','--collect-only','-q',"
        "'--continue-on-collection-errors']+ts,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)\n"
        "co=c.stdout.decode()\n"
        "bad=set()\n"
        "for ln in co.splitlines():\n"
        "    if 'not found:' in ln: bad.add(ln.split('not found:')[1].strip())\n"
        "good=[t for t in ts if not any(t in b or b.endswith('::'+t) or b.endswith(t) for b in bad)]\n"
        "print('DROP=%d'%(len(ts)-len(good)))\n"
        "r=subprocess.run(['python','-m','pytest']+good+['-p','no:cacheprovider','-q'])\n"
        "sys.exit(r.returncode)\n"
    )
    cmd=(f"{steps} && source /opt/miniconda3/bin/activate testbed && "
         f"python -c {shlex.quote(runner)} 2>&1; echo RC=$?")
    r=subprocess.run(["apptainer","exec","--contain","--writable-tmpfs","--bind",f"{pdir}:/P",
                      sif,"bash","-c",cmd],capture_output=True,text=True,timeout=1800)
    txt=r.stdout+r.stderr
    rc=None; dropped=0
    for ln in txt.splitlines():
        if ln.startswith("RC="): rc=int(ln[3:])
        if ln.startswith("DROP="): dropped=int(ln[5:])
    import re
    m=re.search(r"(\d+) passed",txt); passed=int(m.group(1)) if m else 0
    m=re.search(r"(\d+) failed",txt); failed=int(m.group(1)) if m else 0
    m=re.search(r"(\d+) error",txt); errored=int(m.group(1)) if m else 0
    return rc,passed,failed,errored,dropped
try:
    # 空跑: 只跑 F2P(应有失败, rc!=0) + 单独跑 P2P(应全过, rc==0)
    pre_f2p_rc,pf,pff,pfe,d1=run(False,"f2p.txt")
    pre_p2p_rc,pp,ppf,ppe,d2=run(False,"p2p.txt")
    # gold跑: F2P+P2P 合跑(应全过, rc==0) — 与最终判定一致
    open(f"{pdir}/all.txt","w").write("\n".join(f2p+p2p))
    post_rc,ap,af,ae,d3=run(True,"all.txt")
except subprocess.TimeoutExpired:
    out(pull_rc=0,sif_mb=sif_mb,clean=False,reason="test_timeout",elapsed=round(time.time()-t0))
diag={"pre_f2p_rc":pre_f2p_rc,"pre_f2p_failed":pff,"pre_p2p_rc":pre_p2p_rc,"pre_p2p_passed":pp,
      "post_rc":post_rc,"post_passed":ap,"post_failed":af,"f2p_n":len(f2p),"p2p_n":len(p2p),
      "dropped":{"pre_f2p":d1,"pre_p2p":d2,"post":d3}}
# clean: 空跑F2P有失败(rc!=0) + 空跑P2P全过(rc==0) + gold后合跑全过(rc==0)
clean = (pre_f2p_rc not in (0,None)) and (pre_p2p_rc==0) and (post_rc==0)
reason="ok" if clean else "failed_validation"
if clean:
    subprocess.run(["cp",sif,dest]); os.remove(sif)   # 拷到/data后删scratch副本(collect_filter读/data)
else: os.remove(sif)
out(pull_rc=0,sif_mb=sif_mb,clean=clean,reason=reason,diag=diag,elapsed=round(time.time()-t0))
PY
