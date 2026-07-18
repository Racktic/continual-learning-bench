#!/usr/bin/env python3
"""诊断单题官方 grading：为什么 gold patch 被判 tests_failed。
打印 eval_cmd、原始 log 末尾、解析出的 status_map(与 F2P/P2P 的交集)、report。
用法: uv run python scripts/swecl/diag_one.py <instance_id>
"""
import json, os, subprocess, sys

SC = "/scratch/qixinx/swecl"
os.environ["CLBENCH_CONTAINER_BACKEND"] = "singularity"
os.environ["APPTAINER_CACHEDIR"] = f"{SC}/cache"
os.environ["APPTAINER_TMPDIR"] = f"{SC}/tmp"
os.environ["TMPDIR"] = f"{SC}/tmp"
os.environ.setdefault(
    "CLBENCH_SINGULARITY_EXEC_ARGS",
    "--contain --cleanenv --fakeroot --no-mount hostfs,bind-paths "
    "--env PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
    "--env PYTHONIOENCODING=utf-8 --env LANG=C.UTF-8 --env LC_ALL=C.UTF-8",
)
sys.path.insert(0, "src")
from src.tasks.swe_bench_cl.official_grading import (  # noqa: E402
    official_eval_command, grade_log,
)
from src.tasks.codebase_adaptation.generic_runtime import (  # noqa: E402
    _apply_patch_text, _cleanup_container, _docker_exec,
    _start_generic_container, initialize_generic_pr_container,
    instance_cwd, strip_patch_paths, test_patch_paths,
)
from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: E402
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER  # noqa: E402

D = "/data/group_data/rl/yuxiaoq/qixinx"
iid = sys.argv[1]
inst = next(json.loads(l) for l in open(f"{D}/swecl_djsy.jsonl") if json.loads(l)["instance_id"] == iid)
VER = {t["instance_id"]: str(t["version"])
       for k, v in json.load(open("/home/qixinx/codebase-data/agents-never-forget/data/SWE-Bench-CL-Task-Stream.json")).items()
       if isinstance(v, list) for t in v if "instance_id" in t and "version" in t}

sif = f"{SC}/sifs/{iid}.sif"
if not os.path.exists(sif):
    img = f"docker://swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}"
    subprocess.run(["apptainer", "pull", "--force", sif, img], check=True, timeout=1800)
inst["image_name"] = sif
inst["version"] = VER.get(iid)
inst["workdir"] = "/testbed"
inst["runtime_mode"] = "git_pr"

cwd = instance_cwd(inst)
patch = strip_patch_paths(inst["patch"], test_patch_paths(inst))
cid = None
try:
    cid = _start_generic_container(sif, cwd=cwd, docker_executable="docker")
    initialize_generic_pr_container(cid, inst, docker_executable="docker", timeout=900)
    _apply_patch_text(cid, cwd, patch, timeout=900, docker_executable="docker")
    eval_cmd = official_eval_command(inst)
    print("EVAL_CMD:", eval_cmd)
    r = _docker_exec(cid, eval_cmd, cwd, timeout=900, docker_executable="docker")
    print("RETURNCODE:", r.returncode)
    print("=== LOG TAIL (2000) ===")
    print(r.stdout[-2000:])
    spec = make_test_spec(inst)
    smap = MAP_REPO_TO_PARSER[inst["repo"]](r.stdout, spec)
    print("=== PARSED status_map size:", len(smap))
    f2p, p2p = inst["FAIL_TO_PASS"], inst["PASS_TO_PASS"]
    print("F2P count:", len(f2p), "P2P count:", len(p2p))
    print("--- F2P 中缺失(parser没解析到)：")
    for t in f2p:
        if t not in smap:
            print("   MISSING:", repr(t))
    print("--- F2P 中非PASSED：")
    for t in f2p:
        if t in smap and smap[t] != "PASSED":
            print("  ", smap[t], repr(t))
    print("--- P2P 中非PASSED(前10)：")
    n = 0
    for t in p2p:
        if smap.get(t) != "PASSED":
            print("  ", smap.get(t, "MISSING"), repr(t)); n += 1
            if n >= 10: break
    res, report = grade_log(inst, r.stdout)
    print("RESOLUTION:", res)
finally:
    _cleanup_container(cid, docker_executable="docker")
