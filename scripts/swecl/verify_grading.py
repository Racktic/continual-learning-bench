#!/usr/bin/env python3
"""端到端验证: 用 full.jsonl 真实行 + task 的判分路由，gold patch 应满分。
每个指定 instance_id: django/sympy 走官方 grading，其余走父类 returncode grading，
断言 gold patch → success。用法: uv run python scripts/swecl/verify_grading.py <iid> [<iid> ...]
"""
import json, os, sys

os.environ["CLBENCH_CONTAINER_BACKEND"] = "singularity"
os.environ["APPTAINER_CACHEDIR"] = "/scratch/qixinx/swecl/cache"
os.environ["APPTAINER_TMPDIR"] = "/scratch/qixinx/swecl/tmp"
os.environ["TMPDIR"] = "/scratch/qixinx/swecl/tmp"
sys.path.insert(0, "src")

# 与 task.__init__ 一致的 exec args（testbed PATH + UTF-8 locale）
from src.tasks.swe_bench_cl.task import _SWECL_SINGULARITY_EXEC_ARGS, _OFFICIAL_GRADING_REPOS  # noqa: E402
os.environ["CLBENCH_SINGULARITY_EXEC_ARGS"] = _SWECL_SINGULARITY_EXEC_ARGS

from src.tasks.codebase_adaptation.evaluator import evaluate_submission  # noqa: E402
from src.tasks.swe_bench_cl.official_grading import evaluate_official_submission  # noqa: E402

rows = {json.loads(l)["instance_id"]: json.loads(l)
        for l in open("data/swe_bench_cl/full.jsonl")}

ok = True
for iid in sys.argv[1:]:
    inst = rows[iid]
    route = "official" if inst["repo"] in _OFFICIAL_GRADING_REPOS else "returncode"
    if route == "official":
        res = evaluate_official_submission(inst["patch"], inst)
    else:
        res = evaluate_submission(model_patch=inst["patch"], instance=inst)
    good = bool(res.success)
    ok = ok and good
    print(f"[{'PASS' if good else 'FAIL'}] {iid} repo={inst['repo']} route={route} "
          f"success={res.success} status={getattr(res,'status',None)}")
    if not good:
        print("   error:", getattr(res, "error", None))
        print("   output_tail:", (getattr(res, "output", "") or "")[-500:])

print("===== 全部满分" if ok else "===== 有失败")
sys.exit(0 if ok else 1)
