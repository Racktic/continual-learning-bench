#!/usr/bin/env python3
"""SWE-smith instance pool 构建(先池后序工作流, 2026-07-20)。

对全部 repo 应用五道过滤(F2P∈[1,20], 改动文件∈[1,3], 局部化P2P∈[3,500],
problem_statement>=100字), 产出:
  pool.jsonl     全部合格 instance(含 tier/cluster/局部化P2P 等派生特征)
  repo_stats.tsv 每 repo 的合格题数/簇数(选 repo 与拉 SIF 的依据)
序列构造推迟到 pass@k baseline 之后(经验难度替代启发式 tier)。
"""
import argparse
import collections
import glob
import json
import os
import re

import pyarrow.parquet as pq

# heldout 污染排除(2026-07-20): jd/tenacity 是 codebase_adaptation 域外探针 repo,
# 训练池混入其合成 bug 会破坏"未见 repo"前提。tablib 不在 SWE-smith 中。
EXCLUDED_REPOS = {"jd__tenacity.0d40e76f"}

STRAT_TIER = {
    "func_basic": 0, "other": 0,
    "func_pm_ctrl_shuffle": 1, "func_pm_ctrl_invert_if": 1, "func_pm_remove_cond": 1,
    "func_pm_op_swap": 1, "func_pm_op_change": 1, "func_pm_op_change_const": 1,
    "func_pm_op_break_chains": 1, "func_pm_remove_wrapper": 1, "lm_rewrite": 1, "lm_modify": 1,
    "func_pm_remove_assign": 2, "func_pm_remove_loop": 2,
    "func_pm_class_rm_funcs": 3, "func_pm_class_rm_base": 3, "func_pm_class_shuffle_funcs": 3,
    "combine_file": 4, "combine_module": 5,
}


def strategy_of(iid):
    m = re.search(r"\.[0-9a-f]{8}\.([a-z_]+?)__", iid)
    return m.group(1) if m else "other"


def module_cluster(patch):
    m = re.search(r"^diff --git a/(\S+)", patch, re.M)
    if not m:
        return "?"
    parts = m.group(1).split("/")
    if parts and parts[0] in ("src", "lib"):
        parts = parts[1:]
    # 扁平包(如 src/jinja2/x.py)降级到按文件聚簇
    return "/".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    n_total = n_pool = 0
    repo_stats = collections.defaultdict(lambda: [0, set()])
    with open(os.path.join(args.out_dir, "pool.jsonl"), "w") as fp:
        for f in sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet"))):
            t = pq.read_table(f)
            cols = {c: t.column(c).to_pylist() for c in t.column_names}
            for i in range(t.num_rows):
                n_total += 1
                f2p = cols["FAIL_TO_PASS"][i]
                f2p = json.loads(f2p) if isinstance(f2p, str) else f2p
                p2p = cols["PASS_TO_PASS"][i]
                p2p = json.loads(p2p) if isinstance(p2p, str) else p2p
                patch = cols["patch"][i]
                ps = (cols["problem_statement"][i] or "").strip()
                nfile = patch.count("diff --git")
                f2p_files = {x.split("::")[0] for x in f2p}
                p2p_local = [x for x in p2p if x.split("::")[0] in f2p_files]
                if not (1 <= len(f2p) <= 20 and 3 <= len(p2p_local) <= 500
                        and 1 <= nfile <= 3 and len(ps) >= 100):
                    continue
                n_pool += 1
                iid = cols["instance_id"][i]
                strat = strategy_of(iid)
                cluster = module_cluster(patch)
                repo_short = cols["repo"][i].split("/")[-1]
                if repo_short in EXCLUDED_REPOS:
                    continue
                repo_stats[repo_short][0] += 1
                repo_stats[repo_short][1].add(cluster)
                fp.write(json.dumps({
                    "instance_id": iid,
                    "repo": cols["repo"][i],
                    "image_name": cols["image_name"][i],
                    "problem_statement": ps,
                    "FAIL_TO_PASS": f2p,
                    "PASS_TO_PASS": p2p_local,
                    "strategy": strat,
                    "tier": STRAT_TIER.get(strat, 1),
                    "n_f2p": len(f2p),
                    "n_p2p_local": len(p2p_local),
                    "n_files": nfile,
                    "cluster": cluster,
                    "branch": iid,
                    "workdir": "/testbed",
                    "conda_env": "testbed",
                }, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out_dir, "repo_stats.tsv"), "w") as fs:
        fs.write("repo\tn_pool\tn_clusters\n")
        for r, (n, cl) in sorted(repo_stats.items(), key=lambda kv: -kv[1][0]):
            fs.write(f"{r}\t{n}\t{len(cl)}\n")

    print(f"总题 {n_total} -> 池 {n_pool} ({n_pool/n_total:.0%}), 覆盖 repo {len(repo_stats)}")
    ge = {th: sum(1 for _, (n, _) in repo_stats.items() if n >= th) for th in (12, 24, 48, 96)}
    print("合格题数≥12/24/48/96 的 repo 数:", ge)


if __name__ == "__main__":
    main()
