#!/usr/bin/env python3
"""pool.jsonl + test_patches -> swe_smith 任务数据集(generic_runtime 兼容, 2026-07-20)。

字段映射(设计见 notes/DATA_EXPANSION_SURVEY_0720 与讨论记录):
  base_commit   := instance 分支名(git checkout 对分支同样生效, bug 态即分支态)
  test_patch    := extract_test_patches.py 产物(分支->main 的测试文件 diff;
                   同时驱动判分链的防作弊剥离与测试恢复)
  image_name    := 该 repo 的 SIF 绝对路径(per-repo 共享)
  runtime_mode  := git_pr(走 codebase_adaptation 的 generic PR 全链路)
  patch         := ""(swesmith 的 bug 由分支承载, 无需注入补丁)
用法:
  python3 convert_pool.py --pool pool.jsonl --test-patch-dir .../test_patches \
      --sif-dir .../swe_smith_sifs --out swe_smith_pilot.jsonl [--repos r1 ...]
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--test-patch-dir", required=True)
    ap.add_argument("--sif-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repos", nargs="*", default=None)
    ap.add_argument("--test-timeout", type=int, default=300)
    args = ap.parse_args()

    tp = {}
    for f in glob.glob(os.path.join(args.test_patch_dir, "*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            tp[d["instance_id"]] = d["test_patch"]

    n_out = n_skip_tp = 0
    with open(args.out, "w") as fo:
        for line in open(args.pool):
            d = json.loads(line)
            rs = d["repo"].split("/")[-1]
            if args.repos and rs not in args.repos:
                continue
            iid = d["instance_id"]
            if iid not in tp:
                n_skip_tp += 1
                continue
            sif = os.path.join(args.sif_dir, rs.split("__")[-1] + ".sif")
            fo.write(json.dumps({
                "instance_id": iid,
                "repo": d["repo"],
                "base_commit": iid,          # 分支名即 bug 态
                "patch": "",
                "test_patch": tp[iid],
                # Strip the upstream "../dev/" CWD quirk (mido): pytest ids must
                # resolve from workdir=/testbed for both grading and anti-cheat.
                "FAIL_TO_PASS": [x.removeprefix("../dev/") for x in d["FAIL_TO_PASS"]],
                "PASS_TO_PASS": [x.removeprefix("../dev/") for x in d["PASS_TO_PASS"]],
                "runtime_mode": "git_pr",
                "problem_statement": d["problem_statement"],
                "image_name": sif,
                "test_command": "python -m pytest",
                "workdir": "/testbed",
                "test_timeout": args.test_timeout,
                "init_submodules": False,
                # 组序列会用到的派生特征(透传)
                "strategy": d["strategy"],
                "tier": d["tier"],
                "cluster": d["cluster"],
                "n_f2p": d["n_f2p"],
                "n_p2p_local": d["n_p2p_local"],
            }, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"写出 {n_out} 题 -> {args.out}(缺 test_patch 跳过 {n_skip_tp})")


if __name__ == "__main__":
    main()
