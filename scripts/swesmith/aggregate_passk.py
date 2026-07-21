#!/usr/bin/env python3
"""合并 pass@k 各批 traj -> per-instance 通过率(v2 难度)+ baseline artifact(2026-07-21)。

输入: 若干个 eval traj 目录(ep_*.json, 单题 episode)。
口径: rollout_step_budget 剔除(预算腰斩非题目难度); error/timeout 计失败(保守)。
输出:
  difficulty.jsonl   {instance_id, n_trials, n_success, passrate, repo, strategy}
  baseline.json      与 baseline_4b_textfmt.json 同构(instance_outcomes: [{instance_id, reward}])
                     reward 取该题各 trial reward 均值(gain 管线口径)
  summary 打印: 总体/分repo/分策略
用法: python3 aggregate_passk.py --traj-dirs <dir1> <dir2> ... --out-dir <dir>
"""
import argparse
import collections
import glob
import json
import os
import re

STRAT_RE = re.compile(r"\.[0-9a-f]{8}\.([a-z_]+?)__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    per = collections.defaultdict(lambda: {"n": 0, "s": 0, "rewards": []})
    dropped = 0
    for d in args.traj_dirs:
        for f in glob.glob(os.path.join(d, "ep_*.json")):
            data = json.load(open(f))
            for o in data.get("outcomes") or []:
                iid = o.get("instance_id")
                if str(o.get("eval_status")) == "rollout_step_budget":
                    dropped += 1
                    continue
                rec = per[iid]
                rec["n"] += 1
                rec["s"] += 1 if o.get("success") else 0
                rec["rewards"].append(float(o.get("reward") or 0.0))

    diff_path = os.path.join(args.out_dir, "difficulty.jsonl")
    base_path = os.path.join(args.out_dir, "baseline.json")
    by_repo = collections.defaultdict(lambda: [0, 0])
    by_strat = collections.defaultdict(lambda: [0, 0])
    outcomes = []
    with open(diff_path, "w") as fd:
        for iid, rec in sorted(per.items()):
            if rec["n"] == 0:
                continue
            pr = rec["s"] / rec["n"]
            repo = iid.split(".")[0]
            m = STRAT_RE.search(iid)
            strat = m.group(1) if m else "other"
            fd.write(json.dumps({
                "instance_id": iid, "n_trials": rec["n"], "n_success": rec["s"],
                "passrate": round(pr, 4), "repo": repo, "strategy": strat,
            }) + "\n")
            by_repo[repo][0] += rec["n"]
            by_repo[repo][1] += rec["s"]
            by_strat[strat][0] += rec["n"]
            by_strat[strat][1] += rec["s"]
            outcomes.append({"instance_id": iid,
                             "reward": round(sum(rec["rewards"]) / len(rec["rewards"]), 4)})
    json.dump({"instance_outcomes": outcomes}, open(base_path, "w"), indent=1)

    n_inst = len(outcomes)
    tot_n = sum(v["n"] for v in per.values())
    tot_s = sum(v["s"] for v in per.values())
    p_at_k = sum(1 for v in per.values() if v["n"] and v["s"] > 0) / max(n_inst, 1)
    print(f"instance={n_inst} trial={tot_n} (剔腰斩 {dropped}) pass@1={tot_s/max(tot_n,1):.1%} pass@k={p_at_k:.1%}")
    for r, (n, s) in sorted(by_repo.items()):
        print(f"  repo {r:12s} n={n:5d} pass@1={s/n:.0%}")
    for st, (n, s) in sorted(by_strat.items(), key=lambda kv: -(kv[1][1] / kv[1][0])):
        print(f"  strat {st:24s} n={n:5d} pass@1={s/n:.0%}")
    print(f"输出: {diff_path} / {base_path}")


if __name__ == "__main__":
    main()
