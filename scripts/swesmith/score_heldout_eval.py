#!/usr/bin/env python3
"""held-out 长序列评测打分(2026-09-21)。clbench 口径,按 sequence 分组。

reward 用的就是官方 regret 公式(codebase_rollout.py:1154):
    regret = max(0, turns_used - 1 - STEP_GRACE);  reward = 1 - regret / max_steps  (未解出 = 0)
本轮评测 STEP_GRACE=0、max_steps=40,所以 solved 时 reward = 1 - (turns-1)/40。

题目归属哪条 sequence 从 FINAL_*.jsonl 反查(轨迹里没保留源 sequence 字段)。
replace 模式额外按题序分箱,用来看"后段"表现 —— 这是整个实验的主要目的。

用法:
  python3 score_heldout_eval.py                          # 自动发现所有 hoeval-* run
  python3 score_heldout_eval.py --runs hoeval-xxx ...    # 指定
  python3 score_heldout_eval.py --json out.json --md out.md
"""
import argparse
import collections
import glob
import json
import os
import statistics

RUNROOT = "/data/group_data/rl/yuxiaoq/qixinx/codebase_adaption_runs"
SEQDIR = "/data/user_data/qixinx/swe_smith/heldout_v4"
SEQFILES = ("FINAL_2x13_26issues.jsonl", "FINAL_3x9_27issues.jsonl")
MAX_STEPS = 40


def load_seq_map():
    """{数据集 tag: {instance_id: (tag, episode 下标, 题序 pos, repo)}}

    必须按数据集分开存:两份 FINAL 之间有 9 道题重叠,合成一张表会被后读的文件覆盖,
    导致 26 题集的 run 里有题被算到 27 题集的 sequence 上(2026-09-21 踩到)。
    """
    m = collections.defaultdict(dict)
    for f in SEQFILES:
        p = os.path.join(SEQDIR, f)
        if not os.path.exists(p):
            continue
        for row in (json.loads(l) for l in open(p) if l.strip()):
            md = row["metadata"]
            tag = f.replace("FINAL_", "").replace(".jsonl", "")
            for pos, (iid, repo) in enumerate(zip(md["instance_ids"], md["stage_labels"])):
                m[tag][iid] = (tag, md["episode_index"], pos, repo)
    return m


def reward_of(success, reward, turns):
    if not success:
        return 0.0
    if reward is not None:
        return float(reward)
    return round(1.0 - max(0, (turns or 1) - 1) / MAX_STEPS, 4)


def score_run(run, seqmaps):
    # 用 run 名字里的数据集 tag 选对应的映射表(避免两份数据重叠题串台)
    tag = next((t for t in seqmaps if t in run), None)
    seqmap = seqmaps.get(tag, {})
    d = f"{RUNROOT}/{run}/traj/eval/heldout/rollout_0"
    files = sorted(glob.glob(d + "/ep_*.json"))
    if not files:
        return None
    trials = []          # 每个 trial:题目、成败、reward、轮数、状态、在本 episode 里的实际位置
    for p in files:
        ep = json.load(open(p))
        n_in_ep = len(ep["trials"])
        for i, tr in enumerate(ep["trials"]):
            o = tr.get("outcome") or {}
            trials.append({
                "iid": tr["instance_id"], "success": bool(o.get("success")),
                "reward": reward_of(bool(o.get("success")), o.get("reward"), o.get("turns")),
                "turns": o.get("turns"), "status": o.get("eval_status"),
                "pos": i, "ep_len": n_in_ep,
            })
    mode = "replace" if len(json.load(open(files[0]))["trials"]) > 1 else "stateless"
    # 每题聚合(stateless 是 avg@N;replace 是 5 个 run 的平均)
    per = collections.defaultdict(list)
    for t in trials:
        per[t["iid"]].append(t)
    out = {"run": run, "mode": mode, "n_traj": len(files), "n_trial": len(trials),
           "n_issue": len(per)}
    out["pass_rate"] = statistics.mean(statistics.mean(1.0 if x["success"] else 0.0 for x in v) for v in per.values())
    out["mean_reward"] = statistics.mean(statistics.mean(x["reward"] for x in v) for v in per.values())
    st = [x["turns"] for x in trials if x["success"] and x["turns"]]
    out["turns_success_median"] = sorted(st)[len(st) // 2] if st else None
    out["turns_success_mean"] = round(statistics.mean(st), 1) if st else None
    out["status_mix"] = dict(collections.Counter(str(x["status"]) for x in trials))
    # 按 sequence 分组
    seqs = collections.defaultdict(lambda: {"issues": [], "rewards": [], "succ": []})
    for iid, v in per.items():
        info = seqmap.get(iid)
        key = f"{info[0]} ep{info[1]}" if info else "(未匹配)"
        seqs[key]["issues"].append(iid)
        seqs[key]["rewards"].append(statistics.mean(x["reward"] for x in v))
        seqs[key]["succ"].append(statistics.mean(1.0 if x["success"] else 0.0 for x in v))
    out["sequences"] = {
        k: {"n_issue": len(s["issues"]), "cum_reward": round(sum(s["rewards"]), 2),
            "cum_success": round(sum(s["succ"]), 1),
            "mean_reward": round(statistics.mean(s["rewards"]), 4),
            "pass_rate": round(statistics.mean(s["succ"]), 4)}
        for k, s in sorted(seqs.items())
    }
    # 按题序三分段。
    #  replace:用 trial 在该 run 里的**实际位置**(5 个 run 块内打乱,同一题落在不同位置,难度被摊平)
    #           —— 这是"后段表现"的正确读法。
    #  stateless:单题 episode 没有序列位置,只能退回原始顺序位置,且原始顺序段内易→难,仅作难度梯度参考。
    def third_of(pos, n):
        return "1-前段" if pos < n / 3 else ("2-中段" if pos < 2 * n / 3 else "3-后段")
    bins = collections.defaultdict(list)
    if mode == "replace":
        for t in trials:
            bins[third_of(t["pos"], t["ep_len"])].append((t["reward"], 1.0 if t["success"] else 0.0))
    else:
        for iid, v in per.items():
            info = seqmap.get(iid)
            if not info:
                continue
            seqlen = 26 if "2x13" in info[0] else 27
            bins[third_of(info[2], seqlen)].append((statistics.mean(x["reward"] for x in v),
                                                   statistics.mean(1.0 if x["success"] else 0.0 for x in v)))
    out["by_third"] = {k: {"n": len(v), "mean_reward": round(statistics.mean(r for r, _ in v), 4),
                           "pass_rate": round(statistics.mean(s for _, s in v), 4)}
                       for k, v in sorted(bins.items())}
    out["by_third_basis"] = "实际位置(已去难度)" if mode == "replace" else "原始位置(仅难度参考)"
    out["per_issue"] = {iid: {"reward": statistics.mean(x["reward"] for x in v),
                              "succ": statistics.mean(1.0 if x["success"] else 0.0 for x in v)}
                        for iid, v in per.items()}
    # 按 repo
    byrepo = collections.defaultdict(list)
    for iid, v in per.items():
        info = seqmap.get(iid)
        byrepo[(info[3] if info else "?").split("__")[-1]].append(
            statistics.mean(1.0 if x["success"] else 0.0 for x in v))
    out["by_repo"] = {k: {"n": len(v), "pass_rate": round(statistics.mean(v), 4)} for k, v in sorted(byrepo.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--json")
    ap.add_argument("--md")
    args = ap.parse_args()
    seqmap = load_seq_map()
    print("数据集映射:", {k: len(v) for k, v in seqmap.items()},
          "| 两份重叠题数:", len(set(seqmap.get("2x13_26issues", {})) & set(seqmap.get("3x9_27issues", {}))))
    runs = args.runs or sorted(os.path.basename(p) for p in glob.glob(RUNROOT + "/hoeval-*") if "smoke" not in p)
    res = [r for r in (score_run(x, seqmap) for x in runs) if r]
    # gain = replace − stateless,逐题配对(只用两边都有的题)
    idx = {(r["run"].replace("hoeval-", "").split("-")[0], r["mode"],
            next((t for t in ("2x13_26issues", "3x9_27issues") if t in r["run"]), "?")): r for r in res}
    gains = []
    for (ck, mode, ds), r in idx.items():
        if mode != "replace":
            continue
        st = idx.get((ck, "stateless", ds))
        if not st:
            continue
        common = set(r["per_issue"]) & set(st["per_issue"])
        if not common:
            continue
        g_rw = statistics.mean(r["per_issue"][i]["reward"] - st["per_issue"][i]["reward"] for i in common)
        g_sc = statistics.mean(r["per_issue"][i]["succ"] - st["per_issue"][i]["succ"] for i in common)
        gains.append((ck, ds, len(common), g_rw, g_sc))
    lines = ["| run | mode | 题数 | 通过率 | 每题 reward | 成功轮数中位 |", "|---|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['run'].replace('hoeval-','')} | {r['mode']} | {r['n_issue']} | "
                     f"{r['pass_rate']*100:.1f}% | {r['mean_reward']:.4f} | {r['turns_success_median']} |")
    md = "\n".join(lines)
    print(md)
    if gains:
        print("\n== gain(replace − stateless,逐题配对)")
        for ck, ds, n, gr, gs in gains:
            print(f"   {ck:12s} {ds:16s} 配对 {n} 题  Δreward {gr:+.4f}  Δ通过率 {gs*100:+.1f}pp")
    for r in res:
        print(f"\n== {r['run']}  ({r['n_traj']} 份轨迹 / {r['n_trial']} trial)")
        print("   按 sequence:")
        for k, s in r["sequences"].items():
            print(f"     {k:24s} {s['n_issue']:2d} 题  Cum Reward {s['cum_reward']:6.2f}  "
                  f"Cum Success {s['cum_success']:5.1f}  每题 reward {s['mean_reward']:.4f}  通过率 {s['pass_rate']*100:.1f}%")
        print(f"   按题序三分段[{r['by_third_basis']}]:",
              {k: f"{v['pass_rate']*100:.1f}%/{v['mean_reward']:.3f}(n={v['n']})" for k, v in r["by_third"].items()})
        print("   按 repo:", {k: f"{v['pass_rate']*100:.0f}%" for k, v in r["by_repo"].items()})
        print("   status:", r["status_mix"])
    if args.json:
        for r in res:
            r.pop("per_issue", None)
        json.dump(res, open(args.json, "w"), ensure_ascii=False, indent=1)
        print("\nJSON ->", args.json)
    if args.md:
        open(args.md, "w").write(md + "\n")


if __name__ == "__main__":
    main()
