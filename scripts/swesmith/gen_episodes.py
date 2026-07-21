#!/usr/bin/env python3
"""SWE-smith -> 6p6 CL episodes 生成器(v1, 2026-07-20 设计定稿)。

设计(与用户逐项确认):
- 过滤: F2P∈[1,20], patch 改动文件∈[1,3], 局部化P2P∈[3,500], problem_statement>=100字;
- 序列形态: 6+6 双 repo, 每题最多进 1 条序列(曝光预算);
- 难度梯度: 序列内 easy->hard。难度 = (合成策略阶梯, 类内 F2P 数)。策略阶梯按全库
  F2P 中位数实测排序(func_basic 4 -> ctrl 6-7 -> remove_assign 11 -> class_rm 13
  -> combine_file 14 -> combine_module 17), 未知策略按其 F2P 数落到最近档;
- 知识共享(序内无关, shuffle 稳健): 每条序列一侧的 6 题尽量取自同一模块簇
  (patch 首个改动文件路径的前两段, 如 pygments/lexers), 簇不足 6 题时按簇大小
  降序合并相邻簇, 最终不足则整 repo 池兜底。

用法:
  python3 gen_episodes.py --parquet-dir /project/flame/qixinx/swe_smith/data \
    --repos pygments__pygments.27649ebb pallets__jinja.ada0a9a6 tobymao__sqlglot.036601ba \
    --sif-dir /project/flame/qixinx/swe_smith_sifs --out-dir /project/flame/qixinx/swe_smith/gen
输出: <out>/instances.jsonl(题注册表)、<out>/episodes_6p6.jsonl、<out>/baseline_todo.txt
"""
import argparse
import collections
import glob
import itertools
import json
import os
import re

import pyarrow.parquet as pq

STRATEGY_ORDER = [
    ("func_basic", 0), ("other", 0),
    ("func_pm_ctrl_shuffle", 1), ("func_pm_ctrl_invert_if", 1), ("func_pm_remove_cond", 1),
    ("func_pm_op_swap", 1), ("func_pm_op_change", 1), ("func_pm_op_change_const", 1),
    ("func_pm_op_break_chains", 1), ("func_pm_remove_wrapper", 1), ("lm_rewrite", 1), ("lm_modify", 1),
    ("func_pm_remove_assign", 2), ("func_pm_remove_loop", 2),
    ("func_pm_class_rm_funcs", 3), ("func_pm_class_rm_base", 3), ("func_pm_class_shuffle_funcs", 3),
    ("combine_file", 4),
    ("combine_module", 5),
]
STRAT_TIER = dict(STRATEGY_ORDER)


def strategy_of(instance_id):
    m = re.search(r"\.[0-9a-f]{8}\.([a-z_]+?)__", instance_id)
    return m.group(1) if m else "other"


def module_cluster(patch):
    m = re.search(r"^diff --git a/(\S+)", patch, re.M)
    if not m:
        return "?"
    parts = m.group(1).split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else parts[0]


def load_pool(parquet_dir, want_repos):
    pool = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        t = pq.read_table(f)
        cols = {c: t.column(c).to_pylist() for c in t.column_names}
        for i in range(t.num_rows):
            repo_short = cols["repo"][i].split("/")[-1]
            if repo_short not in want_repos:
                continue
            f2p = cols["FAIL_TO_PASS"][i]
            f2p = json.loads(f2p) if isinstance(f2p, str) else f2p
            p2p = cols["PASS_TO_PASS"][i]
            p2p = json.loads(p2p) if isinstance(p2p, str) else p2p
            patch = cols["patch"][i]
            nfile = patch.count("diff --git")
            f2p_files = {x.split("::")[0] for x in f2p}
            p2p_local = [x for x in p2p if x.split("::")[0] in f2p_files]
            ps = (cols["problem_statement"][i] or "").strip()
            if not (1 <= len(f2p) <= 20 and len(p2p_local) <= 500 and 1 <= nfile <= 3
                    and len(ps) >= 100  # 22.5% 的题无 issue 描述(2026-07-20 实测), 盲解不可用
                    and len(p2p_local) >= 3):  # 无回归约束=reward-hacking 口子(pygments 快照测试全灭于此)
                continue
            iid = cols["instance_id"][i]
            strat = strategy_of(iid)
            pool[repo_short].append({
                "instance_id": iid,
                "repo": cols["repo"][i],
                "image_name": cols["image_name"][i],
                "problem_statement": cols["problem_statement"][i],
                "FAIL_TO_PASS": f2p,
                "PASS_TO_PASS": p2p_local,  # 局部化(F2P 相邻测试文件), 与 SWE-bench/现管线同语义
                "strategy": strat,
                "tier": STRAT_TIER.get(strat, min(4, max(0, len(f2p) // 5))),
                "n_f2p": len(f2p),
                "n_files": nfile,
                "cluster": module_cluster(patch),
            })
    return pool


def build_sequences(items, per_seq=6, max_seq_per_repo=3):
    """多 repo 广覆盖方针(2026-07-20): 每 repo 最多 max_seq_per_repo 条序列,
    一序列一簇且簇不复用(不同序列覆盖 repo 的不同模块簇); 簇内按难度 easy->hard。"""
    by_cluster = collections.defaultdict(list)
    for it in items:
        by_cluster[it["cluster"]].append(it)
    usable = sorted((cl for cl in by_cluster.values() if len(cl) >= per_seq),
                    key=len, reverse=True)
    seqs = []
    for cl in usable[:max_seq_per_repo]:
        cl.sort(key=lambda x: (x["tier"], x["n_f2p"]))
        # 从簇内按难度分层取 per_seq 题(保持梯度跨度, 而非取最简单的 6 题)
        step = max(1, len(cl) // per_seq)
        chunk = [cl[min(i * step, len(cl) - 1)] for i in range(per_seq)]
        seen = set(); uniq = []
        for x in chunk:
            if x["instance_id"] not in seen:
                uniq.append(x); seen.add(x["instance_id"])
        i = 0
        while len(uniq) < per_seq and i < len(cl):
            if cl[i]["instance_id"] not in seen:
                uniq.append(cl[i]); seen.add(cl[i]["instance_id"])
            i += 1
        uniq.sort(key=lambda x: (x["tier"], x["n_f2p"]))
        seqs.append(uniq)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--sif-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-episodes", type=int, default=0, help="0=不限")
    ap.add_argument("--max-seq-per-repo", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pool = load_pool(args.parquet_dir, set(args.repos))
    for r, items in pool.items():
        print(f"[pool] {r}: {len(items)} 题(过滤后), 簇数 {len(set(i['cluster'] for i in items))}")

    seqs = {r: build_sequences(items, max_seq_per_repo=args.max_seq_per_repo) for r, items in pool.items()}
    for r, s in seqs.items():
        print(f"[seq]  {r}: {len(s)} 条 6 题序列")

    # repo 两两配对, 轮转消耗各自的序列; 每题只进一条 episode
    episodes = []
    pairs = list(itertools.combinations(sorted(seqs), 2))
    cursors = {r: 0 for r in seqs}
    progress = True
    while progress:
        progress = False
        for a, b in pairs:
            if cursors[a] < len(seqs[a]) and cursors[b] < len(seqs[b]):
                sa, sb = seqs[a][cursors[a]], seqs[b][cursors[b]]
                cursors[a] += 1
                cursors[b] += 1
                episodes.append((a, b, sa, sb))
                progress = True
        if args.max_episodes and len(episodes) >= args.max_episodes:
            break

    inst_seen = {}
    with open(os.path.join(args.out_dir, "episodes_6p6.jsonl"), "w") as fe:
        for idx, (a, b, sa, sb) in enumerate(episodes):
            ids = [x["instance_id"] for x in sa + sb]
            labels = [a] * len(sa) + [b] * len(sb)
            fe.write(json.dumps({
                "prompt": [{"role": "user", "content": [{"type": "text",
                            "text": f"SWE-smith 6p6 episode {idx}: {a} -> {b}."}]}],
                "metadata": {"split": "train", "episode_index": idx, "repoA": a, "repoB": b,
                             "instance_ids": ids, "stage_labels": labels},
            }, ensure_ascii=False) + "\n")
            for x in sa + sb:
                inst_seen[x["instance_id"]] = x

    with open(os.path.join(args.out_dir, "instances.jsonl"), "w") as fi:
        for x in inst_seen.values():
            repo_short = x["repo"].split("/")[-1]
            sif = os.path.join(args.sif_dir, repo_short.split("__")[-1] + ".sif")
            fi.write(json.dumps({
                **{k: x[k] for k in ("instance_id", "repo", "image_name", "problem_statement",
                                      "FAIL_TO_PASS", "PASS_TO_PASS", "strategy", "tier",
                                      "n_f2p", "n_files", "cluster")},
                "sif_path": sif,
                "branch": x["instance_id"],
                "workdir": "/testbed",
                "test_command": "python -m pytest",
                "conda_env": "testbed",
            }, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out_dir, "baseline_todo.txt"), "w") as fb:
        for iid in inst_seen:
            fb.write(iid + "\n")

    print(f"[out] episodes={len(episodes)} instances={len(inst_seen)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
