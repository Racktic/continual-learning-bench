#!/usr/bin/env python3
"""SWE-smith -> 6p6 episodes 大规模生成器 v2(2026-07-21, 实测难度版)。

与 v1 的差异(用户定稿):
- 难度 = pass@k 实测策略通过率(passk_v1), 不再用启发式 tier;
- 每 repo-segment 按难度分层抽样: 易/中/难 三档各 2 个不同策略 × 各 1 题,
  segment 内按策略通过率降序 => easy->hard;
- 规模: 能造多少造多少(全部 33 个 smith repo, 题目跨 episode 不重复);
- 函数去重: segment 内同函数禁止; 全集同函数 <=2 次且策略必须不同
  (函数 key 从原始 parquet patch 的 def 名提取);
- repo 配对: 贪心取剩余 segment 最多的两个不同 repo, 逼近 floor(total/2)。

用法:
  python3 gen_episodes_v2.py \
    --instances data/swe_smith/top32.jsonl \
    --parquet-dir /project/flame/qixinx/swe_smith/data \
    --out <episodes.jsonl>
"""
import argparse
import collections
import glob
import json
import random
import re

# passk_v1 实测策略 pass@1(2,391 trial; 3 试点 repo 代理全库)
STRAT_PASSRATE = {
    "func_pm_class_rm_base": 0.75, "func_pm_ctrl_shuffle": 0.42, "func_basic": 0.40,
    "func_pm_op_swap": 0.33, "func_pm_op_change": 0.32, "func_pm_op_change_const": 0.30,
    "func_pm_ctrl_invert_if": 0.23, "other": 0.17, "combine_file": 0.16,
    "func_pm_remove_assign": 0.13, "func_pm_remove_cond": 0.11, "func_pm_class_rm_funcs": 0.11,
    "lm_rewrite": 0.08, "func_pm_remove_loop": 0.04, "combine_module": 0.01,
    "func_pm_remove_wrapper": 0.00,
}
UNMEASURED_DEFAULT = 0.15  # 未实测策略落中档


def bucket_of(strategy):
    p = STRAT_PASSRATE.get(strategy, UNMEASURED_DEFAULT)
    return "easy" if p >= 0.25 else ("mid" if p >= 0.10 else "hard")


DEF_RE = re.compile(r"^[-+@].*?\bdef\s+(\w+)", re.M)
FILE_RE = re.compile(r"^diff --git a/(\S+)", re.M)


def func_key(repo_short, patch):
    """函数身份 = repo + 首个改动文件 + patch 涉及的 def 名集合。"""
    defs = sorted(set(DEF_RE.findall(patch or "")))
    m = FILE_RE.search(patch or "")
    return f"{repo_short}|{m.group(1) if m else '?'}|{','.join(defs) or '?'}"


def load_patches(parquet_dir, want_ids):
    import pyarrow.parquet as pq
    out = {}
    for f in sorted(glob.glob(f"{parquet_dir}/*.parquet")):
        t = pq.read_table(f, columns=["instance_id", "patch"])
        for iid, patch in zip(t.column("instance_id").to_pylist(), t.column("patch").to_pylist()):
            if iid in want_ids:
                out[iid] = patch
    return out


def build_segments(insts, fkey, fuse, rng):
    """一个 repo 的题 -> 尽可能多的 6 题 segment(2易+2中+2难, 段内策略/函数不重)。"""
    by_bucket = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in insts:
        by_bucket[bucket_of(x["strategy"])][x["strategy"]].append(x)
    for b in by_bucket.values():
        for lst in b.values():
            rng.shuffle(lst)
    segments = []
    while True:
        seg, seg_funcs, seg_strats = [], set(), set()
        ok = True
        for bucket in ("easy", "mid", "hard"):
            picked = 0
            # 优先段内新策略; 单策略档允许复用策略(但函数必须不同)
            for allow_dup_strat in (False, True):
                if picked == 2:
                    break
                strats = sorted(by_bucket[bucket],
                                key=lambda s: -len(by_bucket[bucket][s]))
                for s in strats:
                    if picked == 2:
                        break
                    if not allow_dup_strat and s in seg_strats:
                        continue
                    lst = by_bucket[bucket][s]
                    while lst:
                        x = lst.pop()
                        fk = fkey[x["instance_id"]]
                        used = fuse[fk]
                        if fk in seg_funcs or len(used) >= 2 or x["strategy"] in used:
                            continue
                        seg.append(x); seg_funcs.add(fk); seg_strats.add(s)
                        used.append(x["strategy"]); picked += 1
                        break
            if picked < 2:
                ok = False
                break
        if not ok:
            # 回滚本段的函数占用
            for x in seg:
                fuse[fkey[x["instance_id"]]].remove(x["strategy"])
            break
        seg.sort(key=lambda x: -STRAT_PASSRATE.get(x["strategy"], UNMEASURED_DEFAULT))
        segments.append(seg)
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260721)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    by_repo = collections.defaultdict(list)
    ids = set()
    for l in open(args.instances):
        d = json.loads(l)
        by_repo[d["repo"].split("/")[-1]].append(d)
        ids.add(d["instance_id"])
    patches = load_patches(args.parquet_dir, ids)
    fkey = {iid: func_key(iid.split(".")[0], patches.get(iid)) for iid in ids}

    fuse = collections.defaultdict(list)  # func_key -> 已用策略列表(全集 <=2)
    seg_pool = {}
    for repo in sorted(by_repo):
        seg_pool[repo] = build_segments(by_repo[repo], fkey, fuse, rng)

    # 贪心配对: 每次取剩余 segment 最多的两个不同 repo
    episodes = []
    while True:
        avail = sorted((r for r in seg_pool if seg_pool[r]),
                       key=lambda r: -len(seg_pool[r]))
        if len(avail) < 2:
            break
        a, b = avail[0], avail[1]
        episodes.append((a, b, seg_pool[a].pop(), seg_pool[b].pop()))
    rng.shuffle(episodes)

    used = set()
    with open(args.out, "w") as fe:
        for idx, (a, b, sa, sb) in enumerate(episodes):
            ids_ = [x["instance_id"] for x in sa + sb]
            used.update(ids_)
            fe.write(json.dumps({
                "prompt": [{"role": "user", "content": [{"type": "text",
                            "text": f"SWE-smith 6p6 v2 episode {idx}: {a} -> {b}."}]}],
                "metadata": {"split": "train", "episode_index": idx, "repoA": a, "repoB": b,
                             "instance_ids": ids_,
                             "stage_labels": [a] * 6 + [b] * 6,
                             "strategies": [x["strategy"] for x in sa + sb],
                             "shuffle_seed": idx},
            }, ensure_ascii=False) + "\n")

    n_rep = collections.Counter(x for _, _, sa, sb in episodes for x in ())
    cover = {r: 0 for r in by_repo}
    for a, b, _, _ in episodes:
        cover[a] += 1; cover[b] += 1
    leftover = sum(len(v) for v in seg_pool.values())
    print(f"[v2] episodes={len(episodes)} instances={len(used)} "
          f"(池 {len(ids)}, 利用率 {len(used)/len(ids):.0%}) 剩余落单 segment={leftover}")
    print("[v2] repo 覆盖(episode 数):",
          " ".join(f"{r.split('.')[0].split('__')[-1]}={c}" for r, c in sorted(cover.items(), key=lambda kv: -kv[1])))
    zero = [r for r, c in cover.items() if c == 0]
    if zero:
        print("[v2] ⚠ 未覆盖 repo:", zero)


if __name__ == "__main__":
    main()
