#!/usr/bin/env python3
"""v4 episode generator: longer sequences over held-out repos (2026-09-20).

Why a v4 at all: v3's segment recipes are 6-slot module constants
(MIXED_PLAN / EASYMID_PLAN / PURE_EASY_PLAN), because training episodes were
fixed at 2 repos x 6 issues. To measure how the memory behaves *later* in a
sequence we need the same structure with a longer horizon, so the recipe becomes
a CLI argument and everything else is reused verbatim from v2/v3:

  - difficulty proxy      : STRAT_PASSRATE (per-strategy measured pass@1, v2)
  - segment construction  : build_segments (v3) -- shared function-usage state,
                            FUNC_CAP=2, no repeated function inside a segment,
                            within-segment ordering easiest -> hardest
  - repo pairing          : pair_top_k_random (v3), top-k=5
  - metadata schema       : identical to v3's write_episodes, so the rollout
                            reads these rows unchanged (split/episode_index/
                            repoA/repoB/instance_ids/stage_labels/strategies/
                            difficulty_tier/shuffle_seed)

Held-out means: repo not among the 53 used for training. Pass --repos-file to
restrict generation to the repos whose SIFs exist.

Usage:
  # yield report for several candidate recipes (no output written)
  python3 gen_episodes_v4.py --instances heldout.jsonl --parquet-dir <dir> \
      --repos-file repos.txt --report

  # generate 2 x 12 = 24-issue episodes
  python3 gen_episodes_v4.py --instances heldout.jsonl --parquet-dir <dir> \
      --repos-file repos.txt --plan easy:4,mid:4,hard:4 --tier heldout24 \
      --out data/swe_smith/episodes_v4_heldout24.jsonl
"""
import argparse
import collections
import json
import random

from gen_episodes_v2 import STRAT_PASSRATE, UNMEASURED_DEFAULT, bucket_of, func_key, load_patches
from gen_episodes_v3 import FUNC_CAP, PAIR_TOP_K, build_segments, pairing_stats

CANDIDATE_PLANS = {
    "easy:2,mid:2,hard:2": (("easy", 2), ("mid", 2), ("hard", 2)),   # v3 MIXED_PLAN, 训练用过
    "easy:3,mid:3": (("easy", 3), ("mid", 3)),                        # v3 EASYMID_PLAN
    "easy:4,mid:4,hard:4": (("easy", 4), ("mid", 4), ("hard", 4)),
    "easy:3,mid:5,hard:4": (("easy", 3), ("mid", 5), ("hard", 4)),
    "easy:2,mid:5,hard:5": (("easy", 2), ("mid", 5), ("hard", 5)),
    "easy:6,mid:6": (("easy", 6), ("mid", 6)),
    "easy:8,mid:8,hard:8": (("easy", 8), ("mid", 8), ("hard", 8)),
    # 9/12 题段的候选: hard 桶只有 4 种 strategy(段内不许重复策略), 所以 hard 槽位多了就掉产,
    # 下面这几个把 hard 压到 2-3 个、把长度加在 easy/mid 上。
    "easy:3,mid:3,hard:3": (("easy", 3), ("mid", 3), ("hard", 3)),
    "easy:4,mid:3,hard:2": (("easy", 4), ("mid", 3), ("hard", 2)),
    "easy:5,mid:5,hard:2": (("easy", 5), ("mid", 5), ("hard", 2)),
    "easy:4,mid:5,hard:3": (("easy", 4), ("mid", 5), ("hard", 3)),
    "easy:6,mid:4,hard:2": (("easy", 6), ("mid", 4), ("hard", 2)),
    "easy:7,mid:6,hard:2": (("easy", 7), ("mid", 6), ("hard", 2)),
    "easy:5,mid:5,hard:3": (("easy", 5), ("mid", 5), ("hard", 3)),
    "easy:4,mid:6,hard:3": (("easy", 4), ("mid", 6), ("hard", 3)),
    "easy:5,mid:6,hard:2": (("easy", 5), ("mid", 6), ("hard", 2)),
}


def group_top_k_random(seg_pool, rng, per_episode):
    """pair_top_k_random 的推广: 每条 episode 取 `per_episode` 个不同 repo,
    仍在"剩余段数最多的 top-k"里均匀抽,保证配对多样性(v3 的 PAIR_TOP_K=5 语义不变)。
    per_episode=2 时行为与 v3 的 pair_top_k_random 一致。"""
    episodes = []
    while True:
        avail = sorted((r for r in seg_pool if seg_pool[r]), key=lambda r: -len(seg_pool[r]))
        if len(avail) < per_episode:
            break
        k = max(per_episode, min(PAIR_TOP_K, len(avail)))
        repos = rng.sample(avail[:k], per_episode)
        episodes.append((repos, [seg_pool[r].pop() for r in repos]))
    rng.shuffle(episodes)
    return episodes


def parse_plan(spec):
    plan = []
    for part in spec.split(","):
        bucket, _, count = part.partition(":")
        bucket = bucket.strip()
        if bucket not in ("easy", "mid", "hard"):
            raise SystemExit(f"unknown bucket {bucket!r} in --plan (easy/mid/hard)")
        plan.append((bucket, int(count)))
    return tuple(plan)


def load_instances(path, repos_allow, exclude_ids=frozenset()):
    by_repo = collections.defaultdict(list)
    ids = set()
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        repo = d["repo"].split("/")[-1]
        if repos_allow and repo not in repos_allow:
            continue
        if d["instance_id"] in exclude_ids:
            continue
        by_repo[repo].append(d)
        ids.add(d["instance_id"])
    return by_repo, ids


def segments_for(by_repo, fkey, plan, seed):
    """Fresh function-usage state per call so report mode stays independent."""
    rng = random.Random(seed)
    fuse = collections.defaultdict(list)
    return {r: build_segments(list(v), fkey, fuse, rng, plan) for r, v in sorted(by_repo.items())}, rng


def segments_for_plans(by_repo, fkey, plans, seed):
    """异构配方: 每个 repo 槽位一个 plan(如 12 题 + 13 题, 对应 clbench 那条 9+10 的形状)。
    共享同一个 fuse, 所以 FUNC_CAP=2 与"同函数须不同策略"在所有槽位间仍然全局成立;
    先长后短地建池, 让长段优先拿到题(短段更容易凑)。"""
    rng = random.Random(seed)
    fuse = collections.defaultdict(list)
    remaining = {r: list(v) for r, v in sorted(by_repo.items())}
    order = sorted(range(len(plans)), key=lambda i: -sum(n for _, n in plans[i]))
    pools = [None] * len(plans)
    for i in order:
        pools[i] = {r: build_segments(remaining[r], fkey, fuse, rng, plans[i]) for r in remaining}
    return pools, rng


def group_hetero(pools, rng):
    """每条 episode 从每个槽位的池里各取一个段, 且 repo 互不相同。"""
    episodes = []
    while True:
        chosen_repos, chosen_segs = [], []
        for pool in pools:
            avail = sorted((r for r in pool if pool[r] and r not in chosen_repos),
                           key=lambda r: -len(pool[r]))
            if not avail:
                break
            k = min(PAIR_TOP_K, len(avail))
            r = rng.choice(avail[:k])
            chosen_repos.append(r); chosen_segs.append(pool[r].pop())
        if len(chosen_segs) < len(pools):
            for r, seg in zip(chosen_repos, chosen_segs):   # 回滚未成条的
                pools[chosen_repos.index(r)][r].append(seg)
            break
        episodes.append((chosen_repos, chosen_segs))
    rng.shuffle(episodes)
    return episodes


def write_episodes(path, episodes, tier, split):
    """metadata 的键与 v3 逐字段一致(rollout 不用改)。repoA/repoB 保留为前两个 repo,
    repos 字段给出完整列表; stage_labels 逐题标注所属 repo, 是跨题切分的真实依据。"""
    used = set()
    with open(path, "w") as fe:
        for idx, (repos, segs) in enumerate(episodes):
            flat = [x for seg in segs for x in seg]
            ids_ = [x["instance_id"] for x in flat]
            assert not (set(ids_) & used), "instance reuse across episodes"
            used.update(ids_)
            labels = [r for r, seg in zip(repos, segs) for _ in seg]
            chain = " -> ".join(repos)
            fe.write(json.dumps({
                "prompt": [{"role": "user", "content": [{"type": "text",
                            "text": f"SWE-smith heldout v4-{tier} episode {idx}: {chain}."}]}],
                "metadata": {"split": split, "episode_index": idx,
                             "repoA": repos[0], "repoB": repos[1] if len(repos) > 1 else repos[0],
                             "repos": repos,
                             "instance_ids": ids_,
                             "stage_labels": labels,
                             "strategies": [x["strategy"] for x in flat],
                             "difficulty_tier": tier,
                             "shuffle_seed": idx},
            }, ensure_ascii=False) + "\n")
    return len(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, help="converted registry jsonl (convert_pool.py output)")
    ap.add_argument("--parquet-dir", required=True, help="SWE-smith parquet dir (for patches -> func_key)")
    ap.add_argument("--repos-file", help="one repo_short per line; restricts generation")
    ap.add_argument("--exclude-instances", help="每行一个 instance_id 的黑名单(例如判分验证不通过的题), 生成时跳过")
    ap.add_argument("--plan", default="easy:4,mid:4,hard:4", help="segment recipe, e.g. easy:4,mid:4,hard:4")
    ap.add_argument("--plans", help="异构配方, 分号分隔, 每个 repo 槽位一个(如 'easy:4,mid:5,hard:3;easy:5,mid:5,hard:3');"
                                    " 给了它就忽略 --plan 与 --repos-per-episode")
    ap.add_argument("--tier", default="heldout24", help="difficulty_tier label written into metadata")
    ap.add_argument("--split", default="train", help="metadata.split (rollout default is 'train')")
    ap.add_argument("--out")
    ap.add_argument("--repos-per-episode", type=int, default=2,
                    help="一条 episode 串几个 repo(训练是 2);总题数 = 该值 x 段长")
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--report", action="store_true", help="print per-recipe yield, write nothing")
    args = ap.parse_args()

    repos_allow = None
    if args.repos_file:
        repos_allow = {l.strip() for l in open(args.repos_file) if l.strip() and not l.startswith("#")}
    exclude_ids = set()
    if args.exclude_instances:
        exclude_ids = {l.strip() for l in open(args.exclude_instances) if l.strip() and not l.startswith("#")}
        print(f"instance 黑名单: {len(exclude_ids)} 条")
    by_repo, ids = load_instances(args.instances, repos_allow, exclude_ids)
    print(f"instances: {len(ids)} over {len(by_repo)} repos (FUNC_CAP={FUNC_CAP}, PAIR_TOP_K={PAIR_TOP_K})")
    patches = load_patches(args.parquet_dir, ids)
    missing = len(ids) - len(patches)
    if missing:
        print(f"  warning: {missing} instances without a patch in parquet (func identity falls back to '?')")
    fkey = {iid: func_key(iid.split(".")[0], patches.get(iid)) for iid in ids}
    buckets = collections.Counter(bucket_of(d["strategy"]) for v in by_repo.values() for d in v)
    print(f"  buckets: easy={buckets['easy']} mid={buckets['mid']} hard={buckets['hard']}"
          f" (unmeasured strategies default to {UNMEASURED_DEFAULT:.2f} -> mid)")

    if args.report:
        print(f"\n(repos_per_episode={args.repos_per_episode})")
        print(f"{'recipe':22s} {'seg_len':>7s} {'segments':>9s} {'episodes':>9s} {'issues':>7s}  repos contributing")
        per = args.repos_per_episode
        for spec, plan in CANDIDATE_PLANS.items():
            pool, rng = segments_for(by_repo, fkey, plan, args.seed)
            seg_len = sum(n for _, n in plan)
            nseg = sum(len(v) for v in pool.values())
            contrib = sum(1 for v in pool.values() if v)
            eps = group_top_k_random({k: list(v) for k, v in pool.items()}, rng, per)
            if args.max_episodes:
                eps = eps[: args.max_episodes]
            print(f"{spec:22s} {seg_len:7d} {nseg:9d} {len(eps):9d} {len(eps) * seg_len * per:7d}  {contrib}/{len(by_repo)}")
        return

    if args.plans:
        plans = [CANDIDATE_PLANS.get(x.strip()) or parse_plan(x.strip()) for x in args.plans.split(";")]
        lens = [sum(n for _, n in pl) for pl in plans]
        pools, rng = segments_for_plans(by_repo, fkey, plans, args.seed)
        print(f"\nplans={args.plans} 段长={lens} 题/条={sum(lens)}")
        for i, pool in enumerate(pools):
            print(f"  槽位{i} 段数={sum(len(v) for v in pool.values())} 来自 {sum(1 for v in pool.values() if v)} 个 repo")
        episodes = group_hetero(pools, rng)
        if args.max_episodes:
            episodes = episodes[: args.max_episodes]
        if not args.out:
            raise SystemExit("--out is required unless --report")
        n_used = write_episodes(args.out, episodes, args.tier, args.split)
        print(f"wrote {len(episodes)} episodes x {sum(lens)} issues = {n_used} instances -> {args.out}")
        return

    plan = CANDIDATE_PLANS.get(args.plan) or parse_plan(args.plan)
    seg_len = sum(n for _, n in plan)
    pool, rng = segments_for(by_repo, fkey, plan, args.seed)
    print(f"\nplan={args.plan} seg_len={seg_len} segments={sum(len(v) for v in pool.values())}"
          f" from {sum(1 for v in pool.values() if v)} repos")
    episodes = group_top_k_random(pool, rng, args.repos_per_episode)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    if not args.out:
        raise SystemExit("--out is required unless --report")
    n_used = write_episodes(args.out, episodes, args.tier, args.split)
    print(f"wrote {len(episodes)} episodes x {seg_len * args.repos_per_episode} issues = {n_used} instances -> {args.out}")
    if args.repos_per_episode == 2:
        print("  " + pairing_stats([(r[0], r[1], s[0], s[1]) for r, s in episodes]))


if __name__ == "__main__":
    main()
