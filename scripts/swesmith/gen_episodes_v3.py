#!/usr/bin/env python3
"""v3 episode generator over the full 53-repo registry (2026-07-21).

User-approved design:
- All 53 repos are reshuffled together (old top-33 and batch-2 repos mix freely).
- Two subsets, generated to their respective maxima and delivered SEPARATELY:
    mixed: segment = 2 easy + 2 mid + 2 hard (same shape as v2)
    easy:  segment = 3 easy + 3 mid
  The easy subset is built FIRST (user decision: easy tier maxed out at the
  cost of mixed — the two tiers compete for the shared easy/mid instances and
  function budget; measured frontier: easy-first 226/153 vs mixed-first 9/407).
- Shared global constraints across BOTH subsets:
    * an instance appears in at most one episode overall;
    * function cap = 2 (locked by user): <=2 uses per function key, distinct
      strategies required; no repeated function within a segment.
- Repo pairing: top-k random (k=5) instead of strict greedy — pick two distinct
  repos uniformly among the k with the most remaining segments. Keeps yield
  within a few percent of greedy while making repo pairings diverse
  (user request: avoid the same big-repo pairs recurring).
- Within a segment instances are ordered by measured strategy passrate
  descending, i.e. easiest first (difficulty ascending).

Usage:
  python3 gen_episodes_v3.py \
    --instances data/swe_smith/top53.jsonl \
    --parquet-dir /project/flame/qixinx/swe_smith/data \
    --out-mixed data/swe_smith/episodes_v3_mixed.jsonl \
    --out-easy data/swe_smith/episodes_v3_easy.jsonl
"""
import argparse
import collections
import json
import random

from gen_episodes_v2 import STRAT_PASSRATE, UNMEASURED_DEFAULT, bucket_of, func_key, load_patches

FUNC_CAP = 2  # locked by user decision (2026-07-21); do not raise silently
PAIR_TOP_K = 5

MIXED_PLAN = (("easy", 2), ("mid", 2), ("hard", 2))
EASYMID_PLAN = (("easy", 3), ("mid", 3))
PURE_EASY_PLAN = (("easy", 6),)


def build_segments(insts, fkey, fuse, rng, plan):
    """Build as many segments following `plan` as constraints allow.

    Consumes from `insts` destructively via bucket lists; mutates the shared
    function-usage state `fuse` (rolled back for the final failed attempt).
    """
    by_bucket = collections.defaultdict(lambda: collections.defaultdict(list))
    wanted = {b for b, _ in plan}
    for x in insts:
        b = bucket_of(x["strategy"])
        if b in wanted:
            by_bucket[b][x["strategy"]].append(x)
    for b in by_bucket.values():
        for lst in b.values():
            rng.shuffle(lst)

    segments = []
    while True:
        seg, seg_funcs, seg_strats = [], set(), set()
        ok = True
        for bucket, want in plan:
            picked = 0
            for allow_dup_strat in (False, True):
                if picked == want:
                    break
                strats = sorted(by_bucket[bucket], key=lambda s: -len(by_bucket[bucket][s]))
                for s in strats:
                    if picked == want:
                        break
                    if not allow_dup_strat and s in seg_strats:
                        continue
                    lst = by_bucket[bucket][s]
                    while lst:
                        x = lst.pop()
                        fk = fkey[x["instance_id"]]
                        used = fuse[fk]
                        if fk in seg_funcs or len(used) >= FUNC_CAP or x["strategy"] in used:
                            continue
                        seg.append(x); seg_funcs.add(fk); seg_strats.add(s)
                        used.append(x["strategy"]); picked += 1
                        break
            if picked < want:
                ok = False
                break
        if not ok:
            for x in seg:
                fuse[fkey[x["instance_id"]]].remove(x["strategy"])
            break
        seg.sort(key=lambda x: -STRAT_PASSRATE.get(x["strategy"], UNMEASURED_DEFAULT))
        segments.append(seg)
    return segments


def pair_top_k_random(seg_pool, rng):
    """Repeatedly pair two distinct repos chosen uniformly among the top-k by
    remaining segment count, popping one segment from each."""
    episodes = []
    while True:
        avail = sorted((r for r in seg_pool if seg_pool[r]), key=lambda r: -len(seg_pool[r]))
        if len(avail) < 2:
            break
        a, b = rng.sample(avail[:PAIR_TOP_K], 2) if len(avail) >= 2 else (None, None)
        episodes.append((a, b, seg_pool[a].pop(), seg_pool[b].pop()))
    rng.shuffle(episodes)
    return episodes


def write_episodes(path, episodes, tier, used_ids):
    with open(path, "w") as fe:
        for idx, (a, b, sa, sb) in enumerate(episodes):
            ids_ = [x["instance_id"] for x in sa + sb]
            assert not (set(ids_) & used_ids), "instance reuse across episodes"
            used_ids.update(ids_)
            fe.write(json.dumps({
                "prompt": [{"role": "user", "content": [{"type": "text",
                            "text": f"SWE-smith 6p6 v3-{tier} episode {idx}: {a} -> {b}."}]}],
                "metadata": {"split": "train", "episode_index": idx, "repoA": a, "repoB": b,
                             "instance_ids": ids_,
                             "stage_labels": [a] * 6 + [b] * 6,
                             "strategies": [x["strategy"] for x in sa + sb],
                             "difficulty_tier": tier,
                             "shuffle_seed": idx},
            }, ensure_ascii=False) + "\n")


def pairing_stats(episodes):
    pairs = collections.Counter(tuple(sorted((a.split("__")[-1], b.split("__")[-1])))
                                for a, b, _, _ in episodes)
    top = pairs.most_common(3)
    return f"distinct_pairs={len(pairs)} max_pair_repeat={top[0][1] if top else 0}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out-pure-easy", required=True)
    ap.add_argument("--out-easymid", required=True)
    ap.add_argument("--out-mixed", required=True)
    ap.add_argument("--seed", type=int, default=20260723)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    by_repo = collections.defaultdict(list)
    ids = set()
    for line in open(args.instances):
        d = json.loads(line)
        by_repo[d["repo"].split("/")[-1]].append(d)
        ids.add(d["instance_id"])
    patches = load_patches(args.parquet_dir, ids)
    fkey = {iid: func_key(iid.split(".")[0], patches.get(iid)) for iid in ids}

    fuse = collections.defaultdict(list)
    used_ids: set = set()

    # Three-tier curriculum (user decision 2026-07-21): pure-easy / easy-mid /
    # mixed, as balanced as the shared budget allows. Sequential passes:
    # pure-easy first (tightest constraint: six distinct-ish easy strategies
    # from one repo per segment; hard-capped ~63), then easy-mid to its max
    # (~136 — its own ceiling, no quota needed), then mixed from the rest.
    # Measured outcome: 63 / 136 / 182 (total 381).
    pe_pool = {r: build_segments(list(v), fkey, fuse, rng, PURE_EASY_PLAN) for r, v in sorted(by_repo.items())}
    pe_eps = pair_top_k_random(pe_pool, rng)
    write_episodes(args.out_pure_easy, pe_eps, "pure_easy", used_ids)

    em_pool = {}
    for r, v in sorted(by_repo.items()):
        leftover = [x for x in v if x["instance_id"] not in used_ids]
        em_pool[r] = build_segments(leftover, fkey, fuse, rng, EASYMID_PLAN)
    em_eps = pair_top_k_random(em_pool, rng)
    write_episodes(args.out_easymid, em_eps, "easy_mid", used_ids)

    mx_pool = {}
    for r, v in sorted(by_repo.items()):
        leftover = [x for x in v if x["instance_id"] not in used_ids]
        mx_pool[r] = build_segments(leftover, fkey, fuse, rng, MIXED_PLAN)
    mx_eps = pair_top_k_random(mx_pool, rng)
    write_episodes(args.out_mixed, mx_eps, "mixed", used_ids)

    n_repos = lambda eps: len({r for a, b, _, _ in eps for r in (a, b)})
    for name, eps in (("pure_easy", pe_eps), ("easy_mid", em_eps), ("mixed", mx_eps)):
        print(f"[v3-{name}] episodes={len(eps)} repos={n_repos(eps)} {pairing_stats(eps)}")
    print(f"[v3] instances_used={len(used_ids)}/{len(ids)} ({len(used_ids)/len(ids):.0%})")


if __name__ == "__main__":
    main()
