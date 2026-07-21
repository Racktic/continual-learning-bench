#!/usr/bin/env python3
"""Generate easy 6p6 episodes from the leftover pool for a step-level curriculum.

Motivation (2026-07-21): v2 topped out at 333 episodes because each segment
requires 2 hard-bucket instances and the hard bucket ran dry. The easy and mid
buckets still have plenty of function-cap headroom, so we mint additional
easier episodes (segment = 3 easy + 3 mid, sorted by measured strategy
passrate desc => easy->harder within the segment) and place them BEFORE the v2
episodes in the final training file. With --rollout-shuffle disabled the data
source consumes the file sequentially, giving a step-level curriculum:
easy episodes first, mixed-difficulty (v2) later.

Constraints carried over from v2 and shared with it:
- no instance appears in more than one episode (across easy + v2 combined);
- function cap: <=2 uses per function key across the combined set, distinct
  strategies only (the fuse state is seeded from v2's actual usage);
- within a segment: no repeated function, distinct strategies preferred.

Usage:
  python3 gen_episodes_v2_easy.py \
    --instances data/swe_smith/top32.jsonl \
    --v2-episodes data/swe_smith/episodes_6p6_v2.jsonl \
    --parquet-dir /project/flame/qixinx/swe_smith/data \
    --out data/swe_smith/episodes_6p6_v2_easy.jsonl \
    --combined-out data/swe_smith/episodes_6p6_v3_curriculum.jsonl
"""
import argparse
import collections
import json
import random

from gen_episodes_v2 import STRAT_PASSRATE, UNMEASURED_DEFAULT, bucket_of, func_key, load_patches

SEGMENT_PLAN = (("easy", 3), ("mid", 3))


def build_easy_segments(insts, fkey, fuse, rng, func_cap=2):
    """Build as many (3 easy + 3 mid) segments as the leftover pool allows."""
    by_bucket = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in insts:
        b = bucket_of(x["strategy"])
        if b in ("easy", "mid"):
            by_bucket[b][x["strategy"]].append(x)
    for b in by_bucket.values():
        for lst in b.values():
            rng.shuffle(lst)

    segments = []
    while True:
        seg, seg_funcs, seg_strats = [], set(), set()
        ok = True
        for bucket, want in SEGMENT_PLAN:
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
                        if fk in seg_funcs or len(used) >= func_cap or x["strategy"] in used:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--v2-episodes", required=True)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True, help="easy-only episodes")
    ap.add_argument("--combined-out", required=True, help="easy episodes + v2, reindexed")
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--func-cap", type=int, default=3, help=(
        "max uses per function key across easy+v2 combined (distinct strategies "
        "still required); the curriculum warmup tolerates a 3rd variant since "
        "every episode is seen at most once"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    by_repo = collections.defaultdict(list)
    ids = set()
    inst_by_id = {}
    for line in open(args.instances):
        d = json.loads(line)
        by_repo[d["repo"].split("/")[-1]].append(d)
        ids.add(d["instance_id"])
        inst_by_id[d["instance_id"]] = d

    v2_lines = [json.loads(line) for line in open(args.v2_episodes)]
    used_ids = {i for ep in v2_lines for i in ep["metadata"]["instance_ids"]}

    patches = load_patches(args.parquet_dir, ids)
    fkey = {iid: func_key(iid.split(".")[0], patches.get(iid)) for iid in ids}

    # Seed the function-usage state from what v2 already consumed.
    fuse = collections.defaultdict(list)
    for iid in used_ids:
        fuse[fkey[iid]].append(inst_by_id[iid]["strategy"])

    seg_pool = {}
    for repo in sorted(by_repo):
        leftover = [x for x in by_repo[repo] if x["instance_id"] not in used_ids]
        seg_pool[repo] = build_easy_segments(leftover, fkey, fuse, rng, func_cap=args.func_cap)

    episodes = []
    while True:
        avail = sorted((r for r in seg_pool if seg_pool[r]), key=lambda r: -len(seg_pool[r]))
        if len(avail) < 2:
            break
        a, b = avail[0], avail[1]
        episodes.append((a, b, seg_pool[a].pop(), seg_pool[b].pop()))
    rng.shuffle(episodes)

    used_easy = set()
    with open(args.out, "w") as fe:
        for idx, (a, b, sa, sb) in enumerate(episodes):
            ids_ = [x["instance_id"] for x in sa + sb]
            assert not (set(ids_) & used_ids), "overlap with v2 detected"
            used_easy.update(ids_)
            fe.write(json.dumps({
                "prompt": [{"role": "user", "content": [{"type": "text",
                            "text": f"SWE-smith 6p6 v2-easy episode {idx}: {a} -> {b}."}]}],
                "metadata": {"split": "train", "episode_index": idx, "repoA": a, "repoB": b,
                             "instance_ids": ids_,
                             "stage_labels": [a] * 6 + [b] * 6,
                             "strategies": [x["strategy"] for x in sa + sb],
                             "difficulty_tier": "easy",
                             "shuffle_seed": idx},
            }, ensure_ascii=False) + "\n")

    # Combined curriculum file: easy first, then v2, with episode_index reindexed.
    with open(args.combined_out, "w") as fc:
        idx = 0
        for line in open(args.out):
            d = json.loads(line)
            d["metadata"]["episode_index"] = idx
            d["metadata"]["shuffle_seed"] = idx
            fc.write(json.dumps(d, ensure_ascii=False) + "\n")
            idx += 1
        for d in v2_lines:
            d["metadata"]["episode_index"] = idx
            d["metadata"]["shuffle_seed"] = idx
            d["metadata"].setdefault("difficulty_tier", "mixed")
            fc.write(json.dumps(d, ensure_ascii=False) + "\n")
            idx += 1

    cover = collections.Counter()
    for a, b, _, _ in episodes:
        cover[a] += 1
        cover[b] += 1
    print(f"[v2-easy] episodes={len(episodes)} instances={len(used_easy)} "
          f"repos={len(cover)} | combined={len(episodes) + len(v2_lines)} -> {args.combined_out}")
    strat_freq = collections.Counter(
        s for _, _, sa, sb in episodes for s in (x["strategy"] for x in sa + sb))
    print("[v2-easy] strategy mix:", dict(strat_freq.most_common()))


if __name__ == "__main__":
    main()
