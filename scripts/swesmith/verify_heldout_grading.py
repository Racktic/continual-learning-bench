#!/usr/bin/env python3
"""验证 held-out sequence 里每道题的判分可用性(2026-09-20)。

对每个 instance 跑两次**真实判分链**(src/tasks/codebase_adaptation/evaluator.evaluate_submission,
与训练/评测时判分的是同一条路):

  正控 gold  : model_patch = SWE-smith 数据集的 patch(修好 bug 的那个 diff) -> 期望 success=True
  负控 no-op : model_patch = 只新增一个无关文件的 diff                      -> 期望 success=False

为什么负控不用空 patch: evaluator 对空 patch 直接短路返回 status="empty_patch", 根本不进容器,
证明不了"不修就过不了"。no-op diff 一定能 apply, 且不改任何行为, 于是 F2P 必须仍然失败。

用法(在 miles SIF 里跑, 见 verify_heldout_inner_babel.sh):
  python3 verify_heldout_grading.py --episodes <ep.jsonl> [<ep2.jsonl> ...] \
      --registry <heldout_registry.jsonl> --parquet-dir <swesmith parquet dir> \
      --workers 12 --out <report.jsonl>
"""
import argparse
import collections
import concurrent.futures as cf
import glob
import json
import os
import sys
import threading
import time

NOOP_PATCH = """diff --git a/_clbench_noop_probe.txt b/_clbench_noop_probe.txt
new file mode 100644
index 0000000..a94a3fb
--- /dev/null
+++ b/_clbench_noop_probe.txt
@@ -0,0 +1 @@
+no-op probe: this patch must not fix anything
"""

_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


def load_gold(parquet_dir, want_ids):
    import pyarrow.parquet as pq
    out = {}
    for f in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        t = pq.read_table(f, columns=["instance_id", "patch"])
        for iid, patch in zip(t.column("instance_id").to_pylist(), t.column("patch").to_pylist()):
            if iid in want_ids:
                out[iid] = patch
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="只验证前 N 个 instance(调试用)")
    ap.add_argument("--skip-negative", action="store_true", help="只跑 gold 正控")
    args = ap.parse_args()

    # 与 swecl 的 verify_grading.py 一致: singularity 后端 + testbed conda PATH 的 exec args
    os.environ.setdefault("CLBENCH_CONTAINER_BACKEND", "singularity")
    sys.path.insert(0, os.environ.get("CLBENCH_ROOT", "/home/qixinx/continual-learning-bench"))
    from src.tasks.swe_bench_cl.task import _SWECL_SINGULARITY_EXEC_ARGS  # noqa: E402
    os.environ.setdefault("CLBENCH_SINGULARITY_EXEC_ARGS", _SWECL_SINGULARITY_EXEC_ARGS)
    from src.tasks.codebase_adaptation.evaluator import evaluate_submission  # noqa: E402

    rows = {}
    for line in open(args.registry):
        if line.strip():
            d = json.loads(line)
            rows[d["instance_id"]] = d

    order, seen = [], set()
    for f in args.episodes:
        for line in open(f):
            if not line.strip():
                continue
            m = json.loads(line)["metadata"]
            for iid in m["instance_ids"]:
                if iid not in seen:
                    seen.add(iid)
                    order.append((os.path.basename(f), m["episode_index"], iid))
    if args.limit:
        order = order[: args.limit]
    missing = [iid for _, _, iid in order if iid not in rows]
    log(f"instances to verify: {len(order)} (registry 缺失 {len(missing)})")
    if missing:
        log("  缺失示例:", missing[:3])

    gold = load_gold(args.parquet_dir, {iid for _, _, iid in order})
    no_gold = [iid for _, _, iid in order if not (gold.get(iid) or "").strip()]
    log(f"gold patches: {len(gold)}/{len(order)}  (空 gold {len(no_gold)})")

    def one(job):
        epfile, epidx, iid = job
        inst = rows.get(iid)
        rec = {"episode_file": epfile, "episode_index": epidx, "instance_id": iid,
               "repo": (inst or {}).get("repo"), "strategy": None}
        if inst is None:
            rec["verdict"] = "registry_missing"
            return rec
        t0 = time.time()
        g = evaluate_submission(model_patch=gold.get(iid) or "", instance=inst)
        rec["gold_success"] = bool(g.success)
        rec["gold_status"] = g.status
        rec["gold_error"] = (g.error or "")[:400]
        if not args.skip_negative:
            n = evaluate_submission(model_patch=NOOP_PATCH, instance=inst)
            rec["noop_success"] = bool(n.success)
            rec["noop_status"] = n.status
            rec["noop_error"] = (n.error or "")[:400]
        else:
            rec["noop_success"] = None
            rec["noop_status"] = "skipped"
        rec["seconds"] = round(time.time() - t0, 1)
        good = rec["gold_success"] and (args.skip_negative or rec["noop_success"] is False)
        rec["verdict"] = "ok" if good else ("gold_failed" if not rec["gold_success"] else "noop_passed")
        return rec

    done = 0
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex, open(args.out, "w") as fo:
        for rec in ex.map(one, order):
            done += 1
            results.append(rec)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            if rec["verdict"] != "ok" or done % 10 == 0:
                log(f"  [{done}/{len(order)}] {rec['verdict']:15s} {rec['instance_id']}"
                    f"  gold={rec.get('gold_status')} noop={rec.get('noop_status')} {rec.get('seconds')}s")

    v = collections.Counter(r["verdict"] for r in results)
    log("\n==== 汇总 ====")
    for k, n in v.most_common():
        log(f"  {k:16s} {n:4d}")
    per_repo = collections.defaultdict(collections.Counter)
    for r in results:
        per_repo[(r.get("repo") or "?").split("/")[-1]][r["verdict"]] += 1
    log("\n按 repo:")
    for repo, c in sorted(per_repo.items()):
        bad = sum(n for k, n in c.items() if k != "ok")
        log(f"  {repo:44s} ok={c['ok']:3d} 其它={bad:3d} {dict(c) if bad else ''}")
    log(f"\n报告: {args.out}")
    if v["ok"] != len(results):
        log("有未通过项, 退出码 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
