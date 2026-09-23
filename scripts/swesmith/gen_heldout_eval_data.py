#!/usr/bin/env python3
"""把 held-out sequence 文件转成 miles 的 eval prompt 数据(2026-09-21)。

口径与当年 9+10 那套(`miles/examples/codebase_adaption/scripts/gen_textfmt_eval_data.py`)一致:

  replace   : 整条 sequence 一行, **不带** no_memory -> WRITE 每题覆写的记忆模式。
              --replace-runs 次(默认 5), 每次一行, 配 N_EVAL_SAMPLES=1。
              **题序按当年 9+10 的约定处理**(见 data/replace_textfmt_eval.jsonl 实证):
                · repo 分块保持连续(块的先后不变), 只在块内打乱;
                · run0 = 原始顺序(段内易→难), run1..4 = 块内打乱;
                · 打乱用固定种子 f"{tag}-ep{idx}-run{r}", 所以 5 条题序是确定的、可复现的,
                  且写进 metadata.instance_ids 本身(顺序即记录)。
              附带好处: 打乱后同一道题在 5 个 run 里落在不同位置, 分析"后段"时难度会被摊平,
              不会像段内易→难那样把位置和难度绑在一起。
  stateless : 每题拆成单题行 + `no_memory: true`, 配 N_EVAL_SAMPLES=4 得 avg@4。
              (ACT thinking 下 no_memory 只允许单题 episode, 多题会被守卫拒绝。)

stage_labels 直接沿用 sequence 里的(逐题标注所属 repo), 不重新推断。

用法:
  python3 gen_heldout_eval_data.py --sequences FINAL_2x13_26issues.jsonl FINAL_3x9_27issues.jsonl \
      --src-dir /data/user_data/qixinx/swe_smith/heldout_v4 \
      --out-dir /home/qixinx/miles/examples/codebase_adaption/data --prefix ho
"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="+", required=True, help="sequence 文件名(在 --src-dir 下)")
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="ho")
    ap.add_argument("--replace-runs", type=int, default=5)
    ap.add_argument("--split", default="heldout", help="metadata.split(只作标记, 数据集由 CODEBASE_EVAL_DATASET 决定)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for seq in args.sequences:
        rows = [json.loads(l) for l in open(os.path.join(args.src_dir, seq)) if l.strip()]
        base = os.path.splitext(seq)[0].replace("FINAL_", "").lower()
        # ── replace: 每条 episode × replace-runs 次, 块内打乱 ──
        rep_path = os.path.join(args.out_dir, f"{args.prefix}_{base}_replace.jsonl")
        n_rep = 0
        with open(rep_path, "w") as f:
            for run in range(args.replace_runs):
                for r in rows:
                    m = dict(r["metadata"])
                    ids, labs = list(m["instance_ids"]), list(m["stage_labels"])
                    strat = list(m.get("strategies") or [])
                    # 按 repo 连续分块
                    blocks, cur = [], [0]
                    for i in range(1, len(labs)):
                        if labs[i] == labs[i - 1]:
                            cur.append(i)
                        else:
                            blocks.append(cur); cur = [i]
                    blocks.append(cur)
                    if run > 0:   # run0 保留原始顺序
                        rng = random.Random(f"{base}-ep{m.get('episode_index')}-run{run}")
                        order = []
                        for b in blocks:
                            bb = list(b); rng.shuffle(bb); order += bb
                    else:
                        order = [i for b in blocks for i in b]
                    m["instance_ids"] = [ids[i] for i in order]
                    m["stage_labels"] = [labs[i] for i in order]
                    if strat:
                        m["strategies"] = [strat[i] for i in order]
                    m["split"] = args.split
                    m["order_rank"] = n_rep
                    m["eval_run"] = run
                    m["order_seed"] = f"{base}-ep{m.get('episode_index')}-run{run}" if run else "original"
                    m.pop("no_memory", None)
                    text = f"{base} replace run{run} ep{m.get('episode_index')}: " + " -> ".join(m.get("repos", []))
                    f.write(json.dumps({"prompt": [{"role": "user", "content": [{"type": "text", "text": text}]}],
                                        "metadata": m}, ensure_ascii=False) + "\n")
                    n_rep += 1
        # ── stateless: 每题一行, no_memory ──
        st_path = os.path.join(args.out_dir, f"{args.prefix}_{base}_stateless.jsonl")
        n_st = 0
        seen = set()
        with open(st_path, "w") as f:
            for r in rows:
                m0 = r["metadata"]
                for pos, (iid, label) in enumerate(zip(m0["instance_ids"], m0["stage_labels"])):
                    if iid in seen:
                        continue
                    seen.add(iid)
                    m = {"split": args.split, "instance_ids": [iid], "stage_labels": [label],
                         "no_memory": True, "order_rank": n_st,
                         "src_episode_index": m0.get("episode_index"), "src_position": pos,
                         "difficulty_tier": m0.get("difficulty_tier")}
                    f.write(json.dumps({"prompt": [{"role": "user", "content": [{"type": "text",
                                        "text": f"{base} stateless: {iid}"}]}], "metadata": m},
                                       ensure_ascii=False) + "\n")
                    n_st += 1
        print(f"{seq}: replace {n_rep} 行({len(rows)} ep × {args.replace_runs} run) -> {os.path.basename(rep_path)}")
        print(f"{'':{len(seq)}s}  stateless {n_st} 行(去重单题, 配 N_EVAL_SAMPLES=4) -> {os.path.basename(st_path)}")


if __name__ == "__main__":
    main()
