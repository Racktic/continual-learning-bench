#!/usr/bin/env python3
"""从 v4 episode 文件里按下标挑出若干条(2026-09-20)。原文件不动, 挑出的写到新文件。
metadata 原样保留(episode_index / shuffle_seed 不重编号), 便于回溯到完整池子里的哪一条。
用法: python3 pick_episodes.py --in <ep.jsonl> --indices 0,1 --out <pick.jsonl>
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--indices", required=True, help="逗号分隔的 episode 下标(文件内行号, 从 0 起)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

want = [int(x) for x in a.indices.split(",")]
rows = [l for l in open(a.inp) if l.strip()]
with open(a.out, "w") as f:
    for i in want:
        r = json.loads(rows[i])
        m = r["metadata"]
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  picked #{i}: {len(m['instance_ids'])} issues, repos={[x.split('__')[-1] for x in m['repos']]}")
print(f"wrote {len(want)} episodes -> {a.out}  (源文件 {a.inp} 未改动)")
