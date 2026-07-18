#!/usr/bin/env python3
"""从 SWE-Bench-CL Curriculum 抽取指定 repo 的题, 输出拉镜像+清洗所需的精简 jsonl。
用法: python extract_instances.py <curriculum.json> <out.jsonl> <repo_substr>...
每行字段: instance_id, repo, base_commit, test_patch, patch(gold), FAIL_TO_PASS, PASS_TO_PASS, image_name
"""
import json, sys

def norm(x):
    return json.loads(x) if isinstance(x, str) else x

def image_name(iid):
    return f"swebench/sweb.eval.x86_64.{iid.replace('__','_1776_')}"

def main():
    cur, out = sys.argv[1], sys.argv[2]
    repo_subs = sys.argv[3:]
    d = json.load(open(cur))
    rows = []
    for seq in d["sequences"]:
        for t in seq.get("tasks", []):
            md, ev = t["metadata"], t["evaluation"]
            iid = md["instance_id"]
            if repo_subs and not any(s in iid or s in md["repo"] for s in repo_subs):
                continue
            rows.append({
                "instance_id": iid,
                "repo": md["repo"],
                "base_commit": md["base_commit"],
                "test_patch": ev["test_patch"],
                "patch": ev["patch"],
                "FAIL_TO_PASS": norm(ev["FAIL_TO_PASS"]),
                "PASS_TO_PASS": norm(ev["PASS_TO_PASS"]),
                "image_name": image_name(iid),
            })
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"抽取 {len(rows)} 题 → {out}")
    from collections import Counter
    for repo, n in Counter(r["repo"] for r in rows).most_common():
        print(f"  {repo}: {n}")

if __name__ == "__main__":
    main()
