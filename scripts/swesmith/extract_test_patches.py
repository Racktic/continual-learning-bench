#!/usr/bin/env python3
"""为 SWE-smith 池实例批量提取 test_patch(2026-07-20)。

背景: swesmith 实例分支删除了 F2P 测试文件(防作弊)。判分时需把测试恢复到 main
版本。现有 generic_runtime 判分链靠 instance["test_patch"] 完成恢复(restore 到
base(=分支) 后 apply test_patch), 且 test_patch_paths 同时驱动防作弊剥离。
故 test_patch := `git diff <branch> main -- <F2P/P2P 测试文件>`(应用于分支态即
得到 main 的测试)。

实现: 每 repo 一次 `apptainer exec`(SIF 只读即可, git diff 不写盘), 容器内循环
处理该 repo 全部实例, 以分隔符批量输出, 宿主侧切分落盘。

用法:
  python3 extract_test_patches.py --pool pool.jsonl --sif-dir <dir> \
      --out-dir <dir>/test_patches [--repos r1 r2 ...]
输出: <out-dir>/<repo_short>.jsonl, 每行 {"instance_id":..., "test_patch":...}
"""
import argparse
import collections
import json
import os
import subprocess
import tempfile

SEP = "===SWESMITH_PATCH_SEP==="


def repo_sif_name(repo_short):
    return repo_short.split("__")[-1] + ".sif"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--sif-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repos", nargs="*", default=None, help="限定 repo_short 列表")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    by_repo = collections.defaultdict(list)
    for line in open(args.pool):
        d = json.loads(line)
        rs = d["repo"].split("/")[-1]
        if args.repos and rs not in args.repos:
            continue
        files = sorted({x.split("::")[0].removeprefix("../dev/") for x in d["FAIL_TO_PASS"] + d["PASS_TO_PASS"]})
        by_repo[rs].append((d["instance_id"], files))

    for rs, items in by_repo.items():
        sif = os.path.join(args.sif_dir, repo_sif_name(rs))
        out_path = os.path.join(args.out_dir, rs + ".jsonl")
        if not os.path.exists(sif):
            print(f"[skip] {rs}: SIF 不存在 ({sif})")
            continue
        if os.path.exists(out_path):
            done = sum(1 for _ in open(out_path))
            if done >= len(items):
                print(f"[skip] {rs}: 已完成 {done}/{len(items)}")
                continue
        # 容器内批处理脚本: 每实例输出 SEP / instance_id / patch
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False,
                                         dir=os.path.expanduser("~")) as tf:
            tf.write("cd /testbed || exit 1\n")
            for iid, files in items:
                fl = " ".join(f"'{f}'" for f in files)
                tf.write(f"echo '{SEP}'\necho '{iid}'\n"
                         f"git diff 'origin/{iid}' main -- {fl}\n")
            script = tf.name
        try:
            r = subprocess.run(
                ["apptainer", "exec", sif, "bash", script],
                capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                print(f"[fail] {rs}: rc={r.returncode} {r.stderr[:200]}")
                continue
            blocks = r.stdout.split(SEP + "\n")[1:]
            with open(out_path, "w") as fo:
                for b in blocks:
                    iid, _, patch = b.partition("\n")
                    fo.write(json.dumps({"instance_id": iid.strip(),
                                         "test_patch": patch}) + "\n")
            print(f"[ok] {rs}: {len(blocks)}/{len(items)} patches -> {out_path}")
        finally:
            os.unlink(script)


if __name__ == "__main__":
    main()
