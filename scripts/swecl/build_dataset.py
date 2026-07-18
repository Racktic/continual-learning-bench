#!/usr/bin/env python3
"""把清洗好的 swecl_clean.jsonl 转成 clbench git_pr 契约的 dataset jsonl。
补齐 5 个字段: runtime_mode / problem_statement(从 Curriculum join) / image_name(绝对SIF路径) /
test_command(testbed绝对python) / sequence_rank。

用法:
  python build_dataset.py <clean.jsonl> <curriculum.json> <sif_dir> <out.jsonl>
"""
import json, sys

# 跨 pytest 版本安全的最小 flag：--no-header/-rA 是 pytest 6.0+，老 repo(如 astropy 3.x)
# 不认会报 unrecognized arguments。SWE-bench 跨很多版本，只用全版本通过的 -p no:cacheprovider。
# 判定看退出码，不需要报告类 flag。绝对 python 路径绕开 conda 激活(shlex.join 会引号化 &&)。
# -p no:rerunfailures: matplotlib 的 testbed 装了该插件，它在 pytest 配置阶段绑 localhost
# socket，而 agent/判分环境与 codebase_adaptation 一致(无 /etc/hosts,localhost 不可解析) →
# INTERNALERROR 整体崩。禁掉它即可(判分不需要重跑失败用例);无该插件的 repo 上是 no-op。
TEST_COMMAND = "/opt/miniconda3/envs/testbed/bin/python -m pytest -p no:cacheprovider -p no:rerunfailures"

# Screened out by the full gold-patch gate (scripts/swecl/verify_all.py): in our
# environment — which is kept byte-identical to codebase_adaptation's, so `localhost`
# does not resolve — the gold patch does NOT make every F2P/P2P test pass, so the
# instance is not cleanly gradeable and would give even a correct model < full reward.
# This mirrors codebase_adaptation's own screening step. Reason kept per id.
SCREEN_EXCLUDE = {
    # tests need a working localhost (sphinx linkcheck opens local sockets); the
    # parity env has no /etc/hosts, exactly like codebase_adaptation.
    "sphinx-doc__sphinx-8269": "linkcheck test needs localhost",
    "sphinx-doc__sphinx-8475": "linkcheck test needs localhost",
    # image ships a newer pytest than the pinned repo era, so nose-/distutils-style
    # deprecation warnings escalate to errors on P2P tests → unpassable in this image.
    "astropy__astropy-8707": "pytest version drift: nose-setup warning -> error",
    "astropy__astropy-8872": "pytest version drift: distutils warning -> error",
    "astropy__astropy-7336": "pytest version drift: warning -> error",
    # a malformed parametrized node id ("[a:::c]") that pytest cannot select.
    "pytest-dev__pytest-7324": "malformed node id not selectable",
    # >2000 test nodes overflow the single-arg length limit (MAX_ARG_STRLEN 128KB)
    # of the container exec; grading the full node list is not possible as-is.
    "matplotlib__matplotlib-25122": "node list exceeds arg length limit",
    "pydata__xarray-4687": "node list exceeds arg length limit",
}


def load_versions(curriculum_path):
    """从同目录的 Task-Stream 按 instance_id 取 version(官方 test_cmd 依赖版本)。
    Curriculum 本身不含 version, Task-Stream 含且覆盖全部 Curriculum 题。"""
    import os
    ts_path = os.path.join(os.path.dirname(curriculum_path), "SWE-Bench-CL-Task-Stream.json")
    ver = {}
    if os.path.exists(ts_path):
        d = json.load(open(ts_path))
        for k, v in d.items():
            if isinstance(v, list):
                for t in v:
                    if "instance_id" in t and "version" in t:
                        ver[t["instance_id"]] = str(t["version"])
    return ver


def load_curriculum_meta(curriculum_path):
    """从原始 Curriculum(嵌套) 按 instance_id 取 problem_statement + 元数据。

    带回与 codebase_adaptation 对齐的元数据(difficulty / hints_text / created_at 等)
    以及 SWE-Bench-CL 独有的 CL 元数据(modified_files / dependencies / sequence_position)。
    注意: codebase_adaptation 的 difficulty_passed_by / difficulty_success_steps 等是它用
    自己模型跑出来的经验难度, SWE-Bench-CL 没有对应值, 不补(以后自跑再填)。
    """
    d = json.load(open(curriculum_path))
    meta = {}
    for seq in d["sequences"]:
        for t in seq.get("tasks", []):
            md, task, cl = t["metadata"], t.get("task", {}), t.get("continual_learning", {})
            meta[md["instance_id"]] = {
                "problem_statement": task.get("problem_statement", ""),
                "hints_text": task.get("hints_text", ""),
                "created_at": md.get("created_at"),
                "difficulty": md.get("difficulty"),            # 如 "<15 min fix"
                "difficulty_score": cl.get("difficulty_score"),
                "modified_files": cl.get("modified_files"),
                "dependencies": cl.get("dependencies"),
                "sequence_position": cl.get("sequence_position"),
            }
    return meta


def load_collectable(path):
    """读 collect-filter 产出的 map: instance_id -> {f2p:[可选中], p2p:[可选中]}。
    只保留 pytest 真能选中的 node，剔除畸形 node(如参数化空串 test_x[])。"""
    if not path:
        return {}
    m = {}
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if "error" in r:
            continue
        m[r["instance_id"]] = {"f2p": r["f2p"], "p2p": r["p2p"]}
    return m


def main():
    clean_path, cur_path, sif_dir, out_path = sys.argv[1:5]
    collectable_path = sys.argv[5] if len(sys.argv) > 5 else None
    sif_dir = sif_dir.rstrip("/")
    meta_map = load_curriculum_meta(cur_path)
    ver_map = load_versions(cur_path)
    coll_map = load_collectable(collectable_path)

    rows = [json.loads(l) for l in open(clean_path) if l.strip()]
    out = []
    missing_ps = []
    empty_f2p = []
    screened = []
    total_dropped = 0
    for rank, r in enumerate(rows, start=1):
        iid = r["instance_id"]
        if iid in SCREEN_EXCLUDE:
            screened.append(iid)
            continue
        m = meta_map.get(iid)
        if m is None or not m.get("problem_statement"):
            missing_ps.append(iid)
            continue
        ps = m["problem_statement"]
        # 若有 collectable map，用过滤后的 F2P/P2P(剔除畸形 node)；否则用原始
        if iid in coll_map:
            f2p, p2p = coll_map[iid]["f2p"], coll_map[iid]["p2p"]
            total_dropped += (len(r["FAIL_TO_PASS"]) - len(f2p)) + (len(r["PASS_TO_PASS"]) - len(p2p))
        else:
            f2p, p2p = r["FAIL_TO_PASS"], r["PASS_TO_PASS"]
        # F2P 过滤后为空 → 无可判定的 fail→pass 目标(原始 node id 被 SWE-Bench-CL 截断，
        # 如参数化值在逗号处切断)，returncode 判分对它无意义，剔除。
        if not f2p:
            empty_f2p.append(iid)
            continue
        out.append({
            # 已有字段直接透传
            "instance_id": iid,
            "repo": r["repo"],
            "base_commit": r["base_commit"],
            "test_patch": r["test_patch"],
            "patch": r["patch"],
            "FAIL_TO_PASS": f2p,
            "PASS_TO_PASS": p2p,
            # 补齐的 5 个契约字段
            "runtime_mode": "git_pr",
            "problem_statement": ps,
            "image_name": f"{sif_dir}/{iid}.sif",
            "test_command": TEST_COMMAND,
            "version": ver_map.get(iid),   # 官方 grading 的 test_cmd 依赖版本
            "sequence_rank": rank,
            # 默认可省但显式写清楚
            "workdir": "/testbed",
            "test_timeout": 600,
            "init_submodules": False,
            # 与 codebase_adaptation 对齐 + SWE-Bench-CL 元数据(判定不用, 供课程设计/分析)
            "hints_text": m.get("hints_text", ""),
            "created_at": m.get("created_at"),
            "difficulty": m.get("difficulty"),
            "difficulty_score": m.get("difficulty_score"),
            "modified_files": m.get("modified_files"),
            "dependencies": m.get("dependencies"),
            "sequence_position": m.get("sequence_position"),
        })

    with open(out_path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"写 {len(out)} 题 → {out_path}")
    if coll_map:
        print(f"  已按 collectable map 剔除畸形 node 共 {total_dropped} 条")
    if screened:
        print(f"⚠️ {len(screened)} 题被 gold 全量判分门槛剔除(见 SCREEN_EXCLUDE): {screened}")
    if empty_f2p:
        print(f"⚠️ {len(empty_f2p)} 题 F2P 过滤后为空(node id 被截断，已跳过): {empty_f2p}")
    if missing_ps:
        print(f"⚠️ {len(missing_ps)} 题在 Curriculum 找不到 problem_statement(已跳过): {missing_ps}")
    # 分 repo 统计
    from collections import Counter
    for repo, n in Counter(r["repo"] for r in out).most_common():
        print(f"  {repo}: {n}")


if __name__ == "__main__":
    main()
