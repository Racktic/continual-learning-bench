"""SWE-rebench V2 grading core, ported from the official evaluator
(github.com/SWE-rebench/SWE-rebench-V2: scripts/eval.py + lib/agent/log_parsers.py).

Python subset uses a single parser (parse_log_pytest). Verdict semantics follow
upstream exactly: the set of actually-PASSED test names must equal
normalize(PASS_TO_PASS + FAIL_TO_PASS) — strict equality, not subset.
Container-side flow (caller's job): git apply agent patch + test_patch (3way,
ignore-space) -> run install_config.test_cmd -> feed combined stdout/stderr here.
"""
from __future__ import annotations

import re
from enum import Enum


class TestStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def ansi_escape(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def parse_log_pytest(log: str) -> dict[str, str]:
    """pytest -rA report lines -> {test_name: status}. Verbatim port."""
    test_status_map: dict[str, str] = {}
    for line in ansi_escape(log).split("\n"):
        if any(line.startswith(x.value) for x in TestStatus):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) <= 1:
                continue
            test_status_map[parts[1]] = parts[0]
    return test_status_map


def grade(log: str, fail_to_pass: list[str], pass_to_pass: list[str]) -> dict:
    """Upstream verdict: actually-PASSED set == normalized(F2P + P2P)."""
    parsed = {normalize_test_name(k): v for k, v in parse_log_pytest(log).items()}
    passed = sorted(k for k, v in parsed.items() if v == TestStatus.PASSED.value)
    failed = sorted(k for k, v in parsed.items() if v == TestStatus.FAILED.value)
    expected_passed = sorted(normalize_test_name(n) for n in list(pass_to_pass) + list(fail_to_pass))
    resolved = passed == expected_passed
    return {
        "resolved": resolved,
        "passed_actual": passed,
        "failed_actual": failed,
        "passed_expected": expected_passed,
        "n_parsed": len(parsed),
    }


def build_eval_script(base_commit, install_steps, test_cmds, patch_files=("model.patch", "test.patch")):
    """Container-side judging script (validated on gold: isort/babel/cookiecutter 3/3).

    Flow (order is load-bearing — each step was a debugged failure mode):
      1. HOME=/tmp                — images set HOME to a nonexistent host path
      2. discover repo dir        — repo lives at /<basename>, .git at depth<=3
      3. git reset --hard base    — build leaves the worktree dirty ("does not match index")
      4. install_config.install   — per-repo env fixups live here (e.g. babel's
                                     setup.cfg [pytest]->[tool:pytest] sed); omitting
                                     it breaks pytest config parsing
      5. git apply model+test     — official 3way/recount/ignore-space args
      6. test_cmds                — output is fed to parse_log_pytest + grade()

    patch_files: filenames under /patches the caller has written into the sandbox.
    Returns a bash script string; run it with `apptainer exec --writable <sb> bash <script>`.
    """
    inst = install_steps if isinstance(install_steps, (list, tuple)) else ([install_steps] if install_steps else [])
    inst_str = " ; ".join(inst) if inst else "true"
    tcmd_str = " && ".join(test_cmds) if test_cmds else "true"
    applies = "\n".join(
        f'git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/{p} || exit 4'
        for p in patch_files
    )
    return f'''#!/bin/bash
export HOME=/tmp
repo=""
for c in /*; do [ -d "$c/.git" ] && repo="$c" && break; done
[ -z "$repo" ] && repo=$(dirname "$(find / -maxdepth 3 -name .git -type d -not -path "/tmp/*" -not -path "/patches/*" -not -path "/proc/*" 2>/dev/null | head -1)")
cd "$repo" || exit 3
git reset --hard {base_commit} >/dev/null 2>&1 || git reset --hard >/dev/null 2>&1
{inst_str}
{applies}
{tcmd_str}
'''
