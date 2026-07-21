"""SWE-rebench V2 task for the miles eval path.

Reuses codebase_adaptation's generic-PR runtime wholesale (textfmt scaffold, ACT
loop, singularity sandbox, base_commit checkout + test_patch reconstruction). Two
deltas vs the parent:
  1. after PR init, run install_config.install (per-repo env fixups: editable
     install, babel's setup.cfg [pytest]->[tool:pytest] sed, cldr download, ...);
  2. grade with rebench's own parse_log_pytest + strict set-equality verdict
     instead of codebase's returncode check.
Dataset rows carry: workdir=/<repo basename>, image_name=abs .sif path,
base_commit, test_patch, FAIL_TO_PASS, PASS_TO_PASS, and rebench extras
install_cmds / test_cmds / test_timeout.
"""
from __future__ import annotations

import logging
from typing import Any

from ..codebase_adaptation.task import CodebaseAdaptationTask
from ..codebase_adaptation.task_loader import TaskInstance
from ..codebase_adaptation.generic_runtime import (
    GenericEvalResult,
    _apply_patch_text,
    _cleanup_container,
    _docker_exec,
    _start_generic_container,
    initialize_generic_pr_container,
    instance_cwd,
    strip_patch_paths,
    test_patch_paths,
)
from .grading import grade

logger = logging.getLogger(__name__)


def _run_install(container_id, instance, *, timeout, cwd):
    for cmd in instance.get("install_cmds") or []:
        r = _docker_exec(container_id, cmd, cwd, timeout=timeout)
        if r.returncode != 0:
            logger.warning("install step nonzero (%s): %s", cmd, (r.stdout or "")[-200:])


# rebench 镜像基于 python:3.9/3.10-slim(老 glibc),外层 miles SIF 的 libfakeroot 需 GLIBC_2.38,
# --fakeroot 会注入它导致 "GLIBC_2.38 not found"(RUNBOOK W6.5)。故禁用 --fakeroot;沙箱以本用户
# 解包,git reset/clean/写文件均无需 root(swe_bench_cl 已验证同款做法)。PATH/LANG 不写死——
# 各仓 python 版本不同,依赖镜像自带 env(--cleanenv 保留镜像 ENV)。
_REBENCH_SINGULARITY_EXEC_ARGS = "--contain --cleanenv --no-mount hostfs,bind-paths --env LANG=C.UTF-8"


class SweRebenchTask(CodebaseAdaptationTask):
    r_max = 40

    def __init__(self, *args, **kwargs):
        import os
        os.environ["CLBENCH_SINGULARITY_EXEC_ARGS"] = _REBENCH_SINGULARITY_EXEC_ARGS
        super().__init__(*args, **kwargs)

    def _initialize_generic_pr_workspace(self, instance: TaskInstance) -> None:
        # parent: reset --hard + checkout base_commit + apply test_patch
        super()._initialize_generic_pr_workspace(instance)
        from ..container_backend import environment_container_ref

        ref = environment_container_ref(self._env) if self._env is not None else None
        if ref:
            _run_install(ref, instance.raw_data, timeout=int(instance.raw_data.get("test_timeout", 600)),
                         cwd=instance_cwd(instance.raw_data))

    def _evaluate_submission(self, model_patch: str, instance: "TaskInstance"):
        raw = instance.raw_data
        cwd = instance_cwd(raw)
        timeout = int(raw.get("test_timeout", 600))
        sanitized = strip_patch_paths(model_patch, test_patch_paths(raw))
        cid = None
        try:
            cid = _start_generic_container(raw["image_name"], cwd=cwd)
            initialize_generic_pr_container(cid, raw, timeout=timeout)  # reset+checkout+test_patch
            _run_install(cid, raw, timeout=timeout, cwd=cwd)
            try:
                _apply_patch_text(cid, cwd, sanitized, timeout=timeout)
            except RuntimeError as exc:
                return GenericEvalResult(success=False, status="apply_failed", error=str(exc))
            tcmds = raw.get("test_cmds") or []
            if isinstance(tcmds, str):
                tcmds = [tcmds]
            r = _docker_exec(cid, " && ".join(tcmds) if tcmds else "true", cwd, timeout=timeout)
            v = grade(r.stdout, raw.get("FAIL_TO_PASS") or [], raw.get("PASS_TO_PASS") or [])
            return GenericEvalResult(
                success=v["resolved"],
                status="completed" if v["resolved"] else "tests_failed",
                output=r.stdout,
            )
        except Exception as exc:
            return GenericEvalResult(success=False, status="error", error=str(exc))
        finally:
            _cleanup_container(cid)
