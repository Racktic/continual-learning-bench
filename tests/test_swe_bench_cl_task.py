import os

from src.tasks.swe_bench_cl.task import SweBenchCLTask


def test_swe_bench_cl_uses_compatible_singularity_exec_args(monkeypatch):
    monkeypatch.delenv("CLBENCH_SINGULARITY_EXEC_ARGS", raising=False)

    SweBenchCLTask(dataset_path="unused.jsonl", schedule=None)

    args = os.environ["CLBENCH_SINGULARITY_EXEC_ARGS"].split()
    assert "--contain" in args
    assert "--cleanenv" in args
    assert "--fakeroot" not in args
    assert any(
        value.startswith("PATH=/opt/miniconda3/envs/testbed/bin:") for value in args
    )


def test_swe_bench_cl_preserves_explicit_singularity_exec_args(monkeypatch):
    monkeypatch.setenv("CLBENCH_SINGULARITY_EXEC_ARGS", "--cleanenv --compat")

    SweBenchCLTask(dataset_path="unused.jsonl", schedule=None)

    assert os.environ["CLBENCH_SINGULARITY_EXEC_ARGS"] == "--cleanenv --compat"
