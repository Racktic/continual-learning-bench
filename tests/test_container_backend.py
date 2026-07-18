import subprocess
import time

import pytest

from src.tasks.container_backend import _run_process_tree


def test_run_process_tree_timeout_kills_descendants_without_hanging():
    start = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        _run_process_tree(
            ["bash", "-c", "sleep 30 & wait"],
            timeout=0.1,
        )

    assert time.monotonic() - start < 2
