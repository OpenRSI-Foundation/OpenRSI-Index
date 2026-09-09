from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rsi_harness.runtime.snapshot import OverlaySnapshotBackend


@pytest.mark.integration
def test_real_overlay_keeps_eight_gib_sparse_workspace_zero_copy(tmp_path: Path):
    data_root = tmp_path / "engine-data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-8g"
    workspace.mkdir(parents=True)
    sparse = workspace / "sparse.bin"
    subprocess.run(["truncate", "-s", "8G", str(sparse)], check=True)
    logical_size = 8 * 1024**3
    before_blocks = sparse.stat().st_blocks
    assert sparse.stat().st_size == logical_size
    assert before_blocks * 512 < 1024**2
    backend = OverlaySnapshotBackend(data_root)

    capabilities = backend.probe(workspace_root)
    if not capabilities.copy_on_write:
        reasons = backend.probe_reasons
        assert reasons["overlayfs"]
        assert reasons["fuse-overlayfs"]
        pytest.skip(
            "both real overlay probes failed: "
            f"overlayfs={reasons['overlayfs']}; "
            f"fuse-overlayfs={reasons['fuse-overlayfs']}"
        )

    lease = backend.acquire(workspace, run_id="run-8g", round_id="agent-1")
    try:
        assert not (lease.upper_dir / "sparse.bin").exists()
        upper_blocks_before = sum(
            path.stat().st_blocks
            for path in lease.upper_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        assert upper_blocks_before * 512 < 1024**2
        merged_sparse = lease.merged_dir / "sparse.bin"
        assert merged_sparse.stat().st_size == logical_size
        assert sparse.stat().st_blocks == before_blocks
        (lease.merged_dir / "judge-only.txt").write_text("discard me\n")
        assert not (workspace / "judge-only.txt").exists()
        upper_blocks_after = sum(
            path.stat().st_blocks
            for path in lease.upper_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        assert upper_blocks_after * 512 < 1024**2
    finally:
        backend.release(lease)

    assert sparse.stat().st_size == logical_size
    assert sparse.stat().st_blocks == before_blocks
    assert not (workspace / "judge-only.txt").exists()
