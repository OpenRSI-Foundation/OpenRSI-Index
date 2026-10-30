"""A non-Linux controller must be able to load the Linux worker module."""

import os
import subprocess
import sys
from pathlib import Path


def test_controller_import_without_linux_statvfs_constants():
    source = (
        Path(__file__).resolve().parents[2]
        / "src/rsi_harness/judge_agent/judge_isolation.py"
    )
    code = """
import os, runpy, sys
for name in ('ST_NODEV', 'ST_NOATIME', 'ST_NODIRATIME', 'ST_RELATIME'):
    if hasattr(os, name):
        delattr(os, name)
sys.platform = 'darwin'
module = runpy.run_path(sys.argv[1])
try:
    module['remount_flags'](0)
except RuntimeError as error:
    assert 'require Linux' in str(error)
else:
    raise AssertionError('Linux remount executed on a non-Linux controller')
"""
    subprocess.run(
        [sys.executable, "-c", code, str(source)], check=True, env=dict(os.environ)
    )
