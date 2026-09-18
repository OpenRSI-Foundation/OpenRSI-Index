#!/usr/bin/env python3
"""Retired serial/full-BRIGHT calibration: cannot produce compatible baselines."""


def measure(*args, **kwargs):
    raise RuntimeError("serial calibration superseded; use calibrate_baseline_multi_gpu.py")


def main():
    print('{"status":"calibration_failed","error":"superseded_serial_protocol"}', flush=True)
    raise SystemExit(1)


if __name__ == "__main__": main()
