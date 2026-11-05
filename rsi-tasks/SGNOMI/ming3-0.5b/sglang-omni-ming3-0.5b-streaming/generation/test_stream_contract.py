"""Streaming contract regressions; synthetic records never replace GPU validation."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_protocol import PROTOCOL, common, execute_nodes, lane, pinned_nodes
from test_model_lane import calibration, noise, ROOT, runs


def stream_row(chunks=2, underrun=0.01):
    return dict(audio_ttfp_s=0.2, latency_s=2.0, audio_chunk_count=chunks,
                chunk_audio_duration_s=[0.5] * chunks, inter_chunk_s=[0.51] * (chunks - 1),
                first_audio_payload_bytes=24000, audio_duration_s=0.5 * chunks,
                max_playback_underrun_s=underrun if chunks > 1 else None)


class StreamingContractTests(unittest.TestCase):
    def test_streaming_semantics_reject_nonstreaming_or_low_concurrency(self):
        config = dict(stream=True, warmup=16, concurrency=16, max_new_tokens=2048,
                      max_samples=None, sample_offset=0, temperature=None, top_p=None,
                      top_k=None, repetition_penalty=None, seed=42, arrival_seed=None,
                      max_running_requests=64, cuda_graph_max_bs=64)
        lane.check_tts_config(config, 42, 'synthetic', stream=True)
        for wrong in ({'stream': False}, {'concurrency': 1}, {'concurrency': 48}, {'warmup': 1}):
            with self.subTest(wrong=wrong), self.assertRaises(common.Failure):
                lane.check_tts_config({**config, **wrong}, 42, 'synthetic', stream=True)


    def test_continuity_keeps_single_chunk_na_and_rejects_incomplete_evidence(self):
        rows = [stream_row(), stream_row(1)]
        summary = dict(playback_continuity_requests=1, playback_continuity_na_requests=1, c50=100.0)
        lane.check_streaming(rows, summary, 'synthetic')
        for key in ('audio_ttfp_s', 'audio_chunk_count', 'chunk_audio_duration_s',
                    'first_audio_payload_bytes', 'max_playback_underrun_s'):
            broken = copy.deepcopy(rows)
            del broken[0][key]
            with self.subTest(missing=key), self.assertRaises(common.Failure):
                lane.check_streaming(broken, summary, 'synthetic')
        for wrong in ({'c50': 0}, {'playback_continuity_na_requests': 0}, {'c50': 101}):
            with self.subTest(wrong=wrong), self.assertRaises(common.Failure):
                lane.check_streaming(rows, {**summary, **wrong}, 'synthetic')
        with self.assertRaises(common.Failure):
            lane.check_streaming([stream_row(1)], dict(
                playback_continuity_requests=0, playback_continuity_na_requests=1, c50=None), 'synthetic')
        broken = copy.deepcopy(rows)
        broken[0]['chunk_audio_duration_s'] = [0.1, 0.1]
        with self.assertRaises(common.Failure):
            lane.check_streaming(broken, summary, 'synthetic')


    def test_c50_boundary_respects_native_rounding(self):
        summary = dict(playback_continuity_requests=1, playback_continuity_na_requests=0)
        for c50 in (0.0, 100.0):
            lane.check_streaming([stream_row(underrun=0.05)], dict(summary, c50=c50), 'synthetic')
        lane.check_streaming([stream_row(underrun=0.0501)], dict(summary, c50=0.0), 'synthetic')
        with self.assertRaises(common.Failure):
            lane.check_streaming([stream_row(underrun=0.0501)], dict(summary, c50=100.0), 'synthetic')

