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


class StreamingTests(unittest.TestCase):
    def test_only_c16_commands_and_separate_outputs_preserve_nonstreaming(self):
        all_cases = common.evaluation_cases()
        self.assertEqual(len(all_cases), 2)
        self.assertEqual(len({case['output'] for case in all_cases}), 2)
        for model in PROTOCOL['lanes']:
            normal, stream = common.evaluation_cases(model)
            self.assertEqual(normal['argv'], model['argv'])
            self.assertNotIn('--stream', normal['argv'])
            self.assertEqual(stream['argv'].count('--stream'), 1)
            for index, seed in enumerate((42, 43, 44)):
                for case in (normal, stream):
                    cmd = lane.benchmark_argv(case, index)
                    self.assertEqual(cmd[cmd.index('--max-concurrency') + 1], '16')
                    self.assertEqual(cmd[cmd.index('--output-dir') + 1], case['output'])
                    self.assertEqual(cmd[cmd.index('--seed') + 1], str(seed))
                    self.assertEqual(lane.benchmark_argv(case, index, transcribe_only=True),
                                     cmd + ['--transcribe-only'])

    def test_pinned_streaming_request_uses_pcm_and_preserves_conditioning(self):
        _, nodes = pinned_nodes('benchmarks/tasks/tts.py',
                                {'_build_tts_payload', '_resolve_tts_generation_kwargs'})
        ns = {}
        execute_nodes(nodes, ns)
        sample = SimpleNamespace(target_text='Speech test.', ref_audio='ref.wav', ref_text='Reference.')
        for model in PROTOCOL['lanes']:
            kwargs = dict(seed=42, max_new_tokens=2048,
                          ref_format='references')
            normal = ns['_build_tts_payload'](sample, model['model'], **kwargs)
            stream = ns['_build_tts_payload'](sample, model['model'], stream=True, **kwargs)
            self.assertEqual(stream, {**normal, 'response_format': 'pcm', 'stream': True})

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

    def test_streaming_metrics_cannot_be_missing_or_pooled_into_normal_runs(self):
        values = runs()
        values['cosy'][0]['metrics']['latency_p99_s'] = 1.0
        values['cosy_stream'][0]['metrics']['latency_p99_s'] = 100.0
        self.assertNotEqual(common.means(values)['cosy']['latency_p99_s'],
                            common.means(values)['cosy_stream']['latency_p99_s'])
        for key in common.metrics_for('cosy_stream'):
            broken = copy.deepcopy(values)
            del broken['cosy_stream'][0]['metrics'][key]
            with self.subTest(missing=key), self.assertRaises(common.Failure):
                common.validate_runs(broken, 'synthetic')
        record = calibration()
        for name in list(record['runs']):
            if name.endswith('_stream'):
                del record['runs'][name]
                del record['means'][name]
        with patch.object(noise, 'HERE', ROOT / 'tests'), self.assertRaises(common.Failure):
            noise.summarize([record])

    def test_one_gpu_lane_executes_both_modes_and_does_not_publish_partial_success(self):
        for fail in (False, True):
            calls = []
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = dict(root=temp, lane=PROTOCOL['lanes'][0], cpu=[0],
                              gpu={'index': '0', 'uuid': 'GPU-test'}, original=temp,
                              cancel=str(root / 'cancel'))
                cfg = root / 'config.json'; cfg.write_text(json.dumps(config))

                def repetition(conf, index):
                    calls.append((conf['lane']['name'], index))
                    if fail and conf['lane']['stream']:
                        raise common.Failure('infrastructure', 'synthetic_failure', root, 'synthetic')
                    return {'repetition': index}

                with (patch.object(lane.sys, 'argv', ['lane.py', str(cfg)]),
                      patch.object(lane.os, 'sched_setaffinity'),
                      patch.object(lane.os, 'sched_getaffinity', return_value={0}),
                      patch.object(lane, 'clean_env', return_value={}),
                      patch.object(lane.subprocess, 'run'),
                      patch.object(lane, 'repetition', repetition),
                      patch.object(lane, 'safe_exception')):
                    self.assertEqual(lane.main(), 1 if fail else 0)
                result = json.loads((root / 'lane-result.json').read_text())
                if fail:
                    self.assertNotEqual(result['status'], 'complete')
                    self.assertNotIn('runs', result)
                else:
                    self.assertEqual(calls, [('cosy', 0), ('cosy', 1), ('cosy', 2)] +
                                     [('cosy_stream', 0), ('cosy_stream', 1), ('cosy_stream', 2)])
                    self.assertEqual(set(result['runs']), {'cosy_stream', 'cosy'})


if __name__ == '__main__':
    unittest.main()
