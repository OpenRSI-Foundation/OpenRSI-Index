"""CPU-only regression checks; no serving, GPU inference, or scored workload."""

import ast
import asyncio
import copy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import time


TASK = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


contract = load("adaptation_contract", TASK / "tests/contract.py")
launcher = load("adaptation_launcher", TASK / "tools/launch.py")
streaming = load("adaptation_streaming", TASK / "environment/exact_text_streaming.py")
adapter = load("adaptation_source", TASK / "environment/adapt_baseline.py")


def measurement(error=0.0667, speed=1.0):
    repeats = {}
    for workload in contract.WORKLOADS:
        metrics = {name: 1.0 for name in contract.BASE_METRICS}
        metrics.update(qps=10 * speed, rtfx=50 * speed, corpus_wer=error,
                       coverage=contract.COUNTS[workload.split('/')[1]],
                       warmup_coverage=contract.COUNTS[workload.split('/')[1]])
        if workload.startswith('stream/'):
            metrics.update({name: 1.0 for name in contract.STREAM_METRICS})
        repeats[workload] = [dict(metrics) for _ in range(3)]
    means, ranges = contract.aggregate(repeats)
    return dict(repeats=repeats, means=means, ranges=ranges)


class QualityTests(unittest.TestCase):
    def test_relative_boundary_and_independent_cells(self):
        initial = measurement()['means']
        current = measurement(error=0.0667 * 1.03, speed=1.1)['means']
        self.assertEqual(contract.compare(current, initial, initial)['failed'], [])
        current['stream/zh']['corpus_wer'] += 1e-8
        self.assertEqual(contract.compare(current, initial, initial)['failed'],
                         ['stream/zh:corpus_wer>103% of initial B0'])

    def test_ceiling_does_not_roll_with_bk(self):
        initial = measurement(error=0.05)['means']
        previous = measurement(error=0.0515)['means']
        current = measurement(error=0.052, speed=1.1)['means']
        failures = contract.compare(current, initial, previous)['failed']
        self.assertEqual(len(failures), 4)
        self.assertTrue(all('initial B0' in failure for failure in failures))

    def test_zero_baseline_has_no_epsilon(self):
        initial = measurement(error=0)['means']
        current = measurement(error=0, speed=1.1)['means']
        self.assertEqual(contract.compare(current, initial, initial)['failed'], [])
        current['nonstream/en']['corpus_wer'] = 1e-12
        self.assertTrue(contract.compare(current, initial, initial)['failed'])

    def test_both_modes_and_weighted_reward(self):
        initial = measurement()['means']
        current = copy.deepcopy(initial)
        for workload in current:
            ratio = 1.2 if workload.startswith('stream/') else 1.1
            current[workload]['qps'] *= ratio
        result = contract.compare(current, initial, initial)
        self.assertEqual(result['failed'], [])
        self.assertAlmostEqual(result['reward'], 1.2 ** (2/3) * 1.1 ** (1/3))
        for workload in ('nonstream/en', 'nonstream/zh'):
            current[workload]['qps'] = initial[workload]['qps']
        self.assertIn('nonstream:mode ratio must exceed 1 against Bk',
                      contract.compare(current, initial, initial)['failed'])

    def test_actual_host_calibration_and_aa_accept_model_own_error(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = launcher.Controller.__new__(launcher.Controller)
            controller.root = Path(directory)
            controller.state_path = controller.root / 'state.json'
            controller.run_id = 'cpu-regression'
            controller.args = SimpleNamespace(prepare=True)
            controller.contract = contract
            controller.state = {'phase': 'calibration', 'generation': 0,
                                'protocol': 'synthetic', 'run_id': controller.run_id,
                                'hardware': {'synthetic': True}}
            value = measurement()
            for phase, round_id in [('calibration', 'r0'), ('aa', 'r1')]:
                pending = {**controller.state, 'round_id': round_id, 'nonce': round_id}
                controller.pending = pending
                result = {key: pending[key] for key in
                          ('phase', 'generation', 'protocol', 'run_id', 'hardware', 'round_id', 'nonce')}
                result.update(value, scored_requests=18648, warmup_requests=18648)
                if phase == 'calibration':
                    result.update(failed=[], reward=1.0)
                else:
                    result.update(contract.compare(value['means'], value['means'], value['means']))
                result['accepted'] = not result['failed']
                if not result['accepted']:
                    result['reward'] = None
                status = 'completed' if result['accepted'] else 'verifier_error'
                logs = controller.root / 'logs/runs' / controller.run_id / 'task'
                report_path = logs / 'submissions' / round_id / 'report.json'
                report_path.parent.mkdir(parents=True)
                report = {'status': status, 'score': result['reward'],
                          'metrics': {'reward': result['reward']} if result['accepted'] else {},
                          'full_output_captured': True, 'timed_out': False, 'exit_code': 0}
                report_path.write_text(json.dumps(report))
                feedback = logs / 'feedback' / (round_id + '.log')
                feedback.parent.mkdir(exist_ok=True)
                feedback.write_text('ASR_AGGREGATE ' + json.dumps(result) + '\n')
                if result['accepted']:
                    reward = logs / 'verifier' / round_id / 'reward.json'
                    reward.parent.mkdir(parents=True)
                    reward.write_text(json.dumps({'reward': result['reward']}))
                controller.finish({'round_id': round_id, 'status': status, 'score': result['reward']})
            self.assertEqual(controller.state['phase'], 'ready')
            self.assertEqual(controller.state['b0']['means']['stream/zh']['corpus_wer'], 0.0667)


class StreamingTests(unittest.TestCase):
    def test_unicode_whitespace_and_early_emission(self):
        text = '  Hello 世界 🐈!  '
        raw = text.encode('utf-8')
        state = streaming.ExactTextStreamState(
            decoder=lambda ids: bytes(ids).decode('utf-8', errors='replace'),
            interval_seconds=0.05)
        deltas = [state.step(byte, terminal=False, now_seconds=1 + index * 0.06)
                  for index, byte in enumerate(raw)]
        deltas.append(state.step(None, terminal=True, now_seconds=10))
        self.assertEqual(''.join(deltas), text.strip())
        self.assertTrue(any(deltas[:len(raw)//2]))
        self.assertNotIn('\ufffd', ''.join(deltas))

    def test_terminal_flush_bypasses_interval_without_eos(self):
        state = streaming.ExactTextStreamState(
            decoder=lambda ids: ''.join(chr(value) for value in ids), interval_seconds=0.05)
        self.assertEqual(state.step(ord('A'), terminal=False, now_seconds=1), 'A')
        self.assertEqual(state.step(ord('B'), terminal=False, now_seconds=1.01), '')
        self.assertEqual(state.step(None, terminal=True, now_seconds=1.02), 'B')
        self.assertEqual(state.step(None, terminal=True, now_seconds=1.03), '')

    def test_incomplete_terminal_bytes_match_final_decode(self):
        state = streaming.ExactTextStreamState(
            decoder=lambda ids: bytes(ids).decode('utf-8', errors='replace'), interval_seconds=0.05)
        self.assertEqual(state.step(0xe4, terminal=False, now_seconds=1), '')
        self.assertEqual(state.step(None, terminal=True, now_seconds=2), '\ufffd')

    def test_baseline_patch_and_actual_stream_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with tarfile.open(TASK / 'environment/source.tar.gz') as archive:
                for member in archive:
                    parts = member.name.split('/', 1)
                    if len(parts) == 2 and parts[1]:
                        member.name = parts[1]
                        archive.extract(member, source, filter='data')
            adapter.adapt_source(source)
            model = source / 'sglang_omni/models' / adapter.MODEL_CODE
            tree = ast.parse((model / 'request_builders.py').read_text())
            builder_node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                                and n.name == 'ExactASRStreamOutputBuilder')
            namespace = {'__name__': __name__, 'time': SimpleNamespace(perf_counter=lambda: 10),
                         'ExactTextStreamState': streaming.ExactTextStreamState,
                         'OutgoingMessage': lambda **values: SimpleNamespace(**values),
                         'RequestOutput': lambda **values: SimpleNamespace(**values),
                         'decode_token_ids': lambda tokenizer, ids, skip_special_tokens: tokenizer.decode(ids),
                         'postprocess_moss_transcribe_diarize_text': lambda text: text.strip()}
            module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), builder_node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), '<actual patched stream builder>', 'exec'), namespace)
            tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: ''.join(chr(value) for value in ids))
            builder = namespace['ExactASRStreamOutputBuilder'](tokenizer, 0.05)
            request = SimpleNamespace(inflight_middle_chunks=0, finished=lambda: False)
            data = SimpleNamespace(req=request,
                                   stage_payload=SimpleNamespace(request=SimpleNamespace(params={'stream': True})),
                                   exact_text_stream_state=None)
            messages = []
            for value in ' A B ':
                messages += builder('one', data, SimpleNamespace(data=ord(value), finished=False))
            messages += builder.flush('one', data)
            self.assertEqual(''.join(message.data['text'] for message in messages), 'A B')
            self.assertEqual(builder.flush('one', data), [])
            data.stage_payload.request.params['stream'] = False
            self.assertEqual(builder('one', data, SimpleNamespace(data=ord('X'), finished=False)), [])
            policy = json.loads((TASK / 'tests/policy.json').read_text())
            for relative, functions in policy['fixed_nodes'].items():
                current = ast.parse((source / relative).read_text())
                for name, expected in functions.items():
                    matches = [ast.dump(n, include_attributes=False) for n in ast.walk(current)
                               if isinstance(n, ast.FunctionDef) and n.name == name]
                    self.assertEqual(matches, [expected], (relative, name))

            if adapter.MODEL_CODE == 'moss_transcribe_diarize':
                gates = load('adaptation_gates', TASK / 'tests/gates.py')
                self.assertIsNone(gates.FIXED['asr.factory.max_new_tokens'])
                self.assertIsNone(gates.FIXED['asr.factory.context_length'])
                gates.WORK = source / 'guard-probe/workspace'
                gates.ROOT = source / 'guard-probe/authority'
                gates.ROOT.mkdir(parents=True)
                for relative in set(policy['fixed_nodes']) | set(policy['fixed_assignments']):
                    destination = gates.WORK / 'sglang-omni' / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text((source / relative).read_text())
                (gates.WORK / 'candidate.yaml').write_text('{}\n')
                inventory = {p.relative_to(gates.WORK).as_posix(): gates.describe(p)
                             for p in gates.WORK.rglob('*')}
                (gates.ROOT / 'build-inventory.json').write_text(json.dumps(
                    {'workspace': inventory, 'assets': {}}))
                (gates.ROOT / 'policy.json').write_text(json.dumps(policy))
                self.assertEqual(gates.check(), {})
                stage = gates.WORK / 'sglang-omni/sglang_omni/models/moss_transcribe_diarize/stages.py'
                original = stage.read_text()
                stage.write_text(original.replace('return 5120', 'return 128'))
                with self.assertRaises(gates.CandidateInvalid) as raised:
                    gates.check()
                self.assertTrue(any(error['field'] == 'default_max_new_tokens'
                                    for error in raised.exception.errors))
                stage.write_text(original)
                (gates.WORK / 'candidate.yaml').write_text('asr.factory.max_new_tokens: 128\n')
                with self.assertRaises(gates.CandidateInvalid) as raised:
                    gates.check()
                self.assertTrue(any(error['code'] == 'unknown_execution_knob'
                                    for error in raised.exception.errors))


@unittest.skipUnless((TASK / 'tests/moss_text.py').exists(), 'MOSS only')
class MossParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = load('adaptation_moss_parser', TASK / 'tests/moss_text.py')

    def test_preserves_word_boundaries_and_overlapping_speakers(self):
        raw = '[0][S01]Hello world[2] [1][S02]Good morning[3]'
        self.assertEqual(self.parser.speech_text(raw, 4), 'Hello world Good morning')
        self.assertEqual(self.parser.speech_text('[0][S01]你好世界[1]<|im_end|>', 2), '你好世界')

    def test_speaker_only_retains_every_span(self):
        # Same three-speaker structure as both failed diagnostic probes; synthetic text.
        raw = '[S01]First span.[S02]Second span.[S01]Final span.'
        self.assertEqual(self.parser.speech_text(raw, 10),
                         'First span. Second span. Final span.')
        self.assertEqual(self.parser.speech_text(
            '<|im_start|>[S01]你好 世界[S123]Hello 🐈 [laughter]<|im_end|>', 3),
            '你好 世界 Hello 🐈 [laughter]')

    def test_never_rejects_and_removes_only_native_markers(self):
        # Like the other ASR tasks, quality is WER on the text; structure is not validated.
        for raw, expected in (('plain transcript', 'plain transcript'),
                              ('prefix[S01]secret speech', 'prefix secret speech'),
                              ('[S01]secret speech[0.5]', 'secret speech'),
                              ('[1][S01]hello[2][0][S02]world[1]', 'hello world'),
                              ('[0][S01]hello[20]junk', 'hello junk'),
                              ('[S1]hello', '[S1]hello'),
                              ('[S01]hello[S02', 'hello[S02'),
                              ('[S01]hello[-1]', 'hello[-1]')):
            with self.subTest(raw=raw):
                self.assertEqual(self.parser.speech_text(raw, 3), expected)

    def test_duration_is_not_validated(self):
        for duration in (0, -1, float('inf'), float('nan'), None):
            with self.subTest(duration=duration):
                self.assertEqual(self.parser.speech_text('[0][S01]hello[1]', duration), 'hello')


class OutputContractIntegrationTests(unittest.TestCase):
    def test_http_and_sse_use_same_parser_and_preserve_wire_output(self):
        parser = load('integration_moss_parser', TASK / 'tests/moss_text.py')
        gates = load('integration_moss_gates', TASK / 'tests/gates.py')
        tree = ast.parse((TASK / 'tests/evaluate.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                    and n.name == 'request')
        namespace = dict(asyncio=asyncio, json=json, time=time, MODEL_NAME='synthetic',
                         URL='http://unused', speech_text=parser.speech_text,
                         EvaluationFailure=gates.EvaluationFailure)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual request>', 'exec'), namespace)

        class Content:
            def __init__(self, chunks):
                self.chunks = chunks

            def __aiter__(self):
                async def iterate():
                    for chunk in self.chunks:
                        yield chunk
                return iterate()

            def iter_chunked(self, size):
                return self.__aiter__()

        class Response:
            status = 200

            def __init__(self, text, stream):
                self.headers = {'Content-Type': 'text/event-stream' if stream else 'application/json'}
                chunks = ([('data: ' + json.dumps({'type': 'transcript.text.delta', 'delta': text}) + '\n\n').encode(),
                           ('data: ' + json.dumps({'type': 'transcript.text.done', 'text': text}) + '\n\n').encode(),
                           b'data: [DONE]\n\n'] if stream
                          else [json.dumps({'text': text}).encode()])
                self.content = Content(chunks)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class Form:
            def add_field(self, *args, **kwargs):
                pass

        fake_http = SimpleNamespace(FormData=Form, ClientError=type('ClientError', (Exception,), {}))
        with mock.patch.dict(sys.modules, aiohttp=fake_http):
            for stream in (False, True):
                # Malformed markup is no longer an output-contract failure.
                for text in ('[S01]First.[S02]Second.[S01]Last.', '[0][S01]你好[1]',
                             '[S01]private answer[0.5]', 'plain text'):
                    session = SimpleNamespace(post=lambda *a, **kw: Response(text, stream))
                    protected = []
                    result = asyncio.run(namespace['request'](
                        session, (b'synthetic audio', 'unused reference', 3), 'en', stream, protected))
                    self.assertEqual(result['raw_text'], text)
                    self.assertEqual(result['text'], text)
                    self.assertIn(text, protected)
                    if stream:
                        self.assertEqual(''.join(result['deltas']), text)
                        self.assertIsNotNone(result['ttft'])

    def test_old_protocol_state_is_rejected_and_new_state_needs_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(cpu_set=None, prepare=True)
            with mock.patch.object(launcher, 'reserve_affinity', return_value=[]):
                controller = launcher.Controller(args, root, TASK, None, contract)
                self.assertEqual(controller.state['phase'], 'calibration')
                self.assertEqual(controller.state['protocol'], launcher.digest(TASK / 'tests/immutable-manifest.json'))
                (root / 'state.json').write_text(json.dumps({'phase': 'ready', 'protocol': 'old-protocol'}))
                with self.assertRaisesRegex(RuntimeError, 'state belongs to another protocol'):
                    launcher.Controller(args, root, TASK, None, contract)


if __name__ == '__main__':
    unittest.main(verbosity=2)
