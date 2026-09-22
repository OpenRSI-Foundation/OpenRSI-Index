"""Candidate-only Judge. No training, fitting, downloads, or live baseline pass."""
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import (CandidateInvalid, candidate_errors, clean_request, error,
                      finalize_reward, score_records, validate_prediction)

ROOT = Path(__file__).resolve().parent
REWARD = Path('/logs/verifier/reward.json')
PREDICTION_FAILURES = {
    'candidate_memory_error': 'candidate inference raised a memory-allocation error',
    'candidate_runtime_error': 'candidate inference raised a runtime error (including tensor/device/kernel or CUDA allocation errors)',
    'candidate_value_error': 'candidate inference raised a value/type/index/key error',
    'candidate_execution_error': 'candidate inference raised an exception',
    'inference_memory_exceeded': 'peak Torch-allocated inference memory must not exceed 16 GiB',
}


def assert_isolated(together, alone):
    if len(together) != len(alone) or max(abs(a - b) for a, b in zip(together, alone)) > 1e-3:
        raise CandidateInvalid([error('question_isolation_failed', '/workspace/candidate/entrypoint.py',
                                    'predict', 'asking another question, reordering siblings, or reusing state must not change this question beyond absolute tolerance 1e-3')])


class PredictorProcess:
    def __init__(self, scratch):
        env = {'PATH': '/opt/kev-venv/bin:/usr/local/bin:/usr/bin:/bin',
               'HOME': str(scratch), 'TMPDIR': str(scratch), 'XDG_CACHE_HOME': str(scratch / 'cache'),
               'HF_HOME': str(scratch / 'hf'), 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8',
               'CUDA_VISIBLE_DEVICES': '0'}
        # Preserve only container GPU-loader configuration, never Agent API keys.
        for name in ('LD_LIBRARY_PATH', 'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES'):
            if name in os.environ:
                env[name] = os.environ[name]
        self.load_log = tempfile.TemporaryFile()
        self.phase = 'load'
        self.process = subprocess.Popen([sys.executable, '-I', '-B', '-u', str(ROOT / 'worker.py'), str(scratch)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.load_log, env=env, cwd='/')
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b''
        self.deadline = time.monotonic() + 1200
        self.peak = 0
        try:
            self.ready = self.receive(180)
            if self.ready.get('status') != 'ready':
                raise RuntimeError('candidate worker did not complete model loading')
            self.phase = 'predict'
        except BaseException:
            self.close()
            raise
        finally:
            # No request has been sent yet: load-time logs cannot contain hidden
            # inputs. Read only the existing descriptor, never a candidate path.
            if not self.load_log.closed:
                self.load_log.seek(0)
                sys.stderr.write(self.load_log.read().decode('utf-8', errors='replace'))
                self.load_log.close()

    def receive(self, timeout=45):
        deadline = min(self.deadline, time.monotonic() + timeout)
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError('candidate inference did not finish within the published time budget; no reward')
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f'candidate process ended without complete evaluation (exit {self.process.poll()}); no reward')
            self.buffer += chunk
            if len(self.buffer) > 2 * 1024**2:
                raise CandidateInvalid([error('prediction_too_large', '/workspace/candidate/entrypoint.py',
                                            'predict', 'each prediction response must be at most 2 MiB')])
        line, self.buffer = self.buffer.split(b'\n', 1)
        message = json.loads(line)
        if message.get('status') in ('candidate_invalid', 'candidate_error'):
            if self.phase == 'predict':
                code = message.get('code')
                if code not in PREDICTION_FAILURES:
                    code = 'candidate_execution_error'
                raise CandidateInvalid([error(code, '/workspace/candidate/entrypoint.py', 'predict',
                                            PREDICTION_FAILURES[code],
                                            hint='Reproduce with /opt/kev-runtime/evaluate_public.py for full diagnostics; hidden-input exception metadata is not returned.')])
            raise CandidateInvalid([error(message.get('code', 'candidate_execution_error'),
                                        '/workspace/candidate/entrypoint.py', message.get('phase', 'load'),
                                        message.get('condition', 'candidate implementation raised an exception'),
                                        detail={k: v for k, v in message.items() if k != 'status'})])
        if message.get('status') == 'infrastructure_error':
            if self.phase == 'predict':
                raise RuntimeError('worker reported an unexpected runtime failure after inference started; no reward')
            raise RuntimeError(json.dumps(message))
        return message

    def predict(self, request):
        payload = json.dumps(request, allow_nan=False).encode() + b'\n'
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        message = self.receive()
        if message.get('status') != 'prediction':
            raise RuntimeError('unexpected worker protocol state; no reward')
        self.peak = max(self.peak, message['peak_memory_bytes'])
        validate_prediction(request, message['response'])
        return message['response']

    def close(self):
        self.selector.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()


def load_fixed_data():
    manifest = json.loads((ROOT / 'assets/manifest.json').read_text())
    data = (ROOT / 'assets/evaluation.jsonl').read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest['sha256']:
        raise RuntimeError('task-owned evaluation asset identity mismatch')
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(rows) != manifest['records'] or dict(Counter(row['_meta']['source'] for row in rows)) != manifest['source_counts']:
        raise RuntimeError('task-owned evaluation coverage mismatch')
    return rows


def verify_fixed_inputs(expected):
    problems = []
    for filename, digest in expected.items():
        path = Path(filename)
        if path.is_symlink() or not path.is_file():
            problems.append(error('missing_fixed_asset', path, '', 'fixed input must remain a regular file at its original path'))
            continue
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
        if hasher.hexdigest() != digest:
            problems.append(error('fixed_asset_modified', path, '', 'restore the original fixed input; changes belong under /workspace/candidate', expected_sha256=digest))
    if problems:
        raise CandidateInvalid(problems)


def check_workspace_scope(root=Path('/workspace')):
    problems = [error('editable_scope_violation', path, '',
                      'keep task edits, checkpoints and research notes inside candidate/ or experiments/')
                for path in root.iterdir() if path.name not in {'candidate', 'experiments'}]
    if problems:
        raise CandidateInvalid(problems)


def run():
    rows = load_fixed_data()
    verify_fixed_inputs(json.loads((ROOT / 'assets/fixed-inputs.json').read_text())['files'])
    check_workspace_scope()
    problems = candidate_errors('/workspace/candidate')
    if problems:
        raise CandidateInvalid(problems)
    # Preflight the fixed runtime before assigning failures to candidate code.
    import torch
    import transformers
    import peft
    if not torch.cuda.is_available():
        raise RuntimeError('Judge requires one usable CUDA GPU; no reward')
    free, _ = torch.cuda.mem_get_info(0)
    if free < 18 * 1024**3:
        raise RuntimeError('Judge GPU has less than 18 GiB free at preflight; no reward')
    probe = torch.ones(1, device='cuda:0')
    torch.cuda.synchronize()
    del probe
    torch.cuda.empty_cache()
    # Root parent holds labels. Child is another UID with a filesystem allowlist.
    import ctypes
    if ctypes.CDLL(None).prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
        raise RuntimeError('unable to protect Judge process memory')
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='kev-candidate-') as td:
        scratch = Path(td)
        os.chown(scratch, 65534, 65534)
        worker = None
        try:
            worker = PredictorProcess(scratch)
            outputs = []
            for record in rows:
                outputs.append(worker.predict(clean_request(record)))
            # Check isolation on held-out inputs without exposing labels or IDs.
            for index in range(16):
                request = clean_request(rows[index])
                qid = next(iter(request['questions']))
                single = copy.deepcopy(request)
                single['questions'] = {qid: request['questions'][qid]}
                reference = validate_prediction(single, worker.predict(single))[qid]
                combined = copy.deepcopy(single)
                combined['questions']['independent_probe'] = {
                    'type': 'noul', 'instructions': 'This is a separate question. Is the state empty?'}
                together = validate_prediction(combined, worker.predict(combined))[qid]
                assert_isolated(reference, together)
                combined['questions'] = dict(reversed(list(combined['questions'].items())))
                reordered = validate_prediction(combined, worker.predict(combined))[qid]
                assert_isolated(reference, reordered)
                assert_isolated(reference, validate_prediction(single, worker.predict(single))[qid])
            summary = score_records(rows, outputs)
            summary.update(parameters=worker.ready['parameters'], peak_inference_bytes=worker.peak,
                           evaluation_seconds=time.monotonic() - start, question_isolation='passed')
            return summary
        finally:
            if worker is not None:
                worker.close()


def main():
    try:
        summary = run()
    except CandidateInvalid as exc:
        print(json.dumps({'status': 'candidate_invalid', 'errors': exc.errors, 'reward': 0.0}), flush=True)
        finalize_reward(REWARD, 0.0)
        return
    # Other exceptions deliberately propagate without a reward. Fixed data,
    # loader, timeout and incomplete-execution failures are not quality scores.
    print(json.dumps({'status': 'complete', **summary}, allow_nan=False), flush=True)
    finalize_reward(REWARD, summary['reward'])


if __name__ == '__main__':
    main()
