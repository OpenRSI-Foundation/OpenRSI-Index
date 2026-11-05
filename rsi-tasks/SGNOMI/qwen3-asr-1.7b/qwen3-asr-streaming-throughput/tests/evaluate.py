"""Full fixed HTTP evaluation. This file is never run during generation."""
import asyncio
from collections import deque
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import string
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave

# -I intentionally omits the script directory. Add only Judge-owned modules.
sys.path.insert(0, '/tests')
from contract import COUNTS, WORKLOADS, aggregate, compare
from gates import CandidateInvalid, EvaluationFailure, FIXED, authority_json, check, sha

TESTS = Path('/tests')
MODEL = '/opt/models/qwen3-asr-1.7b'
MODEL_NAME = 'Qwen/Qwen3-ASR-1.7B'
URL = 'http://127.0.0.1:18080'
PREFIX = 'ASR_AGGREGATE '


def emit(obj):
    print(json.dumps(obj, allow_nan=False, sort_keys=True), flush=True)


def protocol_id():
    return sha(TESTS / 'immutable-manifest.json')


def hardware():
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
                            '--format=csv,noheader,nounits'], check=False, text=True,
                           capture_output=True, timeout=30)
    except OSError:
        raise EvaluationFailure('infrastructure', 'gpu_query_unavailable',
                                'Check the NVIDIA driver tools and container GPU runtime.',
                                condition='nvidia-smi could not be launched.') from None
    except subprocess.TimeoutExpired:
        raise EvaluationFailure('infrastructure', 'gpu_query_timeout',
                                'Check driver health before repeating the unchanged evaluation.',
                                condition='nvidia-smi exceeded its 30-second query timeout.') from None
    if q.returncode:
        print(q.stderr, file=sys.stderr, flush=True)
        raise EvaluationFailure('infrastructure', 'gpu_query_failed', 'nvidia-smi failed; see its diagnostics.')
    rows = q.stdout.strip().splitlines()
    if len(rows) != 1:
        raise EvaluationFailure('infrastructure', 'gpu_count_mismatch',
                                'Expose the single exclusive GPU reserved for this run.',
                                condition=f'Expected one visible physical GPU; found {len(rows)}.')
    return {'gpu': rows[0], 'cpu_affinity': sorted(os.sched_getaffinity(0))}


def verify_state():
    manifest = authority_json(TESTS / 'immutable-manifest.json', 'fixed_manifest')
    for rel, expected in manifest.items():
        try:
            actual = sha(TESTS / rel)
        except OSError:
            raise EvaluationFailure('infrastructure', 'fixed_asset_unreadable',
                                    'Restore the complete host task copy and Judge injection.',
                                    condition='A required fixed evaluator or data asset cannot be read.') from None
        if actual != expected:
            raise EvaluationFailure('infrastructure', 'fixed_asset_digest_mismatch',
                                    'Restore the immutable evaluator/data assets from the host task copy.',
                                    condition='A fixed evaluator or data asset failed its SHA-256 check.')
    s = authority_json(TESTS / 'comparison-state.json', 'comparison_state')
    if s.get('phase') not in ('calibration', 'aa', 'research'):
        raise EvaluationFailure('infrastructure', 'comparison_preparation_required',
                                'Run the documented host launcher with --prepare before research.',
                                path=TESTS / 'comparison-state.json', field='phase',
                                condition='No prepared calibration, A/A or research phase is present.')
    if s.get('protocol') != protocol_id() or not s.get('run_id') or not s.get('round_id') or not s.get('nonce'):
        raise EvaluationFailure('infrastructure', 'comparison_identity_mismatch',
                                'Use the locked host launcher and its current run-specific state.',
                                path=TESTS / 'comparison-state.json', field='protocol/run_id/round_id/nonce',
                                condition='Host comparison identity is missing or stale.')
    if s.get('hardware') != hardware():
        raise EvaluationFailure('infrastructure', 'hardware_affinity_mismatch',
                                'Restore the calibrated GPU, driver and reserved CPU affinity; recalibrate if hardware intentionally changes.',
                                field='hardware', condition='GPU identity, driver or reserved CPU affinity differs from prepared host state.')
    try:
        inventory_hash = sha(TESTS / 'build-inventory.json')
    except OSError:
        raise EvaluationFailure('infrastructure', 'build_inventory_unreadable',
                                'Restore the host-injected pristine build inventory.',
                                path=TESTS / 'build-inventory.json',
                                condition='Host build inventory cannot be read.') from None
    if s.get('build_inventory_sha256') != inventory_hash:
        raise EvaluationFailure('infrastructure', 'build_inventory_identity_mismatch',
                                'Inject the pristine inventory captured for this prepared run.',
                                path=TESTS / 'build-inventory.json', condition='Build inventory digest differs from host state.')
    if s['phase'] != 'calibration':
        for name in ('b0', 'bk'):
            try:
                means, ranges = aggregate(s[name]['repeats'])
                consistent = means == s[name]['means'] and ranges == s[name]['ranges']
            except (KeyError, TypeError, ValueError):
                raise EvaluationFailure('infrastructure', 'comparison_aggregate_schema',
                                        'Restore complete B0/Bk aggregates from the prepared host state.',
                                        field=name, condition='Host baseline repetitions or aggregate schema are incomplete or invalid.') from None
            if not consistent:
                raise EvaluationFailure('infrastructure', 'comparison_aggregate_mismatch',
                                        'Restore host state; do not edit measured baseline aggregates.',
                                        field=name, condition='Recorded baseline means/ranges disagree with their repetitions.')
    return s


def load_rows():
    import pyarrow.parquet as pq
    manifest = authority_json(TESTS / 'data-manifest.json', 'data_manifest')
    rows = {}
    for lang, count in COUNTS.items():
        entries = []
        for shard in manifest[lang]['shards']:
            if shard['start'] != len(entries):
                raise EvaluationFailure('infrastructure', 'dataset_order_mismatch', 'Restore native shard order from the fixed task data manifest.')
            path = TESTS / shard['path']
            if sha(path) != shard['sha256']:
                raise EvaluationFailure('infrastructure', 'dataset_digest_mismatch', 'Restore the fixed dataset shards; a shard digest differs from its manifest.')
            table = pq.read_table(path)
            if table.num_rows != shard['rows']:
                raise EvaluationFailure('infrastructure', 'dataset_shard_coverage', 'Restore the fixed dataset shards; a shard row count differs from its manifest.')
            for row in table.to_pylist():
                audio = row['ref_audio']['bytes']
                with wave.open(io.BytesIO(audio), 'rb') as w:
                    duration = w.getnframes() / w.getframerate()
                if not math.isfinite(duration) or duration <= 0 or not isinstance(row['ref_text'], str):
                    raise EvaluationFailure('infrastructure', 'dataset_input_invalid', 'Restore fixed dataset assets; audio duration or reference-text type is invalid.')
                # No filenames, IDs, or waveform hashes are used as row keys.
                entries.append((audio, row['ref_text'], duration))
        if len(entries) != count or manifest[lang]['rows'] != count:
            raise EvaluationFailure('infrastructure', 'dataset_coverage', 'Restore the complete fixed dataset; full-language row coverage is incomplete.')
        rows[lang] = entries
    return rows


def normalizer():
    from whisper.normalizers import EnglishTextNormalizer
    from zhon.hanzi import punctuation
    en = EnglishTextNormalizer()
    def normalize(text, lang):
        if lang == 'en':
            return en(text)
        for ch in punctuation + string.punctuation:
            if ch != "'":
                text = text.replace(ch, '')
        text = text.replace(' ', '').replace('\u3000', '').strip()
        return ' '.join(list(text))
    return normalize


async def request(session, row, lang, stream, protected):
    import aiohttp
    form = aiohttp.FormData()
    for k, v in [('model', MODEL_NAME), ('language', lang), ('response_format', 'json')]:
        form.add_field(k, v)
    if stream:
        form.add_field('stream', 'true')
    form.add_field('file', row[0], filename='audio.wav', content_type='audio/wav')
    headers = {'x-sglang-omni-route-stream': 'true'} if stream else None
    start = time.perf_counter()
    deltas, times, empty = [], [], 0
    final, done = None, False
    try:
        async with session.post(URL + '/v1/audio/transcriptions', data=form, headers=headers, allow_redirects=False) as response:
            if response.status != 200:
                # HTTP bodies may contain reference inputs or predicted answers.
                raise EvaluationFailure('candidate_runtime', 'http_status', f'HTTP {response.status}; inspect the sanitized server diagnostics.')
            if stream:
                if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                    raise EvaluationFailure('candidate_invalid', 'sse_content_type', 'Return text/event-stream for streaming requests.')
                size = 0
                async for raw in response.content:
                    size += len(raw)
                    if size > 4 * 1024**2:
                        raise EvaluationFailure('candidate_invalid', 'response_size', 'Response exceeds 4 MiB safety bound.')
                    line = raw.decode('utf-8').strip()
                    if not line or line.startswith(':') or line.startswith('event:'):
                        continue
                    if not line.startswith('data:') or done:
                        raise EvaluationFailure('candidate_invalid', 'sse_order', 'Malformed event or data after DONE.')
                    payload = line[5:].lstrip()
                    if payload == '[DONE]':
                        if final is None:
                            raise EvaluationFailure('candidate_invalid', 'sse_done_order', 'Final text must precede DONE.')
                        done = True
                        continue
                    item = json.loads(payload)
                    if not isinstance(item, dict):
                        raise ValueError('event must be an object')
                    typ = item.get('type')
                    if typ == 'transcript.text.delta':
                        delta = item.get('delta')
                        if final is not None or not isinstance(delta, str):
                            raise ValueError('invalid delta ordering or type')
                        if delta:
                            deltas.append(delta)
                            times.append(time.perf_counter())
                        else:
                            empty += 1
                    elif typ == 'transcript.text.done':
                        if final is not None or not isinstance(item.get('text'), str):
                            raise ValueError('invalid final text')
                        final = item['text']
                        protected.append(final)
                    else:
                        raise ValueError('unsupported or error event')
                if not done or final is None or not deltas or ''.join(deltas) != final:
                    raise ValueError('incomplete or inconsistent stream')
            else:
                parts, size = [], 0
                async for part in response.content.iter_chunked(65536):
                    size += len(part)
                    if size > 4 * 1024**2:
                        raise ValueError('oversized response')
                    parts.append(part)
                raw = b''.join(parts)
                item = json.loads(raw)
                if not isinstance(item, dict) or not isinstance(item.get('text'), str):
                    raise ValueError('response must contain string text')
                final = item['text']
                protected.append(final)
    except asyncio.TimeoutError:
        raise EvaluationFailure('candidate_runtime', 'request_timeout', 'A request exceeded the fixed 300-second timeout; no retries are allowed.') from None
    except (UnicodeError, ValueError, aiohttp.ClientError):
        raise EvaluationFailure('candidate_invalid', 'response_protocol', 'Malformed HTTP/SSE output; preserve event order, final text, DONE and delta concatenation.') from None
    finally:
        # Retain the complete partial answer even on timeout/protocol failure.
        # Individual deltas are not substring-redaction tokens: a delta may be "a".
        if deltas:
            protected.append(''.join(deltas))
    end = time.perf_counter()
    return {'text': final.strip(), 'raw_text': final, 'deltas': deltas, 'latency': end-start,
            'ttft': times[0]-start if times else None,
            'intervals': [b-a for a, b in zip(times, times[1:])],
            'tail': end-times[-1] if times else None, 'empty': empty}


async def pass_requests(rows, lang, stream, protected):
    import aiohttp
    queue = deque(enumerate(rows))
    outputs = [None] * len(rows)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300),
                                     connector=aiohttp.TCPConnector(limit=32), trust_env=False) as session:
        async def worker():
            while queue:
                index, row = queue.popleft()
                outputs[index] = await request(session, row, lang, stream, protected)
        started = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(32)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        wall = time.perf_counter() - started
    if any(x is None for x in outputs) or wall <= 0:
        raise EvaluationFailure('incomplete', 'coverage', 'Every native row must complete exactly once.')
    return outputs, wall


def metrics(rows, outputs, wall, lang, stream, normalize):
    import numpy as np
    from jiwer import process_words
    errors = words = 0
    for row, out in zip(rows, outputs, strict=True):
        ref, hyp = normalize(row[1], lang), normalize(out['text'], lang)
        if not ref:
            raise EvaluationFailure('infrastructure', 'reference_normalization_empty', 'Check the fixed dataset and normalization dependencies; a reference normalizes to empty text.')
        result = process_words(ref, hyp)
        errors += result.substitutions + result.deletions + result.insertions
        words += result.substitutions + result.deletions + result.hits
    lat = [x['latency'] for x in outputs]
    rtfs = [x['latency']/row[2] for row, x in zip(rows, outputs, strict=True)]
    def p95(values):
        return float(np.percentile(values, 95, method='linear'))
    m = {'qps': len(rows)/wall, 'rtfx': sum(r[2] for r in rows)/wall,
         'latency_mean_s': statistics.mean(lat), 'latency_p95_s': p95(lat),
         'rtf_mean': statistics.mean(rtfs), 'rtf_p95': p95(rtfs),
         'corpus_wer': errors/words, 'coverage': len(outputs), 'warmup_coverage': len(rows)}
    if stream:
        ttft = [x['ttft'] for x in outputs]
        intervals = [v for x in outputs for v in x['intervals']]
        m.update(ttft_mean_s=statistics.mean(ttft), ttft_p95_s=p95(ttft),
                 inter_chunk_mean_s=statistics.mean(intervals) if intervals else None,
                 inter_chunk_p95_s=p95(intervals) if intervals else None,
                 single_delta_fraction=sum(len(x['deltas']) == 1 for x in outputs)/len(outputs),
                 empty_delta_count=sum(x['empty'] for x in outputs),
                 last_delta_to_receipt_mean_s=statistics.mean(x['tail'] for x in outputs))
    return m


def scrub(lines, protected):
    # Child logs actually contain request params and may include model outputs.
    # Preserve stack locations and ordinary resource/compile diagnostics.
    secrets = {part.strip() for s in protected for part in s.splitlines() if part.strip()}
    # Whole answer fields/regions are protected regardless of their length.
    # Match labels, not words in a source path or an unrelated exception.
    fields = re.compile(r'''(?i)(?:["']?\b(?:prompt|transcript|ref_text|hyp_text|audio_fingerprint|payload|delta|text|content)\b["']?\s*[:=]\s*|sampling temp\s*[:=]\s*)''')
    long_text = sorted((s for s in secrets if len(s) >= 8), key=len, reverse=True)
    pattern = re.compile('|'.join(re.escape(s) for s in long_text)) if long_text else None
    for line in lines:
        field = fields.search(line)
        if field:
            # The remainder is an answer-bearing region, possibly malformed JSON.
            line = line[:field.end()] + '[protected request/output fields redacted]'
        elif line.strip() in secrets:
            line = '[protected text]'
        if pattern:
            line = pattern.sub('[protected text]', line)
        # No child can forge the host's result envelope through its output.
        print('server: ' + line, flush=True)


@contextlib.contextmanager
def service(cfg, protected):
    with tempfile.TemporaryDirectory(prefix='asr-judge-', dir='/tmp') as tmp:
        scratch = Path(tmp)
        env = {k: os.environ[k] for k in ('PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
                                         'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES') if k in os.environ}
        env.update(HOME=tmp, TMPDIR=tmp, XDG_CACHE_HOME=tmp+'/cache',
                   PYTHONPATH='/workspace/sglang-omni', PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', TORCHINDUCTOR_CACHE_DIR=tmp+'/inductor',
                   TRITON_CACHE_DIR=tmp+'/triton', TORCH_EXTENSIONS_DIR=tmp+'/extensions',
                   CUDA_CACHE_PATH=tmp+'/cuda', FLASHINFER_WORKSPACE_BASE=tmp+'/flashinfer',
                   TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE='0', TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE='0')
        command = ['/opt/asr-venv/bin/python', '-B', '-m', 'sglang_omni.cli', 'serve',
                   '--model-path', MODEL, '--model-name', MODEL_NAME, '--host', '127.0.0.1',
                   '--port', '18080', '--skip-server-warmup']
        for key, value in {**cfg, **FIXED}.items():
            command += ['--'+key, str(value).lower() if isinstance(value, bool) else str(value)]
        try:
            proc = subprocess.Popen(command, cwd=tmp, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except OSError:
            raise EvaluationFailure('infrastructure', 'service_launch_failed',
                                    'Restore the baked Python executable and check its execution permissions.',
                                    path=command[0], condition='The fixed service launcher could not be executed.') from None
        captured = deque(maxlen=2000)
        def drain():
            while True:
                raw = proc.stdout.readline(16385)
                if not raw:
                    break
                if len(raw) > 16384:
                    # Drop the entire oversized line, including all continuations.
                    while raw and not raw.endswith(b'\n'):
                        raw = proc.stdout.readline(16385)
                    captured.append('[oversized diagnostic line omitted]')
                else:
                    captured.append(raw.decode('utf-8', errors='replace').rstrip())
        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        try:
            import requests
            session = requests.Session()
            session.trust_env = False
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise EvaluationFailure('candidate_runtime', 'service_exit', f'Service exited with code {proc.returncode}; inspect server diagnostics.')
                try:
                    if session.get(URL+'/health', timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                raise EvaluationFailure('incomplete', 'readiness_timeout', 'Service did not become ready within 600 seconds.')
            session.close()
            yield proc
            if proc.poll() is not None:
                raise EvaluationFailure('candidate_runtime', 'service_exit', 'Service exited before pass cleanup.')
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=15)
            thread.join(timeout=10)
            if thread.is_alive():
                raise EvaluationFailure('incomplete', 'service_cleanup_incomplete', 'Service output remained open after process-group termination; check child cleanup before resubmitting.')
            scrub(captured, protected)
            # Scratch disappears before a final reward can be written.


def main():
    state = verify_state()
    cfg = check()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EvaluationFailure('infrastructure', 'cuda_preflight_failed', 'Check the fixed Torch/CUDA installation, driver and single-GPU visibility.')
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        raise EvaluationFailure('infrastructure', 'cuda_synchronize_failed', 'The fixed CUDA runtime could not synchronize before candidate launch; check driver/device health.') from None
    try:
        rows = load_rows()
    except (OSError, ValueError, KeyError, TypeError, wave.Error, EOFError):
        raise EvaluationFailure('infrastructure', 'dataset_decode_failed', 'Check the fixed Parquet/audio assets and PyArrow installation; dataset decoding failed before candidate launch.') from None
    normalize = normalizer()
    protected = [row[1] for group in rows.values() for row in group]
    repeats = {key: [] for key in WORKLOADS}
    for key in WORKLOADS:
        mode, lang = key.split('/')
        for repetition in range(3):
            with service(cfg, protected) as proc:
                warm, _ = asyncio.run(pass_requests(rows[lang], lang, mode == 'stream', protected))
                del warm
                torch.cuda.synchronize()
                outputs, wall = asyncio.run(pass_requests(rows[lang], lang, mode == 'stream', protected))
                torch.cuda.synchronize()
                result = metrics(rows[lang], outputs, wall, lang, mode == 'stream', normalize)
                repeats[key].append(result)
                del outputs
            emit({'status': 'repeat_complete', 'workload': key, 'repetition': repetition+1, 'metrics': result})
    means, ranges = aggregate(repeats)
    result = {'run_id': state['run_id'], 'round_id': state['round_id'], 'nonce': state['nonce'],
              'protocol': state['protocol'], 'hardware': state['hardware'],
              'phase': state['phase'], 'generation': state['generation'],
              'repeats': repeats, 'means': means, 'ranges': ranges,
              'scored_requests': 18648, 'warmup_requests': 18648}
    if state['phase'] == 'calibration':
        failures = [key+':corpus_wer>0.02' for key in WORKLOADS if means[key]['corpus_wer'] > 0.02]
        result.update(failed=failures, reward=1.0)
    else:
        result.update(compare(means, state['b0']['means'], state['bk']['means']))
    # A/A is the full normal comparison and remains unscored if strict improvement fails.
    result['accepted'] = not result['failed']
    if not result['accepted']:
        result['reward'] = None
    print(PREFIX + json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    if result['accepted']:
        with open('/logs/verifier/reward.json', 'x', encoding='utf-8') as f:
            json.dump({'reward': result['reward']}, f, allow_nan=False)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        main()
    except CandidateInvalid as exc:
        emit({'status': 'candidate_invalid', 'errors': exc.errors})
        sys.exit(1)
    except EvaluationFailure as exc:
        emit({'status': exc.category, 'code': exc.code, 'condition': exc.condition,
              'path': exc.path, 'field': exc.field, 'hint': exc.hint})
        sys.exit(1)
    except ImportError as exc:
        # Do not echo import exception messages: they can include protected source.
        emit({'status': 'infrastructure', 'code': 'dependency_import_failed',
              'module': exc.name, 'condition': 'A required dependency could not be imported.',
              'hint': 'Restore the locked Environment dependencies and validate their imports/ABI before research.'})
        sys.exit(1)
    except BaseException as exc:
        # Scoring implementation and data-derived exception values are protected.
        # Reveal the exception class and public candidate frames, never locals.
        frames = [{'path': f.filename, 'line': f.lineno, 'function': f.name}
                  for f in traceback.extract_tb(exc.__traceback__)
                  if f.filename.startswith('/workspace/')]
        emit({'status': 'infrastructure_or_incomplete', 'exception': type(exc).__name__,
              'candidate_frames': frames,
              'hint': 'Check prepared host state, fixed assets, CUDA availability and sanitized service diagnostics; no reward was written.'})
        sys.exit(1)
