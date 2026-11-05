"""Host-only production Harness launcher and full calibration preparation.

Must run in the pinned Harness Python environment, outside every task container.
No fake backend or external service supplies evaluation or comparison state.
"""
import argparse
from dataclasses import replace
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tarfile

# Host imports must not add bytecode to the locked, subsequently injected task.
sys.dont_write_bytecode = True


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, obj):
    temporary = path.with_name(path.name + '.' + secrets.token_hex(8))
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(obj, f, allow_nan=False, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def json_read(path):
    return json.loads(path.read_text())


def verify_tree(root, manifest):
    for rel, expected in manifest.items():
        p = root / rel
        if p.is_symlink() or not p.is_file() or digest(p) != expected:
            raise RuntimeError('immutable package changed: ' + str(p))
    expected_paths = set(manifest) | {'tests/comparison-state.json', 'tests/build-inventory.json',
                                      'tools/package-manifest.json'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != expected_paths:
        raise RuntimeError('package file set changed')


def reserve_affinity(raw):
    cpus = sorted(set(int(x) for x in raw.split(',')))
    if len(cpus) != 16 or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError('provide exactly 16 available physical CPU IDs')
    physical = set()
    for c in cpus:
        root = Path(f'/sys/devices/system/cpu/cpu{c}/topology')
        physical.add(((root/'physical_package_id').read_text().strip(), (root/'core_id').read_text().strip()))
    if len(physical) != 16:
        raise ValueError('CPU affinity must contain 16 distinct physical cores')
    return cpus


class Controller:
    def __init__(self, args, state_dir, task, docker, contract):
        self.args, self.root, self.task, self.docker, self.contract = args, state_dir, task, docker, contract
        self.manifest = json_read(task / 'tools/package-manifest.json')
        self.protocol = digest(task / 'tests/immutable-manifest.json')
        self.cpus = reserve_affinity(args.cpu_set)
        self.state_path = state_dir / 'state.json'
        self.run_id = None
        self.pending = None
        self.state = json_read(self.state_path) if self.state_path.exists() else {
            'phase': 'calibration', 'generation': 0, 'protocol': self.protocol}
        if self.state['protocol'] != self.protocol:
            raise RuntimeError('state belongs to another protocol')
        if args.prepare and self.state_path.exists():
            raise RuntimeError('preparation requires a new state directory; never cherry-pick a repeated calibration')
        if not args.prepare and self.state.get('phase') != 'ready':
            raise RuntimeError('complete B0 and A/A preparation is required before research')

    def save(self):
        atomic(self.state_path, self.state)

    def read_container_file(self, container, path):
        stream, _ = container.get_archive(path)
        data = b''.join(stream)
        if len(data) > 64 * 1024**2:
            raise RuntimeError('build inventory exceeds host parser bound')
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            members = [m for m in archive if m.isfile()]
            if len(members) != 1:
                raise RuntimeError('build inventory archive shape mismatch')
            return archive.extractfile(members[0]).read()

    def __call__(self, name, value):
        if name == 'gpu_plan':
            if len(value.authorized_pool.uuids) != 1 or value.work.uuids != value.judge.uuids or value.judge_mode.value != 'release-all':
                raise RuntimeError('requires one exclusive physical GPU in release-all mode')
            self.gpu = value.work.uuids[0]
        elif name == 'run_started':
            self.run_id = value
            self.state['run_id'] = value
            if not self.args.prepare:
                self.state['phase'] = 'research'
            self.save()
        elif name == 'work_started':
            work = self.docker.containers.get(value)
            work.update(cpuset_cpus=','.join(map(str, self.cpus)))
            raw = self.read_container_file(work, '/opt/asr-reference/build-inventory.json')
            build = json.loads(raw)
            if 'workspace' not in build or 'assets' not in build:
                raise RuntimeError('clean image lacks complete post-build inventory')
            if self.args.prepare:
                atomic(self.root / 'build-inventory.json', build)
            elif build != json_read(self.root / 'build-inventory.json'):
                raise RuntimeError('image or dependency bytes changed since calibration')
            atomic(self.task / 'tests/build-inventory.json', build)
            q = work.exec_run(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version', '--format=csv,noheader,nounits'])
            if q.exit_code != 0 or len(q.output.decode().strip().splitlines()) != 1:
                raise RuntimeError('fixed GPU identity preflight failed')
            hardware = {'gpu': q.output.decode().strip(), 'cpu_affinity': self.cpus}
            if not hardware['gpu'].startswith(self.gpu + ','):
                raise RuntimeError('GPU allocation and physical identity disagree')
            if not self.args.prepare and self.state['hardware'] != hardware:
                raise RuntimeError('hardware changed since calibration; recalibrate in a new trajectory')
            self.state['hardware'] = hardware
            self.state['build_inventory_sha256'] = digest(self.task / 'tests/build-inventory.json')
            self.save()
        elif name == 'judge_started':
            verify_tree(self.task, self.manifest)
            if self.state['phase'] not in ('calibration', 'aa', 'research'):
                raise RuntimeError('evaluation cannot start in current state')
            self.pending = {**self.state, 'round_id': value['round_id'], 'nonce': secrets.token_hex(32)}
            atomic(self.task / 'tests/comparison-state.json', self.pending)
        elif name == 'judge_exec_started':
            labels = [f'rsi-harness.run-id={self.run_id}', 'rsi-harness.role=judge',
                      'rsi-harness.round-id=' + value['round_id']]
            containers = self.docker.containers.list(filters={'label': labels})
            if len(containers) != 1:
                raise RuntimeError('cannot identify the active Judge for fixed CPU affinity')
            containers[0].update(cpuset_cpus=','.join(map(str, self.cpus)))
        elif name == 'judge_finished':
            self.finish(value)

    def finish(self, event):
        pending = self.pending
        if pending is None or event['round_id'] != pending['round_id']:
            raise RuntimeError('missing or stale pending evaluation')
        root = self.root / 'logs/runs' / self.run_id
        paths = list(root.rglob('submissions/' + pending['round_id'] + '/report.json'))
        if len(paths) != 1:
            raise RuntimeError('matching completed Harness report is missing')
        report = json_read(paths[0])
        task_log = paths[0].parent.parent.parent
        feedback = task_log / 'feedback' / (pending['round_id'] + '.log')
        self.pending = None
        if report['status'] != event['status'] or report['score'] != event['score']:
            raise RuntimeError('Harness event/report disagreement')
        envelopes = []
        if feedback.is_file():
            with feedback.open() as f:
                for line in f:
                    if line.startswith('ASR_AGGREGATE '):
                        envelopes.append(json.loads(line[len('ASR_AGGREGATE '):]))
        if len(envelopes) != 1:
            if self.args.prepare:
                self.state['phase'] = 'preparation_failed'
                self.save()
                raise RuntimeError('full calibration/A/A aggregate is missing; inspect host diagnostics')
            return
        result = envelopes[0]
        for key in ('run_id', 'round_id', 'nonce', 'protocol', 'hardware', 'phase', 'generation'):
            if result.get(key) != pending.get(key):
                raise RuntimeError('aggregate identity mismatch: ' + key)
        if not report['full_output_captured'] or report['timed_out'] or report['exit_code'] != 0:
            if self.args.prepare:
                raise RuntimeError('preparation did not complete cleanly')
            return
        if result['scored_requests'] != 18648 or result['warmup_requests'] != 18648:
            raise RuntimeError('aggregate request coverage mismatch')
        means, ranges = self.contract.aggregate(result['repeats'])
        if result['means'] != means or result['ranges'] != ranges:
            raise RuntimeError('aggregate arithmetic mismatch')
        reward_path = task_log / 'verifier' / pending['round_id'] / 'reward.json'
        phase = pending['phase']
        if phase == 'calibration':
            expected_reward = 1.0
            expected_failed = [k+':corpus_wer>0.02' for k in self.contract.WORKLOADS if means[k]['corpus_wer'] > 0.02]
        else:
            comparison = self.contract.compare(means, pending['b0']['means'], pending['bk']['means'])
            expected_reward, expected_failed = comparison['reward'], comparison['failed']
            for key in comparison:
                if key != 'reward' and result.get(key) != comparison[key]:
                    raise RuntimeError('comparison feedback mismatch: ' + key)
        if result['failed'] != expected_failed or result['accepted'] != (not expected_failed):
            raise RuntimeError('acceptance decision mismatch')
        if result['accepted']:
            if report['status'] != 'completed' or not reward_path.is_file():
                raise RuntimeError('accepted feedback lacks a matching final reward')
            reward = json_read(reward_path)
            if reward != {'reward': expected_reward} or result['reward'] != expected_reward or report['metrics'] != reward or report['score'] != expected_reward:
                raise RuntimeError('final reward, report and feedback disagree')
        else:
            if result['reward'] is not None or reward_path.exists() or report['status'] != 'verifier_error':
                raise RuntimeError('failed evaluation unexpectedly has a reward')
        measurement = {key: result[key] for key in ('repeats', 'means', 'ranges')}
        if phase == 'calibration':
            if expected_failed:
                raise RuntimeError('B0 failed fixed quality gates; preparation is unscored')
            self.state.update(b0=measurement, bk=measurement, phase='aa')
        elif phase == 'aa':
            self.state['aa'] = measurement
            unstable = []
            for key in self.contract.WORKLOADS:
                for metric, base in pending['b0']['means'][key].items():
                    current = means[key][metric]
                    if metric in ('coverage', 'warmup_coverage', 'corpus_wer', 'single_delta_fraction', 'empty_delta_count', 'last_delta_to_receipt_mean_s'):
                        continue
                    if current is not None and base not in (None, 0) and not 0.97 <= current/base <= 1.03:
                        unstable.append(key+':'+metric)
                if means[key]['corpus_wer'] > 0.02:
                    unstable.append(key+':quality')
            # A/A must not establish an apparent accepted optimization.
            if unstable or result['accepted']:
                self.state.update(phase='noise_review_required', aa_diagnostics=unstable,
                                  aa_would_accept=result['accepted'])
                self.save()
                raise RuntimeError('A/A noise can change the decision; resolve environment before research, without changing gates or selecting a better rerun')
            self.state['phase'] = 'ready'
        elif result['accepted']:
            self.state['bk'] = measurement
            self.state['generation'] += 1
        self.save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--cpu-set', required=True, help='16 reserved physical CPU IDs, comma-separated')
    p.add_argument('--model', required=True)
    p.add_argument('--reasoning-effort', required=True)
    p.add_argument('--primary-reward', choices=['reward'], default='reward')
    p.add_argument('--score-direction', choices=['maximize'], default='maximize')
    p.add_argument('--max-submissions', type=int, choices=[64], default=64)
    p.add_argument('--hours', type=int, choices=[24, 48], default=24)
    args = p.parse_args()
    if ',' in args.gpus:
        p.error('provide exactly one physical GPU selector')
    import docker
    import rsi_harness
    from rsi_harness.models import CompileOptions, RunRequest
    from rsi_harness.runtime.production import ProductionRuntimeServices
    from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
    source = Path(__file__).resolve().parents[1]
    harness_src = Path(rsi_harness.__file__).resolve().parent
    for rel, expected in json_read(source/'tools/harness-manifest.json').items():
        if digest(harness_src / rel) != expected:
            raise RuntimeError('launcher requires the reviewed Harness source interface: ' + rel)
    root = args.state_dir.expanduser().resolve()
    if args.prepare:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError('state directory must be a host-owned real directory')
    os.chmod(root, 0o700)
    with (root/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        task = root / source.name
        if args.prepare:
            verify_tree(source, json_read(source/'tools/package-manifest.json'))
            shutil.copytree(source, task)
        verify_tree(task, json_read(source/'tools/package-manifest.json'))
        spec = importlib.util.spec_from_file_location('asr_contract', task/'tests/contract.py')
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        client = docker.from_env(timeout=120)
        controller = Controller(args, root, task, client, contract)
        class PreparationAdapter(RSILoopAgentAdapter):
            def prepare(self, request):
                prepared = super().prepare(request)
                # Real Work, snapshot, GPU release, Judge and submission path;
                # preparation invokes no model Agent and alters no candidate.
                return replace(prepared, command=('/bin/bash', '-c', 'set -euo pipefail; rsi-submit; rsi-submit'))
        options = CompileOptions(primary_reward='reward', score_direction='maximize',
                                 max_submissions=2 if args.prepare else args.max_submissions,
                                 disable_stop_hook=args.prepare,
                                 agent_timeout_seconds=28800 if args.prepare else args.hours*3600)
        runtime = ProductionRuntimeServices(data_root=root/'runtime', logs_root=root/'logs',
                    docker_client=client, event_callback=controller,
                    **({'agent_adapter_factory': lambda config, work_runtime: PreparationAdapter(config, runtime=work_runtime)} if args.prepare else {}))
        result = runtime.run(RunRequest(task_dir=task, gpu_selectors=(args.gpus,), options=options,
                                       model=args.model, reasoning_effort=args.reasoning_effort))
        if args.prepare and controller.state.get('phase') != 'ready':
            raise RuntimeError('B0/A/A preparation incomplete; research remains disabled')
        if not args.prepare:
            controller.state['phase'] = 'finished'
            controller.save()
        print(result.model_dump_json(), flush=True)


if __name__ == '__main__':
    main()
