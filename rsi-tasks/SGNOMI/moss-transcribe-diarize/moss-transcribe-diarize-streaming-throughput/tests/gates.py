"""Judge-owned structural gates; no candidate module is imported."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import stat

ROOT = Path('/tests')
WORK = Path('/workspace')
SURFACES = ('models/moss_transcribe_diarize', 'scheduling', 'comm', 'http', 'serve',
            'model_runner', 'pipeline', 'relay', 'sampling', 'utils', 'kernels')
PREFIXES = tuple('sglang-omni/sglang_omni/' + s for s in SURFACES)
FIXED = {'asr.factory.dtype': 'bfloat16', 'asr.engine.enable_torch_compile': True, 'asr.engine.disable_radix_cache': True, 'asr.factory.stream_emit_interval_s': 0.05, 'asr.factory.encoder_cache_size_bytes': 0, 'asr.factory.mm_embedding_cache_size_bytes': 0, 'asr.factory.max_new_tokens': None, 'asr.factory.context_length': None}
KNOBS = {
    'asr.factory.encoder_max_batch_size': (int, 1),
    'asr.factory.request_build_max_workers': (int, 1),
    'asr.factory.request_build_max_pending': (int, 1),
    'asr.factory.prefill_coalesce_requests': (int, 0),
    'asr.factory.prefill_coalesce_wait_ms': (float, 0),
    'asr.factory.prefill_coalesce_when_idle': (bool, 0),
    'asr.factory.prefill_coalesce_requires_pending_builds': (bool, 0),
    'asr.factory.prefill_coalesce_after_builds_during_decode': (bool, 0),
    'asr.engine.max_running_requests': (int, 1),
    'asr.engine.torch_compile_max_bs': (int, 1),
    'asr.engine.cuda_graph_max_bs': (int, 1),
    'asr.engine.disable_cuda_graph': (bool, 0),
    'asr.engine.mem_fraction_static': (float, 0),
    'asr.engine.kv_cache_bytes': (int, 1),
}
BYTE_CAPACITY_KEYS = {'asr.engine.kv_cache_bytes'}


class CandidateInvalid(Exception):
    def __init__(self, errors):
        self.errors = errors


class EvaluationFailure(Exception):
    def __init__(self, category, code, hint, *, path='', field='', condition=''):
        self.category, self.code, self.hint = category, code, hint
        self.path, self.field, self.condition = str(path), field, condition or hint


def authority_json(path, code):
    try:
        value = json.loads(path.read_text())
    except OSError:
        raise EvaluationFailure('infrastructure', code + '_unreadable',
                                'Restore the host-injected task inputs and check file access.',
                                path=path, condition='Required authority file cannot be read.') from None
    except (ValueError, UnicodeError):
        raise EvaluationFailure('infrastructure', code + '_invalid_json',
                                'Restore this file from the prepared host task copy.',
                                path=path, condition='Required authority file is not valid JSON.') from None
    if not isinstance(value, dict):
        raise EvaluationFailure('infrastructure', code + '_schema',
                                'Restore this file from the prepared host task copy.',
                                path=path, condition='Authority input must be a JSON object.')
    return value


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def describe(p):
    mode = p.lstat().st_mode
    if stat.S_ISLNK(mode):
        return {'type': 'symlink', 'target': os.readlink(p)}
    if stat.S_ISDIR(mode):
        return {'type': 'directory'}
    if stat.S_ISREG(mode):
        return {'type': 'file', 'sha256': sha(p)}
    return {'type': 'special'}


def allowed(rel):
    return rel == 'candidate.yaml' or rel == 'research' or rel.startswith('research/') or any(
        rel == p or rel.startswith(p + '/') for p in PREFIXES)


def error(code, path, condition, field='', hint='Restore the fixed baseline or repair this candidate file.'):
    return dict(code=code, path=str(path), field=field, condition=condition, hint=hint)


def check():
    import yaml
    baseline = authority_json(ROOT / 'build-inventory.json', 'build_inventory')
    if 'workspace' not in baseline or 'assets' not in baseline:
        raise EvaluationFailure('infrastructure', 'build_inventory_unprepared',
                                'Run host preparation to capture the pristine post-build inventory.',
                                path=ROOT / 'build-inventory.json',
                                condition='Prepared workspace and asset inventories are missing.')
    errors = []
    actual = {p.relative_to(WORK).as_posix(): p for p in WORK.rglob('*')}
    for rel in set(actual) | set(baseline['workspace']):
        p = WORK / rel
        old = baseline['workspace'].get(rel)
        if allowed(rel):
            if rel in actual:
                typ = describe(p)['type']
                if typ not in ('file', 'directory'):
                    errors.append(error('candidate_path_type', p, 'Only regular files and directories are allowed.'))
                elif p.is_file() and p.suffix == '.py' and not rel.startswith('research/'):
                    try:
                        if p.stat().st_size > 8 * 1024**2:
                            raise ValueError('Python source exceeds 8 MiB parser bound')
                        ast.parse(p.read_text())
                    except (SyntaxError, UnicodeError, ValueError, RecursionError) as exc:
                        field = f'line {exc.lineno}' if isinstance(exc, SyntaxError) else ''
                        errors.append(error('candidate_python_syntax', p, type(exc).__name__, field, hint='Check encoding, syntax, nesting and source size.'))
        elif rel not in actual or describe(p) != old:
            errors.append(error('protected_workspace_path', p, 'Protected source or workspace inventory changed.'))
    for root, expected in baseline['assets'].items():
        base = Path(root)
        current = {p.relative_to(base).as_posix(): p for p in base.rglob('*')}
        if root == '/opt/asr-reference':
            current.pop('build-inventory.json', None)
        for rel in set(current) | set(expected):
            if rel not in current or describe(base / rel) != expected.get(rel):
                errors.append(error('protected_environment_asset', base / rel, 'Fixed model, dependency or restoration asset changed.'))
    policy = json.loads((ROOT / 'policy.json').read_text())
    for relative in set(policy['fixed_nodes']) | set(policy['fixed_assignments']):
        symbols = policy['fixed_nodes'].get(relative, {})
        p = WORK / 'sglang-omni' / relative
        try:
            if p.is_symlink() or p.stat().st_size > 8 * 1024**2:
                raise ValueError('fixed semantic source has invalid type or size')
            tree = ast.parse(p.read_text())
            nodes = {}
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nodes.setdefault(n.name, []).append(ast.dump(n, include_attributes=False))
            for name, expected in symbols.items():
                if nodes.get(name) != [expected]:
                    errors.append(error('fixed_semantic_function', p, 'Fixed preprocessing, token budget or cold-input policy changed.', name))
            assignments = policy['fixed_assignments'].get(relative, {})
            found = {name: [] for name in assignments}
            for n in ast.walk(tree):
                if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id in found:
                            found[target.id].append(ast.dump(n, include_attributes=False))
            for name, expected in assignments.items():
                if found[name] != expected:
                    errors.append(error('fixed_decoding_semantics', p, 'Fixed language, greedy decoding or token-budget assignment changed.', name))
        except (OSError, SyntaxError, UnicodeError, ValueError, RecursionError):
            errors.append(error('fixed_semantic_source_missing', p, 'Required semantic functions cannot be read.'))
    p = WORK / 'candidate.yaml'
    cfg = {}
    try:
        if not p.is_file() or p.is_symlink() or p.stat().st_size > 65536:
            raise ValueError('requires a regular YAML file of at most 64 KiB')
        cfg = yaml.safe_load(p.read_text())
        if not isinstance(cfg, dict) or len(cfg) > len(KNOBS):
            raise ValueError('requires a mapping of documented flat configuration keys')
        for key, value in cfg.items():
            if not isinstance(key, str) or key not in KNOBS:
                errors.append(error('unknown_execution_knob', p, 'Only documented execution knobs are permitted.', str(key)[:200]))
                continue
            typ, lower = KNOBS[key]
            valid = type(value) is typ if typ is not float else type(value) in (int, float)
            if valid and typ is not bool:
                upper = 2**63 if key in BYTE_CAPACITY_KEYS else 2**31
                valid = lower <= value < upper and math.isfinite(value)
            if key.endswith('mem_fraction_static'):
                valid = valid and 0 < value < 1
            if key.endswith('prefill_coalesce_wait_ms'):
                valid = valid and value > 0
            if key.endswith('prefill_coalesce_requests'):
                valid = valid and value != 1
            if not valid:
                errors.append(error('execution_knob_type_or_range', p, f'Requires {typ.__name__} in its documented upstream range.', key))
    except (ValueError, UnicodeError, RecursionError, yaml.YAMLError) as exc:
        errors.append(error('candidate_config_schema', p, 'Invalid YAML shape, encoding, type or size.', hint='Use a flat mapping; {} restores defaults.'))
    if isinstance(cfg, dict) and 'asr.engine.mem_fraction_static' in cfg and 'asr.engine.kv_cache_bytes' in cfg:
        errors.append(error('conflicting_kv_capacity', p, 'Set either memory fraction or explicit KV capacity, not both.', 'asr.engine'))
    if errors:
        raise CandidateInvalid(errors)
    return cfg
