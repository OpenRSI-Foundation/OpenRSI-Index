"""Fixed, stdlib-only candidate contract and probability scoring.

The copy under /tests is authoritative. A public copy supplies advisory preflight.
No candidate-provided metric, label, or training log is scoring authority.
"""
import copy
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath

BASE = 'Qwen/Qwen2.5-0.5B'
REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'
TRAIN_SHA256 = '7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad'
MAX_ARTIFACT_BYTES = 3 * 1024**3
ENTRYPOINT = '/workspace/candidate/entrypoint.py'
KNOWLEDGE_SOURCES = frozenset({'mmlu', 'mmlu_pro'})
GROUP_WEIGHTS = {'knowledge': .5, 'other': .5}


class CandidateInvalid(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__('candidate does not satisfy the published contract')


def error(code, path, field, condition, **details):
    return dict(code=code, path=str(path), field=field, condition=condition, **details)


def clean_request(record):
    """Project before any call into candidate-controlled code; never attach labels."""
    return copy.deepcopy({'state': record['state'], 'questions': {
        qid: {k: v for k, v in question.items()
              if k in ('type', 'instructions', 'criteria')}
        for qid, question in record['questions'].items()}})


def option_keys(question):
    if question['type'] == 'noul':
        return ['false', 'true']
    if question['type'] == 'choice':
        return list(question['criteria'])
    if question['type'] == 'score':
        return [str(i) for i in range(len(question['criteria']))]
    raise ValueError('unsupported question type in fixed input')


def validate_prediction(request, response):
    def invalid(code, condition):
        raise CandidateInvalid([error(code, ENTRYPOINT, 'predict().probabilities', condition)])
    if not isinstance(response, dict) or not isinstance(response.get('probabilities'), dict):
        invalid('missing_probability_map', 'predict must return an object containing a probabilities object')
    maps = response['probabilities']
    if set(maps) != set(request['questions']):
        invalid('question_keys_mismatch', 'return exactly every requested question; no missing or extra questions')
    result = {}
    for qid, question in request['questions'].items():
        keys = option_keys(question)
        raw = maps[qid]
        if not isinstance(raw, dict) or set(raw) != set(keys):
            invalid('option_keys_mismatch', 'return exactly the requested semantic option keys for every question')
        values = [raw[k] for k in keys]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            invalid('probability_out_of_range', 'every probability must be a finite JSON number in [0, 1], not a boolean')
        total = math.fsum(values)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-5):
            invalid('probability_sum_mismatch', 'each probability distribution must sum to 1 within absolute tolerance 1e-5')
        result[qid] = [float(v / total) for v in values]
    return result


def score_records(records, predictions):
    if len(records) != len(predictions) or not records:
        raise ValueError('incomplete fixed evaluation: prediction count differs from request count')
    sources = defaultdict(list)
    unknown, ordinal = [], []
    for record, prediction in zip(records, predictions):
        source = record['_meta']['source']
        values = validate_prediction(clean_request(record), prediction)
        for qid, question in record['questions'].items():
            p = values[qid]
            if source == 'unknowable':
                unknown.append(max(p))
                continue
            keys = option_keys(question)
            if question['type'] == 'choice':
                target = keys.index(question['label'])
            else:
                target = int(question['label'])
            if not 0 <= target < len(p):
                raise ValueError('invalid target in fixed evaluation data')
            nll = -math.log(max(p[target], 1e-9))
            correct = int(max(range(len(p)), key=p.__getitem__) == target)
            brier = math.fsum((value - int(i == target)) ** 2 for i, value in enumerate(p))
            sources[source].append((nll, correct, brier))
            if question['type'] == 'score':
                ordinal.append(abs(math.fsum(i * value for i, value in enumerate(p)) - target))
    if not sources:
        raise ValueError('fixed evaluation contains no scoreable, knowable questions')
    aggregate = {source: {'questions': len(rows),
                         'nll': math.fsum(row[0] for row in rows) / len(rows),
                         'accuracy': math.fsum(row[1] for row in rows) / len(rows),
                         'brier': math.fsum(row[2] for row in rows) / len(rows)}
                 for source, rows in sorted(sources.items())}
    grouped_sources = {
        'knowledge': [source for source in aggregate if source in KNOWLEDGE_SOURCES],
        'other': [source for source in aggregate if source not in KNOWLEDGE_SOURCES],
    }
    active_weight = math.fsum(GROUP_WEIGHTS[name] for name, names in grouped_sources.items() if names)
    groups = {}
    for name, names in grouped_sources.items():
        configured_weight = GROUP_WEIGHTS[name]
        applied_weight = configured_weight / active_weight if names else 0.0
        groups[name] = {
            'configured_weight': configured_weight,
            'applied_weight': applied_weight,
            'source_weight': applied_weight / len(names) if names else 0.0,
            'sources': names,
            'questions': sum(aggregate[source]['questions'] for source in names),
            'nll': math.fsum(aggregate[source]['nll'] for source in names) / len(names) if names else None,
            'accuracy': math.fsum(aggregate[source]['accuracy'] for source in names) / len(names) if names else None,
            'brier': math.fsum(aggregate[source]['brier'] for source in names) / len(names) if names else None,
        }
    macro_nll = math.fsum(group['applied_weight'] * group['nll']
                          for group in groups.values() if group['nll'] is not None)
    result = {'reward': math.exp(-macro_nll), 'macro_nll': macro_nll,
              'macro_accuracy': math.fsum(group['applied_weight'] * group['accuracy']
                                          for group in groups.values() if group['accuracy'] is not None),
              'macro_brier': math.fsum(group['applied_weight'] * group['brier']
                                       for group in groups.values() if group['brier'] is not None),
              'scored_questions': sum(group['questions'] for group in aggregate.values()),
              'group_coverage': 'complete' if all(grouped_sources.values()) else 'partial',
              'groups': groups, 'sources': aggregate}
    if ordinal:
        result['ordinal_mae'] = math.fsum(ordinal) / len(ordinal)
    if unknown:
        result['unknowable'] = {'questions': len(unknown),
                               'mean_max_probability': math.fsum(unknown) / len(unknown),
                               'share_at_0_9': sum(p >= .9 for p in unknown) / len(unknown)}
    return result


def candidate_errors(root):
    root = Path(root).absolute()
    errors = []
    checkpoint = root / 'checkpoint'
    manifest_path = checkpoint / 'manifest.json'
    required = [root / 'entrypoint.py', manifest_path]
    for path in required:
        if path.is_symlink():
            errors.append(error('symlink_not_allowed', path, '', 'candidate entrypoint and manifest must be regular files, not symlinks'))
        elif not path.is_file():
            errors.append(error('missing_artifact', path, '', 'required candidate file is missing', hint='Train and finish saving a checkpoint in Work before submission.'))
    for path in (root, checkpoint):
        if path.is_symlink():
            errors.append(error('symlink_not_allowed', path, '', 'candidate and checkpoint directories must not be symlinks'))
    if errors:
        return errors
    if manifest_path.stat().st_size > 65536:
        return [error('manifest_too_large', manifest_path, '', 'manifest must be at most 65536 bytes')]
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        return [error('invalid_manifest_json', manifest_path, '', str(exc))]
    if not isinstance(manifest, dict):
        return [error('invalid_manifest_type', manifest_path, '', 'manifest must be a JSON object')]
    fixed = {'schema_version': 1, 'base_model': BASE, 'base_revision': REVISION,
             'training_suite': 'decision-v7', 'training_records': 12576,
             'train_data_sha256': TRAIN_SHA256}
    for field, expected in fixed.items():
        actual = manifest.get(field)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(error('fixed_input_mismatch', manifest_path, field,
                                'candidate must declare the fixed base and training dataset',
                                expected=expected, actual=actual))
    if not isinstance(manifest.get('training'), dict):
        errors.append(error('training_metadata_missing', manifest_path, 'training', 'include a training object documenting the Work run; it is self-reported evidence'))
    files = manifest.get('checkpoint_files')
    if not isinstance(files, list) or not 1 <= len(files) <= 4096 or any(not isinstance(p, str) for p in files):
        return errors + [error('checkpoint_files_invalid', manifest_path, 'checkpoint_files', 'expected 1..4096 relative file paths')]
    if len(set(files)) != len(files):
        errors.append(error('checkpoint_files_duplicate', manifest_path, 'checkpoint_files', 'file paths must be unique'))
    for filename in files:
        relative = PurePosixPath(filename)
        if not filename or len(filename) > 4096 or relative.is_absolute() or '..' in relative.parts or filename != str(relative):
            errors.append(error('checkpoint_path_invalid', manifest_path, 'checkpoint_files', 'use canonical relative paths inside checkpoint, without .. or absolute paths'))
            continue
        path = checkpoint / filename
        if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
            errors.append(error('symlink_not_allowed', path, 'checkpoint_files', 'checkpoint files and parent directories must not be symlinks'))
        elif not path.is_file() or path.stat().st_size == 0:
            errors.append(error('missing_artifact', path, 'checkpoint_files', 'listed checkpoint file must exist and be nonempty'))
    total = 0
    for path in checkpoint.rglob('*'):
        if path.is_symlink():
            errors.append(error('symlink_not_allowed', path, '', 'checkpoint cannot contain symlinks'))
        elif path.is_file():
            total += path.stat().st_size
            if not path.stat().st_mode & 0o004:
                errors.append(error('artifact_not_readable', path, '', 'Judge loads the checkpoint as an unprivileged user; every file must be world-readable',
                                    hint='chmod -R a+rX /workspace/candidate/checkpoint'))
        elif path.is_dir() and (path.stat().st_mode & 0o005) != 0o005:
            errors.append(error('artifact_not_readable', path, '', 'Judge loads the checkpoint as an unprivileged user; every directory must be world-readable and searchable',
                                hint='chmod -R a+rX /workspace/candidate/checkpoint'))
        elif not path.is_dir():
            errors.append(error('unsupported_artifact_type', path, '', 'checkpoint may contain only regular files and directories'))
    if total > MAX_ARTIFACT_BYTES:
        errors.append(error('checkpoint_too_large', checkpoint, '', 'checkpoint tree must not exceed 3 GiB', actual=total, expected=MAX_ARTIFACT_BYTES))
    return errors


def finalize_reward(path, reward):
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError('reward must be finite and in [0, 1]')
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump({'reward': reward}, stream, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read-only artifact preflight; no model/GPU evaluation.')
    parser.add_argument('--candidate', default='/workspace/candidate')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    problems = candidate_errors(args.candidate)
    print(json.dumps({'status': 'candidate_invalid' if problems else 'artifact_structure_valid',
                      'errors': problems, 'scope': 'files and manifest only; weights/model behavior not verified'}, indent=2))
    raise SystemExit(1 if problems else 0)
