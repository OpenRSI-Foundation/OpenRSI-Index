"""Trusted process wrapper. Reads clean requests from stdin; never opens labels."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox import drop_to_candidate, restrict_filesystem


def load_candidate_module(path):
    spec = importlib.util.spec_from_file_location('candidate_entrypoint', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def failure(exc, phase):
    # During prediction even dynamic exception names and compiled filenames can
    # contain reserved text. Only a fixed error category may cross that boundary.
    if phase == 'predict':
        code = ('candidate_memory_error' if isinstance(exc, MemoryError) else
                'candidate_runtime_error' if isinstance(exc, RuntimeError) else
                'candidate_value_error' if isinstance(exc, (ValueError, TypeError, KeyError, IndexError)) else
                'candidate_execution_error')
        return {'phase': 'predict', 'code': code}
    details = {'exception': type(exc).__name__, 'phase': phase,
               'frames': [{'path': frame.filename, 'line': frame.lineno, 'function': frame.name}
                          for frame in traceback.extract_tb(exc.__traceback__)]}
    details['message'] = str(exc)
    return details


def candidate_failure(exc, phase):
    return {'status': 'candidate_error', **failure(exc, phase)}


def main():
    scratch = Path(sys.argv[1])
    candidate = Path('/workspace/candidate')
    protocol = os.fdopen(os.dup(1), 'w', buffering=1)

    def send(value):
        protocol.write(json.dumps(value, allow_nan=False) + '\n')
        protocol.flush()

    try:
        # The script and trusted imports are already loaded before /tests becomes
        # inaccessible. Child-created processes inherit this restriction.
        # CUDA initialization writes to the process's own /proc entry (a read-only
        # rule there makes cudaGetDeviceCount fail with error 304). Only this
        # process's entry is writable; the label-holding parent stays invisible.
        restrict_filesystem(
            [candidate, '/opt', '/usr', '/lib', '/lib64', '/sys',
             '/etc/ld.so.cache', '/etc/passwd', '/etc/group', '/etc/nsswitch.conf',
             '/proc/cpuinfo', '/proc/meminfo', '/proc/stat', '/proc/uptime'],
            [scratch, '/dev', f'/proc/{os.getpid()}'])
        drop_to_candidate()
        os.chdir(scratch)
    except Exception as exc:
        send({'status': 'infrastructure_error', **failure(exc, 'isolation')})
        return
    # Before receiving any hidden request, stdout/stderr go to the parent's
    # already-open anonymous load log. Parent never follows a child-owned path.
    os.dup2(2, 1)
    try:
        import torch
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.manual_seed(0)
        probe = torch.ones(1, device='cuda:0')
        torch.cuda.synchronize()
        del probe
        torch.cuda.empty_cache()
    except Exception as exc:
        send({'status': 'infrastructure_error', **failure(exc, 'runtime')})
        return
    phase = 'load'
    try:
        sys.path.insert(0, str(candidate))
        module = load_candidate_module(candidate / 'entrypoint.py')
        predictor = module.load(str(candidate / 'checkpoint'), 'cuda:0')
        if not isinstance(predictor.model, torch.nn.Module):
            raise TypeError('entrypoint.load must return an object with a torch.nn.Module in .model')
        parameters = sum(parameter.numel() for parameter in predictor.model.parameters())
        if parameters > 600_000_000:
            send({'status': 'candidate_invalid', 'code': 'parameter_budget_exceeded',
                  'condition': 'loaded model must have at most 600000000 registered parameters',
                  'actual': parameters})
            return
        predictor.model.eval()
        torch.cuda.synchronize()
        sys.stdout.flush()
        sys.stderr.flush()
        # All later raw output may contain held-out inputs: do not expose it.
        with open(os.devnull, 'w') as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        send({'status': 'ready', 'parameters': parameters})
        phase = 'predict'
        for line in sys.stdin:
            request = json.loads(line)
            with torch.inference_mode():
                prediction = predictor.predict(request)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            if peak > 16 * 1024**3:
                send({'status': 'candidate_invalid', 'code': 'inference_memory_exceeded',
                      'condition': 'peak Torch-allocated inference memory must not exceed 16 GiB', 'actual': peak})
                return
            send({'status': 'prediction', 'response': prediction, 'peak_memory_bytes': peak})
    except Exception as exc:
        sys.stdout.flush()
        sys.stderr.flush()
        send(candidate_failure(exc, phase))


if __name__ == '__main__':
    main()
